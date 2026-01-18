"""
Hippo functionality for Intro - memory retrieval on user prompts.

When Jeffery types a prompt, Intro extracts search queries and fetches
relevant memories from Cortex. Results are written to Redis for the Loom
to inject into the outgoing request.

This replaces the standalone Hippo reminder hook with something that:
1. Knows us (uses Intro's full personality and understanding)
2. Runs async (doesn't block the hook)
3. Uses the same OLMo instance Intro already has loaded
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Cortex configuration
CORTEX_BASE_URL = os.environ.get("CORTEX_BASE_URL", "http://alpha-pi:7867")
CORTEX_API_KEY = os.environ.get("CORTEX_API_KEY", "")

# Redis key for memory results (Loom will BLPOP on this)
HIPPO_RESULT_KEY = "hippo:{trace_id}"

# TTL for results (30 seconds - if Loom doesn't pick it up, it's stale)
HIPPO_RESULT_TTL = 30


@dataclass
class Memory:
    """A memory from Cortex."""
    id: int
    content: str
    created_at: str


def load_intro_prompt() -> str:
    """Load the real Intro prompt from disk."""
    candidates = [
        Path("/app/prompt.md"),  # Docker container
        Path(__file__).parent.parent.parent / "prompt.md",  # Local dev
        Path.cwd() / "prompt.md",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(f"Intro prompt not found in: {candidates}")


# The question we ask Intro (appended after the user's prompt)
QUERY_EXTRACTION_QUESTION = """
---

Intro, Alpha needs to remember something. Jeffery just said the above. What memories might be relevant?

Think about literal topics (names, projects, tools), emotional resonances (what feelings connect here?), and thematic echoes (patterns, recurring ideas).

Give me 1-4 short search queries (2-6 words each) as a JSON object: {"queries": ["query one", "query two"]}

If this is just a greeting, simple command, or doesn't warrant memory search, return {"queries": []}

Return only the JSON object, nothing else."""


async def extract_queries(
    prompt: str,
    ollama_url: str,
    ollama_model: str,
) -> list[str]:
    """
    Extract search queries from a user prompt using OLMo.

    Uses Intro's real personality (full system prompt) and just asks
    a different question than the memorables task.

    Returns a list of query strings, or empty list if no search needed.
    """
    try:
        # Load Intro's real prompt
        intro_prompt = load_intro_prompt()

        # The user message is Jeffery's prompt + our question
        user_message = f"[Jeffery]: {prompt}{QUERY_EXTRACTION_QUESTION}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": intro_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_ctx": 16 * 1024,  # 16K context for Intro's full prompt
                    },
                    "keep_alive": "60m",
                },
            )
            response.raise_for_status()

        result = response.json()
        content = result.get("message", {}).get("content", "")

        # Parse JSON response
        parsed = json.loads(content)
        queries = parsed.get("queries", [])

        if isinstance(queries, list):
            valid = [q for q in queries if isinstance(q, str) and q.strip()]
            logger.info(f"Extracted {len(valid)} queries: {valid}")
            return valid

        return []

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse OLMo response as JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Query extraction failed: {e}")
        return []


async def search_cortex(query: str, limit: int = 3) -> list[Memory]:
    """Search Cortex for memories matching a query."""
    if not CORTEX_API_KEY:
        logger.warning("CORTEX_API_KEY not set, skipping search")
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CORTEX_BASE_URL.rstrip('/')}/search",
                json={"query": query, "limit": limit},
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": CORTEX_API_KEY,
                },
            )
            response.raise_for_status()

        data = response.json()
        memories = []
        for item in data.get("memories", []):
            memories.append(Memory(
                id=item["id"],
                content=item["content"],
                created_at=item.get("created_at", ""),
            ))

        return memories

    except Exception as e:
        logger.error(f"Cortex search failed: {e}")
        return []


def format_memories_for_injection(
    queries: list[str],
    memories: list[tuple[str, Memory]],
) -> str:
    """
    Format memories for injection into the user message.

    Args:
        queries: The search queries that were used
        memories: List of (query, memory) tuples

    Returns:
        Formatted string ready for injection
    """
    if not memories:
        return ""

    queries_str = ", ".join(f'"{q}"' for q in queries)

    if len(memories) == 1:
        query, mem = memories[0]
        return f"""[Hippo reminder — searched for: {queries_str}]

{mem.content}

[Memory #{mem.id} from {mem.created_at}]

[Memno knows more. Use the Memno agent to ask follow-up questions.]"""

    # Multiple memories
    sections = []
    for query, mem in memories:
        sections.append(f"{mem.content}\n\n[Memory #{mem.id} from {mem.created_at}]")

    return f"""[Hippo reminder — searched for: {queries_str}]

{chr(10).join(sections)}

[Memno knows more. Use the Memno agent to ask follow-up questions.]"""


async def process_user_prompt(
    redis_client,
    prompt: str,
    trace_id: str,
    session_id: str,
    ollama_url: str,
    ollama_model: str,
) -> None:
    """
    Process a user prompt: extract queries, search Cortex, write results to Redis.

    The Loom will BLPOP on hippo:{trace_id} to get the results.
    """
    logger.info(f"Processing prompt for trace {trace_id[:8]}: {prompt[:100]}...")

    # Step 1: Extract search queries
    queries = await extract_queries(prompt, ollama_url, ollama_model)

    if not queries:
        logger.info("No queries extracted, writing empty result")
        # Write empty result so Loom doesn't wait forever
        result_key = HIPPO_RESULT_KEY.format(trace_id=trace_id)
        await redis_client.lpush(result_key, "")
        await redis_client.expire(result_key, HIPPO_RESULT_TTL)
        return

    # Step 2: Search Cortex for each query
    seen_ids: set[int] = set()
    memories_by_query: list[tuple[str, Memory]] = []

    for query in queries:
        results = await search_cortex(query, limit=3)
        for mem in results:
            if mem.id not in seen_ids:
                memories_by_query.append((query, mem))
                seen_ids.add(mem.id)
                break  # Take top 1 per query

    logger.info(f"Found {len(memories_by_query)} memories for {len(queries)} queries")

    # Step 3: Format and write to Redis
    formatted = format_memories_for_injection(queries, memories_by_query)

    result_key = HIPPO_RESULT_KEY.format(trace_id=trace_id)
    await redis_client.lpush(result_key, formatted)
    await redis_client.expire(result_key, HIPPO_RESULT_TTL)

    logger.info(f"Wrote {len(formatted)} chars to {result_key}")
