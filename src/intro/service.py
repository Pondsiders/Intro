"""
IntroService - the brain of Intro.

Manages conversation state, handles prompt processing, and extracts memorables.
"""

import asyncio
import json
import logging
from pathlib import Path

from .models import MemoryItem, TurnMessage
from . import ollama, cortex

logger = logging.getLogger(__name__)


def load_prompt() -> str:
    """Load the Intro prompt from various possible locations."""
    candidates = [
        Path("/app/prompt.md"),  # Docker container
        Path(__file__).parent.parent.parent / "prompt.md",  # Local dev
        Path.cwd() / "prompt.md",
    ]
    for path in candidates:
        if path.exists():
            text = path.read_text()
            logger.info(f"Loaded prompt from {path} ({len(text)} chars)")
            return text
    raise FileNotFoundError(f"Prompt not found in any of: {candidates}")


# Query extraction prompt (appended after user's message)
QUERY_EXTRACTION_QUESTION = """
---

Intro, Alpha needs to remember something. Jeffery just said the above. What memories might be relevant?

Think about literal topics (names, projects, tools), emotional resonances (what feelings connect here?), and thematic echoes (patterns, recurring ideas).

Give me 1-4 short search queries (2-6 words each) as a JSON object: {"queries": ["query one", "query two"]}

If this is just a greeting, simple command, or doesn't warrant memory search, return {"queries": []}

Return only the JSON object, nothing else."""


# First turn memorables prompt
FIRST_TURN_PROMPT = """<conversation>
{turn_content}
</conversation>

<question>
Intro, what in the <conversation></conversation> block has been memorable so far? Write all the memorable things as a Markdown list (one per line, starting with `-`). If nothing has been memorable, just say "Nothing notable."
</question>
"""


# Follow-up turn memorables prompt
FOLLOWUP_TURN_PROMPT = """<conversation>
{turn_content}
</conversation>

<question>
Consider the whole conversation so far—everything in the <conversation></conversation> blocks, not just what's new. Step back and look at the full shape of it. What matters? What would Alpha want to carry forward? What would hurt to lose?

Output a fresh list of memorables. Don't just add to your previous list—reconsider from scratch given everything you've seen.
</question>
"""


def parse_memorables(output: str) -> list[str]:
    """Parse markdown list output into list of strings."""
    memorables = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line or line.lower().startswith("nothing notable"):
            continue
        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("-"):
            line = line[1:].strip()
        if line:
            memorables.append(line)
    return memorables


class IntroService:
    """
    Stateful service managing Intro's operations.

    State is kept in-memory for speed. Redis is optional for persistence.
    """

    def __init__(self):
        # In-memory state
        self.ollama_history: dict[str, list[dict]] = {}  # session_id → OLMo conversation
        self.seen_memories: dict[str, set[int]] = {}  # session_id → seen memory IDs
        self.memorables: dict[str, list[str]] = {}  # session_id → current memorables

        # Load the prompt once
        self.system_prompt = load_prompt()

        # Optional Redis client (set by app.py if available)
        self.redis = None

    async def prompt(self, message: str, session_id: str) -> tuple[list[MemoryItem], list[str]]:
        """
        Process a user prompt: extract queries, search Cortex, return memories.

        This is the synchronous "what sounds familiar?" operation.

        Returns:
            (list of memories, list of queries used)
        """
        logger.info(f"Processing prompt for session {session_id[:8]}: {message[:100]}...")

        # Extract search queries
        queries = await self._extract_queries(message)

        if not queries:
            logger.info("No queries extracted")
            return [], []

        # Search Cortex with deduplication
        seen = self.seen_memories.get(session_id, set())
        results, new_seen = await cortex.search_with_dedup(queries, seen)
        self.seen_memories[session_id] = new_seen

        memories = [mem for _, mem in results]
        logger.info(f"Found {len(memories)} memories for {len(queries)} queries")

        return memories, queries

    async def stop(self, session_id: str, turn: list[TurnMessage]) -> None:
        """
        Process a stop event: extract memorables from this turn.

        This is the async "what's memorable?" operation. Kicks off processing
        in the background and returns immediately.

        Args:
            session_id: The session identifier
            turn: This turn's messages (user + assistant)
        """
        logger.info(f"Stop event for session {session_id[:8]} with {len(turn)} messages")

        # Fire and forget
        asyncio.create_task(self._process_memorables(session_id, turn))

    async def clear(self, session_id: str) -> None:
        """
        Clear all state for a session.

        Called when Alpha stores a memory (fresh start) or on SessionStart.
        """
        self.ollama_history.pop(session_id, None)
        self.seen_memories.pop(session_id, None)
        self.memorables.pop(session_id, None)
        logger.info(f"Cleared state for session {session_id[:8]}")

    def get_memorables(self, session_id: str) -> list[str]:
        """Get current memorables for a session (for Loom to inject)."""
        return self.memorables.get(session_id, [])

    async def _extract_queries(self, message: str) -> list[str]:
        """Extract search queries from a user message using OLMo."""
        user_content = f"[Jeffery]: {message}{QUERY_EXTRACTION_QUESTION}"

        try:
            output, _ = await ollama.chat(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=self.system_prompt,
                operation="extract_queries",
                context_size=16 * 1024,
                json_format=True,
            )

            parsed = json.loads(output)
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

    async def _process_memorables(self, session_id: str, turn: list[TurnMessage]) -> None:
        """Process a turn to extract memorables."""
        # Format turn content
        turn_content = "\n\n".join(
            f"[{'Alpha' if m.role == 'assistant' else 'Jeffery'}]: {m.content}"
            for m in turn
        )

        # Get existing OLMo conversation history
        history = self.ollama_history.get(session_id, [])
        is_first_turn = len(history) == 0

        if is_first_turn:
            user_msg = FIRST_TURN_PROMPT.format(turn_content=turn_content)
            messages = [{"role": "user", "content": user_msg}]
            logger.info(f"First turn for session {session_id[:8]}")
        else:
            user_msg = FOLLOWUP_TURN_PROMPT.format(turn_content=turn_content)
            messages = history + [{"role": "user", "content": user_msg}]
            logger.info(f"Turn {len(history)//2 + 1} for session {session_id[:8]}")

        try:
            output, _ = await ollama.chat(
                messages=messages,
                system_prompt=self.system_prompt,
                operation="notice",
            )
        except Exception:
            return  # Error already logged

        memorables = parse_memorables(output)
        logger.info(f"OLMo returned {len(memorables)} memorables")

        # Update history
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": output})
        self.ollama_history[session_id] = history

        # Store memorables
        self.memorables[session_id] = memorables

        # Optionally persist to Redis for Loom to read
        if self.redis:
            mem_key = f"intro:memorables:{session_id}"
            await self.redis.delete(mem_key)
            if memorables:
                await self.redis.rpush(mem_key, *memorables)
                await self.redis.expire(mem_key, 60 * 60 * 24)  # 24h TTL

        logger.info(f"Stored {len(memorables)} memorables for session {session_id[:8]}")
