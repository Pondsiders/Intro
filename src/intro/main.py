"""
Intro - Alpha's metacognitive layer.

Subscribes to transcript pubsub, buffers conversation, calls OLMo to notice
what's memorable, writes results to Redis for the Loom to inject.

Also handles memory retrieval on user prompts (the Hippo job) - extracting
search queries and fetching relevant memories from Cortex.

Architecture:
    transcript:* (pubsub) → conversation buffer → OLMo → memorables buffer
    events:user_prompt_submit → OLMo (queries) → Cortex → hippo:{trace_id}
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from pondside.telemetry import init

from .hippo import process_user_prompt

# Initialize telemetry
init("intro")
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Configuration from environment
REDIS_URL = os.environ.get("REDIS_URL")
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "olmo3:7b")

# Load the prompt - check multiple locations for dev vs container
def load_prompt() -> str:
    """Load the Intro prompt from various possible locations."""
    candidates = [
        Path("/app/prompt.md"),  # Docker container
        Path(__file__).parent.parent.parent / "prompt.md",  # Local dev (src/intro/main.py -> prompt.md)
        Path.cwd() / "prompt.md",  # Current working directory
    ]
    for path in candidates:
        if path.exists():
            text = path.read_text()
            logger.info(f"Loaded prompt from {path} ({len(text)} chars)")
            return text
    raise FileNotFoundError(f"Prompt not found in any of: {candidates}")

INTRO_PROMPT = load_prompt()

# Redis key prefixes
CONVERSATION_KEY = "intro:conversation:{session_id}"
MEMORABLES_KEY = "intro:memorables:{session_id}"

# Buffer TTL (24 hours) - safety valve if session dies without storing
BUFFER_TTL = 60 * 60 * 24

# Max characters to send to OLMo (rough proxy for tokens - ~4 chars/token)
# OLMo has 24K context, we want headroom for system prompt + output
# 16K tokens * 4 chars = 64K chars, but let's be conservative
MAX_CONVERSATION_CHARS = 40000

# Cortex store notification channel pattern
CORTEX_STORED_PATTERN = "cortex:stored:*"

# Events channel pattern (for Stop hook signals)
EVENTS_PATTERN = "events:*"

# Compaction summary prefix (filter these out of conversation buffer)
COMPACTION_PREFIX = "This session is being continued from a previous conversation"

# Dedupe prompt for second-pass consolidation
DEDUPE_PROMPT = """Here are the observations you've made so far:

{memorables}

---

Merge any items that describe essentially the same moment. Keep distinct observations separate—don't combine different things just to reduce the count. Output the deduplicated list in the same format (Markdown list with `-` prefix).
"""


def extract_text_content(raw: dict) -> str | None:
    """Extract text content from a transcript message."""
    message = raw.get("message", {})
    content = message.get("content", [])

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts) if texts else None

    return None


def parse_memorables(output: str) -> list[str]:
    """Parse markdown list output into list of strings."""
    memorables = []
    for line in output.strip().split("\n"):
        line = line.strip()
        # Skip empty lines and "nothing notable"
        if not line or line.lower().startswith("nothing notable"):
            continue
        # Strip leading "- " from markdown list items
        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("-"):
            line = line[1:].strip()
        if line:
            memorables.append(line)
    return memorables


async def call_ollama_chat(
    messages: list[dict],
    operation: str = "notice",
) -> tuple[str, dict]:
    """
    Call OLMo with a multi-turn conversation.

    Returns (output_text, token_counts).
    """
    with tracer.start_as_current_span(
        f"llm.{OLLAMA_MODEL}",
        kind=trace.SpanKind.CLIENT,
    ) as span:
        # Core attributes
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", OLLAMA_MODEL)
        span.set_attribute("metadata.source", "intro")

        # gen_ai attributes for Parallax routing
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", OLLAMA_MODEL)
        span.set_attribute("gen_ai.operation.name", operation)

        # Input attributes (OpenInference format) - log all messages
        for i, msg in enumerate(messages):
            span.set_attribute(f"llm.input_messages.{i}.message.role", msg["role"])
            span.set_attribute(f"llm.input_messages.{i}.message.content", msg["content"])

        # Also set input.value for Phoenix
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        if last_user:
            span.set_attribute("input.value", last_user["content"])

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_ctx": 24 * 1024,  # 24K context window
                        },
                        "keep_alive": "60m",  # Keep model loaded for 60 minutes
                    },
                )
                response.raise_for_status()

            result = response.json()
            output = result.get("message", {}).get("content", "")

            # Output attributes
            span.set_attribute("output.value", output)
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute("llm.output_messages.0.message.content", output)

            # Token counts
            prompt_tokens = result.get("prompt_eval_count", 0)
            completion_tokens = result.get("eval_count", 0)
            span.set_attribute("llm.token_count.prompt", prompt_tokens)
            span.set_attribute("llm.token_count.completion", completion_tokens)
            span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)

            span.set_status(Status(StatusCode.OK))

            return output, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        except Exception as e:
            logger.error(f"OLMo call failed: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


async def process_turn(redis_client: aioredis.Redis, session_id: str, conversation: str):
    """
    Process a conversation turn: notice memorables, then dedupe.

    This is the main Intro flow:
    1. Call OLMo to notice what's memorable
    2. Get existing memorables from Redis
    3. If there are multiple items, call OLMo again to dedupe
    4. Write final list back to Redis
    """
    with tracer.start_as_current_span(
        "intro.process_turn",
        attributes={
            "session_id": session_id[:8],
            "metadata.source": "intro",
        }
    ) as parent_span:

        # === Turn 1: Notice what's memorable ===
        logger.info(f"Calling Intro to notice memorables")

        transcript_prompt = f"""{conversation}

---

Intro, what about this conversation is worth remembering? List the memorable moments as a simple Markdown list (one per line, starting with `-`). If nothing stands out, just say "Nothing notable."
"""

        # Reload prompt fresh each time for hot-reload during development
        current_prompt = load_prompt()

        messages = [
            {"role": "system", "content": current_prompt},
            {"role": "user", "content": transcript_prompt},
        ]

        try:
            turn1_output, turn1_tokens = await call_ollama_chat(messages, operation="notice")
        except Exception:
            return  # Error already logged

        new_memorables = parse_memorables(turn1_output)
        logger.info(f"Turn 1: noticed {len(new_memorables)} memorables")

        if not new_memorables:
            logger.info("Nothing notable this turn")
            return

        # === Get existing memorables ===
        mem_key = MEMORABLES_KEY.format(session_id=session_id)
        existing = await redis_client.lrange(mem_key, 0, -1)

        # Combine existing + new
        all_memorables = existing + new_memorables

        # === Turn 2: Dedupe if we have multiple items ===
        if len(all_memorables) > 1:
            logger.info(f"Deduping {len(all_memorables)} memorables")

            # Format as markdown list
            memorables_md = "\n".join(f"- {m}" for m in all_memorables)
            dedupe_user_msg = DEDUPE_PROMPT.format(memorables=memorables_md)

            # Continue the conversation
            messages.append({"role": "assistant", "content": turn1_output})
            messages.append({"role": "user", "content": dedupe_user_msg})

            try:
                turn2_output, turn2_tokens = await call_ollama_chat(messages, operation="dedupe")
                final_memorables = parse_memorables(turn2_output)
                logger.info(f"Turn 2: deduped to {len(final_memorables)} memorables")
            except Exception:
                # If dedupe fails, just use the combined list
                final_memorables = all_memorables
        else:
            final_memorables = all_memorables

        # === Write final list to Redis ===
        # Delete existing and write fresh
        await redis_client.delete(mem_key)
        if final_memorables:
            await redis_client.rpush(mem_key, *final_memorables)
            await redis_client.expire(mem_key, BUFFER_TTL)

        parent_span.set_attribute("memorables.count", len(final_memorables))
        logger.info(f"Stored {len(final_memorables)} memorables for session {session_id[:8]}")


async def handle_cortex_store(redis_client: aioredis.Redis, channel: str, memory_id: str):
    """Handle a cortex:stored:{session_id} event by clearing buffers.

    When Alpha stores a memory, she's already captured what mattered from
    the recent conversation. We clear our buffers and start fresh.
    """
    # Extract session_id from channel name (cortex:stored:{session_id})
    parts = channel.split(":")
    if len(parts) != 3:
        logger.warning(f"Unexpected channel format: {channel}")
        return

    session_id = parts[2]

    # Clear both conversation and memorables buffers for this session
    conv_key = CONVERSATION_KEY.format(session_id=session_id)
    mem_key = MEMORABLES_KEY.format(session_id=session_id)

    deleted = await redis_client.delete(conv_key, mem_key)
    logger.info(f"Cortex store event: cleared {deleted} keys for session {session_id[:8]} (memory_id={memory_id})")


async def process_message(redis_client: aioredis.Redis, data: dict):
    """Process a single transcript message - buffer only, don't trigger introspection.

    Introspection is triggered by Stop events, not by individual messages.
    This prevents hammering OLMo on every assistant message during rapid tool use.
    """
    session_id = data.get("session_id")
    role = data.get("role")

    # Only care about user/assistant messages
    if role not in ("user", "assistant"):
        return

    # Extract text content
    text = extract_text_content(data.get("raw", {}))
    if not text:
        return

    # Filter out compaction summaries - they're not real conversation
    if text.startswith(COMPACTION_PREFIX):
        logger.info(f"Skipping compaction summary for session {session_id[:8]}")
        return

    # Format for the buffer - use names, not roles
    name = "Alpha" if role == "assistant" else "Jeffery"
    formatted = f"[{name}]: {text}"

    # Add to conversation buffer
    conv_key = CONVERSATION_KEY.format(session_id=session_id)
    await redis_client.rpush(conv_key, formatted)
    await redis_client.expire(conv_key, BUFFER_TTL)

    logger.debug(f"Buffered {role} message for session {session_id[:8]}")


async def handle_stop_event(redis_client: aioredis.Redis, session_id: str):
    """Handle a Stop event - this is when we trigger introspection.

    The Stop hook fires when the assistant is done responding (no more tool calls).
    This is the reliable "end of turn" signal.
    """
    conv_key = CONVERSATION_KEY.format(session_id=session_id)

    # Get conversation buffer
    conversation_lines = await redis_client.lrange(conv_key, 0, -1)
    if not conversation_lines:
        logger.debug(f"Stop event but no conversation to process for session {session_id[:8]}")
        return

    conversation = "\n\n".join(conversation_lines)

    # Truncate if too long (take most recent content)
    if len(conversation) > MAX_CONVERSATION_CHARS:
        logger.warning(f"Conversation too long ({len(conversation)} chars), truncating to {MAX_CONVERSATION_CHARS}")
        conversation = conversation[-MAX_CONVERSATION_CHARS:]
        # Find first complete message boundary (look for "\n\n[")
        boundary = conversation.find("\n\n[")
        if boundary > 0:
            conversation = conversation[boundary + 2:]  # Skip the \n\n

    logger.info(f"Stop event: processing {len(conversation_lines)} messages ({len(conversation)} chars) for session {session_id[:8]}")

    await process_turn(redis_client, session_id, conversation)


async def main():
    """Main entry point - subscribe to pubsub channels and process messages.

    Subscriptions:
    - transcript:* — buffer conversation messages (no introspection trigger)
    - events:* — Stop hook signals trigger introspection
    - cortex:stored:* — store events to clear buffers when Alpha stores
    """

    if not REDIS_URL:
        raise ValueError("REDIS_URL environment variable required")
    if not OLLAMA_URL:
        raise ValueError("OLLAMA_URL environment variable required")

    logger.info(f"Intro starting")
    logger.info(f"Redis: {REDIS_URL}")
    logger.info(f"Ollama: {OLLAMA_URL} ({OLLAMA_MODEL})")

    redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()

    # Subscribe to transcript (buffering), events (Stop signals), and cortex store events
    await pubsub.psubscribe("transcript:*", EVENTS_PATTERN, CORTEX_STORED_PATTERN)
    logger.info("Subscribed to transcript:*, events:*, and cortex:stored:*")

    try:
        async for message in pubsub.listen():
            if message["type"] != "pmessage":
                continue

            channel = message.get("channel", "")
            data = message.get("data", "")

            try:
                if channel.startswith("cortex:stored:"):
                    # Cortex store event - data is just the memory_id as string
                    await handle_cortex_store(redis_client, channel, data)
                elif channel.startswith("events:"):
                    # Event from hooks - parse and dispatch by type
                    parsed = json.loads(data)
                    event_type = parsed.get("type")
                    session_id = parsed.get("session_id")

                    if event_type == "stop" and session_id:
                        await handle_stop_event(redis_client, session_id)
                    elif event_type == "user_prompt_submit" and session_id:
                        # Memory retrieval (the Hippo job)
                        prompt = parsed.get("prompt", "")
                        trace_id = parsed.get("trace_id", "")
                        if prompt and trace_id:
                            # Fire and forget - don't await, let it run async
                            asyncio.create_task(
                                process_user_prompt(
                                    redis_client,
                                    prompt,
                                    trace_id,
                                    session_id,
                                    OLLAMA_URL,
                                    OLLAMA_MODEL,
                                )
                            )
                    else:
                        logger.debug(f"Ignoring event type '{event_type}' on {channel}")
                elif channel.startswith("transcript:"):
                    # Transcript message - buffer only, don't trigger introspection
                    parsed = json.loads(data)
                    await process_message(redis_client, parsed)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in pubsub message on {channel}")
            except Exception as e:
                logger.error(f"Error processing message on {channel}: {e}", exc_info=True)

    except asyncio.CancelledError:
        logger.info("Intro shutting down")
    finally:
        await pubsub.close()
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
