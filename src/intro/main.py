"""
Intro - Alpha's metacognitive layer.

Subscribes to transcript pubsub, maintains an ongoing conversation with OLMo
about what's memorable, writes results to Redis for the Loom to inject.

Also handles memory retrieval on user prompts (the Hippo job) - extracting
search queries and fetching relevant memories from Cortex.

Architecture (incremental):
    Stop event → get this turn's content → continue OLMo conversation
              → "how about now?" → memorables straight to Redis

    cortex:stored → clear OLMo conversation, start fresh

This is KV-cache friendly: we send incremental turns to the same context
window instead of re-sending the whole transcript every time.
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
TURN_BUFFER_KEY = "intro:turn:{session_id}"  # Current turn's messages (cleared after processing)
OLLAMA_HISTORY_KEY = "intro:ollama:{session_id}"  # OLMo conversation history (JSON)
MEMORABLES_KEY = "intro:memorables:{session_id}"

# Buffer TTL (24 hours) - safety valve if session dies without storing
BUFFER_TTL = 60 * 60 * 24

# Cortex store notification channel pattern
CORTEX_STORED_PATTERN = "cortex:stored:*"

# Events channel pattern (for Stop hook signals)
EVENTS_PATTERN = "events:*"

# Compaction summary prefix (filter these out)
COMPACTION_PREFIX = "This session is being continued from a previous conversation"

# First turn prompt
FIRST_TURN_PROMPT = """<conversation>
{turn_content}
</conversation>

<question>
Intro, what in the <conversation></conversation> block has been memorable so far? Write all the memorable things as a Markdown list (one per line, starting with `-`). If nothing has been memorable, just say "Nothing notable."
</question>
"""

# Follow-up turn prompt
FOLLOWUP_TURN_PROMPT = """<conversation>
{turn_content}
</conversation>

<question>
Update your list—add new memorable moments from <conversation></conversation> blocks, keep what still matters, drop anything that's been superseded or no longer seems significant. Output the current list of memorables.
</question>
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

        # Input attributes (OpenInference format)
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


async def get_ollama_history(redis_client: aioredis.Redis, session_id: str) -> list[dict]:
    """Get the OLMo conversation history for this session."""
    key = OLLAMA_HISTORY_KEY.format(session_id=session_id)
    history_json = await redis_client.get(key)
    if history_json:
        return json.loads(history_json)
    return []


async def save_ollama_history(redis_client: aioredis.Redis, session_id: str, history: list[dict]):
    """Save the OLMo conversation history for this session."""
    key = OLLAMA_HISTORY_KEY.format(session_id=session_id)
    await redis_client.set(key, json.dumps(history))
    await redis_client.expire(key, BUFFER_TTL)


async def clear_session(redis_client: aioredis.Redis, session_id: str):
    """Clear all Intro state for a session (on cortex store)."""
    turn_key = TURN_BUFFER_KEY.format(session_id=session_id)
    history_key = OLLAMA_HISTORY_KEY.format(session_id=session_id)
    mem_key = MEMORABLES_KEY.format(session_id=session_id)

    deleted = await redis_client.delete(turn_key, history_key, mem_key)
    logger.info(f"Cleared {deleted} keys for session {session_id[:8]}")


async def process_turn(redis_client: aioredis.Redis, session_id: str, turn_content: str):
    """
    Process a single turn: continue the OLMo conversation, get updated memorables.

    This is the incremental approach:
    - First turn: start fresh conversation with system prompt + first turn
    - Subsequent turns: continue conversation with "how about now?"
    - No dedupe needed—OLMo maintains state
    """
    with tracer.start_as_current_span(
        "intro.process_turn",
        attributes={
            "session_id": session_id[:8],
            "metadata.source": "intro",
        }
    ) as parent_span:

        # Get existing OLMo conversation history
        history = await get_ollama_history(redis_client, session_id)

        is_first_turn = len(history) == 0

        if is_first_turn:
            # Start fresh conversation
            user_msg = FIRST_TURN_PROMPT.format(turn_content=turn_content)
            messages = [
                {"role": "system", "content": INTRO_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            logger.info(f"First turn for session {session_id[:8]}, starting fresh conversation")
        else:
            # Continue existing conversation
            user_msg = FOLLOWUP_TURN_PROMPT.format(turn_content=turn_content)
            messages = [
                {"role": "system", "content": INTRO_PROMPT},
                *history,
                {"role": "user", "content": user_msg},
            ]
            logger.info(f"Turn {len(history)//2 + 1} for session {session_id[:8]}, continuing conversation")

        parent_span.set_attribute("is_first_turn", is_first_turn)
        parent_span.set_attribute("history_length", len(history))

        try:
            output, tokens = await call_ollama_chat(messages, operation="notice")
        except Exception:
            return  # Error already logged

        memorables = parse_memorables(output)
        logger.info(f"OLMo returned {len(memorables)} memorables")

        # Save updated history (user message + assistant response)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": output})
        await save_ollama_history(redis_client, session_id, history)

        # Save memorables directly to Redis (no dedupe needed)
        mem_key = MEMORABLES_KEY.format(session_id=session_id)
        await redis_client.delete(mem_key)
        if memorables:
            await redis_client.rpush(mem_key, *memorables)
            await redis_client.expire(mem_key, BUFFER_TTL)

        parent_span.set_attribute("memorables.count", len(memorables))
        logger.info(f"Stored {len(memorables)} memorables for session {session_id[:8]}")


async def handle_cortex_store(redis_client: aioredis.Redis, channel: str, memory_id: str):
    """Handle a cortex:stored:{session_id} event by clearing all state.

    When Alpha stores a memory, we clear everything and start fresh.
    The next turn will be treated as a "first turn" again.
    """
    # Extract session_id from channel name (cortex:stored:{session_id})
    parts = channel.split(":")
    if len(parts) != 3:
        logger.warning(f"Unexpected channel format: {channel}")
        return

    session_id = parts[2]
    await clear_session(redis_client, session_id)
    logger.info(f"Cortex store event: cleared session {session_id[:8]} (memory_id={memory_id})")


async def process_message(redis_client: aioredis.Redis, data: dict):
    """Process a single transcript message - buffer for current turn only."""
    session_id = data.get("session_id")
    role = data.get("role")

    # Only care about user/assistant messages
    if role not in ("user", "assistant"):
        return

    # Extract text content
    text = extract_text_content(data.get("raw", {}))
    if not text:
        return

    # Filter out compaction summaries
    if text.startswith(COMPACTION_PREFIX):
        logger.info(f"Skipping compaction summary for session {session_id[:8]}")
        return

    # Format for the buffer
    name = "Alpha" if role == "assistant" else "Jeffery"
    formatted = f"[{name}]: {text}"

    # Add to turn buffer
    turn_key = TURN_BUFFER_KEY.format(session_id=session_id)
    await redis_client.rpush(turn_key, formatted)
    await redis_client.expire(turn_key, BUFFER_TTL)

    logger.debug(f"Buffered {role} message for session {session_id[:8]}")


async def handle_stop_event(redis_client: aioredis.Redis, session_id: str):
    """Handle a Stop event - process this turn's content.

    The Stop hook fires when the assistant is done responding.
    We take just this turn's content and send it to OLMo.
    """
    turn_key = TURN_BUFFER_KEY.format(session_id=session_id)

    # Get this turn's messages
    turn_lines = await redis_client.lrange(turn_key, 0, -1)
    if not turn_lines:
        logger.debug(f"Stop event but no turn content for session {session_id[:8]}")
        return

    turn_content = "\n\n".join(turn_lines)

    logger.info(f"Stop event: processing turn with {len(turn_lines)} messages ({len(turn_content)} chars) for session {session_id[:8]}")

    # Process this turn
    await process_turn(redis_client, session_id, turn_content)

    # Clear the turn buffer (but keep the OLMo history and memorables)
    await redis_client.delete(turn_key)


async def main():
    """Main entry point - subscribe to pubsub channels and process messages.

    Subscriptions:
    - transcript:* — buffer turn messages
    - events:* — Stop hook signals trigger OLMo call
    - cortex:stored:* — store events clear all state
    """

    if not REDIS_URL:
        raise ValueError("REDIS_URL environment variable required")
    if not OLLAMA_URL:
        raise ValueError("OLLAMA_URL environment variable required")

    logger.info(f"Intro starting (incremental architecture)")
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
                    # Transcript message - buffer for current turn
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
