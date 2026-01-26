"""OLMo client for Intro.

Wraps Ollama API calls with gen_ai.* semantic conventions for Logfire Model Run panel.

Two modes:
1. When called within a FastAPI request (e.g., POST /prompt): attaches gen_ai.* attributes
   to the existing span created by instrument_fastapi().
2. When called from a background task (e.g., asyncio.create_task from /stop): creates a
   manual logfire.span() with gen_ai.* attributes since no request span exists.

The key insight: gen_ai.operation.name MUST be "chat" for the Model Run panel to render.
"""

import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
import logfire
from opentelemetry import trace

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "olmo3:7b")


@asynccontextmanager
async def _span_context(operation: str, system_prompt: str, input_msgs: list):
    """
    Context manager that either attaches to an existing span or creates a new one.

    If there's an active recording span (FastAPI request), we attach attributes to it.
    If not (background task), we create a manual logfire.span() with the gen_ai.* attributes.
    """
    current_span = trace.get_current_span()

    # Prepare the gen_ai.* request attributes
    request_attrs = {
        "gen_ai.operation.name": "chat",  # MUST be "chat" for Model Run panel
        "intro.operation": operation,  # Our semantic name
        "gen_ai.provider.name": "ollama",
        "gen_ai.request.model": OLLAMA_MODEL,
        "gen_ai.system_instructions": json.dumps([{
            "type": "text",
            "content": system_prompt[:1000] + "..." if len(system_prompt) > 1000 else system_prompt
        }]),
        "gen_ai.input.messages": json.dumps(input_msgs) if input_msgs else "[]",
    }

    if current_span.is_recording():
        # Attach to existing FastAPI span
        for key, value in request_attrs.items():
            current_span.set_attribute(key, value)
        yield current_span
    else:
        # Create a manual span for background tasks
        with logfire.span(f"ollama:{operation}", **request_attrs) as span:
            yield span


def _set_response_attrs(span, output: str, prompt_tokens: int, completion_tokens: int):
    """Set gen_ai.* response attributes on a span."""
    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
    span.set_attribute("gen_ai.response.model", OLLAMA_MODEL)
    span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
    span.set_attribute("gen_ai.output.type", "text")
    span.set_attribute("gen_ai.output.messages", json.dumps([{
        "role": "assistant",
        "parts": [{"type": "text", "content": output}],
        "finish_reason": "stop"
    }]))


async def chat(
    messages: list[dict],
    system_prompt: str,
    operation: str = "chat",
    context_size: int = 16 * 1024,
    json_format: bool = False,
) -> tuple[str, dict]:
    """
    Call OLMo with a multi-turn conversation.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}
        system_prompt: The system prompt to use
        operation: Name for telemetry (e.g., "notice", "extract_queries")
        context_size: Context window size in tokens
        json_format: Whether to request JSON output

    Returns:
        (output_text, token_counts)
    """
    # Format input messages for gen_ai attribute (just the last user message)
    input_msgs = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            input_msgs = [{"role": "user", "parts": [{"type": "text", "content": msg.get("content", "")}]}]
            break

    async with _span_context(operation, system_prompt, input_msgs) as span:
        # Build full message list for the call
        full_messages = [{"role": "system", "content": system_prompt}, *messages]

        try:
            request_body = {
                "model": OLLAMA_MODEL,
                "messages": full_messages,
                "stream": False,
                "options": {
                    "num_ctx": context_size,
                },
                "keep_alive": "60m",
            }
            if json_format:
                request_body["format"] = "json"

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=request_body,
                )
                response.raise_for_status()

            result = response.json()
            output = result.get("message", {}).get("content", "")

            # Token counts (gen_ai semantic conventions)
            prompt_tokens = result.get("prompt_eval_count", 0)
            completion_tokens = result.get("eval_count", 0)

            # Set gen_ai.* response attributes
            _set_response_attrs(span, output, prompt_tokens, completion_tokens)

            logfire.info("OLMo call complete", model=OLLAMA_MODEL, operation=operation,
                         input_tokens=prompt_tokens, output_tokens=completion_tokens)

            return output, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        except Exception as e:
            logger.error(f"OLMo call failed: {e}")
            logfire.error("OLMo call failed", error=str(e))
            raise
