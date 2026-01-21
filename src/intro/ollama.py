"""
OLMo client for Intro.

Wraps Ollama API calls with proper telemetry.
"""

import logging
import os

import httpx
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "olmo3:7b")


async def chat(
    messages: list[dict],
    system_prompt: str,
    operation: str = "chat",
    context_size: int = 24 * 1024,
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
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        for i, msg in enumerate(full_messages):
            span.set_attribute(f"llm.input_messages.{i}.message.role", msg["role"])
            span.set_attribute(f"llm.input_messages.{i}.message.content", msg["content"][:1000])

        # Also set input.value for Phoenix
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        if last_user:
            span.set_attribute("input.value", last_user["content"][:1000])

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

            # Output attributes
            span.set_attribute("output.value", output[:1000])
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute("llm.output_messages.0.message.content", output[:1000])

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
