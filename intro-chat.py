#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "redis",
#     "opentelemetry-api",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-http",
# ]
# ///
"""
intro-chat: Have a conversation with Intro.

Two modes:
1. Standalone chat (default): Separate conversation history in intro:chat:history
2. Session injection (--session): Inject into Intro's live session buffer

Usage:
    intro-chat "Hey Intro, how are you?"           # Standalone chat
    intro-chat --session "Why didn't you flag X?"  # Inject into current session
    intro-chat --session abc123 "Question"         # Inject into specific session
    intro-chat --clear                             # Clear standalone history
    intro-chat --history                           # Show standalone history
    intro-chat --session --history                 # Show session history

Session mode lets Alpha talk back to Intro in context—asking why something
wasn't flagged as memorable while Intro still has the full conversation
in front of them.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add pondside SDK to path
sys.path.insert(0, "/Pondside/Basement/SDK")

import httpx
import redis
from opentelemetry import trace

from pondside.telemetry import init

# Initialize telemetry
init("intro-chat")
tracer = trace.get_tracer(__name__)

# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "olmo3:7b")

# Redis keys
STANDALONE_KEY = "intro:chat:history"
SESSION_KEY_TEMPLATE = "intro:ollama:{session_id}"

# Buffer TTL (24 hours)
BUFFER_TTL = 60 * 60 * 24

# Path to prompt
PROMPT_PATH = Path(__file__).parent / "prompt.md"


def load_prompt() -> str:
    """Load the Intro prompt."""
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text()
    raise FileNotFoundError(f"Prompt not found at {PROMPT_PATH}")


def get_history(r: redis.Redis, key: str) -> list[dict]:
    """Get conversation history from Redis."""
    history_json = r.get(key)
    if history_json:
        return json.loads(history_json)
    return []


def save_history(r: redis.Redis, key: str, history: list[dict]):
    """Save conversation history to Redis."""
    r.set(key, json.dumps(history))
    r.expire(key, BUFFER_TTL)


def clear_history(r: redis.Redis, key: str):
    """Clear conversation history."""
    r.delete(key)
    print(f"Conversation history cleared ({key}).")


def show_history(r: redis.Redis, key: str, session_mode: bool = False):
    """Display conversation history."""
    history = get_history(r, key)
    if not history:
        print("No conversation history.")
        return

    if session_mode:
        print(f"=== Session Conversation History ({key}) ===\n")
        # Session history has longer user messages (turn content + prompt)
        for i, msg in enumerate(history):
            role = msg["role"]
            content = msg["content"]
            # Truncate long session messages for readability
            if len(content) > 500:
                content = content[:500] + "...[truncated]"
            label = "[Turn prompt]" if role == "user" else "[Intro]"
            print(f"{label}: {content}\n")
    else:
        print("=== Standalone Conversation History ===\n")
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                print(f"[Alpha]: {content}\n")
            else:
                print(f"[Intro]: {content}\n")


def chat(r: redis.Redis, message: str, key: str) -> str:
    """Send a message to Intro and get a response."""

    # Load prompt and history
    system_prompt = load_prompt()
    history = get_history(r, key)

    # Build messages for Ollama
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": message},
    ]

    # Call Ollama with OTel span
    with tracer.start_as_current_span(
        f"llm.{OLLAMA_MODEL}",
        kind=trace.SpanKind.CLIENT,
    ) as span:
        # OpenInference attributes for Phoenix
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", OLLAMA_MODEL)

        # gen_ai attributes for Parallax routing
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", OLLAMA_MODEL)
        span.set_attribute("gen_ai.operation.name", "chat")

        # Input attributes
        for i, msg in enumerate(messages):
            span.set_attribute(f"llm.input_messages.{i}.message.role", msg["role"])
            # Truncate long messages for span attributes
            content = msg["content"]
            if len(content) > 1000:
                content = content[:1000] + "...[truncated]"
            span.set_attribute(f"llm.input_messages.{i}.message.content", content)

        span.set_attribute("input.value", message)

        # Make the call
        response = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_ctx": 24 * 1024,
                },
                "keep_alive": "60m",
            },
            timeout=60.0,
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

    # Update history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": output})
    save_history(r, key, history)

    return output


def get_session_id(session_arg: str | None) -> str:
    """Get session ID from argument or environment."""
    if session_arg and session_arg != "true":
        # Explicit session ID provided
        return session_arg

    # Try to get from environment
    session_id = os.environ.get("CLAUDE_SESSION_ID")
    if session_id:
        return session_id

    raise ValueError(
        "No session ID available. Either:\n"
        "  - Provide explicit session ID: --session abc123\n"
        "  - Set CLAUDE_SESSION_ID environment variable\n"
        "  - Run from within a Claude Code session"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Chat with Intro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  intro-chat "What makes something memorable?"     # Standalone chat
  intro-chat --session "Why didn't you flag the emoji?"  # Inject into current session
  intro-chat --session abc123 "Question"           # Inject into specific session
  intro-chat --history                             # Show standalone history
  intro-chat --session --history                   # Show current session's Intro history
        """
    )
    parser.add_argument("message", nargs="?", help="Message to send to Intro")
    parser.add_argument(
        "--session", "-s",
        nargs="?",
        const="true",
        metavar="SESSION_ID",
        help="Inject into session's Intro buffer (uses CLAUDE_SESSION_ID if no ID given)"
    )
    parser.add_argument("--clear", action="store_true", help="Clear conversation history")
    parser.add_argument("--history", action="store_true", help="Show conversation history")
    args = parser.parse_args()

    r = redis.from_url(REDIS_URL, decode_responses=True)

    # Determine which Redis key to use
    if args.session:
        try:
            session_id = get_session_id(args.session)
            key = SESSION_KEY_TEMPLATE.format(session_id=session_id)
            print(f"[Session mode: {session_id[:8]}...]\n")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return
    else:
        key = STANDALONE_KEY

    if args.clear:
        clear_history(r, key)
        return

    if args.history:
        show_history(r, key, session_mode=bool(args.session))
        return

    if not args.message:
        # Read from stdin if no message provided
        if not sys.stdin.isatty():
            args.message = sys.stdin.read().strip()
        else:
            parser.print_help()
            return

    if not args.message:
        print("No message provided.", file=sys.stderr)
        return

    print(f"[Alpha]: {args.message}\n")

    response = chat(r, args.message, key)

    print(f"[Intro]: {response}")


if __name__ == "__main__":
    main()
