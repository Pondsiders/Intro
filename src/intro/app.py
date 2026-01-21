"""
Intro FastAPI application.

HTTP API for memory retrieval and memorables extraction.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, Response, Request
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from pondside.telemetry import init, get_tracer

from .models import (
    PromptRequest,
    PromptResponse,
    StopRequest,
    ClearRequest,
    MemoryItem,
)
from .service import IntroService

# Initialize telemetry
init("intro")
logger = logging.getLogger(__name__)
tracer = get_tracer()

# Optional Redis for persistence
REDIS_URL = os.environ.get("REDIS_URL")

# Global service instance
intro = IntroService()


def extract_parent_context(request: Request):
    """Extract OTel parent context from incoming request headers."""
    traceparent = request.headers.get("traceparent")
    if traceparent:
        carrier = {"traceparent": traceparent}
        return TraceContextTextMapPropagator().extract(carrier=carrier)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Intro starting up")

    # Optionally connect to Redis
    if REDIS_URL:
        import redis.asyncio as aioredis
        intro.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        logger.info(f"Connected to Redis: {REDIS_URL}")
    else:
        logger.info("Running without Redis (state is ephemeral)")

    yield

    # Cleanup
    if intro.redis:
        await intro.redis.close()
    logger.info("Intro shut down")


app = FastAPI(
    title="Intro",
    description="Alpha's metacognitive layer",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "active_sessions": len(intro.ollama_history),
        "ollama_model": os.environ.get("OLLAMA_MODEL", "olmo3:7b"),
    }


@app.post("/prompt", response_model=PromptResponse)
async def prompt(body: PromptRequest, request: Request) -> PromptResponse:
    """
    Process a user prompt: extract queries, search Cortex, return memories.

    This is synchronous - the hook waits for the response.
    """
    parent_context = extract_parent_context(request)

    with tracer.start_as_current_span("intro.prompt", context=parent_context) as span:
        span.set_attribute("session_id", body.session_id[:8])
        span.set_attribute("message_length", len(body.message))

        memories, queries = await intro.prompt(body.message, body.session_id)

        span.set_attribute("memories_returned", len(memories))
        span.set_attribute("queries_used", len(queries))

        return PromptResponse(memories=memories, queries=queries)


@app.post("/stop", status_code=202)
async def stop(body: StopRequest, request: Request):
    """
    Process a stop event: extract memorables from this turn.

    This is async - returns 202 Accepted immediately, processing happens in background.
    """
    parent_context = extract_parent_context(request)

    with tracer.start_as_current_span("intro.stop", context=parent_context) as span:
        span.set_attribute("session_id", body.session_id[:8])
        span.set_attribute("turn_messages", len(body.turn))

        await intro.stop(body.session_id, body.turn)

    return Response(status_code=202)


@app.post("/session/clear", status_code=204)
async def clear_session(body: ClearRequest):
    """
    Clear all state for a session.

    Called when Alpha stores a memory or on SessionStart.
    """
    await intro.clear(body.session_id)
    return Response(status_code=204)


@app.get("/memorables/{session_id}")
async def get_memorables(session_id: str) -> list[str]:
    """
    Get current memorables for a session.

    This is a convenience endpoint - the Loom can also read from Redis directly.
    """
    return intro.get_memorables(session_id)


# For running directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
