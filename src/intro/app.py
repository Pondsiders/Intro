"""
Intro FastAPI application.

HTTP API for memory retrieval and memorables extraction.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse

from pondside.telemetry import init

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

# Optional Redis for persistence
REDIS_URL = os.environ.get("REDIS_URL")

# Global service instance
intro = IntroService()


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
async def prompt(request: PromptRequest) -> PromptResponse:
    """
    Process a user prompt: extract queries, search Cortex, return memories.

    This is synchronous - the hook waits for the response.
    """
    memories, queries = await intro.prompt(request.message, request.session_id)
    return PromptResponse(memories=memories, queries=queries)


@app.post("/stop", status_code=202)
async def stop(request: StopRequest):
    """
    Process a stop event: extract memorables from this turn.

    This is async - returns 202 Accepted immediately, processing happens in background.
    """
    await intro.stop(request.session_id, request.turn)
    return Response(status_code=202)


@app.post("/session/clear", status_code=204)
async def clear_session(request: ClearRequest):
    """
    Clear all state for a session.

    Called when Alpha stores a memory or on SessionStart.
    """
    await intro.clear(request.session_id)
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
