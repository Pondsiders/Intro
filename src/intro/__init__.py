# Intro - Alpha's metacognitive layer
#
# v0.2.0 - FastAPI refactor
# - HTTP API replaces pubsub daemon
# - /prompt for memory retrieval (sync)
# - /stop for memorables extraction (async)
# - /session/clear for lifecycle management

from .app import app
from .service import IntroService
from .models import (
    PromptRequest,
    PromptResponse,
    StopRequest,
    ClearRequest,
    MemoryItem,
    TurnMessage,
)

__all__ = [
    "app",
    "IntroService",
    "PromptRequest",
    "PromptResponse",
    "StopRequest",
    "ClearRequest",
    "MemoryItem",
    "TurnMessage",
]
