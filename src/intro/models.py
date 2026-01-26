"""
Pydantic models for Intro API.
"""

from pydantic import BaseModel


class TurnMessage(BaseModel):
    """A single message in a conversation turn."""
    role: str  # "user" or "assistant"
    content: str


class PromptRequest(BaseModel):
    """Request to /prompt endpoint."""
    message: str
    session_id: str


class MemoryItem(BaseModel):
    """A memory returned from Cortex."""
    id: int
    content: str
    created_at: str
    query: str | None = None  # The query that surfaced this memory


class PromptResponse(BaseModel):
    """Response from /prompt endpoint."""
    memories: list[MemoryItem]  # Each memory now includes the query that surfaced it
    queries: list[str]  # Kept for backward compatibility / logging


class StopRequest(BaseModel):
    """Request to /stop endpoint."""
    session_id: str
    turn: list[TurnMessage]  # This turn's user + assistant messages


class ClearRequest(BaseModel):
    """Request to /session/clear endpoint."""
    session_id: str
