"""Request schemas with strict validation."""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request from a user."""

    username: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Username for session tracking",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Session ID. Auto-generated if not provided.",
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question to answer",
    )
    image_id: Optional[str] = Field(
        default=None,
        description="Optional pre-uploaded image ID for visual question answering",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Optional image URL for visual question answering",
    )


class IngestRequest(BaseModel):
    """Document ingestion request."""

    source_dir: str = Field(
        default="rag_docs/rag_docs",
        description="Directory containing markdown documents",
    )
    force_reindex: bool = Field(
        default=False,
        description="Force re-indexing even if index exists",
    )


class FeedbackRequest(BaseModel):
    """User feedback on a response."""

    trace_id: str = Field(..., description="Trace ID of the response")
    username: str = Field(..., min_length=1, max_length=64)
    rating: str = Field(..., pattern=r"^(good|bad|neutral)$")
    notes: Optional[str] = Field(default=None, max_length=500)


class ImageSearchRequest(BaseModel):
    """Image search request."""

    query: str = Field(..., description="Text query to find matching images")
    top_k: Optional[int] = Field(default=3, ge=1, le=10)
