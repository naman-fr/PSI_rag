"""Response schemas for API endpoints."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """Reference to a source document chunk."""

    source: str = Field(description="Source document filename")
    chunk_id: int = Field(description="Chunk index within document")
    score: float = Field(description="Retrieval similarity score")
    text_preview: str = Field(default="", description="First 100 chars of chunk")


class VerificationVerdict(BaseModel):
    """Result of hallucination verification check."""

    supported: bool = Field(description="Whether answer is supported by context")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Explanation for the verdict")


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class ChatResponse(BaseModel):
    """Full chat response with metadata."""

    trace_id: str = Field(description="Unique trace ID for this request")
    mode: Literal["direct", "retrieval", "guardrail_refusal", "cached"] = Field(
        description="Response generation mode"
    )
    answer: str = Field(description="The generated answer")
    confidence: float = Field(ge=0.0, le=1.0, description="Answer confidence")
    sources: List[SourceReference] = Field(default_factory=list)
    verdict: VerificationVerdict = Field(
        description="Hallucination verification result"
    )
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cached: bool = Field(default=False, description="Whether response was from cache")
    timestamp: str = Field(description="ISO timestamp of response")


class IngestResponse(BaseModel):
    """Document ingestion result."""

    documents_loaded: int = Field(description="Number of documents loaded")
    chunks_indexed: int = Field(description="Number of chunks indexed")
    status: str = Field(description="Ingestion status")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    vector_store: str = Field(description="Vector store status")
    redis: str = Field(description="Redis connection status")


class ConversationMessage(BaseModel):
    """A single conversation message."""

    role: Literal["user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")
    timestamp: str = Field(description="ISO timestamp")


class ConversationHistory(BaseModel):
    """Full conversation history for a user."""

    username: str
    session_id: str
    messages: List[ConversationMessage] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None, description="Compressed summary")
