"""
Application settings loaded from environment variables.

All configuration is centralized here. No hardcoded keys anywhere.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # --- API Keys (required) ---
    groq_api_key: str = Field(..., description="Groq API key for LLM calls")
    gemini_api_key: str = Field(..., description="Google Gemini API key for embeddings")

    # --- API Keys (optional) ---
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_index_name: str = Field(default="self-rag-index", description="Pinecone index name")
    langsmith_api_key: str = Field(default="", description="LangSmith API key")

    # --- Model Configuration ---
    groq_model: str = Field(default="llama-3.1-8b-instant", description="Groq model name")
    embed_model: str = Field(default="gemini-embedding-2", description="Gemini embedding model")
    embed_dimension: int = Field(default=3072, description="Embedding vector dimension")

    # --- Vision & Multimodal Configuration ---
    vision_model: str = Field(default="gemini-2.5-flash", description="Primary multimodal model")
    vision_embed_model: str = Field(default="multimodal-embedding-001", description="Gemini multimodal embedding model")
    vision_embed_dimension: int = Field(default=1408, description="Dimension for multimodal embeddings")
    local_clip_model: str = Field(default="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", description="Local fallback CLIP model")
    max_image_size_mb: int = Field(default=10, description="Max allowed image size in MB")
    max_image_dimension: int = Field(default=1024, description="Max resolution for vision input")
    max_image_tiles: int = Field(default=9, description="Maximum number of grid tiles to process")
    tile_size: int = Field(default=512, description="Image grid tile size")

    # --- Retrieval Settings ---
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    retrieval_score_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    vector_backend: Literal["faiss", "pinecone"] = Field(
        default="faiss", description="Vector store backend"
    )

    # --- Chunking Settings ---
    max_chunk_chars: int = Field(default=1000, ge=200, description="Max characters per chunk")
    min_chunk_chars: int = Field(default=200, ge=50, description="Min characters per chunk")
    chunk_overlap_sentences: int = Field(default=1, ge=0, description="Overlap sentences")

    # --- Context Settings ---
    max_context_chars: int = Field(
        default=3000, ge=500, description="Max characters in assembled context"
    )

    # --- Generation Settings ---
    max_completion_tokens: int = Field(default=384, ge=64, description="Max tokens for answers")
    verification_max_tokens: int = Field(default=128, ge=32, description="Max tokens for verification")
    direct_max_tokens: int = Field(default=128, ge=32, description="Max tokens for direct replies")
    verification_confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Min confidence to accept answer"
    )

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Default cache TTL")
    semantic_cache_threshold: float = Field(
        default=0.95, ge=0.8, le=1.0, description="Cosine similarity threshold for semantic cache"
    )

    # --- Memory ---
    conversation_window_size: int = Field(
        default=10, ge=2, description="Recent messages to keep in context"
    )
    summary_max_tokens: int = Field(
        default=200, ge=50, description="Max tokens for conversation summary"
    )
    summary_interval: int = Field(
        default=5, ge=2, description="Summarize every N turns"
    )

    # --- Paths ---
    docs_dir: str = Field(default="rag_docs/rag_docs", description="Document source directory")
    index_dir: str = Field(default="index_store", description="FAISS index storage")
    log_dir: str = Field(default="logs", description="Log file directory")

    # --- Observability ---
    log_level: str = Field(default="INFO", description="Logging level")
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_project: str = Field(default="psi", description="LangSmith project name")

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(default="psi-rag", description="MLflow experiment name")

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")

    # --- Input Limits ---
    max_question_length: int = Field(default=2000, ge=100, description="Max question characters")
    max_username_length: int = Field(default=64, ge=1, description="Max username characters")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def auto_detect_backend(self) -> "Settings":
        import os
        if self.pinecone_api_key and "VECTOR_BACKEND" not in os.environ:
            self.vector_backend = "pinecone"
        if self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        elif self.langsmith_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        return self


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
