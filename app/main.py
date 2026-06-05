"""
PSI RAG - Production Guardrailed Self-RAG System

FastAPI application factory with lifespan management.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger("main")

# --- Global State (set during lifespan, read-only after) ---
_orchestrator = None
_cache_backend = None
_conversation_manager = None
_summary_manager = None
_metrics_collector = None


def get_orchestrator():
    return _orchestrator


def get_cache_backend():
    return _cache_backend


def get_conversation_manager():
    return _conversation_manager


def get_summary_manager():
    return _summary_manager


def get_metrics_collector():
    return _metrics_collector


async def run_ingestion(source_dir: str, force_reindex: bool = False) -> dict:
    """Run document ingestion pipeline."""
    global _orchestrator

    settings = get_settings()
    source = source_dir or settings.docs_dir

    from app.rag.ingestion import load_markdown_documents, index_documents
    from app.rag.embeddings import EmbeddingService
    from app.vision.retrieval.vector_search import DualRetriever

    embedding_service = EmbeddingService()
    retriever = DualRetriever(
        text_dimension=settings.embed_dimension,
        visual_dimension=settings.vision_embed_dimension
    )

    # Load documents
    records = load_markdown_documents(source)
    if not records:
        raise FileNotFoundError(f"No markdown documents found in {source}")

    # Index
    result = await index_documents(records, embedding_service, retriever)

    # Save index
    from pathlib import Path

    Path(settings.index_dir).mkdir(parents=True, exist_ok=True)
    await retriever.save_index(f"{settings.index_dir}/faiss.index")

    # Build orchestrator with the new retriever
    from app.rag.generation import LLMService
    from app.services.orchestrator import RAGOrchestrator

    llm_service = LLMService()

    _orchestrator = RAGOrchestrator(
        embedding_service=embedding_service,
        retriever=retriever,
        llm_service=llm_service,
        cache_backend=_cache_backend,
        conversation_manager=_conversation_manager,
        summary_manager=_summary_manager,
        metrics_collector=_metrics_collector,
    )

    logger.info(
        "ingestion_complete",
        documents=result["documents_loaded"],
        chunks=result["chunks_indexed"],
    )

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize and cleanup resources."""
    global _cache_backend, _conversation_manager, _summary_manager, _metrics_collector

    setup_logging()
    settings = get_settings()
    logger.info("starting_application", vector_backend=settings.vector_backend)

    # Initialize cache backend
    from app.cache.redis_client import get_cache_backend as _get_cache

    _cache_backend = await _get_cache()

    # Initialize conversation manager
    from app.memory.conversation import ConversationManager

    _conversation_manager = ConversationManager(_cache_backend)

    # Initialize summary manager
    from app.memory.summary import SummaryManager

    _summary_manager = SummaryManager(_cache_backend)

    # Initialize metrics
    from app.observability.metrics import MetricsCollector

    _metrics_collector = MetricsCollector()

    # Try to load existing index
    from pathlib import Path

    index_path = f"{settings.index_dir}/text_index.faiss"
    if Path(index_path).exists():
        try:
            from app.rag.embeddings import EmbeddingService
            from app.vision.retrieval.vector_search import DualRetriever
            from app.rag.generation import LLMService
            from app.services.orchestrator import RAGOrchestrator

            global _orchestrator
            embedding_service = EmbeddingService()
            retriever = DualRetriever(
                text_dimension=settings.embed_dimension,
                visual_dimension=settings.vision_embed_dimension
            )
            await retriever.load_indices(settings.index_dir)
            llm_service = LLMService()

            _orchestrator = RAGOrchestrator(
                embedding_service=embedding_service,
                retriever=retriever,
                llm_service=llm_service,
                cache_backend=_cache_backend,
                conversation_manager=_conversation_manager,
                summary_manager=_summary_manager,
                metrics_collector=_metrics_collector,
            )
            logger.info("loaded_existing_index", path=index_path)
        except Exception as e:
            logger.warning("failed_loading_index", error=str(e))

    logger.info("application_ready")

    yield

    # Cleanup
    logger.info("shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PSI RAG - Guardrailed Self-RAG System",
        description="Production-grade RAG system for GlobalFreight logistics document QA",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from app.routers import chat, ingest, admin

    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(admin.router)

    return app


# Application instance
app = create_app()
