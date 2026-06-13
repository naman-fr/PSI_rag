#!/usr/bin/env python
"""MCP Server stdio runner for the PSI RAG pipeline."""

import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

# Ensure loggers don't write to stdout (it corrupts the stdio JSON-RPC protocol)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# Satisfy environment variables for startup
if not os.environ.get("GROQ_API_KEY") and not os.path.exists(".env"):
    os.environ["GROQ_API_KEY"] = "missing-key-placeholder"
if not os.environ.get("GEMINI_API_KEY") and not os.path.exists(".env"):
    os.environ["GEMINI_API_KEY"] = "missing-key-placeholder"

# Load settings and init RAG components locally
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.cache.redis_client import InMemoryCache
from app.memory.conversation import ConversationManager
from app.memory.summary import SummaryManager
from app.rag.embeddings import EmbeddingService
from app.rag.retrieval import FAISSRetriever
from app.rag.generation import LLMService
from app.services.orchestrator import RAGOrchestrator
import app.main

async def init_orchestrator():
    # Force setup loggers to stderr so we don't break JSON-RPC stdio protocol
    setup_logging()
    # Override logging stream to stderr
    for handler in logging.root.handlers:
        if hasattr(handler, 'stream') and handler.stream == sys.stdout:
            handler.stream = sys.stderr

    cache = InMemoryCache()
    conv_mgr = ConversationManager(cache)
    sum_mgr = SummaryManager(cache)

    embedding_service = EmbeddingService()
    
    # Try loading existing FAISS index
    settings = get_settings()
    index_path = Path(settings.index_save_dir)
    retriever = FAISSRetriever(dimension=embedding_service.dimension)
    
    if index_path.exists():
        try:
            await retriever.load_index(index_path)
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to load index from {index_path}: {e}\n")
    else:
        sys.stderr.write(f"Warning: Index directory {index_path} not found. Running with empty index.\n")

    llm_service = LLMService()

    orchestrator = RAGOrchestrator(
        embedding_service=embedding_service,
        retriever=retriever,
        llm_service=llm_service,
        cache_backend=cache,
        conversation_manager=conv_mgr,
        summary_manager=sum_mgr,
    )
    
    # Populate the main module global state
    app.main._orchestrator = orchestrator
    sys.stderr.write("PSI RAG Orchestrator successfully initialized for MCP stdio.\n")

if __name__ == "__main__":
    import asyncio
    # Initialize components
    asyncio.run(init_orchestrator())
    
    # Run MCP server using stdio transport
    from app.routers.mcp import mcp
    sys.stderr.write("Starting MCP stdio transport server...\n")
    mcp.run(transport="stdio")
