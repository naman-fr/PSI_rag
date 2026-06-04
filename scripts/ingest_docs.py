#!/usr/bin/env python
"""CLI script for ingesting documents into the vector database."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()
setup_logging()
logger = get_logger("ingest_cli")


async def main():
    parser = argparse.ArgumentParser(description="Ingest markdown documents into PSI RAG vector store.")
    parser.add_argument(
        "--source",
        type=str,
        default="rag_docs/rag_docs",
        help="Directory containing markdown files (default: rag_docs/rag_docs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing even if index already exists",
    )
    args = parser.parse_args()

    settings = get_settings()
    logger.info("Starting document ingestion", source_dir=args.source, force=args.force)

    # Initialize cache mock or backend so lifespan/import dependencies don't fail
    from app.cache.redis_client import get_cache_backend
    cache_backend = await get_cache_backend()

    from app.main import run_ingestion
    try:
        # Run the ingestion pipeline
        result = await run_ingestion(source_dir=args.source, force_reindex=args.force)
        logger.info(
            "Ingestion completed successfully!",
            documents_loaded=result["documents_loaded"],
            chunks_indexed=result["chunks_indexed"],
        )
    except Exception as e:
        logger.exception("Ingestion failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
