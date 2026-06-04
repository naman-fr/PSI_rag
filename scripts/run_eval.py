#!/usr/bin/env python
"""Evaluation script for testing all 46 questions against the RAG pipeline."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Set mock environment variables first if mock flag is present to bypass Pydantic validation
if "--mock" in sys.argv:
    os.environ.setdefault("GROQ_API_KEY", "mock-groq-key")
    os.environ.setdefault("GEMINI_API_KEY", "mock-gemini-key")
    os.environ.setdefault("PINECONE_API_KEY", "mock-pinecone-key")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

load_dotenv()
setup_logging()
logger = get_logger("evaluation")


async def run_eval(mock_mode: bool = False):
    settings = get_settings()

    # Load test questions
    questions_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info("Loaded questions", count=len(questions), mock_mode=mock_mode)

    # Initialize RAG components
    from app.cache.redis_client import InMemoryCache
    from app.memory.conversation import ConversationManager
    from app.memory.summary import SummaryManager
    from app.rag.retrieval import FAISSRetriever
    from app.services.orchestrator import RAGOrchestrator

    cache = InMemoryCache()
    conv_mgr = ConversationManager(cache)
    sum_mgr = SummaryManager(cache)

    if mock_mode:
        from unittest.mock import AsyncMock, MagicMock
        import numpy as np

        # Mock embedding
        embedding_service = MagicMock()
        mock_vec = np.zeros(settings.embed_dimension, dtype=np.float32)
        embedding_service.embed_query.return_value = mock_vec
        embedding_service.embed_documents.return_value = [mock_vec]

        # Mock retriever
        retriever = MagicMock(spec=FAISSRetriever)
        retriever.search.return_value = [
            {
                "score": 0.95,
                "text": "Gold tier delay tolerance is 24 hours. Delhi to NY transit is 15 days.",
                "metadata": {"source": "DOC1_carrier_sla_agreement.md"},
            }
        ]

        # Mock LLM
        llm_service = AsyncMock()
        llm_service.generate.return_value = (
            '{"supported": true, "confidence": 0.95, "reason": "Matched"}',
            {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )
    else:
        # Real services
        from app.rag.embeddings import EmbeddingService
        from app.rag.generation import LLMService

        embedding_service = EmbeddingService()
        retriever = FAISSRetriever(dimension=settings.embed_dimension)

        # Load index
        index_path = f"{settings.index_dir}/faiss.index"
        if not Path(index_path).exists():
            logger.info("Index not found. Running ingestion first...")
            from app.main import run_ingestion
            await run_ingestion(source_dir=settings.docs_dir)

        await retriever.load_index(index_path)
        llm_service = LLMService()

    orchestrator = RAGOrchestrator(
        embedding_service=embedding_service,
        retriever=retriever,
        llm_service=llm_service,
        cache_backend=cache,
        conversation_manager=conv_mgr,
        summary_manager=sum_mgr,
    )

    results = []
    print("\n" + "=" * 80)
    print(f"RUNNING RAG EVALUATION ({'MOCKED' if mock_mode else 'REAL'})")
    print("=" * 80)

    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            # We use a unique session and user for each evaluation run
            response = await orchestrator.process_query(
                question=q,
                username=f"eval_user_{i}",
                session_id=f"eval_sess_{i}",
            )
            elapsed = time.time() - t0

            # Print to stdout
            print(f"[{i+1}/{len(questions)}] Q: {q}")
            print(f"       -> Mode: {response['mode']}, Confidence: {response['confidence']:.2f}, Time: {elapsed:.2f}s")
            print(f"       -> Answer: {response['answer'][:100]}...")

            results.append({
                "index": i + 1,
                "question": q,
                "mode": response["mode"],
                "confidence": response["confidence"],
                "answer": response["answer"],
                "latency_sec": elapsed,
                "tokens": response.get("usage", {}).get("total_tokens", 0),
            })
        except Exception as e:
            logger.exception("Error processing question", question=q, error=str(e))
            results.append({
                "index": i + 1,
                "question": q,
                "error": str(e),
            })

    # Save output
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Evaluation complete! Results saved to data/eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run with mock services")
    args = parser.parse_args()

    asyncio.run(run_eval(mock_mode=args.mock))
