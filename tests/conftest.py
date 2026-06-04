import os
import sys
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import Settings, get_settings
from app.rag.embeddings import EmbeddingService
from app.rag.retrieval import FAISSRetriever
from app.rag.generation import LLMService


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("PINECONE_API_KEY", "mock-pinecone-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")


@pytest.fixture
def mock_embedding_service():
    """Mock Gemini embedding service."""
    service = MagicMock(spec=EmbeddingService)
    # Return L2 normalized mock vector of dimension 3072
    mock_vector = np.random.randn(3072)
    mock_vector /= np.linalg.norm(mock_vector)
    service.embed_query.return_value = mock_vector
    service.embed_documents.return_value = [mock_vector]
    return service


@pytest.fixture
def mock_retriever():
    """Mock FAISS retriever."""
    retriever = MagicMock(spec=FAISSRetriever)
    retriever.is_ready = True
    retriever.count = 1
    retriever.search.return_value = [
        {
            "score": 0.9,
            "text": "Gold tier shipments have a 24-hour transit time tolerance. Claims must be filed within 14 business days.",
            "metadata": {
                "source": "DOC1_carrier_sla_agreement.md",
                "page": 1,
                "chunk_id": 1,
            },
        }
    ]
    return retriever


@pytest.fixture
def mock_llm_service():
    """Mock Groq LLM service."""
    service = AsyncMock(spec=LLMService)
    service.generate.return_value = (
        '{"supported": true, "confidence": 0.95, "reason": "Claim fits Gold tier documentation."}',
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    )
    return service


@pytest.fixture
def mock_cache_backend():
    """Mock Redis/InMemory cache."""
    cache = AsyncMock()
    cache_dict = {}

    async def get(key):
        return cache_dict.get(key)

    async def set(key, val, ttl=None):
        cache_dict[key] = val
        return True

    async def exists(key):
        return key in cache_dict

    async def delete(key):
        cache_dict.pop(key, None)
        return True

    cache.get.side_effect = get
    cache.set.side_effect = set
    cache.exists.side_effect = exists
    cache.delete.side_effect = delete
    return cache
