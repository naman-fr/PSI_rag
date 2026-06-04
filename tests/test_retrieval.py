import numpy as np
import pytest
import tempfile
from pathlib import Path
from app.rag.retrieval import FAISSRetriever


@pytest.mark.asyncio
async def test_faiss_retriever_build_and_search():
    retriever = FAISSRetriever(dimension=8)
    
    # 2 vectors of dimension 8, L2 normalized
    vectors = np.random.randn(2, 8)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    metadata = [
        {"text": "First vector content", "source": "doc1.md"},
        {"text": "Second vector content", "source": "doc2.md"},
    ]
    
    await retriever.build_index(vectors, metadata)
    assert retriever.count == 2
    assert retriever.is_ready is True
    
    # Search query
    query_vector = vectors[0]  # should match first chunk perfectly
    results = retriever.search(query_vector, top_k=1, score_threshold=0.5)
    
    assert len(results) == 1
    assert results[0]["text"] == "First vector content"
    assert results[0]["score"] > 0.99


@pytest.mark.asyncio
async def test_faiss_retriever_save_and_load():
    retriever = FAISSRetriever(dimension=8)
    vectors = np.random.randn(2, 8)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata = [
        {"text": "First vector content", "source": "doc1.md"},
        {"text": "Second vector content", "source": "doc2.md"},
    ]
    await retriever.build_index(vectors, metadata)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        await retriever.save_index(path)
        
        # Load in a new retriever
        new_retriever = FAISSRetriever(dimension=8)
        await new_retriever.load_index(path)
        assert new_retriever.count == 2
        assert new_retriever.is_ready is True
        
        # Check if search works on loaded index
        query_vector = vectors[1]
        results = new_retriever.search(query_vector, top_k=1, score_threshold=0.5)
        assert len(results) == 1
        assert results[0]["text"] == "Second vector content"
