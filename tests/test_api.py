import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_ready_endpoint_not_initialized():
    with TestClient(app) as client:
        # Before ingestion or loading index, it will be degraded
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"


@patch("app.main.run_ingestion")
def test_ingest_endpoint(mock_run):
    mock_run.return_value = {"documents_loaded": 3, "chunks_indexed": 15}
    with TestClient(app) as client:
        response = client.post("/api/v1/ingest", json={"source_dir": "rag_docs/rag_docs"})
        assert response.status_code == 200
        assert response.json() == {
            "documents_loaded": 3,
            "chunks_indexed": 15,
            "status": "success",
        }


@patch("app.main.get_orchestrator")
def test_chat_endpoint(mock_get_orch):
    mock_orch = AsyncMock()
    mock_get_orch.return_value = mock_orch
    
    mock_orch.process_query.return_value = {
        "trace_id": "test-trace",
        "mode": "retrieval",
        "answer": "Mocked answer payload",
        "confidence": 0.9,
        "sources": [{"source": "DOC1_carrier_sla_agreement.md", "chunk_id": 1, "score": 0.9, "text_preview": "preview"}],
        "verdict": {"supported": True, "confidence": 0.9, "reason": "supported"},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "cached": False,
        "timestamp": "2026-06-04T12:00:00",
    }
    
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"username": "user123", "question": "What is the transit tolerance?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Mocked answer payload"
        assert data["trace_id"] == "test-trace"
