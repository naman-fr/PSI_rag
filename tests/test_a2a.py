import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app

def test_a2a_agent_card():
    """Verify that the Agent Card endpoint returns correct capability metadata."""
    with TestClient(app) as client:
        response = client.get("/api/v1/a2a/agent-card")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "PSI-Logistics-RAG-Agent"
        assert len(data["skills"]) > 0
        assert data["skills"][0]["id"] == "logistics_sla_advisor"

def test_a2a_well_known_redirect():
    """Verify that /.well-known/agent-card.json redirects to the api versioned endpoint."""
    with TestClient(app) as client:
        response = client.get("/.well-known/agent-card.json", follow_redirects=False)
        assert response.status_code in [302, 307]
        assert "/api/v1/a2a/agent-card" in response.headers["location"]

def test_a2a_task_lifecycle():
    """Verify stateful A2A task creation, retrieval, and background execution."""
    with TestClient(app) as client:
        # 1. Create a task
        payload = {
            "skillId": "logistics_sla_advisor",
            "input": {
                "question": "What is the compensation for Platinum Express?",
                "username": "client_agent"
            }
        }
        res_create = client.post("/api/v1/a2a/tasks", json=payload)
        assert res_create.status_code == 200
        task_data = res_create.json()
        task_id = task_data["taskId"]
        assert task_data["status"] == "created"
        assert task_data["input"]["question"] == "What is the compensation for Platinum Express?"
        
        # 2. Get the task
        res_get = client.get(f"/api/v1/a2a/tasks/{task_id}")
        assert res_get.status_code == 200
        assert res_get.json()["status"] == "created"
        
        # 3. Execute the task
        mock_orch = AsyncMock()
        mock_orch.process_query.return_value = {"answer": "Platinum Express compensation is 15% refund per 24 hours."}
        
        with patch("app.routers.a2a.get_orchestrator", return_value=mock_orch):
            res_exec = client.put(f"/api/v1/a2a/tasks/{task_id}/execute")
            assert res_exec.status_code == 200
            assert res_exec.json()["status"] == "running"
            
            # In TestClient, background tasks run synchronously. Thus, task execution completes immediately.
            res_completed = client.get(f"/api/v1/a2a/tasks/{task_id}")
            assert res_completed.status_code == 200
            comp_data = res_completed.json()
            assert comp_data["status"] == "completed"
            assert "15%" in comp_data["output"]
            assert len(comp_data["artifacts"]) == 1
            assert comp_data["artifacts"][0]["name"] == "grounded_response"
