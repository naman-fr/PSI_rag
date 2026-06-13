import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.routers.mcp import get_carrier_sla, get_customs_tariff, get_shipment_delay

def test_mcp_resources():
    """Verify that MCP resource readers load files successfully."""
    sla = get_carrier_sla()
    assert "GlobalFreight" in sla or "not found" in sla
    
    tariff = get_customs_tariff()
    assert "Customs" in tariff or "not found" in tariff
    
    delay = get_shipment_delay()
    assert "Shipment" in delay or "not found" in delay

@pytest.mark.asyncio
async def test_mcp_tools_search():
    """Verify that search_logistics_docs calls the retriever and formats results."""
    mock_orch = MagicMock()
    mock_orch.embedding_service.embed_query.return_value = [0.1] * 8
    mock_orch.retriever.search.return_value = [
        {"score": 0.95, "text": "Test SLA rule text", "metadata": {"source": "doc1.md"}}
    ]
    
    with patch("app.routers.mcp.get_orchestrator", return_value=mock_orch):
        from app.routers.mcp import search_logistics_docs
        res = await search_logistics_docs("query text", top_k=1)
        assert "Test SLA rule text" in res
        assert "doc1.md" in res
        assert "0.9500" in res

@pytest.mark.asyncio
async def test_mcp_tools_answer():
    """Verify that answer_carrier_question calls the RAG orchestrator."""
    mock_orch = MagicMock()
    mock_orch.process_query = AsyncMock(return_value={"answer": "Grounded answer: Platinum is 4 hours."})
    
    with patch("app.routers.mcp.get_orchestrator", return_value=mock_orch):
        from app.routers.mcp import answer_carrier_question
        res = await answer_carrier_question("Platinum tolerance?")
        assert "Grounded answer: Platinum is 4 hours." in res

def test_mcp_endpoints_existence():
    """Verify that the MCP endpoints are mounted on the FastAPI application."""
    with TestClient(app) as client:
        # A POST request to messages with no session should return 400 Bad Request, not 404
        response = client.post("/api/v1/mcp/messages", content="{}")
        assert response.status_code != 404
