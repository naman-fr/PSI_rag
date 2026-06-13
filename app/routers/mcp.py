import logging
from pathlib import Path
from fastapi import APIRouter, Request
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

from app.core.config import get_settings
from app.main import get_orchestrator

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("PSI-Logistics-RAG")

# Base directory for docs
DOCS_DIR = Path("rag_docs/rag_docs")

# --- Resources ---
@mcp.resource("resource://carrier_sla")
def get_carrier_sla() -> str:
    """Read the carrier SLA agreement document (DOC1)."""
    path = DOCS_DIR / "DOC1_carrier_sla_agreement.md"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Carrier SLA agreement file not found."

@mcp.resource("resource://customs_tariff")
def get_customs_tariff() -> str:
    """Read the customs tariff reference document (DOC2)."""
    path = DOCS_DIR / "DOC2_customs_tariff_reference.md"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Customs tariff reference file not found."

@mcp.resource("resource://shipment_delay")
def get_shipment_delay() -> str:
    """Read the shipment delay & exception policy document (DOC3)."""
    path = DOCS_DIR / "DOC3_shipment_delay_policy.md"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Shipment delay policy file not found."

# --- Tools ---
@mcp.tool()
async def search_logistics_docs(query: str, top_k: int = 5) -> str:
    """Search the GlobalFreight Carrier SLAs, customs tariffs, and delay policies.
    
    Returns matching document chunks with their similarity scores and sources.
    """
    orchestrator = get_orchestrator()
    if not orchestrator:
        return "RAG orchestrator not initialized on server."
        
    try:
        # Perform retrieval
        emb = orchestrator.embedding_service.embed_query(query)
        results = orchestrator.retriever.search(emb, top_k=top_k, score_threshold=0.0)
        
        if not results:
            return "No matching document chunks found."
            
        formatted = []
        for i, res in enumerate(results):
            source = res.get("metadata", {}).get("source", "unknown")
            score = res.get("score", 0.0)
            text = res.get("text", "")
            formatted.append(f"[{i+1}] Source: {source} (Score: {score:.4f})\nContent: {text}\n")
            
        return "\n---\n".join(formatted)
    except Exception as e:
        logger.exception("Error searching docs via MCP")
        return f"Error executing search: {str(e)}"

@mcp.tool()
async def answer_carrier_question(question: str, username: str = "mcp_client") -> str:
    """Ask a question to the verified and guardrailed RAG QA pipeline.
    
    Returns the grounded answer or a refusal response if not supported.
    """
    orchestrator = get_orchestrator()
    if not orchestrator:
        return "RAG orchestrator not initialized on server."
        
    try:
        response = await orchestrator.process_query(
            question=question,
            username=username,
        )
        return response.get("answer", "No answer generated.")
    except Exception as e:
        logger.exception("Error processing RAG query via MCP")
        return f"Error executing RAG: {str(e)}"

# --- FastAPI SSE Routing ---
router = APIRouter(prefix="/mcp", tags=["mcp"])
sse = SseServerTransport("/api/v1/mcp/messages")

@router.get("/sse")
async def handle_sse(request: Request):
    """Establishes the SSE stream connection for MCP."""
    async with sse.connect_sse(
        request.scope, 
        request.receive, 
        request._send
    ) as (read_stream, write_stream):
        await mcp.run(
            read_stream, 
            write_stream, 
            mcp.create_initialization_options()
        )

# Mounted directly on app in app/main.py
