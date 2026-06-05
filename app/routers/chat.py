"""Chat API endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.requests import ChatRequest, FeedbackRequest
from app.schemas.responses import ChatResponse, ConversationHistory

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message through the RAG pipeline.

    Returns a grounded answer with verification metadata.
    """
    from app.main import get_orchestrator

    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized. Run /api/v1/ingest first.")

    try:
        result = await orchestrator.process_query(
            question=request.question,
            username=request.username,
            session_id=request.session_id,
            image_id=request.image_id,
            image_url=request.image_url,
        )
        return ChatResponse(
            trace_id=result["trace_id"],
            mode=result["mode"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result.get("sources", []),
            verdict=result.get("verdict", {"supported": False, "confidence": 0.0, "reason": "unknown"}),
            usage=result.get("usage", {}),
            cached=result.get("cached", False),
            timestamp=result["timestamp"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{username}", response_model=ConversationHistory)
async def get_history(username: str, session_id: str = None):
    """Get conversation history for a user."""
    from app.main import get_conversation_manager

    manager = get_conversation_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    sid = session_id or f"session_{username}"
    messages = await manager.get_full_history(username, sid)

    from app.main import get_summary_manager

    summary_mgr = get_summary_manager()
    summary = await summary_mgr.get_summary(username) if summary_mgr else None

    return ConversationHistory(
        username=username,
        session_id=sid,
        messages=messages,
        summary=summary,
    )


@router.delete("/history/{username}")
async def clear_history(username: str, session_id: str = None):
    """Clear conversation history for a user."""
    from app.main import get_conversation_manager

    manager = get_conversation_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    sid = session_id or f"session_{username}"
    await manager.clear_conversation(username, sid)
    return {"status": "cleared", "username": username, "session_id": sid}
