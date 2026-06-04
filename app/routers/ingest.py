"""Document ingestion endpoints."""

from fastapi import APIRouter, HTTPException

from app.schemas.requests import IngestRequest
from app.schemas.responses import IngestResponse

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


@router.post("", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest = None):
    """
    Ingest markdown documents into the vector store.

    Loads documents from the configured directory, chunks them,
    generates embeddings, and indexes them.
    """
    if request is None:
        request = IngestRequest()

    from app.main import run_ingestion

    try:
        result = await run_ingestion(
            source_dir=request.source_dir,
            force_reindex=request.force_reindex,
        )
        return IngestResponse(
            documents_loaded=result["documents_loaded"],
            chunks_indexed=result["chunks_indexed"],
            status="success",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
