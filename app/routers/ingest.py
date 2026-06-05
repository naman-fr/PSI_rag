"""Document ingestion endpoints."""

from fastapi import APIRouter, HTTPException, File, Form, UploadFile

from app.schemas.requests import IngestRequest, ImageSearchRequest
from app.schemas.responses import IngestResponse, UploadImageResponse, ImageSearchResult

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


@router.post("/image", response_model=UploadImageResponse)
async def upload_image(
    username: str = Form(..., description="Username associated with upload"),
    file: UploadFile = File(..., description="Binary image file (JPEG/PNG/WEBP)")
):
    """
    Ingest an image into the visual database.
    Performs validation, layout-aware OCR extraction, object/tag detection,
    caches visual features, and indexes the image vector.
    """
    from app.vision.ingestion.image_loader import validate_image_format, load_image_from_bytes, preprocess_image
    from app.main import get_orchestrator

    # 1. Validate format
    if not validate_image_format(file.content_type or ""):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {file.content_type}. Only JPEG, PNG, and WEBP are allowed."
        )

    # 2. Check orchestrator
    orchestrator = get_orchestrator()
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Vector retriever and orchestrator are not initialized.")

    try:
        # 3. Read bytes & load image
        contents = await file.read()
        pil_img = load_image_from_bytes(contents)
        pil_img = preprocess_image(pil_img)

        # 4. Generate unique ID
        import uuid
        image_id = f"img_{uuid.uuid4().hex[:12]}"

        # 5. Extract OCR & tags
        ocr_text = await orchestrator.ocr_extractor.extract_text(pil_img)
        tags = await orchestrator.ocr_extractor.detect_objects_and_tags(pil_img)

        # 6. Compute embedding
        image_vector = orchestrator.multimodal_embedder.embed_image(pil_img)

        # 7. Cache features
        await orchestrator.image_feature_cache.set_features(
            image_id=image_id,
            embedding=image_vector,
            ocr_text=ocr_text,
            tags=tags,
            ttl=orchestrator.settings.cache_ttl_seconds * 2
        )

        # 8. Dynamically index into DualRetriever's visual index
        # Store metadata for similarity queries
        metadata = {
            "image_id": image_id,
            "username": username,
            "text": ocr_text,
            "caption": ", ".join(tags),
            "tags": tags,
            "source": image_id,
        }
        await orchestrator.retriever.add_visual_vector(image_vector, metadata)

        return UploadImageResponse(
            image_id=image_id,
            ocr_text=ocr_text,
            tags=tags,
            status="Image ingested and indexed successfully."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image ingestion failed: {str(e)}")


@router.post("/search-images", response_model=list[ImageSearchResult])
async def search_images(request: ImageSearchRequest):
    """
    Search the visual index using a text query.
    Returns matching image metadata and similarity scores.
    """
    from app.main import get_orchestrator

    orchestrator = get_orchestrator()
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Vector retriever and orchestrator are not initialized.")

    try:
        # 1. Embed query in multimodal space
        query_vector = orchestrator.multimodal_embedder.embed_text(request.query)

        # 2. Search FAISS visual index
        results = orchestrator.retriever.search_visual(
            query_vector=query_vector,
            top_k=request.top_k or 3,
            score_threshold=orchestrator.settings.retrieval_score_threshold
        )

        # 3. Format response
        search_results = []
        for r in results:
            meta = r.get("metadata") or {}
            search_results.append(
                ImageSearchResult(
                    image_id=meta.get("image_id", "unknown"),
                    score=r["score"],
                    caption=r["text"],
                    tags=meta.get("tags", [])
                )
            )
        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")
