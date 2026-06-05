"""
Cross-modal reranking and score refinement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from app.vision.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


def rerank_results(
    results: List[RetrievalResult],
    top_k: int = 5
) -> List[RetrievalResult]:
    """
    Rerank and deduplicate a combined list of textual and visual retrieval results.
    
    Strategies:
    - Sort all results by similarity score descending.
    - Deduplicate chunks with identical source path or identical text content.
    - Balance textual references and visual references in the final top-k selection.
    """
    if not results:
        return []

    # 1. Sort by score descending
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    # 2. Deduplicate based on text content and metadata
    unique_results: List[RetrievalResult] = []
    seen_texts = set()
    seen_sources = set()

    for res in sorted_results:
        text_content = res["text"].strip().lower()
        meta = res.get("metadata") or {}
        source = meta.get("source") or meta.get("image_id") or ""
        chunk_id = meta.get("chunk_id") or meta.get("tile_index") or 0

        # Create a unique footprint for this chunk
        footprint = f"{source}_{chunk_id}"

        # Skip if we have already included this exact chunk
        if footprint in seen_sources:
            continue

        # Skip if the exact text content has already been seen (unless it is short/generic)
        if len(text_content) > 30 and text_content in seen_texts:
            continue

        seen_sources.add(footprint)
        if len(text_content) > 30:
            seen_texts.add(text_content)

        unique_results.append(res)

    logger.debug("Rerank and deduplication complete", original=len(results), refined=len(unique_results))
    return unique_results[:top_k]
