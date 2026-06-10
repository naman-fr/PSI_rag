"""Post-retrieval reranking and deduplication.

Applies score-threshold filtering, removes near-duplicate chunks, and
returns the top-k results.
"""

from __future__ import annotations

import logging
from typing import List

from app.core.config import get_settings
from app.rag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _text_overlap_ratio(a: str, b: str) -> float:
    """Compute character-level overlap ratio between *a* and *b*.

    Uses the Jaccard coefficient on whitespace-delimited word sets,
    which is fast and sufficient for near-duplicate detection.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _deduplicate(
    results: List[RetrievalResult],
    overlap_threshold: float = 0.90,
) -> List[RetrievalResult]:
    """Remove results whose text overlaps an already-kept result by ≥ *overlap_threshold*.

    Results are assumed to be sorted by descending score so that the
    highest-scoring version of near-duplicate content is always kept.
    """
    kept: List[RetrievalResult] = []
    for result in results:
        text = result["text"]
        is_dup = any(
            _text_overlap_ratio(text, k["text"]) >= overlap_threshold
            for k in kept
        )
        if not is_dup:
            kept.append(result)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rerank_results(
    results: List[RetrievalResult],
    top_k: int | None = None,
) -> List[RetrievalResult]:
    """Rerank, deduplicate, and filter retrieval results.

    Pipeline:
        1. Sort by descending score (already expected, but enforced).
        2. Apply score-threshold filter (``Settings.retrieval_score_threshold``).
        3. Deduplicate near-identical chunks (≥ 90 % word-level overlap).
        4. Apply Cross-Document Correlation balancing to ensure representation from
           different relevant source files.
        5. Truncate to *top_k*.

    Parameters
    ----------
    results:
        Raw retrieval hits from :class:`app.rag.retrieval.FAISSRetriever`.
    top_k:
        Maximum results to return.  Defaults to ``Settings.top_k``.

    Returns
    -------
    List[RetrievalResult]
        Filtered and deduplicated results, ordered by descending score.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k
    score_threshold: float = settings.retrieval_score_threshold

    if not results:
        return []

    # 1. Sort by score descending.
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)

    # 2. Score-threshold filter.
    ranked = [r for r in ranked if r["score"] >= score_threshold]

    # 3. Deduplicate near-identical chunks.
    ranked = _deduplicate(ranked)

    if len(ranked) <= 2 or top_k <= 2:
        logger.debug(
            "Reranking (no balancing): %d input -> %d output",
            len(results),
            len(ranked[:top_k]),
        )
        return ranked[:top_k]

    # 4. Cross-Document Correlation balancing
    selected: List[RetrievalResult] = []

    # Keep top 2 overall (high priority, must have)
    selected.extend(ranked[:2])

    # Find which documents are represented in the top 2
    represented_docs = {
        r.get("metadata", {}).get("source", "unknown") for r in selected
    }

    # Pull in the highest scoring chunk from any unrepresented document
    # provided its score is above a reasonable minimum relevance threshold (threshold + 0.05 or 0.45)
    cross_doc_candidates = []
    min_cross_score = max(score_threshold + 0.05, 0.45)

    for r in ranked[2:]:
        doc_name = r.get("metadata", {}).get("source", "unknown")
        if doc_name not in represented_docs:
            if r["score"] >= min_cross_score:
                cross_doc_candidates.append(r)
                represented_docs.add(doc_name)

    # Add cross-document chunks up to remaining slots (ordered by score)
    cross_doc_candidates.sort(key=lambda r: r["score"], reverse=True)
    remaining_slots = top_k - len(selected)
    selected.extend(cross_doc_candidates[:remaining_slots])

    # Fill any remaining slots with the leftover highest-scoring chunks overall
    remaining_slots = top_k - len(selected)
    if remaining_slots > 0:
        selected_ids = {id(r) for r in selected}
        leftovers = [r for r in ranked if id(r) not in selected_ids]
        selected.extend(leftovers[:remaining_slots])

    # Sort final selected list by score descending to present logical order to LLM
    selected.sort(key=lambda r: r["score"], reverse=True)

    logger.debug(
        "Reranking (with cross-doc balancing): %d input -> %d output (top_k=%d)",
        len(results),
        len(selected),
        top_k,
    )

    return selected

