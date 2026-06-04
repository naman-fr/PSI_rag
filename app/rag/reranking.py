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
        4. Truncate to *top_k*.

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

    # 4. Truncate.
    ranked = ranked[:top_k]

    logger.debug(
        "Reranking: %d input -> %d output (threshold=%.2f, top_k=%d)",
        len(results),
        len(ranked),
        score_threshold,
        top_k,
    )

    return ranked
