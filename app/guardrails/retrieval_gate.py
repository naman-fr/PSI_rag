"""Retrieval quality gate.

Validates that the retrieved chunks meet minimum quality thresholds
*before* the context is sent to the LLM for answer generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.core.config import get_settings


def check_retrieval_quality(
    results: List[Dict[str, Any]],
    min_score: float | None = None,
    min_chunks: int = 1,
) -> Tuple[bool, str]:
    """Evaluate whether retrieved results are good enough to generate an answer.

    Checks applied (in order):

    1. **Minimum chunk count** – at least *min_chunks* results returned.
    2. **Minimum similarity score** – every result must meet *min_score*.
    3. **Non-empty content** – every chunk must contain meaningful text.

    Parameters
    ----------
    results:
        List of retrieval result dicts.  Each dict is expected to have
        at least ``"score"`` (float) and ``"text"`` (str) keys.
    min_score:
        Minimum acceptable cosine-similarity score.  Defaults to
        ``Settings.retrieval_score_threshold``.
    min_chunks:
        Minimum number of chunks required.  Defaults to ``1``.

    Returns
    -------
    tuple[bool, str]
        ``(passed, reason)`` – *passed* is ``True`` when all checks
        succeed; *reason* describes the first failing check (or
        ``"ok"`` on success).
    """
    settings = get_settings()
    if min_score is None:
        min_score = settings.retrieval_score_threshold

    # --- Check 1: minimum chunk count ---
    if len(results) < min_chunks:
        return (
            False,
            f"Insufficient chunks: got {len(results)}, need >= {min_chunks}",
        )

    # --- Check 2: minimum similarity score ---
    for idx, chunk in enumerate(results):
        score = chunk.get("score", 0.0)
        if score < min_score:
            return (
                False,
                f"Chunk {idx} score {score:.3f} below threshold {min_score:.3f}",
            )

    # --- Check 3: non-empty content ---
    for idx, chunk in enumerate(results):
        text = (chunk.get("text") or "").strip()
        if not text:
            return (
                False,
                f"Chunk {idx} has empty content",
            )

    return (True, "ok")
