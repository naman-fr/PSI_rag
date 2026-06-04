"""Context assembly for the generation stage.

Builds a token-budget-aware context string from retrieved chunks, recent
memory, and conversation summaries, following the memory hierarchy:

    summary → recent memory → retrieved context
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.core.config import get_settings
from app.rag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


def _format_chunk_block(result: RetrievalResult, index: int) -> str:
    """Format a single retrieval hit as a source-annotated block.

    Example output::

        [source=DOC1_carrier_sla.md chunk=3 score=0.82]
        The carrier shall deliver…
    """
    meta = result.get("metadata") or {}
    source = meta.get("source", "unknown")
    chunk_id = meta.get("chunk_id", index)
    score = result.get("score", 0.0)
    text = result.get("text", "")
    return f"[source={source} chunk={chunk_id} score={score:.2f}]\n{text}"


def assemble_context(
    chunks: List[RetrievalResult],
    max_chars: int | None = None,
    memory_text: Optional[str] = None,
    summary_text: Optional[str] = None,
) -> str:
    """Assemble a context string for the LLM prompt.

    The context is packed in **memory-hierarchy order** so that the most
    persistent information appears first (and is least likely to be
    truncated by the model's attention):

        1. **Summary** – compressed conversation history.
        2. **Recent memory** – last few user/assistant turns.
        3. **Retrieved chunks** – ranked by relevance score.

    A running character budget (``max_chars``, from
    :func:`app.core.config.get_settings`) prevents exceeding the token
    limit.

    Parameters
    ----------
    chunks:
        Retrieval results, assumed already reranked.
    max_chars:
        Hard character cap.  Defaults to ``Settings.max_context_chars``.
    memory_text:
        Recent conversational turns (pre-formatted).
    summary_text:
        Compressed conversation summary.

    Returns
    -------
    str
        Fully assembled context ready for injection into the system prompt.
    """
    settings = get_settings()
    budget: int = max_chars or settings.max_context_chars
    sections: List[str] = []
    used: int = 0

    # --- 1. Summary (highest priority) ---
    if summary_text and summary_text.strip():
        header = "=== Conversation Summary ==="
        block = f"{header}\n{summary_text.strip()}"
        if used + len(block) <= budget:
            sections.append(block)
            used += len(block)

    # --- 2. Recent memory ---
    if memory_text and memory_text.strip():
        header = "=== Recent Conversation ==="
        block = f"{header}\n{memory_text.strip()}"
        if used + len(block) <= budget:
            sections.append(block)
            used += len(block)

    # --- 3. Retrieved chunks ---
    if chunks:
        chunk_header = "=== Retrieved Context ==="
        sections.append(chunk_header)
        used += len(chunk_header)

        for idx, chunk in enumerate(chunks):
            block = _format_chunk_block(chunk, idx)
            block_len = len(block) + 2  # account for separating newlines
            if used + block_len > budget:
                logger.debug(
                    "Context budget exhausted after %d/%d chunks",
                    idx,
                    len(chunks),
                )
                break
            sections.append(block)
            used += block_len

    context = "\n\n".join(sections)

    logger.debug(
        "Assembled context: %d chars (%d/%d budget), %d chunks included",
        len(context),
        used,
        budget,
        len(chunks),
    )

    return context
