"""Proposal-based document chunking for the RAG pipeline.

Splits markdown text using structural cues (blank lines, headings), then
merges undersized blocks, splits oversized blocks by sentence, and adds
sentence overlap for coherence.
"""

from __future__ import annotations

import re
from typing import List

from app.core.config import get_settings
from app.utils.text import normalize_whitespace, split_sentences


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
_BLANK_LINE_RE = re.compile(r"\n{2,}")


def _split_by_structure(text: str) -> List[str]:
    """Split *text* on blank lines **and** markdown headings.

    Headings are kept with their following content (the heading line is
    **not** dropped).
    """
    # First, insert an extra newline before every heading so that the
    # subsequent blank-line split always separates headings.
    text = _HEADING_RE.sub(r"\n\n\1 ", text)
    blocks = _BLANK_LINE_RE.split(text)
    return [b.strip() for b in blocks if b.strip()]


def _merge_small_blocks(
    blocks: List[str],
    min_chars: int,
) -> List[str]:
    """Merge consecutive blocks that are shorter than *min_chars*."""
    merged: List[str] = []
    buf: str = ""

    for block in blocks:
        candidate = f"{buf}\n\n{block}".strip() if buf else block
        if len(candidate) < min_chars:
            buf = candidate
        else:
            merged.append(candidate)
            buf = ""

    if buf:
        # Attach leftover to last chunk or keep it standalone.
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{buf}"
        else:
            merged.append(buf)

    return merged


def _split_oversized(
    blocks: List[str],
    max_chars: int,
) -> List[str]:
    """Break blocks exceeding *max_chars* into sentence-level chunks."""
    result: List[str] = []

    for block in blocks:
        if len(block) <= max_chars:
            result.append(block)
            continue

        sentences = split_sentences(block)
        current = ""
        for sent in sentences:
            candidate = f"{current} {sent}".strip() if current else sent
            if len(candidate) > max_chars and current:
                result.append(current)
                current = sent
            else:
                current = candidate

        if current:
            result.append(current)

    return result


def _add_sentence_overlap(
    chunks: List[str],
    overlap: int,
) -> List[str]:
    """Prepend the last *overlap* sentences of each chunk to the next chunk."""
    if overlap <= 0 or len(chunks) < 2:
        return chunks

    result: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_sentences = split_sentences(chunks[i - 1])
        overlap_text = " ".join(prev_sentences[-overlap:])
        merged = f"{overlap_text} {chunks[i]}".strip()
        result.append(merged)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def proposal_based_chunk_text(text: str) -> List[str]:
    """Chunk *text* using the proposal-based strategy.

    Pipeline:
        1. Normalize whitespace.
        2. Split on blank lines and markdown headings.
        3. Merge blocks smaller than ``min_chunk_chars``.
        4. Split blocks larger than ``max_chunk_chars`` by sentence.
        5. Add sentence overlap for cross-chunk coherence.

    All thresholds are read from :func:`app.core.config.get_settings`.

    Parameters
    ----------
    text:
        Raw document text (markdown-formatted).

    Returns
    -------
    List[str]
        Ordered list of text chunks.
    """
    settings = get_settings()
    max_chars: int = settings.max_chunk_chars
    min_chars: int = settings.min_chunk_chars
    overlap: int = settings.chunk_overlap_sentences

    text = normalize_whitespace(text)
    if not text:
        return []

    blocks = _split_by_structure(text)
    blocks = _merge_small_blocks(blocks, min_chars)
    blocks = _split_oversized(blocks, max_chars)
    blocks = _add_sentence_overlap(blocks, overlap)

    return blocks
