"""Document loading and indexing pipeline.

Reads markdown files from disk, chunks them with the proposal-based
strategy, embeds via Gemini, and builds a FAISS index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.core.config import get_settings
from app.rag.chunking import proposal_based_chunk_text
from app.rag.embeddings import EmbeddingService
from app.rag.retrieval import FAISSRetriever
from app.utils.text import estimate_tokens, normalize_whitespace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_markdown_documents(docs_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    """Load all ``*.md`` files from *docs_dir*.

    Each file is returned as a dict with keys:

    - ``source`` – filename (stem + suffix).
    - ``page``   – always ``1`` for markdown (no pagination).
    - ``text``   – normalised full document text.

    Parameters
    ----------
    docs_dir:
        Directory containing markdown files.  Defaults to
        ``Settings.docs_dir``.

    Returns
    -------
    List[Dict[str, Any]]
        One record per markdown file found.
    """
    settings = get_settings()
    dir_path = Path(docs_dir or settings.docs_dir)

    if not dir_path.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {dir_path}")

    records: List[Dict[str, Any]] = []
    for md_file in sorted(dir_path.glob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8")
        clean_text = normalize_whitespace(raw_text)
        if not clean_text:
            logger.warning("Skipping empty file: %s", md_file.name)
            continue
        records.append(
            {
                "source": md_file.name,
                "page": 1,
                "text": clean_text,
            }
        )

    logger.info("Loaded %d markdown documents from %s", len(records), dir_path)
    return records


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


async def index_documents(
    records: List[Dict[str, Any]],
    embedding_service: EmbeddingService,
    retriever: FAISSRetriever,
) -> Dict[str, Any]:
    """Chunk, embed, and index a list of document records.

    Parameters
    ----------
    records:
        Output of :func:`load_markdown_documents`.
    embedding_service:
        An initialised :class:`app.rag.embeddings.EmbeddingService`.
    retriever:
        An initialised :class:`app.rag.retrieval.FAISSRetriever` (will be
        populated in-place).

    Returns
    -------
    Dict[str, Any]
        Summary with keys ``documents_loaded``, ``chunks_indexed``,
        ``sources``.
    """
    all_texts: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    for record in records:
        source = record["source"]
        page = record["page"]
        text = record["text"]

        chunks = proposal_based_chunk_text(text)

        for chunk_idx, chunk_text in enumerate(chunks):
            all_texts.append(chunk_text)
            all_metadata.append(
                {
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_idx,
                    "chars": len(chunk_text),
                    "tokens_est": estimate_tokens(chunk_text),
                    "text": chunk_text,
                }
            )

    logger.info(
        "Chunking complete: %d documents -> %d chunks",
        len(records),
        len(all_texts),
    )

    # --- Embed ---
    vectors = embedding_service.embed_documents(all_texts)
    vectors_array = np.stack(vectors, axis=0)

    # --- Build index ---
    await retriever.build_index(vectors_array, all_metadata)

    # --- Persist index ---
    settings = get_settings()
    await retriever.save_index(settings.index_dir)

    summary: Dict[str, Any] = {
        "documents_loaded": len(records),
        "chunks_indexed": len(all_texts),
        "sources": list({r["source"] for r in records}),
    }

    logger.info("Indexing complete: %s", summary)
    return summary
