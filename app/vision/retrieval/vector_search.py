"""
Dual-index vector search retriever.
Manages a textual document index and a visual/image chunk index.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
import faiss
import numpy as np
from langsmith import traceable

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RetrievalResult(TypedDict):
    """Single retrieval hit."""
    score: float
    text: str
    metadata: Dict[str, Any]
    type: str  # "text" or "visual"


class DualRetriever:
    """
    Manages two FAISS indices:
    - Text index (dimension 3072, e.g. gemini-embedding-2)
    - Visual index (dimension 1408, e.g. multimodal-embedding-001)
    """

    def __init__(
        self,
        text_dimension: int = 3072,
        visual_dimension: int = 1408
    ) -> None:
        settings = get_settings()
        self._text_dim = text_dimension
        self._visual_dim = visual_dimension

        self._text_index: Optional[faiss.IndexFlatIP] = None
        self._text_metadata: List[Dict[str, Any]] = []

        self._visual_index: Optional[faiss.IndexFlatIP] = None
        self._visual_metadata: List[Dict[str, Any]] = []

        self._lock = asyncio.Lock()
        self._backend = settings.vector_backend  # "faiss" fallback for simplicity

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    async def build_text_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Build the textual document index."""
        if vectors.ndim != 2 or vectors.shape[1] != self._text_dim:
            raise ValueError(f"Expected vectors of shape (n, {self._text_dim}), got {vectors.shape}")
        if len(metadata) != vectors.shape[0]:
            raise ValueError(f"metadata size mismatch: len(metadata)={len(metadata)}, vectors.shape[0]={vectors.shape[0]}")

        async with self._lock:
            index = faiss.IndexFlatIP(self._text_dim)
            index.add(vectors.astype(np.float32))
            self._text_index = index
            self._text_metadata = list(metadata)
        logger.info("Built FAISS Text index with %d vectors", vectors.shape[0])

    async def build_visual_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Build the visual image chunk index."""
        if vectors.ndim != 2 or vectors.shape[1] != self._visual_dim:
            raise ValueError(f"Expected vectors of shape (n, {self._visual_dim}), got {vectors.shape}")
        if len(metadata) != vectors.shape[0]:
            raise ValueError(f"metadata size mismatch")

        async with self._lock:
            index = faiss.IndexFlatIP(self._visual_dim)
            index.add(vectors.astype(np.float32))
            self._visual_index = index
            self._visual_metadata = list(metadata)
        logger.info("Built FAISS Visual index with %d vectors", vectors.shape[0])

    async def add_visual_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Dynamically add a single visual vector and metadata to the index."""
        async with self._lock:
            if self._visual_index is None:
                self._visual_index = faiss.IndexFlatIP(self._visual_dim)
            self._visual_index.add(vector.astype(np.float32).reshape(1, -1))
            self._visual_metadata.append(metadata)
        logger.info("Dynamically added vector to FAISS Visual index (total=%d)", self.visual_count)

    # --- Backward compatibility aliases for text-only pipeline scripts ---
    async def build_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Build the text index (alias for backward compatibility)."""
        await self.build_text_index(vectors, metadata)

    async def save_index(self, path: str | Path) -> None:
        """Save indices (alias for backward compatibility)."""
        await self.save_indices(path)

    async def load_index(self, path: str | Path) -> None:
        """Load indices (alias for backward compatibility)."""
        await self.load_indices(path)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @traceable(name="retrieve_text_context")
    def search_text(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        score_threshold: float = 0.3
    ) -> List[RetrievalResult]:
        """Search the textual index."""
        if self._text_index is None or self._text_index.ntotal == 0:
            logger.debug("search_text called on empty index")
            return []

        qv = query_vector.astype(np.float32).reshape(1, -1)
        k = min(top_k, self._text_index.ntotal)
        scores, indices = self._text_index.search(qv, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < score_threshold:
                continue
            meta = self._text_metadata[int(idx)]
            results.append({
                "score": float(score),
                "text": meta.get("text", ""),
                "metadata": meta,
                "type": "text"
            })
        return results

    @traceable(name="retrieve_visual_context")
    def search_visual(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        score_threshold: float = 0.3
    ) -> List[RetrievalResult]:
        """Search the visual index (using a multimodal image or text vector)."""
        if self._visual_index is None or self._visual_index.ntotal == 0:
            logger.debug("search_visual called on empty index")
            return []

        qv = query_vector.astype(np.float32).reshape(1, -1)
        k = min(top_k, self._visual_index.ntotal)
        scores, indices = self._visual_index.search(qv, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < score_threshold:
                continue
            meta = self._visual_metadata[int(idx)]
            results.append({
                "score": float(score),
                "text": meta.get("text", "") or meta.get("caption", ""),
                "metadata": meta,
                "type": "visual"
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save_indices(self, path: str | Path) -> None:
        """Persist indices and metadata to disk."""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        async with self._lock:
            # Text index
            if self._text_index is not None:
                faiss.write_index(self._text_index, str(dir_path / "text_index.faiss"))
                (dir_path / "text_metadata.json").write_text(
                    json.dumps(self._text_metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            # Visual index
            if self._visual_index is not None:
                faiss.write_index(self._visual_index, str(dir_path / "visual_index.faiss"))
                (dir_path / "visual_metadata.json").write_text(
                    json.dumps(self._visual_metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
        logger.info("Saved FAISS indices to %s", dir_path)

    async def load_indices(self, path: str | Path) -> None:
        """Load indices and metadata from disk."""
        dir_path = Path(path)
        
        async with self._lock:
            # Text index
            text_index_file = dir_path / "text_index.faiss"
            text_meta_file = dir_path / "text_metadata.json"
            if text_index_file.exists() and text_meta_file.exists():
                self._text_index = faiss.read_index(str(text_index_file))
                self._text_metadata = json.loads(text_meta_file.read_text(encoding="utf-8"))
                logger.info("Loaded Text index with %d vectors", self._text_index.ntotal)

            # Visual index
            visual_index_file = dir_path / "visual_index.faiss"
            visual_meta_file = dir_path / "visual_metadata.json"
            if visual_index_file.exists() and visual_meta_file.exists():
                self._visual_index = faiss.read_index(str(visual_index_file))
                self._visual_metadata = json.loads(visual_meta_file.read_text(encoding="utf-8"))
                logger.info("Loaded Visual index with %d vectors", self._visual_index.ntotal)

    @property
    def text_count(self) -> int:
        return self._text_index.ntotal if self._text_index else 0

    @property
    def visual_count(self) -> int:
        return self._visual_index.ntotal if self._visual_index else 0
