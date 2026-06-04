"""FAISS-based vector retrieval with async-safe persistence.

Uses ``IndexFlatIP`` (inner-product) which, when combined with L2-normalised
vectors from :class:`app.rag.embeddings.EmbeddingService`, is equivalent to
cosine similarity search.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from langsmith import traceable

import faiss
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RetrievalResult(TypedDict):
    """Single retrieval hit returned by :meth:`FAISSRetriever.search`."""

    score: float
    text: str
    metadata: Dict[str, Any]


class FAISSRetriever:
    """FAISS and Pinecone vector store retriever wrapper.

    Thread-safety for index mutations (``build_index``, ``save_index``,
    ``load_index``) is guaranteed via an :class:`asyncio.Lock`.

    Parameters
    ----------
    dimension:
        Vector dimensionality.  Defaults to ``Settings.embed_dimension``.
    """

    def __init__(self, dimension: int | None = None) -> None:
        settings = get_settings()
        self._dimension: int = dimension or settings.embed_dimension
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._backend = settings.vector_backend

        if self._backend == "pinecone":
            try:
                from pinecone import Pinecone
            except Exception as e:
                if "renamed from `pinecone-client` to `pinecone`" in str(e) or "pinecone-client" in str(e).lower():
                    import subprocess
                    import sys
                    import importlib
                    logger.warning("Detected broken pinecone-client package in retriever. Attempting programmatic uninstall and force-reinstall...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pinecone-client"])
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "pinecone"])
                        importlib.invalidate_caches()
                        from pinecone import Pinecone
                    except Exception as ex:
                        logger.exception("Failed to programmatically fix pinecone packages", error=str(ex))
                        raise e
                else:
                    raise e
            self._pc = Pinecone(api_key=settings.pinecone_api_key)
            self._pinecone_index = self._pc.Index(settings.pinecone_index_name)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    async def build_index(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Build a new index from *vectors* and attach *metadata*.

        Parameters
        ----------
        vectors:
            Array of shape ``(n, dimension)`` with **L2-normalised** vectors.
        metadata:
            Parallel list of dicts carrying chunk text and provenance info.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Expected vectors of shape (n, {self._dimension}), "
                f"got {vectors.shape}"
            )
        if len(metadata) != vectors.shape[0]:
            raise ValueError(
                f"metadata length ({len(metadata)}) != vectors rows ({vectors.shape[0]})"
            )

        if self._backend == "pinecone":
            async with self._lock:
                try:
                    self._pinecone_index.delete(delete_all=True)
                    logger.info("Cleared all existing vectors from Pinecone index")
                except Exception as clear_err:
                    logger.warning("Failed to clear Pinecone index: %s", clear_err)

                upsert_data = []
                for idx, (vec, meta) in enumerate(zip(vectors, metadata)):
                    chunk_meta = dict(meta)
                    chunk_meta["text"] = meta.get("text", "")
                    chunk_id = f"{meta.get('source', 'doc')}_chunk_{meta.get('chunk_id', idx)}"
                    upsert_data.append((str(chunk_id), vec.tolist(), chunk_meta))
                
                batch_size = 100
                for i in range(0, len(upsert_data), batch_size):
                    batch = upsert_data[i:i+batch_size]
                    self._pinecone_index.upsert(vectors=batch)
            logger.info("Upserted %d vectors to Pinecone index", vectors.shape[0])
            return

        async with self._lock:
            index = faiss.IndexFlatIP(self._dimension)
            index.add(vectors.astype(np.float32))
            self._index = index
            self._metadata = list(metadata)

        logger.info("Built FAISS index with %d vectors (dim=%d)", vectors.shape[0], self._dimension)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @traceable(name="retrieve_context")
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> List[RetrievalResult]:
        """Search the index for nearest neighbours.

        Parameters
        ----------
        query_vector:
            L2-normalised query embedding of shape ``(dimension,)``.
        top_k:
            Max results to return.  Defaults to ``Settings.top_k``.
        score_threshold:
            Min cosine score.  Defaults to ``Settings.retrieval_score_threshold``.

        Returns
        -------
        List[RetrievalResult]
            Ranked results (highest score first), each containing
            ``score``, ``text``, and ``metadata``.
        """
        settings = get_settings()
        top_k = top_k or settings.top_k
        score_threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

        if self._backend == "pinecone":
            response = self._pinecone_index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            results: List[RetrievalResult] = []
            for match in response.matches:
                score = match.score
                if score < score_threshold:
                    continue
                meta = match.metadata or {}
                text = meta.pop("text", "")
                results.append(
                    RetrievalResult(
                        score=float(score),
                        text=text,
                        metadata=meta,
                    )
                )
            return results

        if self._index is None or self._index.ntotal == 0:
            logger.warning("search() called on empty index")
            return []

        qv = query_vector.astype(np.float32).reshape(1, -1)
        # Clamp top_k to index size.
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(qv, k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < score_threshold:
                continue
            meta = self._metadata[int(idx)]
            results.append(
                RetrievalResult(
                    score=float(score),
                    text=meta.get("text", ""),
                    metadata=meta,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save_index(self, path: str | Path) -> None:
        """Persist the FAISS index and metadata to disk.

        Creates two files at *path*:
        - ``index.faiss`` – the FAISS binary index.
        - ``metadata.json`` – the metadata list as JSON.

        Parameters
        ----------
        path:
            Directory where index artefacts are written.
        """
        if self._backend == "pinecone":
            logger.info("Pinecone backend is cloud-managed; skipping local save_index")
            return

        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        async with self._lock:
            if self._index is None:
                raise RuntimeError("No index to save – call build_index() first")
            faiss.write_index(self._index, str(dir_path / "index.faiss"))
            (dir_path / "metadata.json").write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logger.info("Saved FAISS index to %s", dir_path)

    async def load_index(self, path: str | Path) -> None:
        """Load a previously saved FAISS index from disk.

        Parameters
        ----------
        path:
            Directory containing ``index.faiss`` and ``metadata.json``.
        """
        if self._backend == "pinecone":
            logger.info("Pinecone backend is cloud-managed; index loaded automatically from remote")
            return

        dir_path = Path(path)
        index_file = dir_path / "index.faiss"
        meta_file = dir_path / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"Index artefacts not found in {dir_path}. "
                "Expected index.faiss and metadata.json."
            )

        async with self._lock:
            self._index = faiss.read_index(str(index_file))
            self._metadata = json.loads(meta_file.read_text(encoding="utf-8"))

        logger.info(
            "Loaded FAISS index from %s (%d vectors)",
            dir_path,
            self._index.ntotal,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return ``True`` if the index is populated and searchable."""
        if self._backend == "pinecone":
            return True
        return self._index is not None and self._index.ntotal > 0

    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        if self._backend == "pinecone":
            try:
                stats = self._pinecone_index.describe_index_stats()
                return stats.total_vector_count
            except Exception:
                return 0
        if self._index is None:
            return 0
        return self._index.ntotal
