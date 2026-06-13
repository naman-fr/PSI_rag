"""Gemini embedding service with batch support and L2 normalisation.

Wraps the ``google.genai`` client to produce task-prefixed embeddings for
the question-answering domain.  All vectors are L2-normalised so that
inner-product search is equivalent to cosine similarity.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from google import genai
from langsmith import traceable

from app.core.config import get_settings
from app.core.constants import EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Produce embeddings via Google Gemini with local fallback."""

    _local_embeddings = None
    _fallback_active = None  # None: untested, True: fallback to local, False: use Gemini

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model: str = model or settings.embed_model
        self._dimension: int = settings.embed_dimension

        if EmbeddingService._fallback_active is None:
            if not settings.gemini_api_key or settings.gemini_api_key in ["mock-gemini-key", "missing-key-placeholder", ""]:
                EmbeddingService._fallback_active = True
                logger.warning("Gemini API key is missing or placeholder. Falling back to local embeddings.")
            else:
                try:
                    logger.info("Testing Gemini embedding API availability...")
                    test_client = genai.Client(api_key=settings.gemini_api_key)
                    test_client.models.embed_content(
                        model=self._model,
                        contents="test",
                    )
                    EmbeddingService._fallback_active = False
                    logger.info("Gemini embedding API is available.")
                except Exception as e:
                    logger.warning(
                        "Gemini embedding API test failed: %s. Falling back to local embeddings.",
                        str(e),
                    )
                    EmbeddingService._fallback_active = True

        if EmbeddingService._fallback_active:
            if EmbeddingService._local_embeddings is None:
                logger.info(
                    "Initializing local HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)..."
                )
                from langchain_community.embeddings import HuggingFaceEmbeddings

                EmbeddingService._local_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        else:
            self._client = genai.Client(api_key=settings.gemini_api_key)

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        if EmbeddingService._fallback_active:
            return 384
        return self._dimension

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """Return the L2-normalised version of *vec*."""
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Call Gemini ``embed_content`` for a batch of texts.

        Returns a list of L2-normalised numpy vectors.
        """
        from concurrent.futures import ThreadPoolExecutor

        def embed_one(text: str) -> np.ndarray:
            response = self._client.models.embed_content(
                model=self._model,
                contents=text,
            )
            emb = response.embeddings[0]
            vec = np.array(emb.values, dtype=np.float32)
            return self._l2_normalize(vec)

        # Use ThreadPoolExecutor to make parallel requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            vectors = list(executor.map(embed_one, texts))

        return vectors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(name="embed_query")
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string.

        The query is prefixed with the task-type prefix defined in
        ``app.core.constants.EMBED_QUERY_PREFIX`` (only for Gemini).

        Parameters
        ----------
        text:
            Raw user query.

        Returns
        -------
        np.ndarray
            L2-normalised embedding vector of shape ``(embed_dimension,)``.
        """
        if EmbeddingService._fallback_active:
            res = EmbeddingService._local_embeddings.embed_query(text)
            vec = np.array(res, dtype=np.float32)
            return self._l2_normalize(vec)

        prefixed = f"{EMBED_QUERY_PREFIX}{text}"
        vectors = self._embed_batch([prefixed])
        return vectors[0]

    @traceable(name="embed_documents")
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of document chunks.

        Each chunk is prefixed with ``EMBED_DOC_PREFIX`` (only for Gemini).

        Parameters
        ----------
        texts:
            Document chunks to embed.

        Returns
        -------
        List[np.ndarray]
            List of L2-normalised embedding vectors.
        """
        if not texts:
            return []

        if EmbeddingService._fallback_active:
            res = EmbeddingService._local_embeddings.embed_documents(texts)
            return [self._l2_normalize(np.array(v, dtype=np.float32)) for v in res]

        prefixed = [f"{EMBED_DOC_PREFIX}{t}" for t in texts]

        # Gemini supports large batches; split only if needed to stay
        # under the API per-request content limit (≈ 2 048 items).
        batch_size = 2048
        all_vectors: List[np.ndarray] = []
        for start in range(0, len(prefixed), batch_size):
            batch = prefixed[start : start + batch_size]
            all_vectors.extend(self._embed_batch(batch))
            logger.debug(
                "Embedded batch %d–%d of %d chunks",
                start,
                start + len(batch),
                len(prefixed),
            )

        return all_vectors

