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
    """Produce embeddings via Google Gemini.

    Parameters
    ----------
    model:
        Override the model name (defaults to ``Settings.embed_model``).
    """

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model: str = model or settings.embed_model
        self._dimension: int = settings.embed_dimension
        self._client = genai.Client(api_key=settings.gemini_api_key)

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
        from tenacity import retry, stop_after_attempt, wait_random_exponential

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, min=2, max=10),
            reraise=True
        )
        def _call_api():
            return self._client.models.embed_content(
                model=self._model,
                contents=texts,
            )

        response = _call_api()
        return [
            self._l2_normalize(np.array(emb.values, dtype=np.float32))
            for emb in response.embeddings
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(name="gemini_embed_query")
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string.

        The query is prefixed with the task-type prefix defined in
        ``app.core.constants.EMBED_QUERY_PREFIX``.

        Parameters
        ----------
        text:
            Raw user query.

        Returns
        -------
        np.ndarray
            L2-normalised embedding vector of shape ``(embed_dimension,)``.
        """
        prefixed = f"{EMBED_QUERY_PREFIX}{text}"
        vectors = self._embed_batch([prefixed])
        return vectors[0]

    @traceable(name="gemini_embed_documents")
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of document chunks in a single batch call.

        Each chunk is prefixed with ``EMBED_DOC_PREFIX``.

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
