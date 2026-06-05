"""
Caching for heavy image embeddings, OCR text, and visual tags.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.cache.redis_client import CacheBackend

logger = logging.getLogger(__name__)


class ImageFeatureCache:
    """Handles caching and retrieval of precomputed image features (embeddings, OCR, tags)."""

    def __init__(self, cache_backend: CacheBackend) -> None:
        self.cache = cache_backend

    async def set_features(
        self,
        image_id: str,
        embedding: np.ndarray,
        ocr_text: str,
        tags: List[str],
        ttl: Optional[int] = None
    ) -> None:
        """Cache embedding, OCR text, and tags for a given image ID."""
        key = f"img_feat:{image_id}"
        payload = {
            "embedding": embedding.tolist(),
            "ocr_text": ocr_text,
            "tags": tags
        }
        try:
            value_str = json.dumps(payload)
            await self.cache.set(key, value_str, ttl=ttl)
            logger.debug("Successfully cached image features for image_id=%s", image_id)
        except Exception as e:
            logger.warning("Failed to cache image features for image_id=%s: %s", image_id, e)

    async def get_features(
        self,
        image_id: str
    ) -> Optional[Tuple[np.ndarray, str, List[str]]]:
        """
        Retrieve cached embedding, OCR text, and tags.
        Returns:
            Tuple of (embedding_numpy_array, ocr_text_str, tags_list) or None if cache miss.
        """
        key = f"img_feat:{image_id}"
        try:
            value_str = await self.cache.get(key)
            if not value_str:
                return None
            
            data = json.loads(value_str)
            embedding = np.array(data["embedding"], dtype=np.float32)
            ocr_text = data.get("ocr_text", "")
            tags = data.get("tags", [])
            logger.debug("Image feature cache hit for image_id=%s", image_id)
            return embedding, ocr_text, tags
        except Exception as e:
            logger.warning("Failed to retrieve image features for image_id=%s: %s", image_id, e)
            return None

    async def delete_features(self, image_id: str) -> None:
        """Invalidate cached features for an image ID."""
        key = f"img_feat:{image_id}"
        await self.cache.delete(key)
