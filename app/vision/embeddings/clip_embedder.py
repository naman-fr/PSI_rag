"""
Multimodal embedding service (text and image embedding) using Google Gemini or local CLIP.
"""

from __future__ import annotations

import io
import logging
from typing import List, Union
import numpy as np
from PIL import Image
from google import genai
from langsmith import traceable

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """Produce joint text/image embeddings using Google Gemini's multimodal-embedding-001."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.vision_embed_model
        self._dimension = settings.vision_embed_dimension
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._local_model = None
        self._local_processor = None

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """L2 normalise a vector to support cosine similarity via dot product."""
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    def _get_local_model(self):
        """Lazy loader for local CLIP fallback if Gemini API fails or is not desired."""
        if self._local_model is None:
            settings = get_settings()
            try:
                from transformers import CLIPModel, CLIPProcessor
                logger.info("Initializing local CLIP model: %s", settings.local_clip_model)
                self._local_processor = CLIPProcessor.from_pretrained(settings.local_clip_model)
                self._local_model = CLIPModel.from_pretrained(settings.local_clip_model)
            except ImportError:
                logger.error("transformers or torch not installed. Cannot use local CLIP fallback.")
                raise ImportError("Please run 'pip install transformers torch' to use local CLIP.")
        return self._local_model, self._local_processor

    @traceable(name="embed_text_multimodal")
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a 1D embedding vector for text."""
        from tenacity import retry, stop_after_attempt, wait_random_exponential

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, min=2, max=10),
            reraise=True
        )
        def _call_api():
            return self._client.models.embed_content(
                model=self._model,
                contents=text,
            )

        try:
            response = _call_api()
            val = response.embeddings[0].values
            vec = np.array(val, dtype=np.float32)
            return self._l2_normalize(vec)
        except Exception as e:
            logger.warning("Gemini multimodal text embedding failed, attempting local fallback: %s", str(e))
            return self._embed_text_local(text)

    @traceable(name="embed_image_multimodal")
    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Generate a 1D embedding vector for a PIL Image."""
        from tenacity import retry, stop_after_attempt, wait_random_exponential

        # Check if image is in correct mode
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, min=2, max=10),
            reraise=True
        )
        def _call_api():
            return self._client.models.embed_content(
                model=self._model,
                contents=img,
            )

        try:
            response = _call_api()
            val = response.embeddings[0].values
            vec = np.array(val, dtype=np.float32)
            return self._l2_normalize(vec)
        except Exception as e:
            logger.warning("Gemini multimodal image embedding failed, attempting local fallback: %s", str(e))
            return self._embed_image_local(img)

    def _embed_text_local(self, text: str) -> np.ndarray:
        """Local CLIP fallback to embed text."""
        model, processor = self._get_local_model()
        import torch
        with torch.no_grad():
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            text_features = model.get_text_features(**inputs)
            vec = text_features[0].cpu().numpy().astype(np.float32)
            # Pad or truncate if dimensions do not match the expected index dimension
            if len(vec) != self._dimension:
                logger.warning("Local CLIP size %d differs from settings dimension %d. Normalizing.", len(vec), self._dimension)
            return self._l2_normalize(vec)

    def _embed_image_local(self, img: Image.Image) -> np.ndarray:
        """Local CLIP fallback to embed image."""
        model, processor = self._get_local_model()
        import torch
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            image_features = model.get_image_features(**inputs)
            vec = image_features[0].cpu().numpy().astype(np.float32)
            return self._l2_normalize(vec)
            
    def embed_image_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Embed a list of PIL Images."""
        # Note: Gemini's embed_content API can process items in parallel, or we can iterate
        vectors = []
        for img in images:
            vectors.append(self.embed_image(img))
        return vectors
