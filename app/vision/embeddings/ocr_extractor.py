"""
OCR text extraction service using Gemini multimodal capabilities.
"""

from __future__ import annotations

import io
import logging
from PIL import Image
from google import genai
from langsmith import traceable

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Extract layout-preserving text from images (e.g. invoices, cargo labels, shipping forms)."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.vision_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    @traceable(name="extract_ocr_text")
    async def extract_text(self, img: Image.Image) -> str:
        """
        Run OCR on a PIL Image.
        Returns layout-preserved markdown text, including tables and columns.
        """
        try:
            logger.info("Running Gemini OCR on image", size=img.size)
            # Check if image is in correct mode
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            prompt = (
                "Extract all readable text and structural data from this image. "
                "Preserve the layout, spacing, and column formatting as much as possible. "
                "If the image contains tables, format them as markdown tables. "
                "Do not add any summary, explanation, introduction, or closing remarks. "
                "Provide only the transcribed text."
            )

            # Call Gemini multimodal generation
            import asyncio
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=[img, prompt]
            )

            extracted_text = response.text or ""
            logger.info("OCR completed successfully", character_count=len(extracted_text))
            return extracted_text.strip()
        except Exception as e:
            logger.exception("Failed to run Gemini OCR, returning empty string")
            # Fallback: if Gemini fails, return empty or try a simple local pytesseract/easyocr if present
            return ""

    @traceable(name="detect_objects_and_tags")
    async def detect_objects_and_tags(self, img: Image.Image) -> list[str]:
        """
        Detect visual elements, objects, and descriptors in the image.
        Returns a list of tags (e.g., ['cargo container', 'damaged door', 'seal lock']).
        """
        try:
            logger.info("Running visual object tagger on image", size=img.size)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            prompt = (
                "Identify the main objects, labels, condition (e.g. damaged, sealed, new), and visual characteristics in this image. "
                "Return ONLY a comma-separated list of short descriptors or tags. "
                "Do not write full sentences or descriptions. Example: cargo container, invoice table, customs stamp, barcode."
            )

            import asyncio
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=[img, prompt]
            )

            tags_text = response.text or ""
            tags = [t.strip().lower() for t in tags_text.split(",") if t.strip()]
            logger.info("Object tagging completed", tags=tags)
            return tags
        except Exception as e:
            logger.warning("Failed to run object tagger: %s", str(e))
            return []
