"""
Multimodal answer generation service.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from google import genai
from langsmith import traceable

from app.core.config import get_settings
from app.core.constants import GROUNDED_ANSWER_PROMPT

logger = logging.getLogger(__name__)


class MultimodalGenerator:
    """Coordinate the multimodal LLM (Gemini) to generate answers grounded in visual and textual evidence."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.vision_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    @traceable(name="generate_multimodal_answer")
    async def generate(
        self,
        question: str,
        context: str,
        image: Optional[Image.Image] = None,
        ocr_text: Optional[str] = None,
        detected_objects: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Invoke the Gemini multimodal model to generate a grounded answer.
        """
        settings = get_settings()
        max_tokens = max_tokens or settings.max_completion_tokens

        # Build prompt elements
        prompt_parts = []
        
        # 1. System Prompt Instruction
        system_instruction = GROUNDED_ANSWER_PROMPT
        
        # 2. Assembled Context Block
        context_block = (
            f"Retrieved Document Context:\n{context}\n\n"
        )
        if ocr_text:
            context_block += f"Extracted Text layout (OCR) from uploaded image:\n{ocr_text}\n\n"
        if detected_objects:
            context_block += f"Detected visual elements/tags in uploaded image:\n{', '.join(detected_objects)}\n\n"
            
        context_block += f"User Question:\n{question}\n"
        
        # Assemble parts list for Gemini Client API
        contents = []
        if image:
            # Check if image is in correct mode
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            contents.append(image)
            
        contents.append(context_block)

        logger.info(
            "Calling Gemini multimodal generator",
            model=self._model,
            has_image=image is not None,
            context_len=len(context_block)
        )

        try:
            # Run in thread pool to prevent blocking FastAPI's main event loop
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=contents,
                config={
                    "system_instruction": system_instruction,
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )

            answer = response.text or ""
            
            # Extract token usage from response metadata if available
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
                usage["completion_tokens"] = response.usage_metadata.candidates_token_count or 0
                usage["total_tokens"] = response.usage_metadata.total_token_count or 0

            logger.info("Successfully generated multimodal response", answer_len=len(answer), usage=usage)
            return answer, usage

        except Exception as e:
            logger.exception("Error in Gemini generate call")
            raise RuntimeError(f"Multimodal generation failed: {str(e)}")
