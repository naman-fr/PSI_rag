"""
Visual hallucination checker and answer grounding verifier.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from langsmith import traceable

from app.core.config import get_settings
from app.core.constants import VERIFICATION_PROMPT
from app.utils.json_parser import parse_json_object

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    """Structured response schema for hallucination verification."""
    supported: bool = Field(description="True if the answer is fully supported by the context and image features without hallucinating.")
    confidence: float = Field(description="Confidence rating of this verification decision, from 0.0 to 1.0.")
    reason: str = Field(description="Explanation of the verdict, highlighting any ungrounded assertions.")


class VisualHallucinationChecker:
    """Post-generation verification of answer grounding against textual and visual context."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.vision_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    @traceable(name="verify_answer_grounding")
    async def verify(
        self,
        question: str,
        context: str,
        answer: str,
        ocr_text: Optional[str] = None,
        detected_objects: Optional[list[str]] = None
    ) -> VerificationResult:
        """
        Verify if the generated answer is grounded in the retrieved text context and visual features.
        """
        try:
            logger.info("Verifying grounding of generated answer")
            
            # Build unified context
            full_context = (
                f"Retrieved Document Context:\n{context}\n\n"
            )
            if ocr_text:
                full_context += f"Image OCR Text:\n{ocr_text}\n\n"
            if detected_objects:
                full_context += f"Image Object Tags:\n{', '.join(detected_objects)}\n\n"

            # Create verification prompt
            verification_payload = (
                f"Question Asked:\n{question}\n\n"
                f"Unified Grounding Context:\n{full_context}\n\n"
                f"Generated Answer to Verify:\n{answer}\n"
            )

            # Call Gemini with structured schema output
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=verification_payload,
                config={
                    "system_instruction": VERIFICATION_PROMPT,
                    "response_mime_type": "application/json",
                    "response_schema": VerificationResult,
                    "temperature": 0.0,
                }
            )

            response_text = response.text or ""
            # Parse response JSON
            data = parse_json_object(response_text)
            
            supported = bool(data.get("supported", False))
            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "No verification reasoning provided"))

            verdict = VerificationResult(
                supported=supported,
                confidence=confidence,
                reason=reason
            )
            logger.info("Verification check complete", supported=supported, confidence=confidence, reason=reason)
            return verdict

        except Exception as e:
            logger.exception("Grounding verification failed. Falling back to cautious verdict.")
            # Cautious fallback: assume unsupported if verification fails
            return VerificationResult(
                supported=False,
                confidence=0.0,
                reason=f"Verification failure: {str(e)}"
            )
