"""Hallucination verification via LLM cross-check.

Sends the question, retrieved context, and generated answer to a
second LLM call that judges whether the answer is fully grounded
in the context.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol
from langsmith import traceable

from app.core.constants import VERIFICATION_PROMPT
from app.schemas.responses import VerificationVerdict
from app.utils.json_parser import parse_json_object

logger = logging.getLogger(__name__)

# Truncation limits for the verification payload.
# Set generously to ensure the verifier sees the full context and answer.
_MAX_CONTEXT_CHARS: int = 4000
_MAX_ANSWER_CHARS: int = 2000


class LLMService(Protocol):
    """Minimal interface expected from the LLM service layer."""

    async def generate(self, prompt: str, system: str, max_tokens: int) -> str:
        ...  # pragma: no cover


@traceable(name="verify_answer")
async def verify_answer(
    question: str,
    context: str,
    answer: str,
    llm_service: Any,
) -> VerificationVerdict:
    """Ask an LLM whether *answer* is supported by *context*.

    The context and answer are truncated to keep the verification
    call cheap and fast.

    Parameters
    ----------
    question:
        The user's original question.
    context:
        The assembled retrieval context (will be truncated to
        ``_MAX_CONTEXT_CHARS``).
    answer:
        The candidate answer to verify (truncated to
        ``_MAX_ANSWER_CHARS``).
    llm_service:
        Any object that exposes an ``async generate(prompt, system, max_tokens)``
        method.

    Returns
    -------
    VerificationVerdict
        Pydantic model with ``supported``, ``confidence``, and ``reason``.
    """
    truncated_context = context[:_MAX_CONTEXT_CHARS]
    truncated_answer = answer[:_MAX_ANSWER_CHARS]

    user_content = (
        f"Question: {question}\n\n"
        f"Context:\n{truncated_context}\n\n"
        f"Answer:\n{truncated_answer}\n\n"
        "Is the answer fully supported by the context? "
        "Return JSON only."
    )

    messages = [
        {"role": "system", "content": VERIFICATION_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw, _ = await llm_service.generate(
            messages=messages,
            max_tokens=128,
            temperature=0.0,
            response_json=True,
        )
        parsed = parse_json_object(raw)
        return VerificationVerdict(
            supported=bool(parsed.get("supported", False)),
            confidence=float(parsed.get("confidence", 0.0)),
            reason=str(parsed.get("reason", "no_reason_provided")),
        )
    except Exception as exc:
        logger.warning("Verification call failed: %s", exc, exc_info=True)
        return VerificationVerdict(
            supported=False,
            confidence=0.0,
            reason=f"verification_error: {exc}",
        )
