"""Refusal and uncertainty policy.

Decides whether to accept or refuse a candidate answer based on the
hallucination-verification verdict and a configurable confidence
threshold.
"""

from __future__ import annotations

from typing import Tuple

from app.core.config import get_settings
from app.core.constants import REFUSAL_RESPONSE
from app.schemas.responses import VerificationVerdict


def apply_refusal_policy(
    verdict: VerificationVerdict,
    confidence_threshold: float | None = None,
    answer: str = "",
) -> Tuple[str, bool]:
    """Apply the refusal policy to a verification *verdict*.

    The answer is **refused** when either:

    * ``verdict.supported`` is ``False``, or
    * ``verdict.confidence`` is below *confidence_threshold*.

    In both cases the standard ``REFUSAL_RESPONSE`` is returned.

    Parameters
    ----------
    verdict:
        The :class:`VerificationVerdict` produced by the hallucination
        checker.
    confidence_threshold:
        Minimum confidence to accept the answer.  Defaults to
        ``Settings.verification_confidence_threshold``.
    answer:
        The candidate answer produced by the generator.  Returned
        unchanged when the policy accepts the answer.

    Returns
    -------
    tuple[str, bool]
        ``(final_answer, was_refused)`` – if *was_refused* is ``True``
        the *final_answer* is the canned refusal string.
    """
    if confidence_threshold is None:
        confidence_threshold = get_settings().verification_confidence_threshold

    # Get supported and confidence dynamically to support both dict and Pydantic Model
    supported = getattr(verdict, "supported", None)
    if supported is None:
        supported = verdict.get("supported", False) if isinstance(verdict, dict) else False

    confidence = getattr(verdict, "confidence", None)
    if confidence is None:
        confidence = verdict.get("confidence", 0.0) if isinstance(verdict, dict) else 0.0

    # Refuse when the verifier says the answer is unsupported.
    if not supported:
        return (REFUSAL_RESPONSE, True)

    # Refuse when confidence is below the threshold.
    if confidence < confidence_threshold:
        return (REFUSAL_RESPONSE, True)

    # Accept – pass the original answer through unchanged.
    return (answer, False)
