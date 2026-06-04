"""Output validation guardrails.

Post-generation checks on the candidate answer to catch obvious
problems *before* sending the response to the user.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Phrases that frequently indicate model hallucination.
_HALLUCINATION_MARKERS: List[str] = [
    "as an ai",
    "as a language model",
    "i cannot access",
    "i don't have access to real-time",
    "my training data",
    "my knowledge cutoff",
    "i was trained",
    "based on my training",
    "i'm unable to browse",
]

# Overconfident language that is suspect when context is short / weak.
_OVERCONFIDENT_PHRASES: List[str] = [
    "it is absolutely certain",
    "without a doubt",
    "there is no question",
    "100% guaranteed",
    "definitively",
    "undeniably",
    "irrefutably",
]

# Maximum reasonable answer length (characters).
_MAX_ANSWER_LENGTH: int = 4000

# Minimum context length (chars) below which overconfident language
# is flagged.
_WEAK_CONTEXT_THRESHOLD: int = 100


def validate_response(answer: str, context: str) -> Tuple[bool, str]:
    """Run post-generation validation on *answer*.

    Checks (in order):

    1. Answer is not empty or whitespace-only.
    2. Answer does not contain hallucination marker phrases.
    3. Answer length is within a reasonable range.
    4. Overconfident language is flagged when context is weak.

    Parameters
    ----------
    answer:
        The generated answer text.
    context:
        The retrieval context that was supplied to the generator.

    Returns
    -------
    tuple[bool, str]
        ``(valid, reason)`` – *valid* is ``True`` when all checks pass;
        *reason* describes the first failure (or ``"ok"``).
    """
    # --- Check 1: non-empty answer ---
    if not answer or not answer.strip():
        return (False, "Answer is empty")

    lowered = answer.lower()

    # --- Check 2: hallucination markers ---
    for marker in _HALLUCINATION_MARKERS:
        if marker in lowered:
            return (
                False,
                f"Hallucination marker detected: '{marker}'",
            )

    # --- Check 3: reasonable length ---
    if len(answer) > _MAX_ANSWER_LENGTH:
        return (
            False,
            f"Answer too long: {len(answer)} chars (max {_MAX_ANSWER_LENGTH})",
        )

    # --- Check 4: overconfident language on weak context ---
    context_len = len((context or "").strip())
    if context_len < _WEAK_CONTEXT_THRESHOLD:
        for phrase in _OVERCONFIDENT_PHRASES:
            if phrase in lowered:
                return (
                    False,
                    f"Overconfident language on weak context: '{phrase}'",
                )

    return (True, "ok")
