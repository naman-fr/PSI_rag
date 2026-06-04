"""Input sanitization and injection detection guardrails.

Validates, sanitizes, and screens user input before it reaches
the retrieval or generation pipeline.  All thresholds come from
``app.core.config.Settings``; adversarial patterns are defined in
``app.core.constants``.
"""

import re
from typing import List

from app.core.config import get_settings
from app.core.constants import ADVERSARIAL_PATTERNS

# Pre-compile adversarial regexes once at import time for performance.
_COMPILED_ADVERSARIAL: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in ADVERSARIAL_PATTERNS
]


def sanitize_input(text: str) -> str:
    """Strip, collapse whitespace, and remove null bytes from user input.

    Parameters
    ----------
    text:
        Raw user input string.

    Returns
    -------
    str
        Cleaned string safe for downstream processing.
    """
    if not isinstance(text, str):
        return ""
    # Remove null bytes and other control characters (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse runs of whitespace into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_injection(text: str) -> bool:
    """Check whether *text* contains a known adversarial / prompt-injection pattern.

    Matches are case-insensitive and evaluated against every pattern in
    ``ADVERSARIAL_PATTERNS``.

    Parameters
    ----------
    text:
        Pre-sanitised user input.

    Returns
    -------
    bool
        ``True`` if any adversarial pattern matches; ``False`` otherwise.
    """
    if not text:
        return False
    for pattern in _COMPILED_ADVERSARIAL:
        if pattern.search(text):
            return True
    return False


def validate_input_length(text: str, max_length: int | None = None) -> bool:
    """Return ``True`` when *text* length is within the allowed limit.

    Parameters
    ----------
    text:
        User input to measure.
    max_length:
        Character limit override.  Defaults to
        ``Settings.max_question_length``.

    Returns
    -------
    bool
        ``True`` if ``0 < len(text) <= max_length``.
    """
    if max_length is None:
        max_length = get_settings().max_question_length
    return 0 < len(text) <= max_length
