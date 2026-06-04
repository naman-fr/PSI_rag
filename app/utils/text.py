"""Text processing utilities extracted from notebook."""

import math
import re
from typing import List


def estimate_tokens(text: str) -> int:
    """Rough token count estimation (1 word ~ 1.33 tokens)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text.split()) * 1.33))


def normalize_whitespace(text: str) -> str:
    """Clean up whitespace: collapse spaces, limit blank lines, strip."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries."""
    text = normalize_whitespace(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p.strip() for p in pieces if p.strip()]


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "..."
