"""Robust JSON extraction from LLM output."""

import json
import re
from typing import Any, Dict


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from LLM output.

    Handles markdown code fences, leading/trailing text, and
    returns a safe default on parse failure.
    """
    text = text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Find the JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {
            "supported": False,
            "confidence": 0.0,
            "reason": "json_parse_failed",
        }
