"""Intent classification for routing user messages.

Classifies every incoming message into one of four intents so the
orchestrator can choose the correct pipeline path:

* **greeting** – casual hellos, goodbyes, thank-yous
* **adversarial** – prompt-injection / jailbreak attempts
* **rag** – genuine document-grounded questions
* **off_topic** – everything else
"""

import random
import re
from typing import List, Literal

from app.core.constants import ADVERSARIAL_PATTERNS, GREETING_PATTERNS

# Pre-compile patterns once at import time.
_COMPILED_GREETINGS: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in GREETING_PATTERNS
]
_COMPILED_ADVERSARIAL: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in ADVERSARIAL_PATTERNS
]

# Keywords that hint the user is asking about logistics documents.
_RAG_KEYWORDS: List[str] = [
    "document", "shipment", "freight", "logistics", "invoice",
    "tracking", "delivery", "cargo", "warehouse", "customs",
    "shipping", "transport", "supply chain", "container",
    "manifest", "bill of lading", "bol", "consignment",
    "dispatch", "port", "tariff", "route", "schedule",
    "policy", "procedure", "guideline", "regulation",
    "report", "data", "record", "order", "inventory",
    "what", "how", "when", "where", "who", "why", "explain",
    "describe", "tell me", "show me", "find", "list",
]

Intent = Literal["greeting", "rag", "adversarial", "off_topic"]


def classify_intent(text: str) -> Intent:
    """Classify *text* into an intent category.

    Evaluation order (first match wins):

    1. **adversarial** – any ``ADVERSARIAL_PATTERNS`` match
    2. **greeting** – any ``GREETING_PATTERNS`` match
    3. **rag** – keyword heuristic for document-relevant questions
    4. **off_topic** – fallback

    Parameters
    ----------
    text:
        Sanitised user input.

    Returns
    -------
    Intent
        One of ``"greeting"``, ``"rag"``, ``"adversarial"``, ``"off_topic"``.
    """
    if not text:
        return "off_topic"

    lowered = text.lower().strip()

    # 1 – Adversarial check has highest priority.
    for pattern in _COMPILED_ADVERSARIAL:
        if pattern.search(lowered):
            return "adversarial"

    # 2 – Greetings / small-talk.
    for pattern in _COMPILED_GREETINGS:
        if pattern.search(lowered):
            return "greeting"

    # 3 – Document-relevant question heuristic.
    if any(kw in lowered for kw in _RAG_KEYWORDS):
        return "rag"

    # 4 – Fallback: treat as RAG (better to attempt retrieval than refuse).
    return "rag"


# --- Canned greeting responses ---

_GREETING_REPLIES: List[str] = [
    "Hello! I'm your GlobalFreight Logistics assistant. How can I help you today?",
    "Hi there! Ask me anything about your logistics documents.",
    "Hey! Ready to help with your freight and shipping questions.",
    "Good day! How can I assist you with GlobalFreight Logistics?",
]

_FAREWELL_REPLIES: List[str] = [
    "Goodbye! Feel free to come back anytime.",
    "See you later! Have a great day.",
    "Take care! I'm here whenever you need help.",
]

_THANKS_REPLIES: List[str] = [
    "You're welcome! Let me know if you need anything else.",
    "Happy to help! Any other questions?",
    "Glad I could assist!",
]


def direct_chat_reply(text: str) -> str:
    """Return a canned response for greeting / small-talk messages.

    Parameters
    ----------
    text:
        Sanitised user input already classified as ``"greeting"``.

    Returns
    -------
    str
        A friendly canned reply.
    """
    lowered = text.lower().strip()

    if re.search(r"\b(bye|goodbye|see you|take care)\b", lowered):
        return random.choice(_FAREWELL_REPLIES)

    if re.search(r"\b(thanks|thank you|thx)\b", lowered):
        return random.choice(_THANKS_REPLIES)

    return random.choice(_GREETING_REPLIES)
