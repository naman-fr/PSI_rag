"""Groq LLM wrapper for text generation.

Wraps the synchronous ``groq.Groq`` client and exposes an async-compatible
``generate`` method via :func:`asyncio.to_thread`.  All configuration is
sourced from :func:`app.core.config.get_settings`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq
from langsmith import traceable

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMService:
    """Groq chat-completion wrapper.

    Parameters
    ----------
    model:
        Override the model name (defaults to ``Settings.groq_model``).
    """

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model: str = model or settings.groq_model
        self._client = Groq(api_key=settings.groq_api_key)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_json: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        """Blocking call to the Groq chat-completion API."""
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        usage: Dict[str, Any] = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }
        return text, usage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(name="generate_grounded_answer")
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        response_json: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a chat completion via Groq.

        The synchronous Groq client is offloaded to a thread so that the
        calling event loop is not blocked.

        Parameters
        ----------
        messages:
            OpenAI-compatible message dicts (``role``, ``content``).
        max_tokens:
            Maximum completion tokens.
            Defaults to ``Settings.max_completion_tokens``.
        temperature:
            Sampling temperature.
        response_json:
            If ``True``, request JSON-mode output from the model.

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            ``(generated_text, usage_dict)`` where *usage_dict* contains
            ``prompt_tokens``, ``completion_tokens``, and ``total_tokens``
            compatible with :class:`app.schemas.responses.TokenUsage`.
        """
        settings = get_settings()
        max_tokens = max_tokens or settings.max_completion_tokens

        logger.debug(
            "LLM generate: model=%s, max_tokens=%d, temperature=%.2f, json=%s",
            self._model,
            max_tokens,
            temperature,
            response_json,
        )

        text, usage = await asyncio.to_thread(
            self._call_sync,
            messages,
            max_tokens,
            temperature,
            response_json,
        )

        logger.debug(
            "LLM response: %d chars, usage=%s",
            len(text),
            usage,
        )

        return text, usage
