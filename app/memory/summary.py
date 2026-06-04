"""
Rolling conversation summaries.

Stores a single summary string per user under ``summary:{username}``
so the LLM can use compressed context for long conversations.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from app.core.config import get_settings

if TYPE_CHECKING:
    from app.cache.redis_client import CacheBackend

logger = logging.getLogger(__name__)

_KEY_PREFIX = "summary"


class SummaryManager:
    """Manage rolling conversation summaries stored in cache.

    Parameters
    ----------
    cache_backend:
        A ``CacheBackend`` (Redis or in-memory).
    """

    def __init__(self, cache_backend: "CacheBackend") -> None:
        self._cache = cache_backend
        self._settings = get_settings()

    # -- key helper ----------------------------------------------------------

    @staticmethod
    def _make_key(username: str) -> str:
        return f"{_KEY_PREFIX}:{username}"

    # -- public API ----------------------------------------------------------

    async def get_summary(self, username: str) -> Optional[str]:
        """Return the current rolling summary for *username*, or ``None``.

        Parameters
        ----------
        username:
            Unique user identifier.
        """
        key = self._make_key(username)
        raw = await self._cache.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data.get("summary")
            return None
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt summary for key=%s – returning None", key)
            return None

    async def update_summary(self, username: str, summary_text: str) -> None:
        """Overwrite the rolling summary for *username*.

        Parameters
        ----------
        username:
            Unique user identifier.
        summary_text:
            The new summary text produced by the LLM.
        """
        key = self._make_key(username)
        payload = json.dumps({"summary": summary_text}, ensure_ascii=False)
        # Keep summaries for 7 days
        await self._cache.set(key, payload, ttl=604_800)
        logger.debug("Updated summary for user=%s (%d chars)", username, len(summary_text))

    @staticmethod
    def should_summarize(message_count: int, interval: Optional[int] = None) -> bool:
        """Decide whether the conversation should be summarised now.

        Parameters
        ----------
        message_count:
            Total number of messages in the current conversation.
        interval:
            Re-summarise every *interval* messages.  Falls back to
            ``settings.summary_interval`` when ``None``.

        Returns
        -------
        bool
            ``True`` if it is time to produce a new summary.
        """
        effective_interval = interval or get_settings().summary_interval
        if message_count < effective_interval:
            return False
        return message_count % effective_interval == 0
