"""
Per-user, per-session conversation history.

Messages are stored as a JSON array under ``conv:{username}:{session_id}``
in whatever ``CacheBackend`` is active (Redis *or* in-memory).
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, List, Optional

from app.core.config import get_settings

if TYPE_CHECKING:
    from app.cache.redis_client import CacheBackend

logger = logging.getLogger(__name__)

_KEY_PREFIX = "conv"


class ConversationManager:
    """Manage turn-by-turn chat history stored in cache.

    Parameters
    ----------
    cache_backend:
        A ``CacheBackend`` (Redis or in-memory).
    """

    def __init__(self, cache_backend: "CacheBackend") -> None:
        self._cache = cache_backend
        self._settings = get_settings()

    # -- key helpers ---------------------------------------------------------

    @staticmethod
    def _make_key(username: str, session_id: str) -> str:
        return f"{_KEY_PREFIX}:{username}:{session_id}"

    # -- read / write --------------------------------------------------------

    async def _load(self, key: str) -> List[dict]:
        raw = await self._cache.get(key)
        if raw is None:
            return []
        try:
            messages: list = json.loads(raw)
            return messages
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt conversation data for key=%s – resetting", key)
            return []

    async def _save(self, key: str, messages: List[dict]) -> None:
        payload = json.dumps(messages, ensure_ascii=False)
        # Long TTL – conversations should persist across the session
        await self._cache.set(key, payload, ttl=86_400)  # 24 h

    # -- public API ----------------------------------------------------------

    async def add_message(
        self,
        username: str,
        session_id: str,
        role: str,
        content: str,
        image_id: Optional[str] = None,
    ) -> None:
        """Append a message to the conversation.

        Parameters
        ----------
        username:
            Unique user identifier.
        session_id:
            Current conversation session ID.
        role:
            ``"user"`` or ``"assistant"``.
        content:
            Message text.
        image_id:
            Optional image identifier associated with this message turn.
        """
        key = self._make_key(username, session_id)
        messages = await self._load(key)
        messages.append(
            {
                "role": role,
                "content": content,
                "image_id": image_id,
                "timestamp": time.time(),
            }
        )
        await self._save(key, messages)
        logger.debug(
            "Conversation %s: added %s message (total=%d, image_id=%s)",
            key, role, len(messages), image_id,
        )

    async def get_recent_messages(
        self,
        username: str,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Return the *limit* most recent messages.

        If *limit* is ``None`` the configured ``conversation_window_size``
        is used.
        """
        effective_limit = limit or self._settings.conversation_window_size
        key = self._make_key(username, session_id)
        messages = await self._load(key)
        return messages[-effective_limit:]

    async def get_full_history(
        self,
        username: str,
        session_id: str,
    ) -> List[dict]:
        """Return the complete conversation history."""
        key = self._make_key(username, session_id)
        return await self._load(key)

    async def clear_conversation(
        self,
        username: str,
        session_id: str,
    ) -> None:
        """Delete the entire conversation."""
        key = self._make_key(username, session_id)
        await self._cache.delete(key)
        logger.info("Cleared conversation %s", key)
