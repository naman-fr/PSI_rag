"""
Redis cache backend with in-memory fallback.

Provides a ``CacheBackend`` protocol and two concrete implementations:
- ``RedisCache``    – production cache backed by Redis.
- ``InMemoryCache`` – dict-based fallback that works without any external service.

Use the ``get_cache_backend()`` factory to obtain the right backend automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional, Protocol, runtime_checkable

import redis.asyncio as aioredis

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CacheBackend(Protocol):
    """Minimal async cache interface."""

    async def get(self, key: str) -> Optional[str]:
        """Return the cached value or ``None``."""
        ...

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Store *value* under *key* with an optional TTL in seconds."""
        ...

    async def delete(self, key: str) -> None:
        """Remove *key* from the cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* is present."""
        ...


# ---------------------------------------------------------------------------
# Redis implementation
# ---------------------------------------------------------------------------

class RedisCache:
    """Async Redis cache backend.

    Parameters
    ----------
    redis_url:
        Redis connection string (e.g. ``redis://localhost:6379/0``).
    default_ttl:
        Default time-to-live in seconds when no explicit TTL is given.
    """

    def __init__(self, redis_url: str, default_ttl: int = 3600) -> None:
        self._url = redis_url
        self._default_ttl = default_ttl
        self._client: aioredis.Redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )

    # -- public API ----------------------------------------------------------

    async def get(self, key: str) -> Optional[str]:
        """Retrieve a value by key."""
        try:
            return await self._client.get(key)
        except aioredis.RedisError as exc:
            logger.warning("RedisCache.get failed for key=%s: %s", key, exc)
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a key with optional TTL (seconds)."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        try:
            await self._client.set(key, value, ex=effective_ttl)
        except aioredis.RedisError as exc:
            logger.warning("RedisCache.set failed for key=%s: %s", key, exc)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        try:
            await self._client.delete(key)
        except aioredis.RedisError as exc:
            logger.warning("RedisCache.delete failed for key=%s: %s", key, exc)

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            return bool(await self._client.exists(key))
        except aioredis.RedisError as exc:
            logger.warning("RedisCache.exists failed for key=%s: %s", key, exc)
            return False

    async def ping(self) -> bool:
        """Return ``True`` if the Redis server is reachable."""
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    def __repr__(self) -> str:
        return f"RedisCache(url={self._url!r})"


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------

class InMemoryCache:
    """Dict-based async cache for development / testing.

    Entries honour TTL: expired items are lazily evicted on access and
    periodically during ``set`` when the cache exceeds ``max_size``.
    """

    def __init__(self, default_ttl: int = 3600, max_size: int = 10_000) -> None:
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._store: dict[str, tuple[str, float]] = {}  # key -> (value, expires_at)
        self._lock = asyncio.Lock()

    # -- helpers -------------------------------------------------------------

    def _is_expired(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return True
        _, expires_at = entry
        return time.monotonic() > expires_at

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired_keys = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired_keys:
            del self._store[k]

    # -- public API ----------------------------------------------------------

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            if key not in self._store or self._is_expired(key):
                self._store.pop(key, None)
                return None
            value, _ = self._store[key]
            return value

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + effective_ttl
        async with self._lock:
            # Lazy eviction when near capacity
            if len(self._store) >= self._max_size:
                self._evict_expired()
            self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        async with self._lock:
            if key not in self._store or self._is_expired(key):
                self._store.pop(key, None)
                return False
            return True

    async def close(self) -> None:
        """No-op for API symmetry with ``RedisCache``."""

    def __repr__(self) -> str:
        return f"InMemoryCache(size={len(self._store)})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def get_cache_backend() -> CacheBackend:
    """Return a ``RedisCache`` if reachable, otherwise an ``InMemoryCache``.

    Configuration is read from ``get_settings()``.
    """
    settings = get_settings()

    redis_cache = RedisCache(
        redis_url=settings.redis_url,
        default_ttl=settings.cache_ttl_seconds,
    )

    if await redis_cache.ping():
        logger.info("Connected to Redis at %s", settings.redis_url)
        return redis_cache

    await redis_cache.close()
    logger.warning(
        "Redis unavailable at %s – falling back to InMemoryCache",
        settings.redis_url,
    )
    return InMemoryCache(default_ttl=settings.cache_ttl_seconds)
