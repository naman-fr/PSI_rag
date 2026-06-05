"""
Response-level caching utilities.

Thin helpers that serialise / deserialise RAG responses on top of any
``CacheBackend`` implementation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.cache.redis_client import CacheBackend

logger = logging.getLogger(__name__)

# Cache key namespace
_RESPONSE_PREFIX = "resp"


def make_cache_key(query: str, image_id: Optional[str] = None, prompt_version: str = "v1") -> str:
    """Derive a deterministic cache key from *query*, *image_id*, and *prompt_version*.

    The key is a SHA-256 hex digest prefixed with ``resp:``.

    Parameters
    ----------
    query:
        The user question (lowered & stripped before hashing).
    image_id:
        Optional uploaded image identifier.
    prompt_version:
        Version tag so cache auto-invalidates when the prompt template changes.
    """
    normalised = f"{query.strip().lower()}::{image_id or ''}::{prompt_version}"
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    return f"{_RESPONSE_PREFIX}:{digest}"


async def cache_response(
    cache: "CacheBackend",
    query_hash: str,
    response_dict: dict,
    ttl: int = 3600,
) -> None:
    """Persist a response dict under *query_hash*.

    Parameters
    ----------
    cache:
        Any ``CacheBackend`` instance.
    query_hash:
        Key returned by :func:`make_cache_key`.
    response_dict:
        Serialisable response payload.
    ttl:
        Time-to-live in seconds.
    """
    try:
        payload = json.dumps(response_dict, ensure_ascii=False)
        await cache.set(query_hash, payload, ttl=ttl)
        logger.debug("Cached response for key=%s (ttl=%ds)", query_hash, ttl)
    except (TypeError, ValueError) as exc:
        logger.warning("Failed to serialise response for caching: %s", exc)


async def get_cached_response(
    cache: "CacheBackend",
    query_hash: str,
) -> Optional[dict]:
    """Retrieve a previously cached response.

    Returns ``None`` on cache miss or deserialisation failure.
    """
    raw = await cache.get(query_hash)
    if raw is None:
        return None
    try:
        data: dict = json.loads(raw)
        logger.debug("Cache hit for key=%s", query_hash)
        return data
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Corrupt cache entry for key=%s: %s", query_hash, exc)
        return None


async def invalidate_cache(cache: "CacheBackend", pattern: str) -> int:
    """Delete all keys matching *pattern*.

    For ``RedisCache`` this uses ``SCAN`` + ``DELETE``.  For the in-memory
    backend we iterate over the internal store.  Returns the number of
    keys removed.

    Parameters
    ----------
    cache:
        Any ``CacheBackend`` instance.
    pattern:
        Glob-style pattern (e.g. ``resp:*``).
    """
    import fnmatch

    from app.cache.redis_client import InMemoryCache, RedisCache

    deleted = 0

    if isinstance(cache, RedisCache):
        try:
            cursor: int | str = 0
            while True:
                cursor, keys = await cache._client.scan(
                    cursor=cursor, match=pattern, count=200,
                )
                if keys:
                    deleted += await cache._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning("invalidate_cache scan/delete failed: %s", exc)

    elif isinstance(cache, InMemoryCache):
        async with cache._lock:
            matching = [k for k in cache._store if fnmatch.fnmatch(k, pattern)]
            for k in matching:
                del cache._store[k]
            deleted = len(matching)
    else:
        logger.warning("invalidate_cache: unsupported backend %r", cache)

    logger.info("Invalidated %d key(s) matching pattern=%s", deleted, pattern)
    return deleted
