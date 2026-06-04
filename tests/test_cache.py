import pytest
from app.cache.response_cache import make_cache_key, cache_response, get_cached_response, invalidate_cache
from app.cache.redis_client import InMemoryCache


def test_make_cache_key():
    k1 = make_cache_key("test question", "v1")
    k2 = make_cache_key("  TEST question  ", "v1")
    k3 = make_cache_key("test question", "v2")
    
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("resp:")


@pytest.mark.asyncio
async def test_cache_backend_in_memory():
    cache = InMemoryCache()
    key = make_cache_key("query")
    response = {"answer": "hello"}
    
    await cache_response(cache, key, response, ttl=10)
    cached = await get_cached_response(cache, key)
    assert cached == response
    
    deleted = await invalidate_cache(cache, "resp:*")
    assert deleted == 1
    
    cached_after = await get_cached_response(cache, key)
    assert cached_after is None
