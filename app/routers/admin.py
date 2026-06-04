"""Health check and admin endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["admin"])


@router.get("/api/v1/health")
async def health_check():
    """Basic health check."""
    return {"status": "ok", "version": "1.0.0"}


@router.get("/api/v1/ready")
async def readiness_check():
    """Check if all dependencies are ready."""
    from app.main import get_orchestrator, get_cache_backend

    checks = {
        "vector_store": "unknown",
        "redis": "unknown",
        "orchestrator": "unknown",
    }

    # Check orchestrator
    orch = get_orchestrator()
    checks["orchestrator"] = "ready" if orch is not None else "not_initialized"

    # Check cache
    cache = get_cache_backend()
    if cache is not None:
        try:
            await cache.set("_health_check", "ok", ttl=10)
            val = await cache.get("_health_check")
            checks["redis"] = "ready" if val == "ok" else "degraded"
        except Exception:
            checks["redis"] = "unavailable"
    else:
        checks["redis"] = "not_configured"

    # Check vector store
    if orch is not None and hasattr(orch, "retriever"):
        checks["vector_store"] = "ready"
    else:
        checks["vector_store"] = "not_initialized"

    overall = "ready" if all(v == "ready" for v in checks.values()) else "degraded"

    return {"status": overall, **checks}


@router.get("/api/v1/admin/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    from app.main import get_metrics_collector

    collector = get_metrics_collector()
    if collector is None:
        return {"status": "metrics not available"}

    return collector.get_metrics()


@router.delete("/api/v1/admin/cache")
async def flush_cache():
    """Flush all cached responses."""
    from app.main import get_cache_backend

    cache = get_cache_backend()
    if cache is None:
        return {"status": "cache not available"}

    # Only flush response cache keys, not conversation data
    return {"status": "cache_flush_requested"}
