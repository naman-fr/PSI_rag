"""
In-memory request metrics.

``MetricsCollector`` accumulates counts, latencies, token usage, and
cache-hit rates.  Data resets on process restart – swap for Prometheus
counters in production if persistence is required.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class _ModeStats:
    """Aggregate stats for a single response mode (e.g. ``rag``, ``direct``)."""

    count: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    cache_hits: int = 0


class MetricsCollector:
    """Thread-safe, in-memory metrics aggregator.

    Usage::

        metrics = MetricsCollector()
        metrics.record_request(mode="rag", latency_ms=120.5, tokens=350, cache_hit=False)
        print(metrics.get_metrics())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at: float = time.time()
        self._mode_stats: Dict[str, _ModeStats] = defaultdict(_ModeStats)
        self._total_requests: int = 0
        self._total_cache_hits: int = 0

    # -- recording -----------------------------------------------------------

    def record_request(
        self,
        mode: str,
        latency_ms: float,
        tokens: int = 0,
        cache_hit: bool = False,
    ) -> None:
        """Register a completed request.

        Parameters
        ----------
        mode:
            Response mode identifier (``"rag"``, ``"direct"``, ``"conversational"``…).
        latency_ms:
            End-to-end latency in milliseconds.
        tokens:
            Total tokens consumed (prompt + completion).
        cache_hit:
            Whether the response was served from cache.
        """
        with self._lock:
            self._total_requests += 1
            if cache_hit:
                self._total_cache_hits += 1

            stats = self._mode_stats[mode]
            stats.count += 1
            stats.total_latency_ms += latency_ms
            stats.total_tokens += tokens
            if cache_hit:
                stats.cache_hits += 1

    # -- reporting -----------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of all collected metrics."""
        with self._lock:
            uptime_s = round(time.time() - self._started_at, 1)
            total = self._total_requests
            cache_hit_rate = (
                round(self._total_cache_hits / total, 4) if total else 0.0
            )

            per_mode: Dict[str, Any] = {}
            for mode, stats in self._mode_stats.items():
                avg_latency = (
                    round(stats.total_latency_ms / stats.count, 2)
                    if stats.count
                    else 0.0
                )
                avg_tokens = (
                    round(stats.total_tokens / stats.count, 1)
                    if stats.count
                    else 0.0
                )
                per_mode[mode] = {
                    "count": stats.count,
                    "avg_latency_ms": avg_latency,
                    "avg_tokens": avg_tokens,
                    "cache_hits": stats.cache_hits,
                }

            return {
                "uptime_seconds": uptime_s,
                "total_requests": total,
                "cache_hit_rate": cache_hit_rate,
                "per_mode": per_mode,
            }

    # -- reset ---------------------------------------------------------------

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._started_at = time.time()
            self._mode_stats.clear()
            self._total_requests = 0
            self._total_cache_hits = 0
