"""Time-series aggregation from persisted events.

Queries events from the store, buckets by time window, and computes
latency percentiles. Prefers SQL-level aggregation when the store supports
it (SQLite/Postgres) for O(1)-memory dashboards. Falls back to the legacy
Python-side scan for MemoryStore.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from stateloom.core.event import (
    CacheHitEvent,
    LLMCallEvent,
)
from stateloom.store.base import Store

logger = logging.getLogger("stateloom.observability.aggregator")

_MAX_AGGREGATION_EVENTS = 50_000

# Window → (timedelta, bucket_resolution_seconds)
_WINDOW_CONFIG: dict[str, tuple[timedelta, int]] = {
    "1h": (timedelta(hours=1), 60),  # 1-minute buckets
    "6h": (timedelta(hours=6), 300),  # 5-minute buckets
    "24h": (timedelta(hours=24), 900),  # 15-minute buckets
    "7d": (timedelta(days=7), 3600),  # 1-hour buckets
}


class TimeSeriesAggregator:
    """Aggregates persisted events into time-series data for dashboard charts."""

    def __init__(self, store: Store) -> None:
        self._store = store
        # Detect whether the store supports SQL-level aggregation
        self._has_sql = hasattr(store, "aggregate_timeseries")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_timeseries(
        self,
        window: str = "1h",
        org_id: str = "",
        team_id: str = "",
    ) -> dict[str, Any]:
        """Get time-bucketed request metrics."""
        td, bucket_secs = _WINDOW_CONFIG.get(window, _WINDOW_CONFIG["1h"])
        now = datetime.now(timezone.utc)
        start = now - td

        if self._has_sql:
            return self._timeseries_sql(start, now, bucket_secs, window, org_id, team_id)
        return self._timeseries_legacy(start, now, bucket_secs, window, org_id, team_id)

    def get_latency(
        self,
        window: str = "1h",
        org_id: str = "",
        team_id: str = "",
    ) -> dict[str, Any]:
        """Get latency percentiles and histogram."""
        td, _ = _WINDOW_CONFIG.get(window, _WINDOW_CONFIG["1h"])
        now = datetime.now(timezone.utc)
        start = now - td

        if self._has_sql:
            latencies = self._store.aggregate_latency(  # type: ignore[attr-defined]
                start.isoformat(),
                org_id=org_id,
                team_id=team_id,
            )
        else:
            latencies = self._latency_legacy(start, org_id, team_id)

        if not latencies:
            return {
                "window": window,
                "p50": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0,
                "avg_ms": 0,
                "histogram_buckets": [],
            }

        n = len(latencies)
        hist_boundaries = [100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000]
        # Use bisect for O(log n) histogram computation instead of O(n*b)
        import bisect

        histogram_buckets = [
            {"le": le, "count": bisect.bisect_right(latencies, le)} for le in hist_boundaries
        ]

        return {
            "window": window,
            "p50": self._percentile(latencies, 50),
            "p90": self._percentile(latencies, 90),
            "p95": self._percentile(latencies, 95),
            "p99": self._percentile(latencies, 99),
            "avg_ms": round(sum(latencies) / n, 2),
            "histogram_buckets": histogram_buckets,
        }

    def get_breakdown(
        self,
        window: str = "1h",
        org_id: str = "",
        team_id: str = "",
    ) -> dict[str, Any]:
        """Get breakdown by model, provider, and cache stats."""
        td, _ = _WINDOW_CONFIG.get(window, _WINDOW_CONFIG["1h"])
        now = datetime.now(timezone.utc)
        start = now - td

        if self._has_sql:
            return self._breakdown_sql(start, window, org_id, team_id)
        return self._breakdown_legacy(start, window, org_id, team_id)

    # ------------------------------------------------------------------
    # SQL-accelerated paths (SQLiteStore / PostgresStore)
    # ------------------------------------------------------------------

    def _timeseries_sql(
        self,
        start: datetime,
        now: datetime,
        bucket_secs: int,
        window: str,
        org_id: str,
        team_id: str,
    ) -> dict[str, Any]:
        rows = self._store.aggregate_timeseries(  # type: ignore[attr-defined]
            start.isoformat(),
            bucket_secs,
            org_id=org_id,
            team_id=team_id,
        )
        # Build full bucket list (fill gaps with zeros)
        buckets = self._build_empty_buckets(start, now, bucket_secs)
        row_map = {r["bucket_idx"]: r for r in rows}
        for idx, bucket in enumerate(buckets):
            if idx in row_map:
                r = row_map[idx]
                bucket["requests"] = r["requests"]
                bucket["cost"] = r["cost"]
                bucket["prompt_tokens"] = r["prompt_tokens"]
                bucket["completion_tokens"] = r["completion_tokens"]
        return {"window": window, "bucket_seconds": bucket_secs, "buckets": buckets}

    def _breakdown_sql(
        self,
        start: datetime,
        window: str,
        org_id: str,
        team_id: str,
    ) -> dict[str, Any]:
        data = self._store.aggregate_breakdown(  # type: ignore[attr-defined]
            start.isoformat(),
            org_id=org_id,
            team_id=team_id,
        )
        by_model = {
            r["model"]: {"requests": r["requests"], "cost": r["cost"], "tokens": r["tokens"]}
            for r in data["by_model"]
        }
        by_provider = {
            r["provider"]: {"requests": r["requests"], "cost": r["cost"], "errors": 0}
            for r in data["by_provider"]
        }
        cache_hits = data["cache_hits"]
        total_requests = sum(v["requests"] for v in by_model.values()) + cache_hits
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        return {
            "window": window,
            "by_model": by_model,
            "by_provider": by_provider,
            "cache": {
                "hits": cache_hits,
                "misses": sum(v["requests"] for v in by_model.values()),
                "hit_rate": round(hit_rate, 4),
            },
        }

    # ------------------------------------------------------------------
    # Legacy paths (MemoryStore fallback)
    # ------------------------------------------------------------------

    def _timeseries_legacy(
        self,
        start: datetime,
        now: datetime,
        bucket_secs: int,
        window: str,
        org_id: str,
        team_id: str,
    ) -> dict[str, Any]:
        events = self._store.get_session_events(
            "", event_type="llm_call", limit=_MAX_AGGREGATION_EVENTS
        )
        llm_events = [e for e in events if isinstance(e, LLMCallEvent) and e.timestamp >= start]
        if org_id or team_id:
            llm_events = self._filter_by_tenant(llm_events, org_id, team_id)

        buckets = self._build_empty_buckets(start, now, bucket_secs)
        for e in llm_events:
            idx = self._bucket_index(e.timestamp, start, bucket_secs)
            if 0 <= idx < len(buckets):
                buckets[idx]["requests"] += 1
                buckets[idx]["cost"] += e.cost
                buckets[idx]["prompt_tokens"] += e.prompt_tokens
                buckets[idx]["completion_tokens"] += e.completion_tokens

        return {"window": window, "bucket_seconds": bucket_secs, "buckets": buckets}

    def _latency_legacy(
        self,
        start: datetime,
        org_id: str,
        team_id: str,
    ) -> list[float]:
        events = self._store.get_session_events(
            "", event_type="llm_call", limit=_MAX_AGGREGATION_EVENTS
        )
        filtered = [e for e in events if isinstance(e, LLMCallEvent) and e.timestamp >= start]
        if org_id or team_id:
            filtered = self._filter_by_tenant(filtered, org_id, team_id)
        latencies = sorted(e.latency_ms for e in filtered if e.latency_ms > 0)
        return latencies

    def _breakdown_legacy(
        self,
        start: datetime,
        window: str,
        org_id: str,
        team_id: str,
    ) -> dict[str, Any]:
        events = self._store.get_session_events(
            "", event_type="llm_call", limit=_MAX_AGGREGATION_EVENTS
        )
        llm_events = [e for e in events if isinstance(e, LLMCallEvent) and e.timestamp >= start]
        if org_id or team_id:
            llm_events = self._filter_by_tenant(llm_events, org_id, team_id)

        by_model: dict[str, dict[str, Any]] = {}
        by_provider: dict[str, dict[str, Any]] = {}
        for e in llm_events:
            if e.model not in by_model:
                by_model[e.model] = {"requests": 0, "cost": 0.0, "tokens": 0}
            by_model[e.model]["requests"] += 1
            by_model[e.model]["cost"] += e.cost
            by_model[e.model]["tokens"] += e.total_tokens

            if e.provider not in by_provider:
                by_provider[e.provider] = {"requests": 0, "cost": 0.0, "errors": 0}
            by_provider[e.provider]["requests"] += 1
            by_provider[e.provider]["cost"] += e.cost

        cache_events = self._store.get_session_events(
            "", event_type="cache_hit", limit=_MAX_AGGREGATION_EVENTS
        )
        cache_hits = sum(
            1 for e in cache_events if isinstance(e, CacheHitEvent) and e.timestamp >= start
        )
        total_requests = len(llm_events) + cache_hits
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        return {
            "window": window,
            "by_model": by_model,
            "by_provider": by_provider,
            "cache": {
                "hits": cache_hits,
                "misses": len(llm_events),
                "hit_rate": round(hit_rate, 4),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_empty_buckets(
        start: datetime, end: datetime, bucket_secs: int
    ) -> list[dict[str, Any]]:
        """Create empty time buckets between start and end."""
        buckets: list[dict[str, Any]] = []
        current = start
        delta = timedelta(seconds=bucket_secs)
        while current < end:
            buckets.append(
                {
                    "timestamp": current.isoformat(),
                    "requests": 0,
                    "cost": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "errors": 0,
                }
            )
            current += delta
        return buckets

    @staticmethod
    def _bucket_index(ts: datetime, start: datetime, bucket_secs: int) -> int:
        """Get the bucket index for a timestamp."""
        diff = (ts - start).total_seconds()
        return int(diff // bucket_secs)

    @staticmethod
    def _percentile(sorted_data: list[float], pct: int) -> float:
        """Compute a percentile from sorted data."""
        if not sorted_data:
            return 0.0
        n = len(sorted_data)
        k = (pct / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(sorted_data[int(k)], 2)
        return round(sorted_data[f] * (c - k) + sorted_data[c] * (k - f), 2)

    def _filter_by_tenant(
        self,
        events: list[Any],
        org_id: str = "",
        team_id: str = "",
    ) -> list[Any]:
        """Filter events by org_id and/or team_id using session lookup."""
        if not org_id and not team_id:
            return events

        matching_sessions: set[str] = set()
        try:
            sessions = self._store.list_sessions(
                org_id=org_id if org_id else None,
                team_id=team_id if team_id else None,
                limit=_MAX_AGGREGATION_EVENTS,
            )
            matching_sessions = {s.id for s in sessions}
        except Exception:
            logger.debug("Tenant filter: session lookup failed", exc_info=True)
            return events

        return [e for e in events if e.session_id in matching_sessions]
