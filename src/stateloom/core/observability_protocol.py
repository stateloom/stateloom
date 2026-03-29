"""Null implementations for observability — replaced by EE at runtime."""

from __future__ import annotations

from typing import Any


class NullMetricsCollector:
    """No-op metrics collector used when EE is not loaded."""

    enabled: bool = False

    def record_llm_call(self, **kw: Any) -> None:
        pass

    def record_cache_hit(self, **kw: Any) -> None:
        pass

    def record_cache_miss(self, **kw: Any) -> None:
        pass

    def record_pii_detection(self, **kw: Any) -> None:
        pass

    def record_budget_violation(self, **kw: Any) -> None:
        pass

    def record_local_routing(self, **kw: Any) -> None:
        pass

    def record_kill_switch_block(self, **kw: Any) -> None:
        pass

    def record_blast_radius_pause(self, **kw: Any) -> None:
        pass

    def record_rate_limit(self, **kw: Any) -> None:
        pass

    def record_circuit_breaker(self, **kw: Any) -> None:
        pass

    def set_active_sessions(self, count: int) -> None:
        pass

    def generate_metrics(self) -> str:
        return ""


class NullAlertManager:
    """No-op alert manager used when EE is not loaded."""

    def __init__(self, webhook_url: str = "", webhook_urls: list[str] | None = None) -> None:
        pass

    def fire(self, event_type: str, payload: dict[str, Any]) -> None:
        pass


class NullTimeSeriesAggregator:
    """No-op aggregator used when EE is not loaded."""

    def __init__(self, store: Any = None) -> None:
        pass

    def get_timeseries(self, *args: Any, **kw: Any) -> dict[str, Any]:
        return {}

    def get_latency(self, *args: Any, **kw: Any) -> dict[str, Any]:
        return {}

    def get_breakdown(self, *args: Any, **kw: Any) -> dict[str, Any]:
        return {}
