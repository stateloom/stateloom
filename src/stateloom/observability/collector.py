"""Prometheus metrics collector for StateLoom.

Uses a dedicated CollectorRegistry to avoid conflicts with user instrumentation.
When prometheus_client is not installed, all record_* methods are silent no-ops.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("stateloom.observability.collector")

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# Latency histogram buckets (seconds)
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


class MetricsCollector:
    """Collects Prometheus metrics for StateLoom.

    When ``enabled=False`` or ``prometheus_client`` is not installed,
    all ``record_*`` methods are no-ops.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self._enabled = enabled and _PROMETHEUS_AVAILABLE
        self._registry: Any = None

        if not enabled:
            return

        if not _PROMETHEUS_AVAILABLE:
            logger.info(
                "prometheus_client not installed — metrics collection disabled. "
                "Install with: pip install stateloom[metrics]"
            )
            return

        self._registry = CollectorRegistry()
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize all Prometheus metric objects."""
        reg = self._registry

        self._request_duration = Histogram(
            "stateloom_llm_request_duration_seconds",
            "LLM call latency in seconds",
            ["model", "provider", "org_id", "team_id"],
            buckets=_LATENCY_BUCKETS,
            registry=reg,
        )

        self._tokens_total = Counter(
            "stateloom_llm_tokens_total",
            "Token usage",
            ["model", "provider", "type", "org_id", "team_id"],
            registry=reg,
        )

        self._request_cost = Counter(
            "stateloom_llm_request_cost_usd",
            "Cumulative cost in USD",
            ["model", "provider", "org_id", "team_id"],
            registry=reg,
        )

        self._requests_total = Counter(
            "stateloom_llm_requests_total",
            "Request count",
            ["model", "provider", "status", "org_id", "team_id"],
            registry=reg,
        )

        self._cache_hits = Counter(
            "stateloom_cache_hits_total",
            "Cache hits",
            ["match_type", "org_id", "team_id"],
            registry=reg,
        )

        self._cache_misses = Counter(
            "stateloom_cache_misses_total",
            "Cache misses",
            ["org_id", "team_id"],
            registry=reg,
        )

        self._pii_detections = Counter(
            "stateloom_pii_detections_total",
            "PII detection count",
            ["pii_type", "action", "org_id", "team_id"],
            registry=reg,
        )

        self._budget_violations = Counter(
            "stateloom_budget_violations_total",
            "Budget violations",
            ["org_id", "team_id"],
            registry=reg,
        )

        self._local_routing = Counter(
            "stateloom_local_routing_total",
            "Routing decisions",
            ["decision", "org_id", "team_id"],
            registry=reg,
        )

        self._kill_switch_blocks = Counter(
            "stateloom_kill_switch_blocks_total",
            "Kill switch blocks",
            ["org_id", "team_id"],
            registry=reg,
        )

        self._blast_radius_pauses = Counter(
            "stateloom_blast_radius_pauses_total",
            "Blast radius pauses",
            ["type", "org_id", "team_id"],
            registry=reg,
        )

        self._rate_limit_requests = Counter(
            "stateloom_rate_limit_requests_total",
            "Rate-limited requests",
            ["team_id", "virtual_key_id", "outcome"],
            registry=reg,
        )

        self._rate_limit_wait_seconds = Histogram(
            "stateloom_rate_limit_wait_seconds",
            "Time spent waiting in rate limiter queue",
            ["team_id", "virtual_key_id"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=reg,
        )

        self._circuit_breaker_transitions = Counter(
            "stateloom_circuit_breaker_transitions_total",
            "Circuit breaker state transitions",
            ["provider", "state", "org_id", "team_id"],
            registry=reg,
        )

        self._active_sessions = Gauge(
            "stateloom_active_sessions",
            "Current active session count",
            registry=reg,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def registry(self) -> Any:
        return self._registry

    def record_llm_call(
        self,
        *,
        model: str,
        provider: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        status: str = "success",
        org_id: str = "",
        team_id: str = "",
    ) -> None:
        """Record an LLM call event."""
        if not self._enabled:
            return
        self._request_duration.labels(
            model=model,
            provider=provider,
            org_id=org_id,
            team_id=team_id,
        ).observe(latency_ms / 1000.0)
        self._tokens_total.labels(
            model=model,
            provider=provider,
            type="prompt",
            org_id=org_id,
            team_id=team_id,
        ).inc(prompt_tokens)
        self._tokens_total.labels(
            model=model,
            provider=provider,
            type="completion",
            org_id=org_id,
            team_id=team_id,
        ).inc(completion_tokens)
        self._request_cost.labels(
            model=model,
            provider=provider,
            org_id=org_id,
            team_id=team_id,
        ).inc(cost)
        self._requests_total.labels(
            model=model,
            provider=provider,
            status=status,
            org_id=org_id,
            team_id=team_id,
        ).inc()

    def record_cache_hit(
        self,
        *,
        match_type: str = "exact",
        org_id: str = "",
        team_id: str = "",
    ) -> None:
        """Record a cache hit."""
        if not self._enabled:
            return
        self._cache_hits.labels(match_type=match_type, org_id=org_id, team_id=team_id).inc()

    def record_cache_miss(self, *, org_id: str = "", team_id: str = "") -> None:
        """Record a cache miss."""
        if not self._enabled:
            return
        self._cache_misses.labels(org_id=org_id, team_id=team_id).inc()

    def record_pii_detection(
        self,
        *,
        pii_type: str,
        action: str,
        org_id: str = "",
        team_id: str = "",
    ) -> None:
        """Record a PII detection."""
        if not self._enabled:
            return
        self._pii_detections.labels(
            pii_type=pii_type,
            action=action,
            org_id=org_id,
            team_id=team_id,
        ).inc()

    def record_budget_violation(self, *, org_id: str = "", team_id: str = "") -> None:
        """Record a budget violation."""
        if not self._enabled:
            return
        self._budget_violations.labels(org_id=org_id, team_id=team_id).inc()

    def record_local_routing(
        self,
        *,
        decision: str,
        org_id: str = "",
        team_id: str = "",
    ) -> None:
        """Record a routing decision (local or cloud)."""
        if not self._enabled:
            return
        self._local_routing.labels(decision=decision, org_id=org_id, team_id=team_id).inc()

    def record_kill_switch_block(self, *, org_id: str = "", team_id: str = "") -> None:
        """Record a kill switch block."""
        if not self._enabled:
            return
        self._kill_switch_blocks.labels(org_id=org_id, team_id=team_id).inc()

    def record_blast_radius_pause(
        self,
        *,
        pause_type: str,
        org_id: str = "",
        team_id: str = "",
    ) -> None:
        """Record a blast radius pause (session or agent)."""
        if not self._enabled:
            return
        self._blast_radius_pauses.labels(
            type=pause_type,
            org_id=org_id,
            team_id=team_id,
        ).inc()

    def record_rate_limit(
        self,
        *,
        team_id: str,
        outcome: str,
        wait_ms: float = 0.0,
        virtual_key_id: str = "",
    ) -> None:
        """Record a rate limit event.

        Args:
            team_id: The team that was rate-limited.
            outcome: One of "passed", "queued", "rejected", "timed_out".
            wait_ms: Queue wait time in milliseconds (for queued requests).
            virtual_key_id: The virtual key ID (for proxy per-key rate limiting).
        """
        if not self._enabled:
            return
        self._rate_limit_requests.labels(
            team_id=team_id,
            virtual_key_id=virtual_key_id,
            outcome=outcome,
        ).inc()
        if wait_ms > 0:
            self._rate_limit_wait_seconds.labels(
                team_id=team_id,
                virtual_key_id=virtual_key_id,
            ).observe(wait_ms / 1000.0)

    def record_circuit_breaker(
        self,
        *,
        provider: str,
        state: str,
        org_id: str = "",
        team_id: str = "",
    ) -> None:
        """Record a circuit breaker state transition."""
        if not self._enabled:
            return
        self._circuit_breaker_transitions.labels(
            provider=provider,
            state=state,
            org_id=org_id,
            team_id=team_id,
        ).inc()

    def set_active_sessions(self, count: int) -> None:
        """Set the current active session count."""
        if not self._enabled:
            return
        self._active_sessions.set(count)

    def generate_metrics(self) -> str:
        """Generate Prometheus exposition format text.

        Returns an empty comment if prometheus_client is not available.
        """
        if not self._enabled or self._registry is None:
            return "# StateLoom metrics disabled (prometheus_client not installed)\n"

        from prometheus_client import generate_latest

        return generate_latest(self._registry).decode("utf-8")
