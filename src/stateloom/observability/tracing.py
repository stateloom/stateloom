"""Optional OpenTelemetry integration for distributed tracing.

When ``opentelemetry-sdk`` is installed, provides span creation for LLM calls
and middleware pipeline execution. When not installed, all operations are
silent no-ops.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("stateloom.observability.tracing")

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


class TracingManager:
    """Manages OpenTelemetry spans for StateLoom.

    When ``opentelemetry-sdk`` is not installed or ``enabled=False``,
    all methods are no-ops.
    """

    def __init__(self, *, enabled: bool = False, service_name: str = "stateloom") -> None:
        self._enabled = enabled and _OTEL_AVAILABLE
        self._tracer: Any = None

        if not enabled:
            return

        if not _OTEL_AVAILABLE:
            logger.info(
                "opentelemetry-sdk not installed — tracing disabled. "
                "Install with: pip install stateloom[tracing]"
            )
            return

        self._tracer = trace.get_tracer(service_name)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Start a new span. Returns a context manager span object.

        If tracing is disabled, returns a no-op context manager.
        """
        if not self._enabled or self._tracer is None:
            return _NoOpSpan()

        return self._tracer.start_as_current_span(
            name,
            attributes=attributes or {},
        )

    def record_llm_span(
        self,
        *,
        model: str,
        provider: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        session_id: str = "",
        status: str = "success",
    ) -> None:
        """Record a completed LLM call as span attributes on the current span."""
        if not self._enabled:
            return

        span = trace.get_current_span()
        if span is None:
            return

        span.set_attribute("llm.model", model)
        span.set_attribute("llm.provider", provider)
        span.set_attribute("llm.latency_ms", latency_ms)
        span.set_attribute("llm.prompt_tokens", prompt_tokens)
        span.set_attribute("llm.completion_tokens", completion_tokens)
        span.set_attribute("llm.cost_usd", cost)
        span.set_attribute("stateloom.session_id", session_id)

        if status != "success":
            span.set_status(StatusCode.ERROR, status)


class _NoOpSpan:
    """No-op span for when tracing is disabled."""

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any) -> None:
        pass
