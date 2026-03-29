"""Tests for the provider circuit breaker middleware."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomCircuitBreakerError,
    StateLoomKillSwitchError,
)
from stateloom.core.event import CircuitBreakerEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.circuit_breaker import (
    _DEFAULT_MODEL_TIERS,
    ProviderCircuitBreakerMiddleware,
    _CircuitState,
    _find_tier_fallback,
    _infer_provider,
)


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "console_output": False,
        "circuit_breaker_enabled": True,
        "circuit_breaker_failure_threshold": 3,
        "circuit_breaker_window_seconds": 300,
        "circuit_breaker_recovery_timeout": 60,
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(
    provider: str = "openai",
    model: str = "gpt-4o",
    **config_overrides,
) -> MiddlewareContext:
    config = _make_config(**config_overrides)
    return MiddlewareContext(
        session=Session(id="test-session"),
        config=config,
        provider=provider,
        model=model,
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )


# ── _CircuitState unit tests ──


class TestCircuitState:
    def test_initial_state_is_closed(self):
        cs = _CircuitState("openai")
        assert cs.state == "closed"
        assert cs.get_failure_count(300) == 0

    def test_record_failure_below_threshold(self):
        cs = _CircuitState("openai")
        result = cs.record_failure(300, threshold=3)
        assert result is None
        assert cs.state == "closed"
        assert cs.get_failure_count(300) == 1

    def test_trip_on_threshold(self):
        cs = _CircuitState("openai")
        cs.record_failure(300, threshold=3)
        cs.record_failure(300, threshold=3)
        result = cs.record_failure(300, threshold=3)
        assert result == "open"
        assert cs.state == "open"

    def test_success_in_half_open_closes(self):
        cs = _CircuitState("openai")
        # Force to half_open
        cs.state = "half_open"
        result = cs.record_success()
        assert result == "closed"
        assert cs.state == "closed"
        assert cs.get_failure_count(300) == 0

    def test_failure_in_half_open_reopens(self):
        cs = _CircuitState("openai")
        cs.state = "half_open"
        result = cs.record_failure(300, threshold=3)
        assert result == "open"
        assert cs.state == "open"

    def test_check_state_open_to_half_open(self):
        cs = _CircuitState("openai")
        cs.state = "open"
        cs.opened_at = time.monotonic() - 120  # 120 seconds ago
        state = cs.check_state(recovery_timeout=60)
        assert state == "half_open"

    def test_check_state_open_stays_open(self):
        cs = _CircuitState("openai")
        cs.state = "open"
        cs.opened_at = time.monotonic() - 10  # 10 seconds ago
        state = cs.check_state(recovery_timeout=60)
        assert state == "open"

    def test_reset(self):
        cs = _CircuitState("openai")
        cs.record_failure(300, threshold=3)
        cs.record_failure(300, threshold=3)
        cs.record_failure(300, threshold=3)
        assert cs.state == "open"
        cs.reset()
        assert cs.state == "closed"
        assert cs.get_failure_count(300) == 0

    def test_success_in_closed_is_noop(self):
        cs = _CircuitState("openai")
        result = cs.record_success()
        assert result is None
        assert cs.state == "closed"

    def test_sliding_window_eviction(self):
        cs = _CircuitState("openai")
        # Add old failures beyond the window
        old_time = time.monotonic() - 400
        cs.failure_timestamps.append(old_time)
        cs.failure_timestamps.append(old_time)
        # These should be evicted
        count = cs.get_failure_count(300)
        assert count == 0


# ── Helper function tests ──


class TestInferProvider:
    def test_openai(self):
        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("gpt-4o-mini") == "openai"
        assert _infer_provider("o1-preview") == "openai"

    def test_anthropic(self):
        assert _infer_provider("claude-sonnet-4-20250514") == "anthropic"
        assert _infer_provider("claude-3-5-haiku-20241022") == "anthropic"

    def test_gemini(self):
        assert _infer_provider("gemini-1.5-pro") == "gemini"
        assert _infer_provider("gemini-2.0-flash-lite") == "gemini"

    def test_unknown(self):
        assert _infer_provider("llama-3") == ""


class TestFindTierFallback:
    def test_same_tier_fallback(self):
        """gpt-4o → claude-sonnet in the same tier when openai is down."""
        result = _find_tier_fallback(
            "gpt-4o", "openai", _DEFAULT_MODEL_TIERS, open_providers={"openai"}
        )
        assert result is not None
        provider = _infer_provider(result)
        assert provider != "openai"
        # Should be a tier-1-flagship model
        assert result in _DEFAULT_MODEL_TIERS["tier-1-flagship"]

    def test_no_fallback_all_open(self):
        """No fallback if all providers in the tier are down."""
        result = _find_tier_fallback(
            "gpt-4o",
            "openai",
            _DEFAULT_MODEL_TIERS,
            open_providers={"openai", "anthropic", "gemini"},
        )
        assert result is None

    def test_no_fallback_unknown_model(self):
        result = _find_tier_fallback("llama-3", "local", _DEFAULT_MODEL_TIERS, open_providers=set())
        assert result is None

    def test_tier_2_fallback(self):
        result = _find_tier_fallback(
            "gpt-4o-mini", "openai", _DEFAULT_MODEL_TIERS, open_providers={"openai"}
        )
        assert result is not None
        assert result in _DEFAULT_MODEL_TIERS["tier-2-fast"]
        assert _infer_provider(result) != "openai"


# ── Middleware integration tests ──


class TestCircuitBreakerPassthrough:
    async def test_disabled(self):
        """When circuit_breaker_enabled is False, calls pass through."""
        mw = ProviderCircuitBreakerMiddleware(config=_make_config(circuit_breaker_enabled=False))
        ctx = _make_ctx()
        called = False

        async def call_next(c):
            nonlocal called
            called = True
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"
        assert called

    async def test_local_provider_passes_through(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        ctx = _make_ctx(provider="local")

        async def call_next(c):
            return "local-response"

        result = await mw.process(ctx, call_next)
        assert result == "local-response"

    async def test_empty_provider_passes_through(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        ctx = _make_ctx(provider="")

        async def call_next(c):
            return "ok"

        result = await mw.process(ctx, call_next)
        assert result == "ok"


class TestCircuitBreakerClosed:
    async def test_successful_call_passes_through(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        ctx = _make_ctx()

        async def call_next(c):
            return "success"

        result = await mw.process(ctx, call_next)
        assert result == "success"
        assert mw._get_circuit("openai").state == "closed"

    async def test_failures_below_threshold(self):
        """Failures below threshold don't trip the circuit."""
        mw = ProviderCircuitBreakerMiddleware(
            config=_make_config(circuit_breaker_failure_threshold=3)
        )

        async def fail_next(c):
            raise ConnectionError("Provider down")

        # Two failures: circuit stays closed
        for _ in range(2):
            ctx = _make_ctx()
            with pytest.raises(ConnectionError):
                await mw.process(ctx, fail_next)

        assert mw._get_circuit("openai").state == "closed"

    async def test_failures_at_threshold_trips_circuit(self):
        mw = ProviderCircuitBreakerMiddleware(
            config=_make_config(circuit_breaker_failure_threshold=3)
        )

        async def fail_next(c):
            raise ConnectionError("Provider down")

        for _ in range(3):
            ctx = _make_ctx()
            with pytest.raises(ConnectionError):
                await mw.process(ctx, fail_next)

        assert mw._get_circuit("openai").state == "open"

    async def test_internal_errors_not_counted(self):
        """StateLoom internal errors should not count as provider failures."""
        mw = ProviderCircuitBreakerMiddleware(
            config=_make_config(circuit_breaker_failure_threshold=2)
        )

        async def budget_error(c):
            raise StateLoomBudgetError(limit=10.0, spent=5.0, session_id="test-session")

        for _ in range(5):
            ctx = _make_ctx()
            with pytest.raises(StateLoomBudgetError):
                await mw.process(ctx, budget_error)

        # Should NOT have tripped
        assert mw._get_circuit("openai").state == "closed"


class TestCircuitBreakerOpen:
    async def test_open_circuit_raises_error(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        # Force circuit open
        circuit = mw._get_circuit("openai")
        circuit.state = "open"
        circuit.opened_at = time.monotonic()

        ctx = _make_ctx()
        with pytest.raises(StateLoomCircuitBreakerError) as exc_info:
            await mw.process(ctx, lambda c: None)

        assert exc_info.value.provider == "openai"

    async def test_open_circuit_includes_fallback(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        circuit = mw._get_circuit("openai")
        circuit.state = "open"
        circuit.opened_at = time.monotonic()

        ctx = _make_ctx(model="gpt-4o")
        with pytest.raises(StateLoomCircuitBreakerError) as exc_info:
            await mw.process(ctx, lambda c: None)

        # Should suggest a tier-1 alternative
        assert exc_info.value.fallback_model != ""
        assert _infer_provider(exc_info.value.fallback_model) != "openai"

    async def test_open_circuit_emits_event(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        circuit = mw._get_circuit("openai")
        circuit.state = "open"
        circuit.opened_at = time.monotonic()

        ctx = _make_ctx()
        with pytest.raises(StateLoomCircuitBreakerError):
            await mw.process(ctx, lambda c: None)

        cb_events = [e for e in ctx.events if isinstance(e, CircuitBreakerEvent)]
        assert len(cb_events) == 1
        assert cb_events[0].state == "open"
        assert cb_events[0].provider == "openai"

    async def test_explicit_fallback_map(self):
        mw = ProviderCircuitBreakerMiddleware(
            config=_make_config(circuit_breaker_fallback_map={"gpt-4o": "claude-sonnet-4-20250514"})
        )
        circuit = mw._get_circuit("openai")
        circuit.state = "open"
        circuit.opened_at = time.monotonic()

        ctx = _make_ctx(model="gpt-4o")
        with pytest.raises(StateLoomCircuitBreakerError) as exc_info:
            await mw.process(ctx, lambda c: None)

        assert exc_info.value.fallback_model == "claude-sonnet-4-20250514"


class TestCircuitBreakerHalfOpen:
    async def test_probe_success_closes_circuit(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        circuit = mw._get_circuit("openai")
        circuit.state = "open"
        circuit.opened_at = time.monotonic() - 120  # Past recovery timeout

        ctx = _make_ctx()

        async def call_next(c):
            return "success"

        # Mock the probe to succeed
        with patch.object(mw, "_send_probe", return_value=True):
            result = await mw.process(ctx, call_next)

        assert result == "success"
        assert circuit.state == "closed"

    async def test_probe_failure_reopens_circuit(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        circuit = mw._get_circuit("openai")
        circuit.state = "open"
        circuit.opened_at = time.monotonic() - 120

        ctx = _make_ctx()

        with patch.object(mw, "_send_probe", return_value=False):
            with pytest.raises(StateLoomCircuitBreakerError):
                await mw.process(ctx, lambda c: None)

        assert circuit.state == "open"

    async def test_probe_success_emits_closed_event(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        circuit = mw._get_circuit("openai")
        circuit.state = "open"
        circuit.opened_at = time.monotonic() - 120

        ctx = _make_ctx()

        async def call_next(c):
            return "ok"

        with patch.object(mw, "_send_probe", return_value=True):
            await mw.process(ctx, call_next)

        cb_events = [e for e in ctx.events if isinstance(e, CircuitBreakerEvent)]
        assert len(cb_events) == 1
        assert cb_events[0].state == "closed"
        assert cb_events[0].previous_state == "half_open"
        assert cb_events[0].probe_success is True


class TestCircuitBreakerPublicAPI:
    def test_get_status_empty(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        status = mw.get_status()
        assert status == {}

    def test_get_status_with_circuits(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        circuit = mw._get_circuit("openai")
        circuit.record_failure(300, 3)

        status = mw.get_status()
        assert "openai" in status
        assert status["openai"]["state"] == "closed"
        assert status["openai"]["failure_count"] == 1

    def test_reset_provider(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        circuit = mw._get_circuit("openai")
        circuit.state = "open"

        assert mw.reset("openai") is True
        assert circuit.state == "closed"

    def test_reset_unknown_provider(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        assert mw.reset("unknown") is False

    def test_reset_all(self):
        mw = ProviderCircuitBreakerMiddleware(config=_make_config())
        mw._get_circuit("openai").state = "open"
        mw._get_circuit("anthropic").state = "open"

        mw.reset_all()
        assert mw._get_circuit("openai").state == "closed"
        assert mw._get_circuit("anthropic").state == "closed"


class TestCircuitBreakerMetrics:
    async def test_metrics_recorded_on_trip(self):
        metrics = MagicMock()
        mw = ProviderCircuitBreakerMiddleware(
            config=_make_config(circuit_breaker_failure_threshold=1),
            metrics=metrics,
        )

        async def fail_next(c):
            raise ConnectionError("down")

        ctx = _make_ctx()
        with pytest.raises(ConnectionError):
            await mw.process(ctx, fail_next)

        # Tripped → should record metric
        metrics.record_circuit_breaker.assert_called()
        call_kwargs = metrics.record_circuit_breaker.call_args
        assert call_kwargs.kwargs["provider"] == "openai"
        assert call_kwargs.kwargs["state"] == "open"


class TestCircuitBreakerEventPersistence:
    async def test_event_persisted_on_trip(self):
        store = MagicMock()
        mw = ProviderCircuitBreakerMiddleware(
            config=_make_config(circuit_breaker_failure_threshold=1),
            store=store,
        )

        async def fail_next(c):
            raise ConnectionError("down")

        ctx = _make_ctx()
        with pytest.raises(ConnectionError):
            await mw.process(ctx, fail_next)

        store.save_event.assert_called_once()
        saved_event = store.save_event.call_args[0][0]
        assert isinstance(saved_event, CircuitBreakerEvent)
        assert saved_event.state == "open"
