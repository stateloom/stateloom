"""Provider circuit breaker middleware — automatic failover on provider outages."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomBlastRadiusError,
    StateLoomBudgetError,
    StateLoomCancellationError,
    StateLoomCircuitBreakerError,
    StateLoomKillSwitchError,
    StateLoomLoopError,
    StateLoomPIIBlockedError,
    StateLoomRateLimitError,
    StateLoomRetryError,
    StateLoomTimeoutError,
)
from stateloom.core.event import CircuitBreakerEvent
from stateloom.middleware.base import MiddlewareContext

if TYPE_CHECKING:
    from stateloom.observability.collector import MetricsCollector
    from stateloom.store.base import Store

logger = logging.getLogger("stateloom.middleware.circuit_breaker")

# Errors from StateLoom itself — NOT provider failures
_INTERNAL_ERRORS = (
    StateLoomBudgetError,
    StateLoomPIIBlockedError,
    StateLoomKillSwitchError,
    StateLoomBlastRadiusError,
    StateLoomRateLimitError,
    StateLoomRetryError,
    StateLoomTimeoutError,
    StateLoomCancellationError,
    StateLoomLoopError,
    StateLoomCircuitBreakerError,
)

# Built-in model tiers — models of equivalent capability across providers
_DEFAULT_MODEL_TIERS: dict[str, list[str]] = {
    "tier-1-flagship": [
        "gpt-4o",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "gemini-1.5-pro",
    ],
    "tier-2-fast": [
        "gpt-4o-mini",
        "claude-haiku-4-5-20251001",
        "claude-3-5-haiku-20241022",
        "gemini-1.5-flash",
        "gemini-2.0-flash-lite",
    ],
    "tier-3-legacy": [
        "gpt-3.5-turbo",
        "claude-3-haiku-20240307",
    ],
}

# Provider probe models (cheapest model per provider for health checks)
_PROBE_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-1.5-flash",
}

# Provider API endpoints for synthetic probes (bypasses SDK patching)
_PROBE_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}


def _infer_provider(model: str) -> str:
    """Infer provider from model name."""
    if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    return ""


def _find_tier_fallback(
    model: str,
    provider: str,
    tiers: dict[str, list[str]],
    open_providers: set[str],
) -> str | None:
    """Find a fallback model in the same tier from a healthy provider."""
    for tier_models in tiers.values():
        if model in tier_models:
            for candidate in tier_models:
                candidate_provider = _infer_provider(candidate)
                if candidate_provider and candidate_provider != provider:
                    if candidate_provider not in open_providers:
                        return candidate
    return None


class _CircuitState:
    """Per-provider circuit breaker state machine."""

    __slots__ = ("provider", "state", "failure_timestamps", "opened_at", "_lock")

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.state: str = "closed"  # closed | open | half_open
        self.failure_timestamps: deque[float] = deque(maxlen=10_000)
        self.opened_at: float = 0.0
        self._lock = threading.Lock()

    def record_failure(self, window_seconds: int, threshold: int) -> str | None:
        """Record a failure. Returns new state if transition occurred."""
        now = time.monotonic()
        with self._lock:
            self.failure_timestamps.append(now)
            # Prune failures outside the sliding window
            cutoff = now - window_seconds
            while self.failure_timestamps and self.failure_timestamps[0] < cutoff:
                self.failure_timestamps.popleft()

            if self.state == "half_open":
                # Probe failed → back to open
                self.state = "open"
                self.opened_at = now
                return "open"

            if self.state == "closed" and len(self.failure_timestamps) >= threshold:
                self.state = "open"
                self.opened_at = now
                return "open"

        return None

    def record_success(self) -> str | None:
        """Record a success. Returns new state if transition occurred."""
        with self._lock:
            if self.state == "half_open":
                self.state = "closed"
                self.failure_timestamps.clear()
                return "closed"
        return None

    def check_state(self, recovery_timeout: int) -> str:
        """Check current state, transitioning open→half_open if timeout elapsed."""
        with self._lock:
            if self.state == "open":
                elapsed = time.monotonic() - self.opened_at
                if elapsed >= recovery_timeout:
                    self.state = "half_open"
            return self.state

    def get_failure_count(self, window_seconds: int) -> int:
        """Get failures in the current sliding window."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            while self.failure_timestamps and self.failure_timestamps[0] < cutoff:
                self.failure_timestamps.popleft()
            return len(self.failure_timestamps)

    def reset(self) -> None:
        """Reset circuit to closed."""
        with self._lock:
            self.state = "closed"
            self.failure_timestamps.clear()
            self.opened_at = 0.0


class ProviderCircuitBreakerMiddleware:
    """Circuit breaker per provider with tier-based failover.

    Tracks LLM call failures per-provider in a sliding time window.
    When failures exceed the threshold, the circuit opens and requests
    are blocked (with fallback model info in the error).

    After recovery_timeout, a synthetic probe is sent to check if the
    provider is healthy before allowing real traffic.
    """

    def __init__(
        self,
        config: StateLoomConfig,
        store: Store | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._metrics = metrics
        self._circuits: dict[str, _CircuitState] = {}
        # Lock ordering: _circuits_lock → _CircuitState._lock (always this order)
        self._circuits_lock = threading.Lock()
        self._probing: set[str] = set()
        self._probing_lock = threading.Lock()

    def _get_circuit(self, provider: str) -> _CircuitState:
        """Get or create a circuit state for a provider."""
        with self._circuits_lock:
            if provider not in self._circuits:
                self._circuits[provider] = _CircuitState(provider)
            return self._circuits[provider]

    def _open_providers(self) -> set[str]:
        """Get set of providers whose circuits are currently open."""
        result: set[str] = set()
        with self._circuits_lock:
            for p, cs in self._circuits.items():
                if cs.state in ("open", "half_open"):
                    result.add(p)
        return result

    def _resolve_fallback(self, model: str, provider: str) -> str | None:
        """Find the best fallback model for the given model/provider.

        Priority: explicit fallback_map > tier-based auto-resolution.
        """
        # 1. Check explicit config map (model-to-model)
        fallback = self._config.circuit_breaker_fallback_map.get(model)
        if fallback:
            fb_provider = _infer_provider(fallback)
            if fb_provider and fb_provider != provider:
                # Ensure fallback provider is healthy
                fb_circuit = self._get_circuit(fb_provider)
                if fb_circuit.state == "closed":
                    return fallback

        # 2. Tier-based auto-resolution
        open_providers = self._open_providers()
        return _find_tier_fallback(model, provider, _DEFAULT_MODEL_TIERS, open_providers)

    async def _send_probe(self, provider: str) -> bool:
        """Send a synthetic probe via httpx (bypasses SDK patching).

        Returns True if the provider responded successfully.
        """
        import asyncio

        # Prevent concurrent probes to the same provider
        with self._probing_lock:
            if provider in self._probing:
                return False
            self._probing.add(provider)

        try:
            return await asyncio.get_event_loop().run_in_executor(None, self._probe_sync, provider)
        finally:
            with self._probing_lock:
                self._probing.discard(provider)

    def _probe_sync(self, provider: str) -> bool:
        """Synchronous probe call using httpx."""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available for circuit breaker probe")
            return False

        try:
            if provider in ("openai",):
                api_key = self._config.provider_api_key_openai
                if not api_key:
                    return False
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        _PROBE_ENDPOINTS["openai"],
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": _PROBE_MODELS["openai"],
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                        },
                    )
                    return resp.status_code == 200

            elif provider in ("anthropic",):
                api_key = self._config.provider_api_key_anthropic
                if not api_key:
                    return False
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        _PROBE_ENDPOINTS["anthropic"],
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": _PROBE_MODELS["anthropic"],
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                        },
                    )
                    return resp.status_code == 200

            elif provider in ("gemini",):
                api_key = self._config.provider_api_key_google
                if not api_key:
                    return False
                model = _PROBE_MODELS["gemini"]
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{model}:generateContent?key={api_key}"
                )
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": "hi"}]}],
                            "generationConfig": {"maxOutputTokens": 1},
                        },
                    )
                    return resp.status_code == 200

        except Exception:
            logger.debug("Probe to %s failed", provider, exc_info=True)

        return False

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        if not self._config.circuit_breaker_enabled:
            return await call_next(ctx)

        provider = ctx.provider
        if not provider or provider == "local":
            return await call_next(ctx)

        circuit = self._get_circuit(provider)
        state = circuit.check_state(self._config.circuit_breaker_recovery_timeout)

        if state == "open":
            # Circuit is open — try fallback
            fallback_model = self._resolve_fallback(ctx.model, provider)
            self._emit_event(
                ctx,
                provider=provider,
                state="open",
                previous_state="open",
                fallback_model=fallback_model or "",
                original_model=ctx.model,
            )
            raise StateLoomCircuitBreakerError(
                provider=provider,
                fallback_model=fallback_model or "",
            )

        if state == "half_open":
            # Send synthetic probe instead of using real request
            probe_ok = await self._send_probe(provider)
            if probe_ok:
                transition = circuit.record_success()
                self._emit_event(
                    ctx,
                    provider=provider,
                    state="closed",
                    previous_state="half_open",
                    probe_success=True,
                    original_model=ctx.model,
                )
                logger.info(
                    "Circuit breaker for '%s': probe succeeded, circuit CLOSED",
                    provider,
                )
                # Probe succeeded — allow the real request through
            else:
                circuit.record_failure(
                    self._config.circuit_breaker_window_seconds,
                    self._config.circuit_breaker_failure_threshold,
                )
                fallback_model = self._resolve_fallback(ctx.model, provider)
                self._emit_event(
                    ctx,
                    provider=provider,
                    state="open",
                    previous_state="half_open",
                    probe_success=False,
                    fallback_model=fallback_model or "",
                    original_model=ctx.model,
                )
                logger.warning(
                    "Circuit breaker for '%s': probe failed, circuit re-OPENED",
                    provider,
                )
                raise StateLoomCircuitBreakerError(
                    provider=provider,
                    fallback_model=fallback_model or "",
                )

        # Circuit is closed (or just recovered) — let the call through
        if ctx.is_streaming:
            result = await call_next(ctx)

            def _on_stream_complete() -> None:
                if ctx._stream_error is not None:
                    exc = ctx._stream_error
                    if not isinstance(exc, _INTERNAL_ERRORS):
                        transition = circuit.record_failure(
                            self._config.circuit_breaker_window_seconds,
                            self._config.circuit_breaker_failure_threshold,
                        )
                        failure_count = circuit.get_failure_count(
                            self._config.circuit_breaker_window_seconds
                        )
                        if transition == "open":
                            fallback_model = self._resolve_fallback(ctx.model, provider)
                            self._emit_event(
                                ctx,
                                provider=provider,
                                state="open",
                                previous_state="closed",
                                failure_count=failure_count,
                                fallback_model=fallback_model or "",
                                original_model=ctx.model,
                            )
                            logger.warning(
                                "Circuit breaker TRIPPED for '%s': %d failures in %ds window",
                                provider,
                                failure_count,
                                self._config.circuit_breaker_window_seconds,
                            )
                        else:
                            self._emit_event(
                                ctx,
                                provider=provider,
                                state="closed",
                                previous_state="closed",
                                failure_count=failure_count,
                                original_model=ctx.model,
                            )
                else:
                    circuit.record_success()

            ctx._on_stream_complete.append(_on_stream_complete)
            return result

        try:
            result = await call_next(ctx)
            circuit.record_success()
            return result
        except Exception as exc:
            # Don't count StateLoom internal errors as provider failures
            if isinstance(exc, _INTERNAL_ERRORS):
                raise

            transition = circuit.record_failure(
                self._config.circuit_breaker_window_seconds,
                self._config.circuit_breaker_failure_threshold,
            )

            failure_count = circuit.get_failure_count(self._config.circuit_breaker_window_seconds)

            if transition == "open":
                fallback_model = self._resolve_fallback(ctx.model, provider)
                self._emit_event(
                    ctx,
                    provider=provider,
                    state="open",
                    previous_state="closed",
                    failure_count=failure_count,
                    fallback_model=fallback_model or "",
                    original_model=ctx.model,
                )
                logger.warning(
                    "Circuit breaker TRIPPED for '%s': %d failures in %ds window",
                    provider,
                    failure_count,
                    self._config.circuit_breaker_window_seconds,
                )
            else:
                # Record each failure so the trace shows count building up
                self._emit_event(
                    ctx,
                    provider=provider,
                    state="closed",
                    previous_state="closed",
                    failure_count=failure_count,
                    original_model=ctx.model,
                )

            raise

    def _emit_event(
        self,
        ctx: MiddlewareContext,
        *,
        provider: str,
        state: str,
        previous_state: str,
        failure_count: int = 0,
        fallback_provider: str = "",
        fallback_model: str = "",
        original_model: str = "",
        probe_success: bool | None = None,
    ) -> None:
        """Create and persist a CircuitBreakerEvent."""
        if fallback_model and not fallback_provider:
            fallback_provider = _infer_provider(fallback_model)

        event = CircuitBreakerEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            provider=provider,
            state=state,
            previous_state=previous_state,
            failure_count=failure_count,
            failure_threshold=self._config.circuit_breaker_failure_threshold,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            original_model=original_model,
            probe_success=probe_success,
        )
        ctx.events.append(event)

        # Persist directly — the error propagates and EventRecorder may not run
        if self._store and failure_count > 0:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("Failed to persist circuit breaker event", exc_info=True)

        # Record metric
        if self._metrics is not None:
            try:
                self._metrics.record_circuit_breaker(
                    provider=provider,
                    state=state,
                    org_id=ctx.session.org_id or "",
                    team_id=ctx.session.team_id or "",
                )
            except Exception:
                pass

    # --- Public API ---

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for all tracked providers."""
        result: dict[str, Any] = {}
        window = self._config.circuit_breaker_window_seconds
        with self._circuits_lock:
            for provider, cs in self._circuits.items():
                state = cs.check_state(self._config.circuit_breaker_recovery_timeout)
                result[provider] = {
                    "state": state,
                    "failure_count": cs.get_failure_count(window),
                    "failure_threshold": self._config.circuit_breaker_failure_threshold,
                    "window_seconds": window,
                    "recovery_timeout": self._config.circuit_breaker_recovery_timeout,
                }
        return result

    def reset(self, provider: str) -> bool:
        """Reset a provider's circuit to closed. Returns True if found."""
        with self._circuits_lock:
            cs = self._circuits.get(provider)
            if cs is None:
                return False
            cs.reset()
            logger.info("Circuit breaker for '%s' manually reset to CLOSED", provider)
            return True

    def reset_all(self) -> None:
        """Reset all circuits to closed."""
        with self._circuits_lock:
            for cs in self._circuits.values():
                cs.reset()
