"""Blast radius containment middleware — auto-pause sessions with repeated failures."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomBlastRadiusError,
    StateLoomBudgetError,
    StateLoomCancellationError,
    StateLoomComplianceError,
    StateLoomKillSwitchError,
    StateLoomRateLimitError,
    StateLoomSuspendedError,
    StateLoomTimeoutError,
)
from stateloom.core.event import BlastRadiusEvent
from stateloom.core.types import SessionStatus
from stateloom.middleware.base import MiddlewareContext

if TYPE_CHECKING:
    from stateloom.observability.collector import MetricsCollector

logger = logging.getLogger("stateloom.middleware.blast_radius")

# Store key for persisted blast radius config state
_STORE_KEY_BLAST_RADIUS_CONFIG = "blast_radius_config_json"

# How often to poll the store for cross-process state (seconds)
_STORE_POLL_INTERVAL = 2.0


class BlastRadiusMiddleware:
    """Auto-pause sessions after repeated failures or budget violations.

    Sits at position 1 in the middleware chain (after kill switch, before
    experiment). Tracks consecutive failures and hourly budget violations
    per session and per agent. When thresholds are breached, the session
    (or all sessions for that agent) is paused and an optional webhook is fired.
    """

    def __init__(
        self,
        config: StateLoomConfig,
        store: Any = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._metrics = metrics
        # Per-session tracking
        self._failure_counts: dict[str, int] = {}
        self._budget_violations: dict[str, list[float]] = {}
        self._paused_sessions: set[str] = set()
        # Per-agent tracking (cross-session)
        self._agent_failure_counts: dict[str, int] = {}
        self._agent_budget_violations: dict[str, list[float]] = {}
        self._paused_agents: set[str] = set()
        # Lock: single _lock guards all tracking dicts
        self._lock = threading.Lock()
        self._last_store_poll: float = 0.0

    def _sync_from_store(self) -> None:
        """Poll persisted blast radius config from the store (cross-process sync)."""
        if not self._store:
            return
        now = time.monotonic()
        if now - self._last_store_poll < _STORE_POLL_INTERVAL:
            return
        self._last_store_poll = now
        try:
            blob = self._store.get_secret(_STORE_KEY_BLAST_RADIUS_CONFIG)
            if not blob:
                return
            data = json.loads(blob)
            if "enabled" in data:
                self._config.blast_radius_enabled = bool(data["enabled"])
            if "consecutive_failures" in data:
                self._config.blast_radius_consecutive_failures = int(data["consecutive_failures"])
            if "budget_violations_per_hour" in data:
                self._config.blast_radius_budget_violations_per_hour = int(
                    data["budget_violations_per_hour"]
                )
        except Exception:
            logger.debug("Failed to sync blast radius config from store", exc_info=True)

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        self._sync_from_store()
        if not self._config.blast_radius_enabled:
            return await call_next(ctx)

        session_id = ctx.session.id
        agent_id = self._get_agent_identity(ctx)

        # If session or agent is already paused, block immediately
        with self._lock:
            if session_id in self._paused_sessions:
                logger.info("Blast radius: blocking already-paused session '%s'", session_id)
                raise StateLoomBlastRadiusError(session_id=session_id, trigger="session_paused")
            if agent_id in self._paused_agents:
                logger.info(
                    "Blast radius: blocking session '%s' — agent '%s' is paused",
                    session_id,
                    agent_id,
                )
                raise StateLoomBlastRadiusError(session_id=session_id, trigger="agent_paused")

        if ctx.is_streaming:
            result = await call_next(ctx)

            def _on_stream_complete() -> None:
                if ctx._stream_error is not None:
                    exc = ctx._stream_error
                    if isinstance(exc, StateLoomBudgetError):
                        self._record_budget_violation(ctx, agent_id)
                    elif isinstance(
                        exc,
                        (
                            StateLoomKillSwitchError,
                            StateLoomBlastRadiusError,
                            StateLoomRateLimitError,
                            StateLoomTimeoutError,
                            StateLoomCancellationError,
                            StateLoomComplianceError,
                            StateLoomSuspendedError,
                        ),
                    ):
                        pass  # Don't count these as failures
                    else:
                        self._record_failure(ctx, agent_id)
                else:
                    with self._lock:
                        self._failure_counts.pop(session_id, None)
                        self._agent_failure_counts.pop(agent_id, None)

            ctx._on_stream_complete.append(_on_stream_complete)
            return result

        try:
            result = await call_next(ctx)
        except StateLoomBudgetError:
            self._record_budget_violation(ctx, agent_id)
            raise
        except (
            StateLoomKillSwitchError,
            StateLoomBlastRadiusError,
            StateLoomRateLimitError,
            StateLoomTimeoutError,
            StateLoomCancellationError,
            StateLoomComplianceError,
            StateLoomSuspendedError,
        ):
            # Don't count these as failures
            raise
        except Exception:
            self._record_failure(ctx, agent_id)
            raise

        # Success — reset consecutive failure counters for both session and agent
        with self._lock:
            self._failure_counts.pop(session_id, None)
            self._agent_failure_counts.pop(agent_id, None)

        return result

    def _get_agent_identity(self, ctx: MiddlewareContext) -> str:
        """Derive agent identity from session metadata or model fallback."""
        agent_name = ctx.session.agent_name or ctx.session.metadata.get("agent_name")
        if agent_name:
            return f"agent:{agent_name}"
        return f"model:{ctx.model}"

    def _record_failure(self, ctx: MiddlewareContext, agent_id: str) -> None:
        """Record a consecutive failure and check thresholds."""
        session_id = ctx.session.id
        threshold = self._config.blast_radius.consecutive_failures

        with self._lock:
            # Session tracking
            session_count = self._failure_counts.get(session_id, 0) + 1
            self._failure_counts[session_id] = session_count

            # Agent tracking
            agent_count = self._agent_failure_counts.get(agent_id, 0) + 1
            self._agent_failure_counts[agent_id] = agent_count

        logger.debug(
            "Blast radius: failure recorded session=%s agent=%s "
            "session_count=%d agent_count=%d threshold=%d",
            session_id,
            agent_id,
            session_count,
            agent_count,
            threshold,
        )

        # Pause whichever hits threshold first
        if session_count >= threshold:
            self._pause_session(ctx, "consecutive_failures", session_count, threshold, agent_id)
        elif agent_count >= threshold:
            self._pause_agent(ctx, "consecutive_failures", agent_count, threshold, agent_id)

    def _record_budget_violation(self, ctx: MiddlewareContext, agent_id: str) -> None:
        """Record a budget violation and check hourly thresholds."""
        session_id = ctx.session.id
        threshold = self._config.blast_radius.budget_violations_per_hour
        now = time.time()
        one_hour_ago = now - 3600

        with self._lock:
            # Session tracking
            session_timestamps = self._budget_violations.get(session_id, [])
            session_timestamps = [t for t in session_timestamps if t > one_hour_ago]
            session_timestamps.append(now)
            self._budget_violations[session_id] = session_timestamps
            session_count = len(session_timestamps)

            # Agent tracking
            agent_timestamps = self._agent_budget_violations.get(agent_id, [])
            agent_timestamps = [t for t in agent_timestamps if t > one_hour_ago]
            agent_timestamps.append(now)
            self._agent_budget_violations[agent_id] = agent_timestamps
            agent_count = len(agent_timestamps)

        if session_count >= threshold:
            self._pause_session(ctx, "budget_violations", session_count, threshold, agent_id)
        elif agent_count >= threshold:
            self._pause_agent(ctx, "budget_violations", agent_count, threshold, agent_id)

    def _pause_session(
        self,
        ctx: MiddlewareContext,
        trigger: str,
        count: int,
        threshold: int,
        agent_id: str,
    ) -> None:
        """Pause a session, save event, fire webhook."""
        session_id = ctx.session.id

        with self._lock:
            if session_id in self._paused_sessions:
                return  # Already paused
            self._paused_sessions.add(session_id)

        # End session with PAUSED status
        ctx.session.end(SessionStatus.PAUSED)

        webhook_url = self._config.blast_radius.webhook_url
        webhook_fired = bool(webhook_url)

        event = BlastRadiusEvent(
            session_id=session_id,
            step=ctx.session.step_counter,
            trigger=trigger,
            count=count,
            threshold=threshold,
            action="paused",
            webhook_fired=webhook_fired,
            webhook_url=webhook_url,
            agent_id=agent_id,
        )

        # Save directly to store (pipeline may not complete)
        if self._store:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("Failed to persist blast radius event", exc_info=True)
            try:
                self._store.save_session(ctx.session)
            except Exception:
                logger.debug("Failed to persist session after blast radius", exc_info=True)

        # Fire webhook in daemon thread (non-blocking, fail-open)
        if webhook_url:
            thread = threading.Thread(
                target=self._fire_webhook,
                args=(webhook_url, session_id, trigger, count, threshold, agent_id),
                daemon=True,
            )
            thread.start()

        if self._metrics is not None:
            try:
                self._metrics.record_blast_radius_pause(
                    pause_type="session",
                    org_id=ctx.session.org_id or "",
                    team_id=ctx.session.team_id or "",
                )
            except Exception:
                pass

        logger.warning(
            "Blast radius: session '%s' paused (%s: %d/%d)",
            session_id,
            trigger,
            count,
            threshold,
        )

        raise StateLoomBlastRadiusError(session_id=session_id, trigger=trigger)

    def _pause_agent(
        self,
        ctx: MiddlewareContext,
        trigger: str,
        count: int,
        threshold: int,
        agent_id: str,
    ) -> None:
        """Pause an agent (cross-session), save event, fire webhook."""
        session_id = ctx.session.id

        with self._lock:
            if agent_id in self._paused_agents:
                return  # Already paused
            self._paused_agents.add(agent_id)

        webhook_url = self._config.blast_radius.webhook_url
        webhook_fired = bool(webhook_url)

        event = BlastRadiusEvent(
            session_id=session_id,
            step=ctx.session.step_counter,
            trigger=trigger,
            count=count,
            threshold=threshold,
            action="agent_paused",
            webhook_fired=webhook_fired,
            webhook_url=webhook_url,
            agent_id=agent_id,
        )

        if self._store:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("Failed to persist blast radius event", exc_info=True)

        if webhook_url:
            thread = threading.Thread(
                target=self._fire_webhook,
                args=(webhook_url, session_id, trigger, count, threshold, agent_id),
                daemon=True,
            )
            thread.start()

        if self._metrics is not None:
            try:
                self._metrics.record_blast_radius_pause(
                    pause_type="agent",
                    org_id=ctx.session.org_id or "",
                    team_id=ctx.session.team_id or "",
                )
            except Exception:
                pass

        logger.warning(
            "Blast radius: agent '%s' paused (%s: %d/%d)",
            agent_id,
            trigger,
            count,
            threshold,
        )

        raise StateLoomBlastRadiusError(session_id=session_id, trigger=trigger)

    def unpause_session(self, session_id: str) -> bool:
        """Unpause a session. Returns True if the session was paused."""
        with self._lock:
            if session_id in self._paused_sessions:
                self._paused_sessions.discard(session_id)
                self._failure_counts.pop(session_id, None)
                self._budget_violations.pop(session_id, None)
                return True
            return False

    def unpause_agent(self, agent_id: str) -> bool:
        """Unpause an agent. Returns True if the agent was paused."""
        with self._lock:
            if agent_id in self._paused_agents:
                self._paused_agents.discard(agent_id)
                self._agent_failure_counts.pop(agent_id, None)
                self._agent_budget_violations.pop(agent_id, None)
                return True
            return False

    def on_session_end(self, session_id: str) -> None:
        """Clean up per-session tracking data when a session ends."""
        with self._lock:
            self._failure_counts.pop(session_id, None)
            self._budget_violations.pop(session_id, None)

    def get_status(self) -> dict[str, Any]:
        """Return current blast radius state."""
        with self._lock:
            return {
                "paused_sessions": list(self._paused_sessions),
                "paused_agents": list(self._paused_agents),
                "session_failure_counts": dict(self._failure_counts),
                "agent_failure_counts": dict(self._agent_failure_counts),
                "session_budget_violations": {
                    k: len(v) for k, v in self._budget_violations.items()
                },
                "agent_budget_violations": {
                    k: len(v) for k, v in self._agent_budget_violations.items()
                },
            }

    @staticmethod
    def _fire_webhook(
        url: str,
        session_id: str,
        trigger: str,
        count: int,
        threshold: int,
        agent_id: str = "",
    ) -> None:
        """Fire a webhook notification (background thread, fail-open)."""
        try:
            import httpx

            payload = {
                "event": "blast_radius",
                "session_id": session_id,
                "trigger": trigger,
                "count": count,
                "threshold": threshold,
                "action": "paused",
                "agent_id": agent_id,
            }
            with httpx.Client(timeout=10.0) as client:
                client.post(url, json=payload)
        except Exception:
            logger.debug("Blast radius webhook failed", exc_info=True)
