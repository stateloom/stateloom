"""Timeout and cancellation checking middleware."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from stateloom.core.errors import (
    StateLoomCancellationError,
    StateLoomSuspendedError,
    StateLoomTimeoutError,
)
from stateloom.core.event import SessionLifecycleEvent
from stateloom.core.types import SessionStatus
from stateloom.middleware.base import MiddlewareContext

logger = logging.getLogger("stateloom.middleware.timeout_checker")


class TimeoutCheckerMiddleware:
    """Check session timeouts and cancellation before each LLM call.

    Sits after RateLimiter, before Experiment in the pipeline.
    Checks:
      1. Session cancellation (immediate)
      2. Session timeout (elapsed since started_at)
      3. Idle timeout (elapsed since last_heartbeat)

    After a successful call, updates the session heartbeat.
    """

    def __init__(self, store: Any = None) -> None:
        self._store = store

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        session = ctx.session

        # Check cancellation first
        if session.is_cancelled:
            self._save_lifecycle_event(ctx, action="cancelled")
            session.end(SessionStatus.CANCELLED)
            raise StateLoomCancellationError(session_id=session.id)

        # Check suspension (human-in-the-loop)
        if session.is_suspended:
            raise StateLoomSuspendedError(session_id=session.id)

        # Check timeouts
        timed_out, timeout_type, elapsed, limit = session.is_timed_out()
        if timed_out:
            self._save_lifecycle_event(
                ctx, action="timed_out", reason=timeout_type,
                elapsed=elapsed, limit=limit,
            )
            session.end(SessionStatus.TIMED_OUT)
            raise StateLoomTimeoutError(
                session_id=session.id,
                timeout_type=timeout_type,
                elapsed=elapsed,
                limit=limit,
            )

        result = await call_next(ctx)

        # Update heartbeat after successful call
        if ctx.is_streaming:

            def _on_complete() -> None:
                if session.timeout is not None or session.idle_timeout is not None:
                    session.heartbeat()

            ctx._on_stream_complete.append(_on_complete)
        else:
            if session.timeout is not None or session.idle_timeout is not None:
                session.heartbeat()

        return result

    def _save_lifecycle_event(
        self,
        ctx: MiddlewareContext,
        *,
        action: str,
        reason: str = "",
        elapsed: float = 0.0,
        limit: float = 0.0,
    ) -> None:
        """Persist a lifecycle event directly to the store."""
        event = SessionLifecycleEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            action=action,
            reason=reason,
            elapsed=round(elapsed, 1),
            limit=limit,
        )
        if self._store:
            try:
                self._store.save_event(event)
                self._store.save_session(ctx.session)
            except Exception:
                logger.debug("Failed to persist lifecycle event", exc_info=True)
