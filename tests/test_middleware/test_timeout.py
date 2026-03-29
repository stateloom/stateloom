"""Tests for the timeout checker middleware and session timeout/heartbeat."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomCancellationError,
    StateLoomTimeoutError,
)
from stateloom.core.session import Session
from stateloom.core.types import SessionStatus
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.timeout_checker import TimeoutCheckerMiddleware


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {"console_output": False}
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(
    session: Session | None = None,
    config: StateLoomConfig | None = None,
) -> MiddlewareContext:
    if session is None:
        session = Session(id="test-session")
    return MiddlewareContext(
        session=session,
        config=config or _make_config(),
        provider="openai",
        model="gpt-4",
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )


class TestSessionTimeoutFields:
    """Session timeout and heartbeat fields."""

    def test_default_timeouts_are_none(self):
        session = Session(id="s1")
        assert session.timeout is None
        assert session.idle_timeout is None
        assert session.last_heartbeat is None

    def test_set_timeouts(self):
        session = Session(id="s1", timeout=60.0, idle_timeout=30.0)
        assert session.timeout == 60.0
        assert session.idle_timeout == 30.0

    def test_heartbeat_updates_timestamp(self):
        session = Session(id="s1", idle_timeout=30.0)
        session.heartbeat()
        assert session.last_heartbeat is not None
        assert isinstance(session.last_heartbeat, datetime)

    def test_is_timed_out_no_timeout_set(self):
        session = Session(id="s1")
        timed_out, _, _, _ = session.is_timed_out()
        assert timed_out is False

    def test_is_timed_out_session_timeout(self):
        session = Session(id="s1", timeout=0.01)
        # Set started_at to the past
        session.started_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        timed_out, timeout_type, elapsed, limit = session.is_timed_out()
        assert timed_out is True
        assert timeout_type == "session_timeout"
        assert elapsed > 0
        assert limit == 0.01

    def test_is_timed_out_not_yet(self):
        session = Session(id="s1", timeout=3600.0)
        timed_out, _, _, _ = session.is_timed_out()
        assert timed_out is False

    def test_is_timed_out_idle_timeout(self):
        session = Session(id="s1", idle_timeout=0.01)
        session.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=5)
        timed_out, timeout_type, elapsed, limit = session.is_timed_out()
        assert timed_out is True
        assert timeout_type == "idle_timeout"

    def test_idle_timeout_no_heartbeat(self):
        """Idle timeout doesn't fire if no heartbeat was ever set."""
        session = Session(id="s1", idle_timeout=0.01)
        timed_out, _, _, _ = session.is_timed_out()
        assert timed_out is False


class TestTimeoutCheckerMiddleware:
    """TimeoutCheckerMiddleware process() behavior."""

    async def test_passthrough_no_timeout(self):
        mw = TimeoutCheckerMiddleware()
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"

    async def test_session_timeout_raises(self):
        mw = TimeoutCheckerMiddleware()
        session = Session(id="s1", timeout=0.01)
        session.started_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        ctx = _make_ctx(session=session)

        async def call_next(c):
            return "response"

        with pytest.raises(StateLoomTimeoutError) as exc_info:
            await mw.process(ctx, call_next)

        assert exc_info.value.session_id == "s1"
        assert exc_info.value.timeout_type == "session_timeout"
        assert session.status == SessionStatus.TIMED_OUT

    async def test_idle_timeout_raises(self):
        mw = TimeoutCheckerMiddleware()
        session = Session(id="s1", idle_timeout=0.01)
        session.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=5)
        ctx = _make_ctx(session=session)

        async def call_next(c):
            return "response"

        with pytest.raises(StateLoomTimeoutError) as exc_info:
            await mw.process(ctx, call_next)

        assert exc_info.value.timeout_type == "idle_timeout"
        assert session.status == SessionStatus.TIMED_OUT

    async def test_heartbeat_updated_on_success(self):
        mw = TimeoutCheckerMiddleware()
        session = Session(id="s1", timeout=3600.0)
        session.last_heartbeat = None
        ctx = _make_ctx(session=session)

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        assert session.last_heartbeat is not None

    async def test_cancelled_session_raises(self):
        mw = TimeoutCheckerMiddleware()
        session = Session(id="s1")
        session.cancel()
        ctx = _make_ctx(session=session)

        async def call_next(c):
            return "response"

        with pytest.raises(StateLoomCancellationError) as exc_info:
            await mw.process(ctx, call_next)

        assert exc_info.value.session_id == "s1"
        assert session.status == SessionStatus.CANCELLED

    async def test_cancellation_checked_before_timeout(self):
        """Cancellation takes priority over timeout."""
        mw = TimeoutCheckerMiddleware()
        session = Session(id="s1", timeout=0.01)
        session.started_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        session.cancel()
        ctx = _make_ctx(session=session)

        async def call_next(c):
            return "response"

        with pytest.raises(StateLoomCancellationError):
            await mw.process(ctx, call_next)


class TestTimeoutErrorAttributes:
    """StateLoomTimeoutError carries useful attributes."""

    def test_attributes(self):
        err = StateLoomTimeoutError(
            session_id="s1",
            timeout_type="session_timeout",
            elapsed=65.3,
            limit=60.0,
        )
        assert err.session_id == "s1"
        assert err.timeout_type == "session_timeout"
        assert err.elapsed == 65.3
        assert err.limit == 60.0
        assert "timed out" in str(err).lower()

    def test_error_code(self):
        err = StateLoomTimeoutError(
            session_id="s1",
            timeout_type="idle_timeout",
            elapsed=35.0,
            limit=30.0,
        )
        assert err.error_code == "SESSION_TIMED_OUT"


class TestCancellationErrorAttributes:
    """StateLoomCancellationError carries useful attributes."""

    def test_attributes(self):
        err = StateLoomCancellationError(session_id="s1")
        assert err.session_id == "s1"
        assert "cancelled" in str(err).lower()

    def test_error_code(self):
        err = StateLoomCancellationError(session_id="s1")
        assert err.error_code == "SESSION_CANCELLED"
