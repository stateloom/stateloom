"""Tests for session suspension (human-in-the-loop)."""

from __future__ import annotations

import threading

import pytest

from stateloom.core.errors import StateLoomSuspendedError
from stateloom.core.event import SuspensionEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType, SessionStatus
from stateloom.retry import _NON_RETRYABLE


class TestSessionSuspendMethod:
    """Session.suspend() method."""

    def test_not_suspended_by_default(self):
        session = Session(id="s1")
        assert session.is_suspended is False

    def test_suspend_sets_flag(self):
        session = Session(id="s1")
        session.suspend()
        assert session.is_suspended is True

    def test_suspend_sets_status(self):
        session = Session(id="s1")
        session.suspend()
        assert session.status == SessionStatus.SUSPENDED

    def test_suspend_with_reason(self):
        session = Session(id="s1")
        session.suspend(reason="Review this draft")
        assert session.metadata["_suspend_reason"] == "Review this draft"

    def test_suspend_with_data(self):
        session = Session(id="s1")
        session.suspend(data={"draft": "hello world"})
        assert session.metadata["_suspend_data"] == {"draft": "hello world"}

    def test_suspend_with_reason_and_data(self):
        session = Session(id="s1")
        session.suspend(reason="Approve", data={"score": 0.95})
        assert session.metadata["_suspend_reason"] == "Approve"
        assert session.metadata["_suspend_data"] == {"score": 0.95}

    def test_suspend_clears_previous_payload(self):
        session = Session(id="s1")
        session.suspend()
        session.signal(payload={"old": True})
        assert session.signal_payload == {"old": True}
        # Re-suspend clears payload
        session.suspend()
        assert session.signal_payload is None

    def test_suspend_no_reason_no_metadata(self):
        """Suspend without reason/data doesn't add metadata keys."""
        session = Session(id="s1")
        session.suspend()
        assert "_suspend_reason" not in session.metadata
        assert "_suspend_data" not in session.metadata


class TestSessionSignalMethod:
    """Session.signal() method."""

    def test_signal_resumes(self):
        session = Session(id="s1")
        session.suspend()
        assert session.is_suspended is True
        session.signal(payload={"approved": True})
        assert session.is_suspended is False
        assert session.status == SessionStatus.ACTIVE

    def test_signal_payload_accessible(self):
        session = Session(id="s1")
        session.suspend()
        session.signal(payload={"decision": "approved", "reviewer": "alice"})
        assert session.signal_payload == {
            "decision": "approved",
            "reviewer": "alice",
        }


class TestWaitForSignal:
    """Session.wait_for_signal() blocking behavior."""

    def test_wait_returns_payload_when_signaled(self):
        session = Session(id="s1")
        session.suspend()

        result = [None]

        def waiter():
            result[0] = session.wait_for_signal(timeout=5.0)

        t = threading.Thread(target=waiter)
        t.start()

        # Signal from "human"
        session.signal(payload={"approved": True})
        t.join(timeout=5.0)
        assert result[0] == {"approved": True}

    def test_wait_returns_none_on_timeout(self):
        session = Session(id="s1")
        session.suspend()
        result = session.wait_for_signal(timeout=0.05)
        assert result is None


class TestSuspensionEvent:
    """SuspensionEvent dataclass."""

    def test_event_type_is_suspension(self):
        event = SuspensionEvent()
        assert event.event_type == EventType.SUSPENSION
        assert event.event_type.value == "suspension"

    def test_suspend_event_fields(self):
        event = SuspensionEvent(
            session_id="s1",
            action="suspended",
            reason="Review draft",
            suspend_data={"draft": "hello"},
        )
        assert event.action == "suspended"
        assert event.reason == "Review draft"
        assert event.suspend_data == {"draft": "hello"}
        assert event.signal_payload is None
        assert event.suspended_duration_ms == 0.0

    def test_resume_event_fields(self):
        event = SuspensionEvent(
            session_id="s1",
            action="resumed",
            reason="Review draft",
            signal_payload={"approved": True},
            suspended_duration_ms=1500.0,
        )
        assert event.action == "resumed"
        assert event.signal_payload == {"approved": True}
        assert event.suspended_duration_ms == 1500.0


class TestSuspendedStatus:
    """SessionStatus.SUSPENDED enum."""

    def test_suspended_status_exists(self):
        assert hasattr(SessionStatus, "SUSPENDED")
        assert SessionStatus.SUSPENDED == "suspended"


class TestNonRetryableErrors:
    """StateLoomSuspendedError is in _NON_RETRYABLE."""

    def test_suspended_error_non_retryable(self):
        assert StateLoomSuspendedError in _NON_RETRYABLE

    def test_suspended_error_instance_matches(self):
        err = StateLoomSuspendedError(session_id="s1")
        assert isinstance(err, _NON_RETRYABLE)


class TestBlastRadiusExclusion:
    """StateLoomSuspendedError excluded from blast radius failure counting."""

    def test_suspended_excluded_from_blast_radius(self):
        from stateloom.middleware.blast_radius import (
            StateLoomSuspendedError as BR_Suspended,
        )

        assert BR_Suspended is StateLoomSuspendedError


class TestTimeoutCheckerSuspension:
    """TimeoutCheckerMiddleware checks suspension."""

    @pytest.mark.asyncio
    async def test_suspended_session_raises_in_middleware(self):
        from stateloom.middleware.timeout_checker import TimeoutCheckerMiddleware

        mw = TimeoutCheckerMiddleware()
        session = Session(id="s1")
        session.suspend()

        # Create a minimal MiddlewareContext
        from stateloom.core.config import StateLoomConfig
        from stateloom.middleware.base import MiddlewareContext

        ctx = MiddlewareContext(
            provider="openai",
            model="gpt-4o",
            request_kwargs={},
            session=session,
            config=StateLoomConfig(),
        )

        async def dummy_next(c):
            pass

        with pytest.raises(StateLoomSuspendedError):
            await mw.process(ctx, dummy_next)


class TestSQLiteRoundTrip:
    """SuspensionEvent serialization/deserialization through SQLite store."""

    def test_suspension_event_round_trip(self, tmp_path):
        from stateloom.store.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))

        # Save a session first
        session = Session(id="test-session")
        store.save_session(session)

        # Save a suspension event
        event = SuspensionEvent(
            session_id="test-session",
            step=1,
            action="suspended",
            reason="Please review",
            suspend_data={"draft": "hello world", "score": 0.9},
        )
        store.save_event(event)

        # Read it back
        events = store.get_session_events("test-session")
        assert len(events) == 1
        loaded = events[0]
        assert isinstance(loaded, SuspensionEvent)
        assert loaded.event_type == EventType.SUSPENSION
        assert loaded.action == "suspended"
        assert loaded.reason == "Please review"
        assert loaded.suspend_data == {"draft": "hello world", "score": 0.9}
        assert loaded.signal_payload is None
        assert loaded.suspended_duration_ms == 0.0

        store.close()

    def test_resumed_event_round_trip(self, tmp_path):
        from stateloom.store.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))

        session = Session(id="test-session")
        store.save_session(session)

        event = SuspensionEvent(
            session_id="test-session",
            step=2,
            action="resumed",
            reason="Review complete",
            signal_payload={"approved": True, "reviewer": "alice"},
            suspended_duration_ms=2500.5,
        )
        store.save_event(event)

        events = store.get_session_events("test-session")
        assert len(events) == 1
        loaded = events[0]
        assert isinstance(loaded, SuspensionEvent)
        assert loaded.action == "resumed"
        assert loaded.signal_payload == {"approved": True, "reviewer": "alice"}
        assert loaded.suspended_duration_ms == 2500.5

        store.close()
