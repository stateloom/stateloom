"""Tests for semantic retry (self-healing) feature."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import stateloom
from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomKillSwitchError,
    StateLoomPIIBlockedError,
    StateLoomRetryError,
)
from stateloom.core.event import SemanticRetryEvent
from stateloom.core.types import EventType
from stateloom.retry import RetryLoop, durable_task, retry_loop
from stateloom.store.memory_store import MemoryStore


def _init_gate() -> stateloom.Gate:
    return stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )


# --- RetryLoop tests ---


class TestRetryLoop:
    def test_succeeds_first_attempt(self):
        """No retries needed when first attempt succeeds."""
        results = []
        for attempt in RetryLoop(retries=3):
            with attempt:
                results.append(attempt.number)
                value = 42

        assert results == [1]
        assert value == 42

    def test_succeeds_on_retry(self):
        """Fails once, succeeds on second attempt."""
        call_count = 0
        for attempt in RetryLoop(retries=3):
            with attempt:
                call_count += 1
                if call_count == 1:
                    raise ValueError("bad JSON")
                value = "ok"

        assert call_count == 2
        assert value == "ok"

    def test_succeeds_on_third_attempt(self):
        """Fails twice, succeeds on third attempt."""
        call_count = 0
        for attempt in RetryLoop(retries=3):
            with attempt:
                call_count += 1
                if call_count < 3:
                    raise ValueError("still bad")
                value = "finally"

        assert call_count == 3
        assert value == "finally"

    def test_exhausted_raises_last_exception(self):
        """All attempts fail — last exception propagates."""
        with pytest.raises(ValueError, match="attempt 3"):
            for attempt in RetryLoop(retries=3):
                with attempt:
                    raise ValueError(f"attempt {attempt.number}")

    def test_stateloom_budget_error_not_retried(self):
        """Budget errors propagate immediately, no retry."""
        call_count = 0
        with pytest.raises(StateLoomBudgetError):
            for attempt in RetryLoop(retries=3):
                with attempt:
                    call_count += 1
                    raise StateLoomBudgetError(limit=5.0, spent=6.0, session_id="s1")

        assert call_count == 1

    def test_stateloom_pii_error_not_retried(self):
        """PII errors propagate immediately, no retry."""
        call_count = 0
        with pytest.raises(StateLoomPIIBlockedError):
            for attempt in RetryLoop(retries=3):
                with attempt:
                    call_count += 1
                    raise StateLoomPIIBlockedError(pii_type="ssn", session_id="s1")

        assert call_count == 1

    def test_stateloom_kill_switch_not_retried(self):
        """Kill switch errors propagate immediately, no retry."""
        call_count = 0
        with pytest.raises(StateLoomKillSwitchError):
            for attempt in RetryLoop(retries=3):
                with attempt:
                    call_count += 1
                    raise StateLoomKillSwitchError()

        assert call_count == 1

    def test_retry_error_not_retried(self):
        """StateLoomRetryError itself propagates immediately."""
        call_count = 0
        with pytest.raises(StateLoomRetryError):
            for attempt in RetryLoop(retries=3):
                with attempt:
                    call_count += 1
                    raise StateLoomRetryError(attempts=3, last_error="x")

        assert call_count == 1

    def test_on_retry_callback(self):
        """on_retry callback is called for each failed attempt."""
        retries_seen: list[tuple[int, str]] = []

        def on_retry(attempt: int, error: Exception) -> None:
            retries_seen.append((attempt, str(error)))

        call_count = 0
        for attempt in RetryLoop(retries=3, on_retry=on_retry):
            with attempt:
                call_count += 1
                if call_count < 3:
                    raise ValueError(f"fail-{call_count}")

        assert len(retries_seen) == 2
        assert retries_seen[0] == (1, "fail-1")
        assert retries_seen[1] == (2, "fail-2")

    def test_retry_loop_factory(self):
        """retry_loop() factory creates a RetryLoop."""
        loop = retry_loop(retries=2)
        assert isinstance(loop, RetryLoop)
        assert loop._max_attempts == 2

    def test_records_events_to_store(self):
        """SemanticRetryEvent is persisted to the store inside a session."""
        gate = _init_gate()

        call_count = 0
        with stateloom.session("retry-test-session") as sess:
            for attempt in RetryLoop(retries=3):
                with attempt:
                    call_count += 1
                    if call_count == 1:
                        raise ValueError("bad output")

        # Check events in store
        events = gate.store.get_session_events("retry-test-session")
        retry_events = [e for e in events if isinstance(e, SemanticRetryEvent)]
        assert len(retry_events) == 1
        assert retry_events[0].attempt == 1
        assert retry_events[0].max_attempts == 3
        assert retry_events[0].error_type == "ValueError"
        assert "bad output" in retry_events[0].error_message
        assert retry_events[0].resolved is False


# --- durable_task decorator tests ---


class TestDurableTask:
    def test_sync_succeeds(self):
        """durable_task decorator works for sync functions on first attempt."""
        _init_gate()

        @durable_task(retries=3, session_id="dt-sync-ok")
        def good_func() -> str:
            return "hello"

        result = good_func()
        assert result == "hello"

    def test_sync_retries_on_failure(self):
        """durable_task retries sync function on exception."""
        _init_gate()

        call_count = 0

        @durable_task(retries=3, session_id="dt-sync-retry")
        def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("not yet")
            return "ok"

        result = flaky_func()
        assert result == "ok"
        assert call_count == 2

    def test_sync_exhausted(self):
        """durable_task raises StateLoomRetryError when all attempts fail."""
        _init_gate()

        @durable_task(retries=2, session_id="dt-sync-exhaust")
        def always_fails() -> str:
            raise ValueError("always bad")

        with pytest.raises(StateLoomRetryError) as exc_info:
            always_fails()

        assert exc_info.value.attempts == 2
        assert "always bad" in exc_info.value.last_error

    def test_sync_with_validate(self):
        """durable_task retries when validate callback returns False."""
        _init_gate()

        call_count = 0

        @durable_task(
            retries=3,
            session_id="dt-sync-validate",
            validate=lambda x: x > 10,
        )
        def incrementing_func() -> int:
            nonlocal call_count
            call_count += 1
            return call_count * 10  # 10, 20, 30...

        result = incrementing_func()
        assert result == 20  # First attempt returns 10 (fails validation), second returns 20
        assert call_count == 2

    def test_sync_budget_error_not_retried(self):
        """Infrastructure errors propagate immediately from durable_task."""
        _init_gate()

        @durable_task(retries=3, session_id="dt-sync-budget")
        def budget_blowout() -> str:
            raise StateLoomBudgetError(limit=1.0, spent=2.0, session_id="s")

        with pytest.raises(StateLoomBudgetError):
            budget_blowout()

    def test_sync_on_retry_callback(self):
        """durable_task on_retry callback is invoked on each retry."""
        _init_gate()

        retries_seen: list[int] = []
        call_count = 0

        @durable_task(
            retries=3,
            session_id="dt-sync-callback",
            on_retry=lambda attempt, error: retries_seen.append(attempt),
        )
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("nope")
            return "done"

        result = flaky()
        assert result == "done"
        assert retries_seen == [1, 2]

    def test_auto_generated_session_id(self):
        """durable_task generates a session ID when none is provided."""
        _init_gate()

        @durable_task(retries=1, name="my_task")
        def simple() -> str:
            return "hi"

        result = simple()
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_async_succeeds(self):
        """durable_task decorator works for async functions."""
        _init_gate()

        @durable_task(retries=3, session_id="dt-async-ok")
        async def async_good() -> str:
            return "async hello"

        result = await async_good()
        assert result == "async hello"

    @pytest.mark.asyncio
    async def test_async_retries(self):
        """durable_task retries async function on exception."""
        _init_gate()

        call_count = 0

        @durable_task(retries=3, session_id="dt-async-retry")
        async def async_flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("async fail")
            return "async ok"

        result = await async_flaky()
        assert result == "async ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_exhausted(self):
        """durable_task raises StateLoomRetryError when async attempts exhausted."""
        _init_gate()

        @durable_task(retries=2, session_id="dt-async-exhaust")
        async def async_always_fails() -> str:
            raise ValueError("always async bad")

        with pytest.raises(StateLoomRetryError) as exc_info:
            await async_always_fails()

        assert exc_info.value.attempts == 2


# --- SemanticRetryEvent serialization ---


class TestSemanticRetryEventSerialization:
    def test_event_type(self):
        """SemanticRetryEvent has correct event type."""
        event = SemanticRetryEvent(
            session_id="s1",
            attempt=2,
            max_attempts=3,
            error_type="ValueError",
            error_message="bad json",
        )
        assert event.event_type == EventType.SEMANTIC_RETRY

    def test_sqlite_round_trip(self, tmp_path):
        """SemanticRetryEvent survives SQLite serialization round-trip."""
        from stateloom.store.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))

        # Save a session first (events need a valid session)
        from stateloom.core.session import Session

        session = Session(id="sr-test")
        store.save_session(session)

        event = SemanticRetryEvent(
            session_id="sr-test",
            step=5,
            attempt=2,
            max_attempts=3,
            error_type="ValueError",
            error_message="invalid JSON at line 1",
            provider="openai",
            model="gpt-4",
            resolved=False,
        )
        store.save_event(event)

        events = store.get_session_events("sr-test")
        retry_events = [e for e in events if isinstance(e, SemanticRetryEvent)]
        assert len(retry_events) == 1

        loaded = retry_events[0]
        assert loaded.event_type == EventType.SEMANTIC_RETRY
        assert loaded.attempt == 2
        assert loaded.max_attempts == 3
        assert loaded.error_type == "ValueError"
        assert loaded.error_message == "invalid JSON at line 1"
        assert loaded.provider == "openai"
        assert loaded.model == "gpt-4"
        assert loaded.resolved is False

    def test_sqlite_round_trip_resolved(self, tmp_path):
        """Resolved SemanticRetryEvent serializes correctly."""
        from stateloom.store.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))

        from stateloom.core.session import Session

        session = Session(id="sr-resolved")
        store.save_session(session)

        event = SemanticRetryEvent(
            session_id="sr-resolved",
            step=10,
            attempt=3,
            max_attempts=3,
            resolved=True,
        )
        store.save_event(event)

        events = store.get_session_events("sr-resolved")
        retry_events = [e for e in events if isinstance(e, SemanticRetryEvent)]
        assert len(retry_events) == 1
        assert retry_events[0].resolved is True
        assert retry_events[0].attempt == 3

    def test_memory_store_round_trip(self):
        """SemanticRetryEvent works with MemoryStore."""
        store = MemoryStore()

        from stateloom.core.session import Session

        session = Session(id="ms-test")
        store.save_session(session)

        event = SemanticRetryEvent(
            session_id="ms-test",
            attempt=1,
            max_attempts=3,
            error_type="JSONDecodeError",
            error_message="Expecting value",
        )
        store.save_event(event)

        events = store.get_session_events("ms-test")
        retry_events = [e for e in events if isinstance(e, SemanticRetryEvent)]
        assert len(retry_events) == 1
        assert retry_events[0].error_type == "JSONDecodeError"


# --- Error class tests ---


class TestStateLoomRetryError:
    def test_attributes(self):
        err = StateLoomRetryError(attempts=3, last_error="bad json", session_id="s1")
        assert err.attempts == 3
        assert err.last_error == "bad json"
        assert err.session_id == "s1"
        assert err.error_code == "RETRY_EXHAUSTED"

    def test_message(self):
        err = StateLoomRetryError(attempts=5, last_error="timeout")
        assert "5 retry attempts exhausted" in str(err)
        assert "timeout" in str(err)

    def test_is_stateloom_error(self):
        from stateloom.core.errors import StateLoomError

        err = StateLoomRetryError(attempts=1, last_error="x")
        assert isinstance(err, StateLoomError)


# --- Public API surface tests ---


class TestPublicAPI:
    def test_retry_loop_in_module(self):
        """retry_loop is accessible from stateloom module."""
        loop = stateloom.retry_loop(retries=2)
        assert loop._max_attempts == 2

    def test_durable_task_in_module(self):
        """durable_task is accessible from stateloom module."""
        decorator = stateloom.durable_task(retries=2)
        assert callable(decorator)

    def test_retry_error_in_module(self):
        """StateLoomRetryError is accessible from stateloom module."""
        assert stateloom.StateLoomRetryError is StateLoomRetryError

    def test_in_all(self):
        """New exports are in __all__."""
        assert "durable_task" in stateloom.__all__
        assert "retry_loop" in stateloom.__all__
        assert "StateLoomRetryError" in stateloom.__all__
