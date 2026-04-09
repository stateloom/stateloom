"""Tests for durable resumption hardening — hash mismatch, concurrency, and buffer mode."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.errors import StateLoomDurableReplayError, StateLoomError
from stateloom.core.session import Session
from stateloom.replay.engine import DurableReplayEngine, _load_durable_steps
from stateloom.replay.step import StepRecord

# ---------------------------------------------------------------------------
# Fix 1: Hash mismatch detection
# ---------------------------------------------------------------------------


class TestDurableReplayError:
    """StateLoomDurableReplayError is properly constructed and classified."""

    def test_error_fields(self):
        err = StateLoomDurableReplayError(
            session_id="sess-1",
            step=5,
            expected_hash="abcdef1234567890",
            actual_hash="1234567890abcdef",
        )
        assert err.step == 5
        assert err.expected_hash == "abcdef1234567890"
        assert err.actual_hash == "1234567890abcdef"
        assert "step 5" in str(err)
        assert "abcdef12" in str(err)
        assert "12345678" in str(err)

    def test_inherits_replay_error(self):
        err = StateLoomDurableReplayError("s", 1, "aaa", "bbb")
        from stateloom.core.errors import StateLoomReplayError

        assert isinstance(err, StateLoomReplayError)

    def test_non_retryable(self):
        from stateloom.retry import _NON_RETRYABLE

        assert StateLoomDurableReplayError in _NON_RETRYABLE


class TestLoadDurableStepsRequestHash:
    """_load_durable_steps populates request_hash from events."""

    def test_request_hash_populated(self):
        """Events with request_hash should propagate to StepRecord."""
        mock_event = MagicMock()
        mock_event.step = 1
        mock_event.event_type = "llm_call"
        mock_event.provider = "openai"
        mock_event.model = "gpt-4o"
        mock_event.cached_response_json = '{"id": "resp-1"}'
        mock_event.prompt_preview = "Hello"
        mock_event.request_hash = "abc123def456"

        mock_gate = MagicMock()
        mock_gate.store.get_session_events.return_value = [mock_event]

        steps = _load_durable_steps(mock_gate, "test-session")
        assert len(steps) == 1
        assert steps[0].request_hash == "abc123def456"

    def test_request_hash_empty_when_missing(self):
        """Events without request_hash should default to empty string."""
        mock_event = MagicMock(spec=[])
        mock_event.step = 1
        mock_event.event_type = "llm_call"
        mock_event.cached_response_json = '{"id": "resp-1"}'

        # Remove request_hash attribute to test getattr fallback
        mock_gate = MagicMock()
        mock_gate.store.get_session_events.return_value = [mock_event]

        steps = _load_durable_steps(mock_gate, "test-session")
        assert len(steps) == 1
        assert steps[0].request_hash == ""

    def test_request_hash_none_becomes_empty(self):
        """Events with request_hash=None should become empty string."""
        mock_event = MagicMock()
        mock_event.step = 1
        mock_event.event_type = "llm_call"
        mock_event.provider = "openai"
        mock_event.model = "gpt-4o"
        mock_event.cached_response_json = '{"id": "resp-1"}'
        mock_event.prompt_preview = "Hello"
        mock_event.request_hash = None

        mock_gate = MagicMock()
        mock_gate.store.get_session_events.return_value = [mock_event]

        steps = _load_durable_steps(mock_gate, "test-session")
        assert steps[0].request_hash == ""


class TestCheckReplayHashValidation:
    """_check_replay validates request_hash for durable replay."""

    def test_matching_hash_returns_cached(self):
        """When hashes match, return the cached response normally."""
        from stateloom.intercept.generic_interceptor import _check_replay

        step_record = StepRecord(
            step=1,
            event_type="llm_call",
            request_hash="abc123",
            cached_response_json='{"id": "resp-1"}',
        )
        engine = MagicMock()
        engine.is_active = True
        engine.should_mock.return_value = True
        engine.get_cached_response.return_value = {"id": "resp-1"}
        engine._step_index = {1: step_record}

        session = MagicMock()
        session.id = "test-sess"

        with patch("stateloom.intercept.generic_interceptor.get_current_replay_engine") as mock_get:
            mock_get.return_value = engine
            result = _check_replay(MagicMock(), 1, session, request_hash="abc123")

        assert result == {"id": "resp-1"}

    def test_mismatched_hash_raises(self):
        """When hashes differ, raise StateLoomDurableReplayError."""
        from stateloom.intercept.generic_interceptor import _check_replay

        step_record = StepRecord(
            step=1,
            event_type="llm_call",
            request_hash="cached_hash_aaa",
            cached_response_json='{"id": "resp-1"}',
        )
        engine = MagicMock()
        engine.is_active = True
        engine.should_mock.return_value = True
        engine._step_index = {1: step_record}

        session = MagicMock()
        session.id = "test-sess"

        with patch("stateloom.intercept.generic_interceptor.get_current_replay_engine") as mock_get:
            mock_get.return_value = engine
            with pytest.raises(StateLoomDurableReplayError) as exc_info:
                _check_replay(MagicMock(), 1, session, request_hash="different_hash")

        assert exc_info.value.step == 1
        assert exc_info.value.expected_hash == "cached_hash_aaa"
        assert exc_info.value.actual_hash == "different_hash"

    def test_empty_cached_hash_skips_validation(self):
        """When cached step has no hash (old session), skip validation."""
        from stateloom.intercept.generic_interceptor import _check_replay

        step_record = StepRecord(
            step=1,
            event_type="llm_call",
            request_hash="",  # old session, no hash stored
            cached_response_json='{"id": "resp-1"}',
        )
        engine = MagicMock()
        engine.is_active = True
        engine.should_mock.return_value = True
        engine.get_cached_response.return_value = {"id": "resp-1"}
        engine._step_index = {1: step_record}

        session = MagicMock()
        session.id = "test-sess"

        with patch("stateloom.intercept.generic_interceptor.get_current_replay_engine") as mock_get:
            mock_get.return_value = engine
            # Should NOT raise even though request_hash differs
            result = _check_replay(MagicMock(), 1, session, request_hash="some_hash")

        assert result == {"id": "resp-1"}

    def test_empty_current_hash_skips_validation(self):
        """When current request hash is empty, skip validation (backward compat)."""
        from stateloom.intercept.generic_interceptor import _check_replay

        step_record = StepRecord(
            step=1,
            event_type="llm_call",
            request_hash="cached_hash",
            cached_response_json='{"id": "resp-1"}',
        )
        engine = MagicMock()
        engine.is_active = True
        engine.should_mock.return_value = True
        engine.get_cached_response.return_value = {"id": "resp-1"}
        engine._step_index = {1: step_record}

        session = MagicMock()
        session.id = "test-sess"

        with patch("stateloom.intercept.generic_interceptor.get_current_replay_engine") as mock_get:
            mock_get.return_value = engine
            # Empty hash = no validation
            result = _check_replay(MagicMock(), 1, session, request_hash="")

        assert result == {"id": "resp-1"}


# ---------------------------------------------------------------------------
# Fix 2: Concurrency guard
# ---------------------------------------------------------------------------


class TestDurableConcurrencyGuard:
    """Session.acquire_durable_step() prevents concurrent calls in durable sessions."""

    def test_sequential_calls_work(self):
        """Sequential acquire/release should work fine."""
        session = Session(id="test-1", durable=True)

        step1 = session.acquire_durable_step()
        assert step1 == 1
        session.release_durable_step()

        step2 = session.acquire_durable_step()
        assert step2 == 2
        session.release_durable_step()

    def test_concurrent_calls_raise(self):
        """Second acquire while first is in-flight should raise."""
        session = Session(id="test-1", durable=True)

        step1 = session.acquire_durable_step()
        assert step1 == 1
        # Don't release — simulate in-flight call

        with pytest.raises(StateLoomError, match="Concurrent LLM calls"):
            session.acquire_durable_step()

        # Clean up
        session.release_durable_step()

    def test_non_durable_allows_concurrent(self):
        """Non-durable sessions should allow concurrent calls."""
        session = Session(id="test-1", durable=False)

        step1 = session.acquire_durable_step()
        assert step1 == 1
        # Should NOT raise for non-durable sessions
        step2 = session.acquire_durable_step()
        assert step2 == 2

    def test_release_after_error(self):
        """Release should work correctly even after errors."""
        session = Session(id="test-1", durable=True)

        session.acquire_durable_step()
        session.release_durable_step()

        # Should not underflow
        session.release_durable_step()
        assert session._durable_in_flight == 0

    def test_next_step_unchanged(self):
        """next_step() should be unaffected by the durable guard."""
        session = Session(id="test-1", durable=True)

        step1 = session.next_step()
        assert step1 == 1
        step2 = session.next_step()
        assert step2 == 2
        # next_step doesn't touch _durable_in_flight
        assert session._durable_in_flight == 0

    def test_thread_safety(self):
        """acquire_durable_step should be thread-safe."""
        session = Session(id="test-1", durable=True)
        errors: list[Exception] = []
        steps: list[int] = []
        barrier = threading.Barrier(2)

        def worker():
            try:
                barrier.wait(timeout=2)
                step = session.acquire_durable_step()
                steps.append(step)
            except Exception as e:
                errors.append(e)

        session.acquire_durable_step()  # Hold in-flight

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)

        # Both should have errored since we held the in-flight lock
        assert len(errors) == 2
        for e in errors:
            assert "Concurrent LLM calls" in str(e)

        session.release_durable_step()


# ---------------------------------------------------------------------------
# Fix 3: Durable stream buffer config
# ---------------------------------------------------------------------------


class TestDurableStreamBufferConfig:
    """durable_stream_buffer config option exists and defaults correctly."""

    def test_default_false(self):
        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig()
        assert config.durable_stream_buffer is False

    def test_set_true(self):
        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(durable_stream_buffer=True)
        assert config.durable_stream_buffer is True


# ---------------------------------------------------------------------------
# Integration: DurableReplayEngine with request_hash
# ---------------------------------------------------------------------------


class TestDurableReplayEngineHashPropagation:
    """DurableReplayEngine preserves request_hash from loaded steps."""

    def test_step_index_preserves_hash(self):
        steps = [
            StepRecord(
                step=1,
                event_type="llm_call",
                request_hash="hash_a",
                cached_response_json='{"id": "r1"}',
            ),
            StepRecord(
                step=2,
                event_type="llm_call",
                request_hash="hash_b",
                cached_response_json='{"id": "r2"}',
            ),
        ]
        engine = DurableReplayEngine(steps)
        # Steps are re-indexed 1,2
        assert engine._step_index[1].request_hash == "hash_a"
        assert engine._step_index[2].request_hash == "hash_b"

    def test_empty_hash_preserved(self):
        steps = [
            StepRecord(
                step=1,
                event_type="llm_call",
                request_hash="",
                cached_response_json='{"id": "r1"}',
            ),
        ]
        engine = DurableReplayEngine(steps)
        assert engine._step_index[1].request_hash == ""
