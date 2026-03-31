"""Tests for ReplayEngine — ContextVar cleanup and event persistence."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import stateloom
from stateloom.core.context import get_current_replay_engine, get_current_session, set_current_session
from stateloom.core.event import CheckpointEvent, LLMCallEvent
from stateloom.replay.engine import DurableReplayEngine, ReplayEngine
from stateloom.replay.schema import serialize_response
from stateloom.replay.step import StepRecord


def _make_gate_with_events(session_id: str, num_events: int = 4):
    """Create a gate with a durable session that has cached LLM events."""
    gate = stateloom.init(
        budget=10.0,
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )

    # Create session and save events with cached responses
    session = gate.session_manager.create(session_id=session_id)
    session.step_counter = num_events
    gate.store.save_session(session)

    for i in range(1, num_events + 1):
        event = LLMCallEvent(
            session_id=session_id,
            step=i,
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
            latency_ms=100.0,
            cached_response_json=serialize_response(
                {"choices": [{"message": {"content": f"Response for step {i}"}}]}
            ),
            prompt_preview=f"Step {i} question",
        )
        gate.store.save_event(event)

    return gate


class TestReplayEngineContextVarCleanup:
    """Test that ReplayEngine properly saves/restores ContextVars."""

    def test_stop_restores_previous_session(self):
        """After stop(), the session ContextVar should be restored to pre-replay value."""
        gate = _make_gate_with_events("test-session", num_events=4)

        # No session active before replay
        set_current_session(None)
        assert get_current_session() is None

        engine = ReplayEngine(gate, session_id="test-session", mock_until_step=2, strict=False)
        engine.start()

        # During replay, session should be set to the replay session
        replay_session = get_current_session()
        assert replay_session is not None
        assert replay_session.id == "replay-test-session"

        engine.stop()

        # After stop, session should be restored to None
        assert get_current_session() is None
        assert get_current_replay_engine() is None

    def test_stop_restores_outer_session(self):
        """If a session was active before replay, it's restored after stop()."""
        gate = _make_gate_with_events("test-session", num_events=4)

        # Set up an outer session
        outer_session = gate.session_manager.create(session_id="outer-session")
        set_current_session(outer_session)

        engine = ReplayEngine(gate, session_id="test-session", mock_until_step=2, strict=False)
        engine.start()

        # During replay, a different session is active
        assert get_current_session().id == "replay-test-session"

        engine.stop()

        # After stop, outer session is restored
        restored = get_current_session()
        assert restored is not None
        assert restored.id == "outer-session"

        # Clean up
        set_current_session(None)

    def test_context_manager_restores_session(self):
        """Using `with engine:` also restores ContextVars."""
        gate = _make_gate_with_events("test-session", num_events=4)
        set_current_session(None)

        engine = ReplayEngine(gate, session_id="test-session", mock_until_step=2, strict=False)
        # engine.start() is called by gate.replay(), then __enter__ skips it
        engine.start()

        with engine:
            assert get_current_session().id == "replay-test-session"
            assert get_current_replay_engine() is engine

        assert get_current_session() is None
        assert get_current_replay_engine() is None

    def test_successive_replays_no_pollution(self):
        """Multiple replays in succession don't pollute each other's ContextVars."""
        gate = _make_gate_with_events("test-session", num_events=4)
        set_current_session(None)

        # First replay
        engine1 = ReplayEngine(gate, session_id="test-session", mock_until_step=2, strict=False)
        engine1.start()
        assert get_current_session().id == "replay-test-session"
        engine1.stop()
        assert get_current_session() is None

        # Second replay — should also start clean
        engine2 = ReplayEngine(gate, session_id="test-session", mock_until_step=3, strict=False)
        engine2.start()
        assert get_current_session().id == "replay-test-session"
        engine2.stop()
        assert get_current_session() is None


class TestReplayEngineEventPersistence:
    """Test that original events survive replay."""

    def test_original_events_persist_after_replay(self):
        """Events for the original session must not be deleted by replay."""
        gate = _make_gate_with_events("test-session", num_events=4)

        # Verify events exist before replay
        events_before = gate.store.get_session_events("test-session")
        assert len(events_before) == 4

        # Run replay
        engine = ReplayEngine(gate, session_id="test-session", mock_until_step=2, strict=False)
        engine.start()
        engine.stop()

        # Verify events still exist after replay
        events_after = gate.store.get_session_events("test-session")
        assert len(events_after) == 4
        assert {e.id for e in events_before} == {e.id for e in events_after}

    def test_replay_session_is_persisted(self):
        """The replay session should be saved to the store."""
        gate = _make_gate_with_events("test-session", num_events=4)

        engine = ReplayEngine(gate, session_id="test-session", mock_until_step=2, strict=False)
        engine.start()

        # Replay session should exist in the store
        replay_session = gate.store.get_session("replay-test-session")
        assert replay_session is not None
        assert replay_session.name == "Replay of test-session"

        engine.stop()

    def test_replay_session_completed_on_stop(self):
        """After stop(), the replay session status should be COMPLETED."""
        gate = _make_gate_with_events("test-session", num_events=4)

        engine = ReplayEngine(gate, session_id="test-session", mock_until_step=2, strict=False)
        engine.start()

        # During replay, session is active
        replay_session = gate.store.get_session("replay-test-session")
        assert replay_session.status == "active"

        engine.stop()

        # After stop, session should be completed and persisted
        replay_session = gate.store.get_session("replay-test-session")
        assert replay_session.status == "completed"
        assert replay_session.ended_at is not None

    def test_idempotent_enter(self):
        """__enter__ should not call start() again if already active."""
        gate = _make_gate_with_events("test-session", num_events=4)
        set_current_session(None)

        engine = ReplayEngine(gate, session_id="test-session", mock_until_step=2, strict=False)
        engine.start()

        # Calling __enter__ when already active should be a no-op
        result = engine.__enter__()
        assert result is engine
        assert engine._active is True

        engine.stop()
        assert get_current_session() is None


class TestDurableReplayEngineStepRenumbering:
    """Test that DurableReplayEngine re-numbers steps sequentially."""

    def test_renumbers_with_checkpoint_gaps(self):
        """When checkpoints create gaps (steps 2,3,4,5,6,7), durable maps to 1,2,3,4,5,6."""
        # Simulate: step 1 = checkpoint, steps 2-7 = LLM, step 8 = checkpoint
        steps = [
            StepRecord(step=2, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q1"),
            StepRecord(step=3, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q2"),
            StepRecord(step=4, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q3"),
            StepRecord(step=5, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q4"),
            StepRecord(step=6, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q5"),
            StepRecord(step=7, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q6"),
        ]

        engine = DurableReplayEngine(steps)

        # After re-numbering, steps 1-6 should all be mocked
        for i in range(1, 7):
            assert engine.should_mock(i), f"Step {i} should be mocked"

        # Step 7 and beyond should NOT be mocked
        assert not engine.should_mock(7)
        assert not engine.should_mock(8)

    def test_renumbers_contiguous_steps(self):
        """Steps already numbered 1,2,3 stay as 1,2,3."""
        steps = [
            StepRecord(step=1, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q1"),
            StepRecord(step=2, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q2"),
            StepRecord(step=3, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"choices":[]}', prompt_preview="q3"),
        ]

        engine = DurableReplayEngine(steps)

        for i in range(1, 4):
            assert engine.should_mock(i)
        assert not engine.should_mock(4)

    def test_empty_steps(self):
        """Engine with no steps should not be active."""
        engine = DurableReplayEngine([])
        assert not engine.is_active
        assert not engine.should_mock(1)

    def test_preserves_order_by_original_step(self):
        """Re-numbered steps preserve the original step ordering."""
        steps = [
            StepRecord(step=5, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"r":"fifth"}', prompt_preview="q5"),
            StepRecord(step=3, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"r":"third"}', prompt_preview="q3"),
            StepRecord(step=10, event_type="llm_call", provider="openai", model="gpt-4o",
                       cached_response_json='{"r":"tenth"}', prompt_preview="q10"),
        ]

        engine = DurableReplayEngine(steps)

        # Sorted by original step: 3, 5, 10 → re-numbered as 1, 2, 3
        assert engine.should_mock(1)  # was step 3
        assert engine.should_mock(2)  # was step 5
        assert engine.should_mock(3)  # was step 10
        assert not engine.should_mock(4)

        # Verify the content is in the right order
        r1 = engine._step_index[1]
        r2 = engine._step_index[2]
        r3 = engine._step_index[3]
        assert r1.prompt_preview == "q3"   # original step 3
        assert r2.prompt_preview == "q5"   # original step 5
        assert r3.prompt_preview == "q10"  # original step 10
