"""Tests for the evaluator-aware refinement loop (``stateloom.refine``)."""

from __future__ import annotations

import math

import pytest

import stateloom
from stateloom.core.errors import StateLoomBudgetError
from stateloom.core.event import RefineAttemptEvent, SemanticRetryEvent
from stateloom.core.types import EventType
from stateloom.refine import (
    RefineNode,
    RefineResult,
    ScoreResult,
    durable_refine,
    format_history_as_messages,
    refine_loop,
)


def _init_gate() -> stateloom.Gate:
    return stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )


# ---------------------------------------------------------------------------
# Plain-float scorer, threshold met on first attempt
# ---------------------------------------------------------------------------


class TestBasicRefineLoop:
    def test_plain_float_threshold_met_attempt_one(self):
        """Scorer returns a plain float that already meets the threshold."""
        call_count = 0

        def gen(history: list[RefineNode]) -> str:
            nonlocal call_count
            call_count += 1
            return "perfect"

        def score(candidate: str) -> float:
            return 1.0

        result = refine_loop(gen, score, max_attempts=5, threshold=1.0)

        assert isinstance(result, RefineResult)
        assert call_count == 1
        assert result.threshold_met is True
        assert result.attempts_used == 1
        assert result.best.result == "perfect"
        assert result.best.score == 1.0
        assert len(result.history) == 1

    def test_score_result_feedback_surfaces_in_history(self):
        """ScoreResult feedback reaches the next generator's history."""
        captured_histories: list[list[RefineNode]] = []

        def gen(history: list[RefineNode]) -> str:
            captured_histories.append(list(history))
            return "attempt-" + str(len(history) + 1)

        def score(candidate: str) -> ScoreResult:
            # Only the second attempt meets the bar.
            if candidate == "attempt-2":
                return ScoreResult(score=1.0, feedback="great!")
            return ScoreResult(
                score=0.0,
                feedback="missing field 'name'",
                metadata={"missing": ["name"]},
            )

        result = refine_loop(gen, score, max_attempts=3, threshold=1.0)

        # Attempt 1 saw empty history; attempt 2 saw attempt 1's feedback.
        assert len(captured_histories) == 2
        assert captured_histories[0] == []
        assert len(captured_histories[1]) == 1
        assert captured_histories[1][0].feedback == "missing field 'name'"
        assert captured_histories[1][0].metadata == {"missing": ["name"]}
        assert result.threshold_met is True
        assert result.attempts_used == 2

    def test_threshold_never_met_returns_best_of_n(self):
        """When threshold is never met, return the best attempt overall."""
        scores = [0.3, 0.7, 0.5]

        def gen(history: list[RefineNode]) -> int:
            return len(history) + 1

        def score(candidate: int) -> float:
            return scores[candidate - 1]

        result = refine_loop(gen, score, max_attempts=3, threshold=0.9)

        assert result.threshold_met is False
        assert result.attempts_used == 3
        assert result.best.attempt == 2
        assert result.best.score == 0.7
        assert [n.score for n in result.history] == [0.3, 0.7, 0.5]

    def test_direction_min(self):
        """direction='min' treats lower scores as better and uses <= threshold."""
        latencies = [500.0, 200.0, 50.0]

        def gen(history: list[RefineNode]) -> int:
            return len(history) + 1

        def score(candidate: int) -> float:
            return latencies[candidate - 1]

        result = refine_loop(gen, score, max_attempts=5, threshold=100.0, direction="min")

        assert result.threshold_met is True
        assert result.attempts_used == 3
        assert result.best.score == 50.0

    def test_direction_min_picks_lowest_when_threshold_missed(self):
        """With direction='min' and no threshold met, best = lowest."""
        latencies = [500.0, 200.0, 300.0]

        def gen(history: list[RefineNode]) -> int:
            return len(history) + 1

        def score(candidate: int) -> float:
            return latencies[candidate - 1]

        result = refine_loop(gen, score, max_attempts=3, direction="min")

        assert result.threshold_met is False
        assert result.best.score == 200.0
        assert result.best.attempt == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_scorer_exception_records_sentinel_and_continues(self):
        """Generic scorer exception → sentinel score recorded, loop continues."""

        def gen(history: list[RefineNode]) -> int:
            return len(history) + 1

        def score(candidate: int) -> float:
            if candidate == 1:
                raise RuntimeError("scorer crashed")
            return 0.8

        result = refine_loop(gen, score, max_attempts=3, threshold=0.7)

        assert result.threshold_met is True
        assert result.attempts_used == 2
        assert math.isinf(result.history[0].score) and result.history[0].score < 0
        assert result.history[1].score == 0.8
        assert result.best.attempt == 2
        assert "scorer raised" in result.history[0].feedback

    def test_scorer_exception_sentinel_direction_min(self):
        """Generic scorer exception with direction='min' → +inf sentinel."""

        def gen(history: list[RefineNode]) -> int:
            return len(history) + 1

        def score(candidate: int) -> float:
            if candidate == 1:
                raise RuntimeError("bad")
            return 50.0

        result = refine_loop(gen, score, max_attempts=2, direction="min")

        assert result.history[0].score == math.inf
        assert result.best.attempt == 2

    def test_non_retryable_error_propagates_immediately(self):
        """StateLoomBudgetError from scorer propagates without being caught."""
        calls = {"gen": 0, "score": 0}

        def gen(history: list[RefineNode]) -> str:
            calls["gen"] += 1
            return "x"

        def score(candidate: str) -> float:
            calls["score"] += 1
            raise StateLoomBudgetError(limit=1.0, spent=2.0, session_id="s")

        with pytest.raises(StateLoomBudgetError):
            refine_loop(gen, score, max_attempts=5)

        assert calls["gen"] == 1
        assert calls["score"] == 1

    def test_non_retryable_from_generator_propagates(self):
        """StateLoomBudgetError raised inside the generator also propagates."""
        calls = {"gen": 0}

        def gen(history: list[RefineNode]) -> str:
            calls["gen"] += 1
            raise StateLoomBudgetError(limit=1.0, spent=2.0, session_id="s")

        with pytest.raises(StateLoomBudgetError):
            refine_loop(gen, lambda c: 1.0, max_attempts=3)

        assert calls["gen"] == 1

    def test_max_attempts_one(self):
        """max_attempts=1 → exactly one generator+scorer call."""
        calls = {"gen": 0}

        def gen(history: list[RefineNode]) -> int:
            calls["gen"] += 1
            return 0

        result = refine_loop(gen, lambda c: 0.0, max_attempts=1)

        assert calls["gen"] == 1
        assert result.attempts_used == 1

    def test_invalid_max_attempts_raises(self):
        with pytest.raises(ValueError):
            refine_loop(lambda h: None, lambda c: 0.0, max_attempts=0)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            refine_loop(
                lambda h: None,
                lambda c: 0.0,
                max_attempts=1,
                direction="weird",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Event persistence
# ---------------------------------------------------------------------------


class TestEventPersistence:
    def test_records_event_per_attempt(self):
        """One RefineAttemptEvent is persisted per attempt with correct fields."""
        gate = _init_gate()

        scores = [0.2, 0.6, 0.4]

        def gen(history: list[RefineNode]) -> int:
            return len(history) + 1

        def score(candidate: int) -> ScoreResult:
            return ScoreResult(
                score=scores[candidate - 1],
                feedback=f"fb-{candidate}",
            )

        with stateloom.session("refine-event-session"):
            result = refine_loop(gen, score, max_attempts=3, threshold=0.9)

        events = gate.store.get_session_events("refine-event-session")
        refine_events = [e for e in events if isinstance(e, RefineAttemptEvent)]

        assert len(refine_events) == 3
        assert [e.attempt for e in refine_events] == [1, 2, 3]
        assert [e.score for e in refine_events] == [0.2, 0.6, 0.4]
        assert [e.best_score_so_far for e in refine_events] == [0.2, 0.6, 0.6]
        assert [e.feedback_summary for e in refine_events] == ["fb-1", "fb-2", "fb-3"]
        # threshold_met is False throughout since 0.9 was never crossed
        assert all(e.threshold_met is False for e in refine_events)
        assert all(e.max_attempts == 3 for e in refine_events)
        assert all(e.direction == "max" for e in refine_events)
        assert all(e.threshold == 0.9 for e in refine_events)
        assert result.threshold_met is False

    def test_event_type_enum_present(self):
        """REFINE_ATTEMPT value exists on the EventType enum."""
        assert EventType.REFINE_ATTEMPT.value == "refine_attempt"

    def test_feedback_truncated_to_500(self):
        """Large feedback strings are truncated in feedback_summary."""
        gate = _init_gate()
        big = "x" * 2000

        def gen(history: list[RefineNode]) -> str:
            return "y"

        def score(candidate: str) -> ScoreResult:
            return ScoreResult(score=1.0, feedback=big)

        with stateloom.session("refine-trunc"):
            refine_loop(gen, score, max_attempts=1, threshold=1.0)

        events = gate.store.get_session_events("refine-trunc")
        refine_events = [e for e in events if isinstance(e, RefineAttemptEvent)]
        assert len(refine_events) == 1
        assert len(refine_events[0].feedback_summary) == 500

    def test_does_not_record_when_outside_session(self):
        """refine_loop outside a session works without raising."""
        _init_gate()

        def gen(history: list[RefineNode]) -> int:
            return 1

        # Should simply return a RefineResult; no event persistence.
        result = refine_loop(gen, lambda c: 1.0, max_attempts=1, threshold=1.0)
        assert result.threshold_met is True

    def test_does_not_record_semantic_retry_events(self):
        """refine_loop must not create SemanticRetryEvents (isolation check)."""
        gate = _init_gate()

        def gen(history: list[RefineNode]) -> int:
            return 1

        with stateloom.session("refine-isolation"):
            refine_loop(gen, lambda c: 0.5, max_attempts=2)

        events = gate.store.get_session_events("refine-isolation")
        assert not any(isinstance(e, SemanticRetryEvent) for e in events)


# ---------------------------------------------------------------------------
# durable_refine decorator
# ---------------------------------------------------------------------------


class TestDurableRefine:
    def test_sync_decorator(self):
        """durable_refine wraps a sync function and returns a RefineResult."""
        _init_gate()

        @durable_refine(
            scorer=lambda c: 1.0 if c == "good" else 0.0,
            max_attempts=3,
            threshold=1.0,
            session_id="dr-sync-1",
        )
        def generate(*, history: list[RefineNode]) -> str:
            return "bad" if len(history) == 0 else "good"

        result = generate()
        assert isinstance(result, RefineResult)
        assert result.threshold_met is True
        assert result.attempts_used == 2
        assert result.best.result == "good"

    def test_sync_decorator_passes_through_args(self):
        """Positional/keyword args flow through to the wrapped function."""
        _init_gate()

        @durable_refine(
            scorer=lambda c: 1.0,
            max_attempts=1,
            threshold=1.0,
            session_id="dr-sync-args",
        )
        def generate(prefix: str, *, suffix: str = "!", history: list[RefineNode]) -> str:
            return prefix + suffix

        result = generate("hi", suffix="?")
        assert result.best.result == "hi?"

    def test_sync_decorator_records_events(self):
        """durable_refine persists RefineAttemptEvents to its durable session."""
        gate = _init_gate()

        @durable_refine(
            scorer=lambda c: len(c) / 10.0,
            max_attempts=2,
            session_id="dr-sync-events",
        )
        def generate(*, history: list[RefineNode]) -> str:
            return "x" * (len(history) + 1)

        generate()

        events = gate.store.get_session_events("dr-sync-events")
        refine_events = [e for e in events if isinstance(e, RefineAttemptEvent)]
        assert len(refine_events) == 2

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """durable_refine supports async functions via inspect.iscoroutinefunction."""
        _init_gate()

        @durable_refine(
            scorer=lambda c: 1.0 if c.endswith("-ok") else 0.0,
            max_attempts=3,
            threshold=1.0,
            session_id="dr-async-1",
        )
        async def generate(*, history: list[RefineNode]) -> str:
            return "try-" + ("ok" if len(history) >= 1 else "no")

        result = await generate()
        assert isinstance(result, RefineResult)
        assert result.threshold_met is True
        assert result.best.result == "try-ok"

    @pytest.mark.asyncio
    async def test_async_decorator_records_events(self):
        """Async variant also persists RefineAttemptEvents."""
        gate = _init_gate()

        @durable_refine(
            scorer=lambda c: 0.5,
            max_attempts=2,
            session_id="dr-async-events",
        )
        async def generate(*, history: list[RefineNode]) -> int:
            return 42

        await generate()

        events = gate.store.get_session_events("dr-async-events")
        refine_events = [e for e in events if isinstance(e, RefineAttemptEvent)]
        assert len(refine_events) == 2


# ---------------------------------------------------------------------------
# Durable integration: cached LLM responses replay on resume
# ---------------------------------------------------------------------------


class TestDurableIntegration:
    def test_resume_replays_cached_llm_for_prior_attempts(self):
        """Durable session inside refine_loop: resume replays LLM responses.

        Simulates a crash mid-refine after attempt 2 by raising.  On resume
        with the same session_id, the first two attempts' cached LLM
        responses are returned from the replay engine — no new LLM calls
        should fire for them; only attempt 3 executes live.
        """
        try:
            from openai.types.chat import ChatCompletion  # noqa: F401
        except ImportError:
            pytest.skip("openai SDK not installed")

        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
            auto_route_semantic_enabled=False,
        )

        from stateloom.core.context import get_current_replay_engine
        from tests.test_e2e.helpers import make_openai_response

        sid = "refine-durable-resume"
        live_calls = {"count": 0}
        boom_sentinel = RuntimeError("simulated crash before attempt 3")

        def make_llm_call(reply: str):
            def _call():
                live_calls["count"] += 1
                return make_openai_response(reply)

            return _call

        def build_generator(sess, crash_at_attempt: int | None = None):
            def generate(history: list[RefineNode]) -> str:
                attempt = len(history) + 1
                if crash_at_attempt is not None and attempt == crash_at_attempt:
                    raise boom_sentinel
                engine = get_current_replay_engine()
                step = sess.next_step()
                if engine is not None and engine.is_active and engine.should_mock(step):
                    return engine.get_cached_response(step)
                reply = f"attempt-{attempt}"
                return gate.pipeline.execute_sync(
                    provider="openai",
                    method="chat.completions.create",
                    model="gpt-3.5-turbo",
                    request_kwargs={"messages": [{"role": "user", "content": reply}]},
                    session=sess,
                    config=gate.config,
                    llm_call=make_llm_call(reply),
                    auto_route_eligible=False,
                )

            return generate

        def scorer(candidate) -> float:
            return 0.0  # never meets threshold → all attempts run

        # --- First run: attempts 1 & 2 hit the pipeline; attempt 3 crashes
        # before the LLM call fires, so only 2 live calls total. ---
        with pytest.raises(RuntimeError) as exc_info:
            with gate.session(session_id=sid, durable=True) as sess:
                gen1 = build_generator(sess, crash_at_attempt=3)
                refine_loop(gen1, scorer, max_attempts=3)
        assert exc_info.value is boom_sentinel
        assert live_calls["count"] == 2

        # --- Second run: resume — attempts 1 & 2 replay from cache,
        # attempt 3 executes live (no crash this time). ---
        live_calls["count"] = 0

        with gate.session(session_id=sid, durable=True) as sess:
            gen2 = build_generator(sess, crash_at_attempt=None)
            result = refine_loop(gen2, scorer, max_attempts=3)

        assert live_calls["count"] == 1
        assert result.attempts_used == 3
        assert len(result.history) == 3


# ---------------------------------------------------------------------------
# Helper: format_history_as_messages
# ---------------------------------------------------------------------------


class TestFormatHistoryAsMessages:
    def _history(self) -> list[RefineNode]:
        return [
            RefineNode(attempt=1, result="first try", score=0.3, feedback="add units"),
            RefineNode(attempt=2, result="final", score=1.0, feedback=""),
        ]

    def test_openai_style(self):
        msgs = format_history_as_messages(self._history(), style="openai")
        assert len(msgs) == 4  # 2 attempts * (assistant + user)
        assert msgs[0] == {"role": "assistant", "content": "first try"}
        assert msgs[1]["role"] == "user"
        assert "0.300" in msgs[1]["content"]
        assert "add units" in msgs[1]["content"]
        assert msgs[2] == {"role": "assistant", "content": "final"}
        assert "1.000" in msgs[3]["content"]

    def test_anthropic_style(self):
        msgs = format_history_as_messages(self._history(), style="anthropic")
        assert len(msgs) == 4
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == [{"type": "text", "text": "first try"}]
        assert msgs[1]["role"] == "user"
        assert isinstance(msgs[1]["content"], list)
        assert msgs[1]["content"][0]["type"] == "text"


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_refine_loop_in_module(self):
        assert callable(stateloom.refine_loop)

    def test_durable_refine_in_module(self):
        assert callable(stateloom.durable_refine)

    def test_format_history_in_module(self):
        assert callable(stateloom.format_history_as_messages)

    def test_dataclasses_in_module(self):
        assert stateloom.ScoreResult is ScoreResult
        assert stateloom.RefineNode is RefineNode
        assert stateloom.RefineResult is RefineResult

    def test_in_all(self):
        for name in (
            "refine_loop",
            "durable_refine",
            "ScoreResult",
            "RefineNode",
            "RefineResult",
            "format_history_as_messages",
        ):
            assert name in stateloom.__all__, f"{name} missing from __all__"
