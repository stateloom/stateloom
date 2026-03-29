"""Tests for the backtest runner."""

from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.session import Session
from stateloom.core.types import SessionStatus
from stateloom.experiment.backtest import BacktestResult, BacktestRunner
from stateloom.experiment.models import VariantConfig
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def gate(store):
    """Create a minimal mock gate for backtest testing."""
    from stateloom.core.session import SessionManager

    gate = MagicMock()
    gate.store = store
    gate.session_manager = SessionManager()
    return gate


@pytest.fixture
def runner(gate):
    return BacktestRunner(gate)


def noop_agent(session):
    """A no-op agent function for testing the runner machinery."""
    pass


class TestBacktestResult:
    def test_to_dict(self):
        result = BacktestResult(
            source_session_id="s1",
            variant_name="fast",
            replay_session_id="backtest-s1-fast",
            total_cost=0.05,
            total_tokens=500,
            call_count=3,
            status="completed",
        )
        d = result.to_dict()
        assert d["source_session_id"] == "s1"
        assert d["variant_name"] == "fast"
        assert d["total_cost"] == 0.05
        assert d["total_tokens"] == 500


class TestBacktestRunner:
    def test_skips_missing_session(self, runner, store):
        results = runner.run_backtest(
            source_session_ids=["nonexistent"],
            variants=[{"name": "v1"}],
            agent_fn=noop_agent,
        )
        assert results == []

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_creates_replay_session_with_variant_metadata(self, mock_engine_cls, runner, store):
        source = Session(id="s1", step_counter=5)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        results = runner.run_backtest(
            source_session_ids=["s1"],
            variants=[VariantConfig(name="fast", model="gpt-4o-mini")],
            agent_fn=noop_agent,
        )

        assert len(results) == 1
        result = results[0]
        assert result.source_session_id == "s1"
        assert result.variant_name == "fast"
        assert result.replay_session_id == "backtest-s1-fast"

        # Verify replay session was created with correct metadata
        replay_session = store.get_session("backtest-s1-fast")
        assert replay_session is not None
        assert replay_session.metadata["experiment_variant_config"]["model"] == "gpt-4o-mini"

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_agent_fn_is_called_with_session(self, mock_engine_cls, runner, store):
        """Verify agent_fn is actually invoked and receives the session."""
        source = Session(id="s1", step_counter=3)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        agent_fn = MagicMock(return_value="agent output")

        results = runner.run_backtest(
            source_session_ids=["s1"],
            variants=[{"name": "v1"}],
            agent_fn=agent_fn,
        )

        agent_fn.assert_called_once()
        # The argument should be a Session
        call_args = agent_fn.call_args
        assert isinstance(call_args[0][0], Session)
        assert call_args[0][0].id == "backtest-s1-v1"

        # Output should be captured
        assert results[0].output == "agent output"

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_agent_fn_called_per_variant(self, mock_engine_cls, runner, store):
        """agent_fn is called once per (session, variant) pair."""
        source = Session(id="s1", step_counter=2)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        agent_fn = MagicMock()

        runner.run_backtest(
            source_session_ids=["s1"],
            variants=[{"name": "v1"}, {"name": "v2"}],
            agent_fn=agent_fn,
        )

        assert agent_fn.call_count == 2

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_multiple_sessions_and_variants(self, mock_engine_cls, runner, store):
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        for sid in ["s1", "s2"]:
            s = Session(id=sid, step_counter=3)
            s.end(SessionStatus.COMPLETED)
            store.save_session(s)

        results = runner.run_backtest(
            source_session_ids=["s1", "s2"],
            variants=[
                {"name": "v1"},
                {"name": "v2", "model": "gpt-4o-mini"},
            ],
            agent_fn=noop_agent,
        )

        assert len(results) == 4  # 2 sessions x 2 variants
        session_variant_pairs = {(r.source_session_id, r.variant_name) for r in results}
        assert ("s1", "v1") in session_variant_pairs
        assert ("s1", "v2") in session_variant_pairs
        assert ("s2", "v1") in session_variant_pairs
        assert ("s2", "v2") in session_variant_pairs

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_evaluator_callback(self, mock_engine_cls, runner, store, gate):
        from stateloom.experiment.manager import ExperimentManager

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        source = Session(id="s1", step_counter=3)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        exp_manager = ExperimentManager(store)
        runner_with_mgr = BacktestRunner(gate, experiment_manager=exp_manager)

        evaluator = MagicMock(return_value=0.85)

        results = runner_with_mgr.run_backtest(
            source_session_ids=["s1"],
            variants=[{"name": "v1"}],
            agent_fn=noop_agent,
            evaluator=evaluator,
        )

        assert len(results) == 1
        evaluator.assert_called_once()

        fb = store.get_feedback("backtest-s1-v1")
        assert fb is not None
        assert fb.score == 0.85
        assert fb.rating == "success"  # >= 0.5

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_evaluator_string_rating(self, mock_engine_cls, runner, store, gate):
        from stateloom.experiment.manager import ExperimentManager

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        source = Session(id="s1", step_counter=3)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        exp_manager = ExperimentManager(store)
        runner_with_mgr = BacktestRunner(gate, experiment_manager=exp_manager)

        evaluator = MagicMock(return_value="partial")

        runner_with_mgr.run_backtest(
            source_session_ids=["s1"],
            variants=[{"name": "v1"}],
            agent_fn=noop_agent,
            evaluator=evaluator,
        )

        fb = store.get_feedback("backtest-s1-v1")
        assert fb is not None
        assert fb.rating == "partial"

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_evaluator_returns_none_skips_feedback(self, mock_engine_cls, runner, store, gate):
        from stateloom.experiment.manager import ExperimentManager

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        source = Session(id="s1", step_counter=3)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        exp_manager = ExperimentManager(store)
        runner_with_mgr = BacktestRunner(gate, experiment_manager=exp_manager)

        evaluator = MagicMock(return_value=None)

        runner_with_mgr.run_backtest(
            source_session_ids=["s1"],
            variants=[{"name": "v1"}],
            agent_fn=noop_agent,
            evaluator=evaluator,
        )

        fb = store.get_feedback("backtest-s1-v1")
        assert fb is None

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_agent_fn_exception_does_not_crash(self, mock_engine_cls, runner, store):
        """If agent_fn raises, the runner catches it and still returns a result."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        source = Session(id="s1", step_counter=3)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        def failing_agent(session):
            raise RuntimeError("agent crashed")

        results = runner.run_backtest(
            source_session_ids=["s1"],
            variants=[{"name": "v1"}],
            agent_fn=failing_agent,
        )
        assert len(results) == 1
        assert results[0].output is None  # no output on failure

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_replay_engine_failure_does_not_crash(self, mock_engine_cls, runner, store):
        mock_engine = MagicMock()
        mock_engine.start.side_effect = Exception("replay error")
        mock_engine_cls.return_value = mock_engine

        source = Session(id="s1", step_counter=3)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        results = runner.run_backtest(
            source_session_ids=["s1"],
            variants=[{"name": "v1"}],
            agent_fn=noop_agent,
        )
        assert len(results) == 1

    @patch("stateloom.replay.engine.ReplayEngine")
    def test_session_context_restored_after_backtest(self, mock_engine_cls, runner, store):
        """Previous session context is restored after backtest completes."""
        from stateloom.core.context import get_current_session, set_current_session

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        source = Session(id="s1", step_counter=2)
        source.end(SessionStatus.COMPLETED)
        store.save_session(source)

        # Set a "current" session before backtest
        original_session = Session(id="original")
        set_current_session(original_session)

        runner.run_backtest(
            source_session_ids=["s1"],
            variants=[{"name": "v1"}],
            agent_fn=noop_agent,
        )

        # Should be restored
        assert get_current_session() is original_session
        set_current_session(None)  # cleanup
