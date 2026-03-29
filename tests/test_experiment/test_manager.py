"""Tests for the experiment manager."""

import pytest

from stateloom.core.session import Session
from stateloom.core.types import ExperimentStatus
from stateloom.experiment.manager import ExperimentManager
from stateloom.experiment.models import VariantConfig
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def manager(store):
    return ExperimentManager(store)


class TestCreateExperiment:
    def test_create_from_dicts(self, manager):
        exp = manager.create_experiment(
            name="test",
            variants=[
                {"name": "control", "weight": 1.0},
                {"name": "fast", "weight": 1.0, "model": "gpt-4o-mini"},
            ],
        )
        assert exp.name == "test"
        assert exp.status == ExperimentStatus.DRAFT
        assert len(exp.variants) == 2
        assert exp.variants[1].model == "gpt-4o-mini"

    def test_create_from_variant_configs(self, manager):
        exp = manager.create_experiment(
            name="test",
            variants=[VariantConfig(name="v1"), VariantConfig(name="v2")],
        )
        assert len(exp.variants) == 2

    def test_create_with_strategy(self, manager):
        exp = manager.create_experiment(
            name="test",
            variants=[{"name": "v1"}],
            strategy="hash",
        )
        from stateloom.core.types import AssignmentStrategy

        assert exp.strategy == AssignmentStrategy.HASH

    def test_create_persisted(self, manager, store):
        exp = manager.create_experiment(
            name="test",
            variants=[{"name": "v1"}],
        )
        retrieved = store.get_experiment(exp.id)
        assert retrieved is not None
        assert retrieved.name == "test"


class TestCreateWithAgentId:
    def test_create_with_agent_id(self, manager, store):
        exp = manager.create_experiment(
            name="agent-test",
            variants=[{"name": "v1"}],
            agent_id="agt-abc123",
        )
        assert exp.agent_id == "agt-abc123"
        retrieved = store.get_experiment(exp.id)
        assert retrieved.agent_id == "agt-abc123"


class TestUpdateExperiment:
    def test_update_name(self, manager, store):
        exp = manager.create_experiment(name="old", variants=[{"name": "v1"}])
        updated = manager.update_experiment(exp.id, name="new")
        assert updated.name == "new"
        assert store.get_experiment(exp.id).name == "new"

    def test_update_description(self, manager):
        exp = manager.create_experiment(name="test", variants=[{"name": "v1"}])
        updated = manager.update_experiment(exp.id, description="new desc")
        assert updated.description == "new desc"

    def test_update_variants(self, manager):
        exp = manager.create_experiment(
            name="test",
            variants=[{"name": "v1"}],
        )
        updated = manager.update_experiment(
            exp.id,
            variants=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
        )
        assert len(updated.variants) == 3
        assert updated.variants[2].name == "c"

    def test_update_strategy(self, manager):
        exp = manager.create_experiment(name="test", variants=[{"name": "v1"}])
        updated = manager.update_experiment(exp.id, strategy="hash")
        from stateloom.core.types import AssignmentStrategy

        assert updated.strategy == AssignmentStrategy.HASH

    def test_update_agent_id(self, manager):
        exp = manager.create_experiment(name="test", variants=[{"name": "v1"}])
        updated = manager.update_experiment(exp.id, agent_id="agt-new")
        assert updated.agent_id == "agt-new"

    def test_update_metadata(self, manager):
        exp = manager.create_experiment(name="test", variants=[{"name": "v1"}])
        updated = manager.update_experiment(exp.id, metadata={"key": "value"})
        assert updated.metadata == {"key": "value"}

    def test_reject_non_draft(self, manager):
        exp = manager.create_experiment(name="test", variants=[{"name": "v1"}])
        manager.start_experiment(exp.id)
        with pytest.raises(ValueError, match="DRAFT"):
            manager.update_experiment(exp.id, name="should-fail")

    def test_reject_nonexistent(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.update_experiment("nonexistent", name="x")

    def test_updates_timestamp(self, manager):
        exp = manager.create_experiment(name="test", variants=[{"name": "v1"}])
        original_updated = exp.updated_at
        updated = manager.update_experiment(exp.id, name="renamed")
        assert updated.updated_at >= original_updated


class TestLifecycle:
    def test_full_lifecycle(self, manager, store):
        exp = manager.create_experiment(
            name="lifecycle-test",
            variants=[{"name": "control"}, {"name": "variant-a"}],
        )
        assert exp.status == ExperimentStatus.DRAFT

        started = manager.start_experiment(exp.id)
        assert started.status == ExperimentStatus.RUNNING

        paused = manager.pause_experiment(exp.id)
        assert paused.status == ExperimentStatus.PAUSED

        restarted = manager.start_experiment(exp.id)
        assert restarted.status == ExperimentStatus.RUNNING

        final_metrics = manager.conclude_experiment(exp.id)
        concluded = store.get_experiment(exp.id)
        assert concluded.status == ExperimentStatus.CONCLUDED
        assert "experiment_id" in final_metrics

    def test_start_nonexistent(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.start_experiment("nonexistent")

    def test_pause_nonexistent(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.pause_experiment("nonexistent")

    def test_conclude_nonexistent(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.conclude_experiment("nonexistent")


class TestAssignment:
    def test_assign_session(self, manager, store):
        exp = manager.create_experiment(
            name="test",
            variants=[{"name": "control"}, {"name": "variant-a"}],
        )
        manager.start_experiment(exp.id)

        assignment = manager.assign_session("s1")
        assert assignment is not None
        assert assignment.experiment_id == exp.id

    def test_no_assignment_for_draft(self, manager):
        manager.create_experiment(
            name="test",
            variants=[{"name": "control"}],
        )
        assert manager.assign_session("s1") is None

    def test_assign_with_explicit_experiment(self, manager):
        exp = manager.create_experiment(
            name="test",
            variants=[{"name": "control"}],
        )
        manager.start_experiment(exp.id)

        assignment = manager.assign_session("s1", experiment_id=exp.id)
        assert assignment is not None
        assert assignment.experiment_id == exp.id


class TestFeedback:
    def test_record_feedback(self, manager, store):
        manager.record_feedback("s1", "success", score=0.9, comment="good")

        fb = store.get_feedback("s1")
        assert fb is not None
        assert fb.rating == "success"
        assert fb.score == 0.9
        assert fb.comment == "good"

    def test_feedback_emits_event(self, manager, store):
        manager.record_feedback("s1", "failure")

        events = store.get_session_events("s1", event_type="feedback")
        assert len(events) == 1
        assert events[0].rating == "failure"


class TestMetrics:
    def test_get_metrics(self, manager, store):
        exp = manager.create_experiment(
            name="test",
            variants=[{"name": "control"}, {"name": "fast"}],
            strategy="manual",
        )
        manager.start_experiment(exp.id)

        # Create sessions and assign
        for i in range(3):
            s = Session(id=f"ctrl-{i}")
            s.total_cost = 0.01 * (i + 1)
            s.total_tokens = 100 * (i + 1)
            store.save_session(s)
            manager.assign_session(f"ctrl-{i}", variant_name="control")

        for i in range(2):
            s = Session(id=f"fast-{i}")
            s.total_cost = 0.005 * (i + 1)
            s.total_tokens = 50 * (i + 1)
            store.save_session(s)
            manager.assign_session(f"fast-{i}", variant_name="fast")

        # Add feedback
        manager.record_feedback("ctrl-0", "success", score=0.8)
        manager.record_feedback("ctrl-1", "failure")
        manager.record_feedback("fast-0", "success", score=0.95)

        metrics = manager.get_metrics(exp.id)
        assert "variants" in metrics
        assert "control" in metrics["variants"]
        assert "fast" in metrics["variants"]

        ctrl = metrics["variants"]["control"]
        assert ctrl["session_count"] == 3
        assert ctrl["success_count"] == 1
        assert ctrl["failure_count"] == 1

        fast = metrics["variants"]["fast"]
        assert fast["session_count"] == 2
        assert fast["success_count"] == 1


class TestLeaderboard:
    def test_leaderboard_ranking(self, manager, store):
        # Create two experiments with different success rates
        exp1 = manager.create_experiment(
            name="exp1",
            variants=[{"name": "v1"}],
            strategy="manual",
        )
        exp2 = manager.create_experiment(
            name="exp2",
            variants=[{"name": "v2"}],
            strategy="manual",
        )
        manager.start_experiment(exp1.id)
        manager.start_experiment(exp2.id)

        # exp1/v1: 2 success, 1 failure (66% success rate)
        for i in range(3):
            s = Session(id=f"e1-{i}")
            s.total_cost = 0.01
            store.save_session(s)
            manager.assign_session(f"e1-{i}", experiment_id=exp1.id, variant_name="v1")
        manager.record_feedback("e1-0", "success")
        manager.record_feedback("e1-1", "success")
        manager.record_feedback("e1-2", "failure")

        # exp2/v2: 3 success, 0 failure (100% success rate)
        for i in range(3):
            s = Session(id=f"e2-{i}")
            s.total_cost = 0.02
            store.save_session(s)
            manager.assign_session(f"e2-{i}", experiment_id=exp2.id, variant_name="v2")
        manager.record_feedback("e2-0", "success")
        manager.record_feedback("e2-1", "success")
        manager.record_feedback("e2-2", "success")

        board = manager.get_leaderboard()
        assert len(board) == 2
        # Higher success rate first
        assert board[0]["variant_name"] == "v2"
        assert board[0]["success_rate"] == 1.0
        assert board[1]["variant_name"] == "v1"


class TestRestoreRunning:
    def test_restore(self, store):
        # Create and start an experiment directly in the store
        exp = ExperimentManager(store).create_experiment(
            name="test",
            variants=[{"name": "v1"}],
        )
        ExperimentManager(store).start_experiment(exp.id)

        # New manager should restore running experiments
        new_manager = ExperimentManager(store)
        new_manager.restore_running_experiments()

        # Should be able to assign sessions
        assignment = new_manager.assign_session("s1")
        assert assignment is not None
        assert assignment.experiment_id == exp.id
