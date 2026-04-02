"""Tests for the experiment assigner."""

import pytest
from stateloom.core.types import AssignmentStrategy, ExperimentStatus
from stateloom.experiment.assigner import ExperimentAssigner
from stateloom.experiment.models import Experiment, VariantConfig
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def assigner(store):
    return ExperimentAssigner(store)


@pytest.fixture
def experiment():
    return Experiment(
        id="exp1",
        name="test-experiment",
        status=ExperimentStatus.RUNNING,
        strategy=AssignmentStrategy.RANDOM,
        variants=[
            VariantConfig(name="control", weight=1.0),
            VariantConfig(name="variant-a", weight=1.0, model="gpt-4o-mini"),
        ],
    )


class TestRandomAssignment:
    def test_assigns_to_variant(self, assigner, experiment, store):
        store.save_experiment(experiment)
        assigner.register(experiment)

        assignment = assigner.assign("s1")
        assert assignment is not None
        assert assignment.experiment_id == "exp1"
        assert assignment.variant_name in ("control", "variant-a")

    def test_weighted_distribution(self, store):
        """With extreme weights, almost all assignments go to the heavy variant."""
        assigner = ExperimentAssigner(store)
        exp = Experiment(
            id="exp1",
            name="weighted",
            status=ExperimentStatus.RUNNING,
            strategy=AssignmentStrategy.RANDOM,
            variants=[
                VariantConfig(name="heavy", weight=100.0),
                VariantConfig(name="light", weight=0.001),
            ],
        )
        store.save_experiment(exp)
        assigner.register(exp)

        counts = {"heavy": 0, "light": 0}
        for i in range(100):
            a = assigner.assign(f"session-{i}")
            assert a is not None
            counts[a.variant_name] += 1

        # With 100:0.001 ratio, almost all should be heavy
        assert counts["heavy"] > 90


class TestHashAssignment:
    def test_deterministic(self, store):
        assigner = ExperimentAssigner(store)
        exp = Experiment(
            id="exp1",
            name="hash-test",
            status=ExperimentStatus.RUNNING,
            strategy=AssignmentStrategy.HASH,
            variants=[
                VariantConfig(name="control", weight=1.0),
                VariantConfig(name="variant-a", weight=1.0),
            ],
        )
        store.save_experiment(exp)
        assigner.register(exp)

        # Same session ID should always get same variant
        assignment1 = assigner.assign("deterministic-session")
        store2 = MemoryStore()
        assigner2 = ExperimentAssigner(store2)
        store2.save_experiment(exp)
        assigner2.register(exp)
        assignment2 = assigner2.assign("deterministic-session")

        assert assignment1 is not None
        assert assignment2 is not None
        assert assignment1.variant_name == assignment2.variant_name

    def test_different_sessions_can_get_different_variants(self, store):
        assigner = ExperimentAssigner(store)
        exp = Experiment(
            id="exp1",
            name="hash-spread",
            status=ExperimentStatus.RUNNING,
            strategy=AssignmentStrategy.HASH,
            variants=[
                VariantConfig(name="a", weight=1.0),
                VariantConfig(name="b", weight=1.0),
            ],
        )
        store.save_experiment(exp)
        assigner.register(exp)

        variants_seen = set()
        for i in range(50):
            a = assigner.assign(f"session-{i}")
            assert a is not None
            variants_seen.add(a.variant_name)

        # With 50 sessions, both variants should appear
        assert len(variants_seen) == 2


class TestManualAssignment:
    def test_requires_variant_name(self, store):
        assigner = ExperimentAssigner(store)
        exp = Experiment(
            id="exp1",
            name="manual",
            status=ExperimentStatus.RUNNING,
            strategy=AssignmentStrategy.MANUAL,
            variants=[
                VariantConfig(name="control", weight=1.0),
                VariantConfig(name="variant-a", weight=1.0),
            ],
        )
        store.save_experiment(exp)
        assigner.register(exp)

        # Without variant_name, returns None
        assert assigner.assign("s1") is None

    def test_explicit_variant(self, store):
        assigner = ExperimentAssigner(store)
        exp = Experiment(
            id="exp1",
            name="manual",
            status=ExperimentStatus.RUNNING,
            strategy=AssignmentStrategy.MANUAL,
            variants=[
                VariantConfig(name="control", weight=1.0),
                VariantConfig(name="variant-a", weight=1.0),
            ],
        )
        store.save_experiment(exp)
        assigner.register(exp)

        assignment = assigner.assign("s1", variant_name="variant-a")
        assert assignment is not None
        assert assignment.variant_name == "variant-a"


class TestDoubleAssignPrevention:
    def test_same_session_gets_same_assignment(self, assigner, experiment, store):
        store.save_experiment(experiment)
        assigner.register(experiment)

        first = assigner.assign("s1")
        second = assigner.assign("s1")
        assert first is not None
        assert second is not None
        assert first.variant_name == second.variant_name

    def test_preserves_original_assignment(self, assigner, experiment, store):
        store.save_experiment(experiment)
        assigner.register(experiment)

        first = assigner.assign("s1", experiment_id=experiment.id)
        assert first is not None

        # Even with explicit variant_name, existing assignment is returned
        # when the experiment is the same.
        second = assigner.assign("s1", experiment_id=experiment.id, variant_name="different")
        assert second.variant_name == first.variant_name


class TestCrossExperimentReassignment:
    """A session assigned to experiment A can be reassigned to experiment B."""

    def test_reassign_to_new_experiment(self, store):
        assigner = ExperimentAssigner(store)
        exp1 = Experiment(
            id="exp1",
            name="first",
            status=ExperimentStatus.RUNNING,
            variants=[VariantConfig(name="v1", weight=1.0)],
        )
        exp2 = Experiment(
            id="exp2",
            name="second",
            status=ExperimentStatus.RUNNING,
            variants=[VariantConfig(name="v2", weight=1.0)],
        )
        store.save_experiment(exp1)
        store.save_experiment(exp2)
        assigner.register(exp1)
        assigner.register(exp2)

        first = assigner.assign("s1", experiment_id="exp1")
        assert first is not None
        assert first.experiment_id == "exp1"
        assert first.variant_name == "v1"

        # Same session, different experiment — should get a new assignment
        second = assigner.assign("s1", experiment_id="exp2")
        assert second is not None
        assert second.experiment_id == "exp2"
        assert second.variant_name == "v2"

        # Store should reflect the latest assignment
        stored = store.get_assignment("s1")
        assert stored is not None
        assert stored.experiment_id == "exp2"


class TestNoRunningExperiment:
    def test_returns_none(self, assigner):
        assert assigner.assign("s1") is None

    def test_unregistered_experiment(self, assigner, experiment, store):
        store.save_experiment(experiment)
        assigner.register(experiment)
        assigner.unregister(experiment.id)
        assert assigner.assign("s2") is None


class TestSpecificExperiment:
    def test_assign_to_specific(self, store):
        assigner = ExperimentAssigner(store)
        exp1 = Experiment(
            id="exp1",
            name="exp1",
            status=ExperimentStatus.RUNNING,
            variants=[VariantConfig(name="v1", weight=1.0)],
        )
        exp2 = Experiment(
            id="exp2",
            name="exp2",
            status=ExperimentStatus.RUNNING,
            variants=[VariantConfig(name="v2", weight=1.0)],
        )
        store.save_experiment(exp1)
        store.save_experiment(exp2)
        assigner.register(exp1)
        assigner.register(exp2)

        assignment = assigner.assign("s1", experiment_id="exp2")
        assert assignment is not None
        assert assignment.experiment_id == "exp2"
        assert assignment.variant_name == "v2"

    def test_nonexistent_experiment_id(self, assigner, experiment, store):
        store.save_experiment(experiment)
        assigner.register(experiment)
        assert assigner.assign("s1", experiment_id="nonexistent") is None
