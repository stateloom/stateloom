"""Tests for experiment data models."""

import pytest

from stateloom.core.event import FeedbackEvent
from stateloom.core.types import EventType, ExperimentStatus, FeedbackRating
from stateloom.experiment.models import (
    Experiment,
    ExperimentAssignment,
    SessionFeedback,
    VariantConfig,
)


class TestVariantConfig:
    def test_create(self):
        v = VariantConfig(name="control", weight=1.0)
        assert v.name == "control"
        assert v.weight == 1.0
        assert v.model is None
        assert v.request_overrides == {}

    def test_with_overrides(self):
        v = VariantConfig(
            name="fast",
            weight=2.0,
            model="gpt-4o-mini",
            request_overrides={"temperature": 0.3},
        )
        assert v.model == "gpt-4o-mini"
        assert v.request_overrides["temperature"] == 0.3

    def test_to_dict(self):
        v = VariantConfig(name="test", weight=1.5, model="gpt-4o")
        d = v.to_dict()
        assert d["name"] == "test"
        assert d["weight"] == 1.5
        assert d["model"] == "gpt-4o"

    def test_from_dict(self):
        d = {
            "name": "test",
            "weight": 2.0,
            "model": "gpt-4o-mini",
            "request_overrides": {"temperature": 0.5},
        }
        v = VariantConfig.from_dict(d)
        assert v.name == "test"
        assert v.weight == 2.0
        assert v.model == "gpt-4o-mini"
        assert v.request_overrides["temperature"] == 0.5

    def test_from_dict_defaults(self):
        d = {"name": "minimal"}
        v = VariantConfig.from_dict(d)
        assert v.weight == 1.0
        assert v.model is None

    def test_roundtrip(self):
        original = VariantConfig(
            name="test",
            weight=3.0,
            model="claude-3",
            request_overrides={"max_tokens": 100},
            metadata={"tag": "prod"},
        )
        restored = VariantConfig.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.weight == original.weight
        assert restored.model == original.model
        assert restored.request_overrides == original.request_overrides
        assert restored.metadata == original.metadata


class TestExperiment:
    def test_create_default(self):
        exp = Experiment(name="test")
        assert exp.status == ExperimentStatus.DRAFT
        assert exp.variants == []
        assert exp.assignment_counts == {}
        assert len(exp.id) == 12

    def test_start(self):
        exp = Experiment(name="test")
        exp.start()
        assert exp.status == ExperimentStatus.RUNNING

    def test_pause(self):
        exp = Experiment(name="test")
        exp.start()
        exp.pause()
        assert exp.status == ExperimentStatus.PAUSED

    def test_conclude_from_running(self):
        exp = Experiment(name="test")
        exp.start()
        exp.conclude()
        assert exp.status == ExperimentStatus.CONCLUDED

    def test_conclude_from_paused(self):
        exp = Experiment(name="test")
        exp.start()
        exp.pause()
        exp.conclude()
        assert exp.status == ExperimentStatus.CONCLUDED

    def test_start_from_paused(self):
        exp = Experiment(name="test")
        exp.start()
        exp.pause()
        exp.start()
        assert exp.status == ExperimentStatus.RUNNING

    def test_cannot_start_concluded(self):
        exp = Experiment(name="test")
        exp.start()
        exp.conclude()
        with pytest.raises(ValueError, match="Cannot start"):
            exp.start()

    def test_cannot_pause_draft(self):
        exp = Experiment(name="test")
        with pytest.raises(ValueError, match="Cannot pause"):
            exp.pause()

    def test_cannot_conclude_draft(self):
        exp = Experiment(name="test")
        with pytest.raises(ValueError, match="Cannot conclude"):
            exp.conclude()

    def test_record_assignment(self):
        exp = Experiment(name="test")
        exp.record_assignment("control")
        exp.record_assignment("control")
        exp.record_assignment("variant-a")
        assert exp.assignment_counts["control"] == 2
        assert exp.assignment_counts["variant-a"] == 1

    def test_thread_safety(self):
        """Test that lifecycle methods are thread-safe."""
        import threading

        exp = Experiment(name="test")
        exp.start()
        errors = []

        def record():
            try:
                for _ in range(100):
                    exp.record_assignment("v1")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert exp.assignment_counts["v1"] == 400


class TestExperimentAssignment:
    def test_create(self):
        variant = VariantConfig(name="fast", model="gpt-4o-mini")
        assignment = ExperimentAssignment.create(
            session_id="s1",
            experiment_id="exp1",
            variant=variant,
        )
        assert assignment.session_id == "s1"
        assert assignment.experiment_id == "exp1"
        assert assignment.variant_name == "fast"
        assert assignment.variant_config["model"] == "gpt-4o-mini"

    def test_config_snapshot(self):
        """Assignment stores a snapshot, not a reference."""
        variant = VariantConfig(name="v1", model="gpt-4o")
        assignment = ExperimentAssignment.create("s1", "exp1", variant)

        # Mutate original — assignment should be unaffected
        variant.model = "changed"
        assert assignment.variant_config["model"] == "gpt-4o"


class TestSessionFeedback:
    def test_create(self):
        fb = SessionFeedback(session_id="s1", rating="success", score=0.95)
        assert fb.session_id == "s1"
        assert fb.rating == "success"
        assert fb.score == 0.95
        assert fb.comment == ""


class TestFeedbackEvent:
    def test_create(self):
        event = FeedbackEvent(
            session_id="s1",
            rating="success",
            score=0.9,
            comment="great",
        )
        assert event.event_type == EventType.FEEDBACK
        assert event.rating == "success"
        assert event.score == 0.9
        assert event.comment == "great"

    def test_defaults(self):
        event = FeedbackEvent(session_id="s1")
        assert event.rating == ""
        assert event.score is None
        assert event.comment == ""


class TestEnums:
    def test_experiment_status_values(self):
        assert ExperimentStatus.DRAFT.value == "draft"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.PAUSED.value == "paused"
        assert ExperimentStatus.CONCLUDED.value == "concluded"

    def test_feedback_rating_values(self):
        assert FeedbackRating.SUCCESS.value == "success"
        assert FeedbackRating.FAILURE.value == "failure"
        assert FeedbackRating.PARTIAL.value == "partial"

    def test_feedback_event_type(self):
        assert EventType.FEEDBACK.value == "feedback"
