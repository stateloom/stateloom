"""Tests for experiment dashboard API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.core.config import StateLoomConfig
from stateloom.core.session import Session, SessionManager
from stateloom.dashboard.api import create_api_router
from stateloom.experiment.manager import ExperimentManager
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def gate():
    """Create a minimal gate-like object for API testing."""
    store = MemoryStore()
    g = MagicMock()
    g.store = store
    g.session_manager = SessionManager()
    g.experiment_manager = ExperimentManager(store)

    def _feedback(session_id, rating, score=None, comment=""):
        g.experiment_manager.record_feedback(session_id, rating, score=score, comment=comment)

    g.feedback = _feedback
    return g


@pytest.fixture
def client(gate):
    app = FastAPI()
    app.include_router(create_api_router(gate))
    return TestClient(app)


class TestExperimentEndpoints:
    def test_list_experiments_empty(self, client):
        resp = client.get("/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert data["experiments"] == []
        assert data["total"] == 0

    def test_list_experiments(self, client, gate):
        mgr = gate.experiment_manager
        mgr.create_experiment(
            name="test1",
            variants=[{"name": "v1"}],
        )
        mgr.create_experiment(
            name="test2",
            variants=[{"name": "v1"}, {"name": "v2"}],
        )

        resp = client.get("/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_list_experiments_filter_status(self, client, gate):
        mgr = gate.experiment_manager
        exp1 = mgr.create_experiment(name="draft", variants=[{"name": "v1"}])
        exp2 = mgr.create_experiment(name="running", variants=[{"name": "v1"}])
        mgr.start_experiment(exp2.id)

        resp = client.get("/experiments?status=running")
        data = resp.json()
        assert data["total"] == 1
        assert data["experiments"][0]["name"] == "running"

    def test_get_experiment(self, client, gate):
        mgr = gate.experiment_manager
        exp = mgr.create_experiment(
            name="test",
            variants=[{"name": "v1", "model": "gpt-4o-mini"}],
            description="A test experiment",
        )

        resp = client.get(f"/experiments/{exp.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test"
        assert data["description"] == "A test experiment"
        assert len(data["variants"]) == 1
        assert data["variants"][0]["model"] == "gpt-4o-mini"

    def test_get_experiment_not_found(self, client):
        resp = client.get("/experiments/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Experiment not found"

    def test_get_experiment_metrics(self, client, gate):
        mgr = gate.experiment_manager
        exp = mgr.create_experiment(
            name="test",
            variants=[{"name": "control"}],
            strategy="manual",
        )
        mgr.start_experiment(exp.id)

        # Create session and assign
        s = Session(id="s1")
        s.total_cost = 0.01
        gate.store.save_session(s)
        mgr.assign_session("s1", variant_name="control")

        resp = client.get(f"/experiments/{exp.id}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "variants" in data
        assert "control" in data["variants"]
        assert data["variants"]["control"]["session_count"] == 1


class TestLeaderboard:
    def test_leaderboard_empty(self, client):
        resp = client.get("/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []

    def test_leaderboard_with_data(self, client, gate):
        mgr = gate.experiment_manager
        exp = mgr.create_experiment(
            name="test",
            variants=[{"name": "v1"}, {"name": "v2"}],
            strategy="manual",
        )
        mgr.start_experiment(exp.id)

        # v1: 1 success
        s1 = Session(id="s1")
        s1.total_cost = 0.01
        gate.store.save_session(s1)
        mgr.assign_session("s1", variant_name="v1")
        mgr.record_feedback("s1", "success")

        # v2: 1 failure
        s2 = Session(id="s2")
        s2.total_cost = 0.02
        gate.store.save_session(s2)
        mgr.assign_session("s2", variant_name="v2")
        mgr.record_feedback("s2", "failure")

        resp = client.get("/leaderboard")
        data = resp.json()
        assert len(data["entries"]) == 2
        # v1 has higher success rate
        assert data["entries"][0]["variant_name"] == "v1"
        assert data["entries"][0]["success_rate"] == 1.0


class TestFeedbackEndpoints:
    def test_post_feedback(self, client, gate):
        resp = client.post(
            "/sessions/s1/feedback",
            json={"rating": "success", "score": 0.9, "comment": "great"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify stored
        fb = gate.store.get_feedback("s1")
        assert fb is not None
        assert fb.rating == "success"
        assert fb.score == 0.9

    def test_get_feedback(self, client, gate):
        gate.experiment_manager.record_feedback("s1", "failure", comment="bad output")

        resp = client.get("/sessions/s1/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["feedback"]["rating"] == "failure"
        assert data["feedback"]["comment"] == "bad output"

    def test_get_feedback_not_found(self, client):
        resp = client.get("/sessions/nonexistent/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["feedback"] is None


class TestSessionWithExperimentInfo:
    def test_session_includes_experiment_metadata(self, client, gate):
        s = Session(id="s1", name="test-session")
        s.metadata = {
            "experiment_id": "exp1",
            "variant": "fast",
        }
        gate.store.save_session(s)

        resp = client.get("/sessions")
        data = resp.json()
        assert data["total"] == 1
        session_data = data["sessions"][0]
        assert session_data["experiment_id"] == "exp1"
        assert session_data["variant"] == "fast"

    def test_session_detail_includes_experiment_metadata(self, client, gate):
        s = Session(id="s1", name="test-session")
        s.metadata = {
            "experiment_id": "exp1",
            "variant": "fast",
        }
        gate.store.save_session(s)

        resp = client.get("/sessions/s1")
        data = resp.json()
        assert data["experiment_id"] == "exp1"
        assert data["variant"] == "fast"
