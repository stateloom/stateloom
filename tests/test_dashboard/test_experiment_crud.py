"""Tests for dashboard experiment CRUD endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.dashboard.api import create_api_router
from stateloom.experiment.manager import ExperimentManager
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def mock_gate():
    """Create a mock Gate with a real ExperimentManager + MemoryStore."""
    gate = MagicMock()
    gate.store = MemoryStore()
    gate.config = MagicMock()
    gate.config.async_jobs_enabled = False
    gate.experiment_manager = ExperimentManager(gate.store)
    return gate


@pytest.fixture
def client(mock_gate):
    app = FastAPI()
    router = create_api_router(mock_gate)
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestCreateExperiment:
    def test_create_basic(self, client):
        resp = client.post(
            "/api/experiments",
            json={
                "name": "test-exp",
                "variants": [
                    {"name": "control", "weight": 1.0},
                    {"name": "treatment", "weight": 1.0, "model": "gpt-4o-mini"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-exp"
        assert data["status"] == "draft"
        assert len(data["variants"]) == 2
        assert data["id"]

    def test_create_with_agent_id(self, client):
        resp = client.post(
            "/api/experiments",
            json={
                "name": "agent-exp",
                "variants": [{"name": "v1"}],
                "agent_id": "agt-abc123",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "agt-abc123"

    def test_create_with_strategy(self, client):
        resp = client.post(
            "/api/experiments",
            json={
                "name": "hash-exp",
                "variants": [{"name": "a"}, {"name": "b"}],
                "strategy": "hash",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["strategy"] == "hash"

    def test_create_requires_name(self, client):
        resp = client.post(
            "/api/experiments",
            json={"variants": [{"name": "v1"}]},
        )
        assert resp.status_code == 422

    def test_create_requires_variants(self, client):
        resp = client.post(
            "/api/experiments",
            json={"name": "test"},
        )
        assert resp.status_code == 422

    def test_create_with_agent_version_id(self, client):
        resp = client.post(
            "/api/experiments",
            json={
                "name": "agent-variant-exp",
                "variants": [
                    {"name": "agent-v1", "agent_version_id": "agv-abc123"},
                ],
            },
        )
        assert resp.status_code == 200
        variants = resp.json()["variants"]
        assert variants[0]["agent_version_id"] == "agv-abc123"


class TestUpdateExperiment:
    def _create_experiment(self, client):
        resp = client.post(
            "/api/experiments",
            json={
                "name": "original",
                "variants": [{"name": "control"}, {"name": "treatment"}],
                "description": "original desc",
            },
        )
        return resp.json()["id"]

    def test_update_name(self, client):
        exp_id = self._create_experiment(client)
        resp = client.patch(
            f"/api/experiments/{exp_id}",
            json={"name": "updated-name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "updated-name"

    def test_update_description(self, client):
        exp_id = self._create_experiment(client)
        resp = client.patch(
            f"/api/experiments/{exp_id}",
            json={"description": "new desc"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "new desc"

    def test_update_variants(self, client):
        exp_id = self._create_experiment(client)
        resp = client.patch(
            f"/api/experiments/{exp_id}",
            json={
                "variants": [
                    {"name": "a", "model": "gpt-4o"},
                    {"name": "b", "model": "gpt-4o-mini"},
                    {"name": "c"},
                ],
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["variants"]) == 3

    def test_update_agent_id(self, client):
        exp_id = self._create_experiment(client)
        resp = client.patch(
            f"/api/experiments/{exp_id}",
            json={"agent_id": "agt-new"},
        )
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "agt-new"

    def test_reject_update_non_draft(self, client):
        exp_id = self._create_experiment(client)
        # Start it
        client.post(f"/api/experiments/{exp_id}/start")
        # Try to update
        resp = client.patch(
            f"/api/experiments/{exp_id}",
            json={"name": "should-fail"},
        )
        assert resp.status_code == 400
        assert "DRAFT" in resp.json()["detail"]

    def test_update_not_found(self, client):
        resp = client.patch(
            "/api/experiments/nonexistent",
            json={"name": "x"},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]


class TestLifecycleEndpoints:
    def _create_experiment(self, client):
        resp = client.post(
            "/api/experiments",
            json={
                "name": "lifecycle-test",
                "variants": [{"name": "a"}, {"name": "b"}],
            },
        )
        return resp.json()["id"]

    def test_start(self, client):
        exp_id = self._create_experiment(client)
        resp = client.post(f"/api/experiments/{exp_id}/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_pause(self, client):
        exp_id = self._create_experiment(client)
        client.post(f"/api/experiments/{exp_id}/start")
        resp = client.post(f"/api/experiments/{exp_id}/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_conclude(self, client):
        exp_id = self._create_experiment(client)
        client.post(f"/api/experiments/{exp_id}/start")
        resp = client.post(f"/api/experiments/{exp_id}/conclude")
        assert resp.status_code == 200
        assert resp.json()["status"] == "concluded"

    def test_start_invalid_state(self, client):
        exp_id = self._create_experiment(client)
        client.post(f"/api/experiments/{exp_id}/start")
        client.post(f"/api/experiments/{exp_id}/conclude")
        resp = client.post(f"/api/experiments/{exp_id}/start")
        assert resp.status_code == 400

    def test_pause_invalid_state(self, client):
        exp_id = self._create_experiment(client)
        resp = client.post(f"/api/experiments/{exp_id}/pause")
        assert resp.status_code == 400

    def test_conclude_invalid_state(self, client):
        exp_id = self._create_experiment(client)
        resp = client.post(f"/api/experiments/{exp_id}/conclude")
        assert resp.status_code == 400

    def test_start_not_found(self, client):
        resp = client.post("/api/experiments/nonexistent/start")
        assert resp.status_code == 400

    def test_full_lifecycle(self, client):
        exp_id = self._create_experiment(client)
        # DRAFT -> RUNNING
        client.post(f"/api/experiments/{exp_id}/start")
        # RUNNING -> PAUSED
        client.post(f"/api/experiments/{exp_id}/pause")
        # PAUSED -> RUNNING
        resp = client.post(f"/api/experiments/{exp_id}/start")
        assert resp.json()["status"] == "running"
        # RUNNING -> CONCLUDED
        resp = client.post(f"/api/experiments/{exp_id}/conclude")
        assert resp.json()["status"] == "concluded"


class TestGetExperimentIncludesAgentId:
    def test_agent_id_in_detail(self, client):
        resp = client.post(
            "/api/experiments",
            json={
                "name": "agent-scoped",
                "variants": [{"name": "v1"}],
                "agent_id": "agt-test",
            },
        )
        exp_id = resp.json()["id"]
        detail = client.get(f"/api/experiments/{exp_id}").json()
        assert detail["agent_id"] == "agt-test"
