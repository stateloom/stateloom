"""Tests for the dashboard API job endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from stateloom.core.job import Job
from stateloom.core.types import JobStatus
from stateloom.dashboard.api import create_api_router
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def mock_gate():
    """Create a mock Gate with async jobs enabled."""
    gate = MagicMock()
    gate.store = MemoryStore()
    gate.config = MagicMock()
    gate.config.async_jobs_enabled = True
    gate.config.async_jobs_default_ttl = 3600

    # Wire up real methods using the store
    def _submit_job(**kwargs):
        job = Job(**kwargs)
        gate.store.save_job(job)
        return job

    def _get_job(job_id):
        return gate.store.get_job(job_id)

    def _list_jobs(status=None, session_id=None, limit=100, offset=0):
        return gate.store.list_jobs(
            status=status, session_id=session_id, limit=limit, offset=offset
        )

    def _cancel_job(job_id):
        job = gate.store.get_job(job_id)
        if job is None or job.status != JobStatus.PENDING:
            return False
        job.status = JobStatus.CANCELLED
        gate.store.save_job(job)
        return True

    def _job_stats():
        return gate.store.get_job_stats()

    gate.submit_job = _submit_job
    gate.get_job = _get_job
    gate.list_jobs = _list_jobs
    gate.cancel_job = _cancel_job
    gate.job_stats = _job_stats

    return gate


@pytest.fixture
def client(mock_gate):
    """Create a test client for the API."""
    from fastapi import FastAPI

    app = FastAPI()
    router = create_api_router(mock_gate)
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


class TestSubmitJob:
    def test_submit_returns_202(self, client):
        resp = client.post(
            "/api/v1/jobs",
            json={
                "provider": "openai",
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["id"].startswith("job_")
        assert data["status"] == "pending"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4"

    def test_submit_with_webhook(self, client):
        resp = client.post(
            "/api/v1/jobs",
            json={
                "model": "gpt-4",
                "webhook_url": "https://example.com/hook",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["webhook_url"] == "https://example.com/hook"

    def test_submit_disabled_returns_400(self, client, mock_gate):
        mock_gate.config.async_jobs_enabled = False
        resp = client.post("/api/v1/jobs", json={"model": "gpt-4"})
        assert resp.status_code == 400
        assert "not enabled" in resp.json()["detail"]

    def test_submit_with_metadata(self, client):
        resp = client.post(
            "/api/v1/jobs",
            json={
                "model": "gpt-4",
                "metadata": {"user": "alice"},
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["metadata"] == {"user": "alice"}


class TestListJobs:
    def test_list_empty(self, client):
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_list_with_jobs(self, client, mock_gate):
        mock_gate.store.save_job(Job(provider="openai"))
        mock_gate.store.save_job(Job(provider="anthropic"))

        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_list_filter_by_status(self, client, mock_gate):
        mock_gate.store.save_job(Job(status=JobStatus.PENDING))
        mock_gate.store.save_job(Job(status=JobStatus.COMPLETED))

        resp = client.get("/api/v1/jobs?status=pending")
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["status"] == "pending"

    def test_list_with_limit(self, client, mock_gate):
        for _ in range(5):
            mock_gate.store.save_job(Job())

        resp = client.get("/api/v1/jobs?limit=2")
        data = resp.json()
        assert len(data["jobs"]) == 2


class TestGetJob:
    def test_get_existing_job(self, client, mock_gate):
        job = Job(provider="openai", model="gpt-4")
        mock_gate.store.save_job(job)

        resp = client.get(f"/api/v1/jobs/{job.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == job.id
        assert data["provider"] == "openai"

    def test_get_nonexistent_job(self, client):
        resp = client.get("/api/v1/jobs/nonexistent")
        assert resp.status_code == 404

    def test_get_job_with_result(self, client, mock_gate):
        job = Job(
            status=JobStatus.COMPLETED,
            result={"choices": [{"text": "hello"}]},
        )
        mock_gate.store.save_job(job)

        resp = client.get(f"/api/v1/jobs/{job.id}")
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] == {"choices": [{"text": "hello"}]}

    def test_get_job_with_error(self, client, mock_gate):
        job = Job(
            status=JobStatus.FAILED,
            error="Budget exceeded",
            error_code="BUDGET_ERROR",
        )
        mock_gate.store.save_job(job)

        resp = client.get(f"/api/v1/jobs/{job.id}")
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "Budget exceeded"
        assert data["error_code"] == "BUDGET_ERROR"


class TestCancelJob:
    def test_cancel_pending_job(self, client, mock_gate):
        job = Job(status=JobStatus.PENDING)
        mock_gate.store.save_job(job)

        resp = client.delete(f"/api/v1/jobs/{job.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

        # Verify in store
        updated = mock_gate.store.get_job(job.id)
        assert updated.status == JobStatus.CANCELLED

    def test_cancel_nonexistent_job(self, client):
        resp = client.delete("/api/v1/jobs/nonexistent")
        assert resp.status_code == 404

    def test_cancel_running_job_fails(self, client, mock_gate):
        job = Job(status=JobStatus.RUNNING)
        mock_gate.store.save_job(job)

        resp = client.delete(f"/api/v1/jobs/{job.id}")
        assert resp.status_code == 400
        assert "not pending" in resp.json()["detail"]


class TestJobStats:
    def test_empty_stats(self, client):
        resp = client.get("/api/v1/jobs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_stats_with_jobs(self, client, mock_gate):
        mock_gate.store.save_job(Job(status=JobStatus.PENDING))
        mock_gate.store.save_job(Job(status=JobStatus.COMPLETED))
        mock_gate.store.save_job(Job(status=JobStatus.FAILED))

        resp = client.get("/api/v1/jobs/stats")
        data = resp.json()
        assert data["total"] == 3
        assert data["by_status"]["pending"] == 1
        assert data["by_status"]["completed"] == 1
        assert data["by_status"]["failed"] == 1
