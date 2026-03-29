"""Production tests: Async Job Queue.

Background job submission, lifecycle, cancellation, and stats.
"""

from __future__ import annotations

import time

import stateloom
from tests.test_production.helpers import make_openai_response


def test_submit_job(e2e_gate, api_client):
    """Submit job → returns job_id, status=pending."""
    gate = e2e_gate(cache=False, async_jobs_enabled=True)
    client = api_client(gate)

    job = gate.submit_job(
        provider="openai",
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Background task"}],
    )

    assert job.id
    assert job.status.value in ("pending", "running")


def test_job_lifecycle(e2e_gate, api_client):
    """Submit → get → verify fields."""
    gate = e2e_gate(cache=False, async_jobs_enabled=True)
    client = api_client(gate)

    job = gate.submit_job(
        provider="openai",
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Lifecycle test"}],
        metadata={"task": "test"},
    )

    retrieved = gate.get_job(job.id)
    assert retrieved is not None
    assert retrieved.id == job.id
    assert retrieved.model == "gpt-3.5-turbo"


def test_cancel_job(e2e_gate, api_client):
    """Cancel pending job → status=cancelled."""
    gate = e2e_gate(cache=False, async_jobs_enabled=True)
    client = api_client(gate)

    job = gate.submit_job(
        provider="openai",
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Cancel me"}],
    )

    # Cancel immediately
    result = gate.cancel_job(job.id)
    assert result is True

    cancelled = gate.get_job(job.id)
    assert cancelled.status.value == "cancelled"


def test_job_stats(e2e_gate, api_client):
    """Multiple jobs → job_stats() shows correct counts."""
    gate = e2e_gate(cache=False, async_jobs_enabled=True)
    client = api_client(gate)

    for i in range(3):
        gate.submit_job(
            provider="openai",
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Job {i}"}],
        )

    stats = gate.job_stats()
    assert stats["total"] >= 3


def test_list_jobs_by_status(e2e_gate, api_client):
    """Filter by status → correct results."""
    gate = e2e_gate(cache=False, async_jobs_enabled=True)
    client = api_client(gate)

    job1 = gate.submit_job(
        provider="openai",
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Stay pending"}],
    )
    job2 = gate.submit_job(
        provider="openai",
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Cancel me"}],
    )
    gate.cancel_job(job2.id)

    cancelled_jobs = gate.list_jobs(status="cancelled")
    cancelled_ids = [j.id for j in cancelled_jobs]
    assert job2.id in cancelled_ids
