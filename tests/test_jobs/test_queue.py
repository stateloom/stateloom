"""Tests for JobQueue protocol and InProcessJobQueue implementation."""

from __future__ import annotations

import pytest

from stateloom.core.job import Job
from stateloom.core.types import JobStatus
from stateloom.jobs.queue import InProcessJobQueue, JobQueue
from stateloom.store.memory_store import MemoryStore


def _make_job(**kwargs) -> Job:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        status=JobStatus.PENDING,
    )
    defaults.update(kwargs)
    return Job(**defaults)


class TestProtocolCompliance:
    def test_isinstance_check(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)
        assert isinstance(queue, JobQueue)


class TestEnqueueAndDequeue:
    def test_enqueue_and_dequeue(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)
        job = _make_job()
        queue.enqueue(job)

        result = queue.dequeue(10)
        assert len(result) == 1
        assert result[0].id == job.id
        assert result[0].status == JobStatus.PENDING

    def test_dequeue_respects_limit(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)
        for _ in range(5):
            queue.enqueue(_make_job())

        result = queue.dequeue(2)
        assert len(result) == 2

    def test_dequeue_empty_queue(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)

        result = queue.dequeue(10)
        assert result == []


class TestMarkRunning:
    def test_mark_running(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)
        job = _make_job()
        queue.enqueue(job)

        queue.mark_running(job)

        updated = store.get_job(job.id)
        assert updated.status == JobStatus.RUNNING
        assert updated.started_at is not None


class TestMarkCompleted:
    def test_mark_completed(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)
        job = _make_job()
        queue.enqueue(job)
        queue.mark_running(job)

        job.result = {"answer": "hello"}
        queue.mark_completed(job)

        updated = store.get_job(job.id)
        assert updated.status == JobStatus.COMPLETED
        assert updated.completed_at is not None


class TestMarkFailed:
    def test_mark_failed(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)
        job = _make_job()
        queue.enqueue(job)
        queue.mark_running(job)

        job.error = "something went wrong"
        queue.mark_failed(job)

        updated = store.get_job(job.id)
        assert updated.status == JobStatus.FAILED
        assert updated.completed_at is not None


class TestRequeue:
    def test_requeue(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)
        job = _make_job()
        queue.enqueue(job)
        queue.mark_running(job)

        job.error = "transient failure"
        queue.requeue(job)

        updated = store.get_job(job.id)
        assert updated.status == JobStatus.PENDING
        assert updated.retry_count == 1
        assert updated.started_at is None


class TestRecoverStale:
    def test_recover_stale(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)

        # Simulate jobs stuck in RUNNING
        job1 = _make_job()
        job1.status = JobStatus.RUNNING
        store.save_job(job1)

        job2 = _make_job()
        job2.status = JobStatus.RUNNING
        store.save_job(job2)

        # Also a PENDING job that should not be touched
        job3 = _make_job()
        store.save_job(job3)

        recovered = queue.recover_stale()
        assert recovered == 2

        assert store.get_job(job1.id).status == JobStatus.PENDING
        assert store.get_job(job2.id).status == JobStatus.PENDING
        assert store.get_job(job3.id).status == JobStatus.PENDING

    def test_recover_stale_empty(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)

        recovered = queue.recover_stale()
        assert recovered == 0


class TestShutdown:
    def test_shutdown_is_noop(self):
        store = MemoryStore()
        queue = InProcessJobQueue(store)
        # Should not raise
        queue.shutdown()
