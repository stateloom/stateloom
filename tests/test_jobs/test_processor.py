"""Tests for JobProcessor — background worker pool that polls and processes jobs."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.errors import StateLoomError
from stateloom.core.job import Job
from stateloom.core.types import JobStatus
from stateloom.jobs.processor import JobProcessor
from stateloom.jobs.queue import InProcessJobQueue
from stateloom.store.memory_store import MemoryStore


def _make_mock_gate(store=None):
    """Create a minimal mock Gate for testing."""
    gate = MagicMock()
    gate.store = store or MemoryStore()
    gate.config = MagicMock()
    gate.config.async_jobs_webhook_timeout = 30.0
    gate.config.async_jobs_webhook_retries = 3
    gate.config.async_jobs_webhook_secret = ""
    # Mock session manager
    session = MagicMock()
    session.id = "test-session"
    gate.session_manager.create.return_value = session
    # Mock pipeline — returns a successful response dict
    gate.pipeline.execute_sync.return_value = {"choices": [{"message": {"content": "hello"}}]}
    return gate


def _make_pending_job(**kwargs) -> Job:
    """Create a pending job with sensible defaults."""
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        status=JobStatus.PENDING,
    )
    defaults.update(kwargs)
    return Job(**defaults)


class TestProcessorStartStop:
    """Test that the processor can start and shut down cleanly."""

    def test_processor_start_stop(self):
        gate = _make_mock_gate()
        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=2)
        try:
            processor.start()
            assert processor._executor is not None
            assert processor._poll_thread is not None
            assert processor._poll_thread.is_alive()
        finally:
            processor.shutdown()

        assert processor._executor is None
        assert processor._poll_thread is None


class TestProcessorProcessesPendingJob:
    """Test that the processor picks up a pending job and completes it."""

    def test_processor_processes_pending_job(self):
        store = MemoryStore()
        gate = _make_mock_gate(store=store)
        job = _make_pending_job()
        store.save_job(job)

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=2)
        try:
            processor.start()
            time.sleep(2.0)
        finally:
            processor.shutdown()

        updated = store.get_job(job.id)
        assert updated is not None
        assert updated.status == JobStatus.COMPLETED
        assert updated.result is not None
        assert updated.completed_at is not None


class TestProcessorRetryOnTransientError:
    """Test that a generic Exception triggers a retry."""

    def test_processor_retry_on_transient_error(self):
        store = MemoryStore()
        gate = _make_mock_gate(store=store)
        gate.pipeline.execute_sync.side_effect = Exception("transient")

        job = _make_pending_job(max_retries=3)
        store.save_job(job)

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=2)
        try:
            processor.start()
            time.sleep(2.0)
        finally:
            processor.shutdown()

        updated = store.get_job(job.id)
        assert updated is not None
        # Job should have been retried at least once
        assert updated.retry_count >= 1
        # Depending on timing, it may still be PENDING (re-queued), FAILED
        # (exhausted retries), or DEAD (moved to DLQ by the queue).
        assert updated.status in (JobStatus.PENDING, JobStatus.FAILED, JobStatus.DEAD)
        assert updated.error == "transient"


class TestProcessorTerminalErrorNoRetry:
    """Test that an StateLoomError causes immediate failure without retry."""

    def test_processor_terminal_error_no_retry(self):
        store = MemoryStore()
        gate = _make_mock_gate(store=store)
        gate.pipeline.execute_sync.side_effect = StateLoomError("test error")

        job = _make_pending_job(max_retries=3)
        store.save_job(job)

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=2)
        try:
            processor.start()
            time.sleep(2.0)
        finally:
            processor.shutdown()

        updated = store.get_job(job.id)
        assert updated is not None
        assert updated.status == JobStatus.FAILED
        assert updated.retry_count == 0  # No retry attempted
        assert "test error" in updated.error
        assert updated.completed_at is not None


class TestProcessorCancelPreventsProcessing:
    """Test that a cancelled job is not picked up by the processor."""

    def test_processor_cancel_prevents_processing(self):
        store = MemoryStore()
        gate = _make_mock_gate(store=store)

        job = _make_pending_job()
        job.status = JobStatus.CANCELLED
        store.save_job(job)

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=2)
        try:
            processor.start()
            time.sleep(2.0)
        finally:
            processor.shutdown()

        updated = store.get_job(job.id)
        assert updated is not None
        assert updated.status == JobStatus.CANCELLED
        # Pipeline should never have been called for a cancelled job
        gate.pipeline.execute_sync.assert_not_called()


class TestProcessorCrashRecovery:
    """Test that jobs stuck in RUNNING status are recovered on startup."""

    def test_processor_crash_recovery(self):
        store = MemoryStore()
        gate = _make_mock_gate(store=store)

        # Simulate a job that was RUNNING when the process crashed
        job = _make_pending_job()
        job.status = JobStatus.RUNNING
        store.save_job(job)

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=2)
        try:
            processor.start()
            # After start(), _recover_stale_jobs should have reset it to PENDING.
            # Give the poll loop time to pick it up and process it.
            time.sleep(2.0)
        finally:
            processor.shutdown()

        updated = store.get_job(job.id)
        assert updated is not None
        # The job should have been recovered (reset to PENDING) and then processed.
        # Since pipeline mock returns a valid response, it should be COMPLETED.
        assert updated.status == JobStatus.COMPLETED


class TestProcessorGracefulShutdown:
    """Test that shutdown completes without crashing even with in-flight jobs."""

    def test_processor_graceful_shutdown(self):
        store = MemoryStore()
        gate = _make_mock_gate(store=store)

        # Make the pipeline call slow so the job is in-flight during shutdown
        def slow_execute(*args, **kwargs):
            time.sleep(0.5)
            return {"choices": [{"message": {"content": "done"}}]}

        gate.pipeline.execute_sync.side_effect = slow_execute

        job = _make_pending_job()
        store.save_job(job)

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=2)
        try:
            processor.start()
            time.sleep(1.5)  # Let the poll loop pick up the job
        finally:
            # Shutdown should not raise or hang
            processor.shutdown(drain_timeout=5.0)

        # Processor state should be cleaned up
        assert processor._executor is None
        assert processor._poll_thread is None


class TestProcessorMaxWorkersLimit:
    """Test that the processor respects the max_workers limit."""

    def test_processor_max_workers_limit(self):
        store = MemoryStore()
        gate = _make_mock_gate(store=store)

        max_workers = 2
        active_count_tracker = {"max_concurrent": 0}
        active_lock = threading.Lock()
        active_counter = {"current": 0}

        def tracking_execute(*args, **kwargs):
            with active_lock:
                active_counter["current"] += 1
                if active_counter["current"] > active_count_tracker["max_concurrent"]:
                    active_count_tracker["max_concurrent"] = active_counter["current"]
            try:
                time.sleep(1.0)  # Hold the slot long enough to overlap
                return {"choices": [{"message": {"content": "ok"}}]}
            finally:
                with active_lock:
                    active_counter["current"] -= 1

        gate.pipeline.execute_sync.side_effect = tracking_execute

        # Submit more jobs than max_workers
        for _ in range(5):
            store.save_job(_make_pending_job())

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=max_workers)
        try:
            processor.start()
            time.sleep(3.0)  # Let jobs start executing
        finally:
            processor.shutdown(drain_timeout=10.0)

        # The max concurrent executions should not exceed max_workers
        assert active_count_tracker["max_concurrent"] <= max_workers
