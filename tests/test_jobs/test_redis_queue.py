"""Tests for RedisJobQueue with mocked Redis client."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.job import Job
from stateloom.core.types import JobStatus
from stateloom.jobs.queue import JobQueue


@pytest.fixture()
def mock_store():
    """Create a mock Store."""
    store = MagicMock()
    return store


@pytest.fixture()
def mock_redis_client():
    """Create a mock Redis client."""
    client = MagicMock()
    client.xgroup_create = MagicMock()
    client.xadd = MagicMock(return_value="1234567890-0")
    client.xreadgroup = MagicMock(return_value=[])
    client.xack = MagicMock(return_value=1)
    client.xpending_range = MagicMock(return_value=[])
    client.xclaim = MagicMock(return_value=[])
    client.close = MagicMock()
    return client


@pytest.fixture()
def redis_queue(mock_store, mock_redis_client):
    """Create a RedisJobQueue with mocked Redis."""
    redis_mod = MagicMock()
    redis_mod.Redis.from_url.return_value = mock_redis_client

    with patch.dict(sys.modules, {"redis": redis_mod}):
        from stateloom.jobs.redis_queue import RedisJobQueue

        queue = RedisJobQueue(mock_store, url="redis://localhost:6379")
    return queue


@pytest.fixture()
def sample_job():
    """Create a sample Job for testing."""
    return Job(
        id="job_test123",
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "hello"}],
    )


class TestRedisJobQueueProtocol:
    """Verify RedisJobQueue satisfies the JobQueue protocol."""

    def test_isinstance_check(self, redis_queue):
        assert isinstance(redis_queue, JobQueue)

    def test_has_all_protocol_methods(self, redis_queue):
        for method in (
            "enqueue",
            "dequeue",
            "mark_running",
            "mark_completed",
            "mark_failed",
            "requeue",
            "recover_stale",
            "shutdown",
        ):
            assert hasattr(redis_queue, method)
            assert callable(getattr(redis_queue, method))


class TestEnqueue:
    """Test enqueue adds to Redis stream and persists to store."""

    def test_enqueue_calls_xadd_and_save(
        self, redis_queue, mock_store, mock_redis_client, sample_job
    ):
        redis_queue.enqueue(sample_job)

        assert sample_job.status == JobStatus.PENDING
        mock_store.save_job.assert_called_once_with(sample_job)
        mock_redis_client.xadd.assert_called_once_with("stateloom:jobs", {"job_id": "job_test123"})


class TestDequeue:
    """Test dequeue reads from Redis stream and loads from store."""

    def test_dequeue_empty(self, redis_queue, mock_redis_client):
        mock_redis_client.xreadgroup.return_value = []
        jobs = redis_queue.dequeue(5)
        assert jobs == []

    def test_dequeue_returns_pending_jobs(
        self, redis_queue, mock_store, mock_redis_client, sample_job
    ):
        sample_job.status = JobStatus.PENDING
        mock_redis_client.xreadgroup.return_value = [
            ("stateloom:jobs", [("msg-1", {"job_id": "job_test123"})])
        ]
        mock_store.get_job.return_value = sample_job

        jobs = redis_queue.dequeue(5)

        assert len(jobs) == 1
        assert jobs[0].id == "job_test123"
        mock_redis_client.xreadgroup.assert_called_once()
        mock_store.get_job.assert_called_once_with("job_test123")

    def test_dequeue_skips_cancelled_jobs(self, redis_queue, mock_store, mock_redis_client):
        cancelled_job = Job(id="job_cancelled", status=JobStatus.CANCELLED)
        mock_redis_client.xreadgroup.return_value = [
            ("stateloom:jobs", [("msg-1", {"job_id": "job_cancelled"})])
        ]
        mock_store.get_job.return_value = cancelled_job

        jobs = redis_queue.dequeue(5)

        assert len(jobs) == 0
        # Should ACK the message for the cancelled job
        mock_redis_client.xack.assert_called_once()

    def test_dequeue_skips_missing_jobs(self, redis_queue, mock_store, mock_redis_client):
        mock_redis_client.xreadgroup.return_value = [
            ("stateloom:jobs", [("msg-1", {"job_id": "job_gone"})])
        ]
        mock_store.get_job.return_value = None

        jobs = redis_queue.dequeue(5)

        assert len(jobs) == 0
        mock_redis_client.xack.assert_called_once()

    def test_dequeue_skips_malformed_entries(self, redis_queue, mock_redis_client):
        mock_redis_client.xreadgroup.return_value = [
            ("stateloom:jobs", [("msg-1", {"bad_field": "value"})])
        ]

        jobs = redis_queue.dequeue(5)

        assert len(jobs) == 0
        mock_redis_client.xack.assert_called_once()

    def test_dequeue_none_result(self, redis_queue, mock_redis_client):
        mock_redis_client.xreadgroup.return_value = None
        jobs = redis_queue.dequeue(5)
        assert jobs == []


class TestMarkRunning:
    """Test mark_running ACKs the message and updates the store."""

    def test_mark_running_acks_and_updates(
        self, redis_queue, mock_store, mock_redis_client, sample_job
    ):
        # Simulate having dequeued this job first
        redis_queue._msg_ids["job_test123"] = "msg-1"

        redis_queue.mark_running(sample_job)

        mock_redis_client.xack.assert_called_once_with(
            "stateloom:jobs", "stateloom-workers", "msg-1"
        )
        assert sample_job.status == JobStatus.RUNNING
        assert sample_job.started_at is not None
        mock_store.save_job.assert_called_once_with(sample_job)

    def test_mark_running_without_msg_id(
        self, redis_queue, mock_store, mock_redis_client, sample_job
    ):
        # No msg_id tracked — should still update status
        redis_queue.mark_running(sample_job)

        mock_redis_client.xack.assert_not_called()
        assert sample_job.status == JobStatus.RUNNING
        mock_store.save_job.assert_called_once()


class TestMarkCompleted:
    """Test mark_completed updates store only (no Redis interaction)."""

    def test_mark_completed(self, redis_queue, mock_store, mock_redis_client, sample_job):
        redis_queue.mark_completed(sample_job)

        assert sample_job.status == JobStatus.COMPLETED
        assert sample_job.completed_at is not None
        mock_store.save_job.assert_called_once_with(sample_job)
        mock_redis_client.xadd.assert_not_called()
        mock_redis_client.xack.assert_not_called()


class TestMarkFailed:
    """Test mark_failed updates store only."""

    def test_mark_failed(self, redis_queue, mock_store, mock_redis_client, sample_job):
        redis_queue.mark_failed(sample_job)

        assert sample_job.status == JobStatus.FAILED
        assert sample_job.completed_at is not None
        mock_store.save_job.assert_called_once_with(sample_job)


class TestRequeue:
    """Test requeue increments retry count and re-adds to stream."""

    def test_requeue(self, redis_queue, mock_store, mock_redis_client, sample_job):
        sample_job.retry_count = 0
        redis_queue.requeue(sample_job)

        assert sample_job.retry_count == 1
        assert sample_job.status == JobStatus.PENDING
        assert sample_job.started_at is None
        mock_store.save_job.assert_called_once_with(sample_job)
        mock_redis_client.xadd.assert_called_once_with("stateloom:jobs", {"job_id": sample_job.id})


class TestRecoverStale:
    """Test stale job recovery via XPENDING + XCLAIM."""

    def test_recover_no_pending(self, redis_queue, mock_redis_client):
        mock_redis_client.xpending_range.return_value = []
        recovered = redis_queue.recover_stale()
        assert recovered == 0

    def test_recover_no_stale(self, redis_queue, mock_redis_client):
        mock_redis_client.xpending_range.return_value = [
            {"message_id": "msg-1", "time_since_delivered": 1000}  # 1 second, not stale
        ]
        recovered = redis_queue.recover_stale()
        assert recovered == 0
        mock_redis_client.xclaim.assert_not_called()

    def test_recover_stale_running_job(self, redis_queue, mock_store, mock_redis_client):
        stale_job = Job(id="job_stale", status=JobStatus.RUNNING)
        mock_redis_client.xpending_range.return_value = [
            {"message_id": "msg-stale", "time_since_delivered": 400_000}  # > 5 min
        ]
        mock_redis_client.xclaim.return_value = [("msg-stale", {"job_id": "job_stale"})]
        mock_store.get_job.return_value = stale_job

        recovered = redis_queue.recover_stale()

        assert recovered == 1
        assert stale_job.status == JobStatus.PENDING
        assert stale_job.started_at is None
        mock_store.save_job.assert_called()
        # Should ACK old message and add a fresh one
        mock_redis_client.xack.assert_called_once()
        mock_redis_client.xadd.assert_called_once()

    def test_recover_skips_missing_job(self, redis_queue, mock_store, mock_redis_client):
        mock_redis_client.xpending_range.return_value = [
            {"message_id": "msg-gone", "time_since_delivered": 400_000}
        ]
        mock_redis_client.xclaim.return_value = [("msg-gone", {"job_id": "job_gone"})]
        mock_store.get_job.return_value = None

        recovered = redis_queue.recover_stale()

        assert recovered == 0
        mock_redis_client.xack.assert_called_once()

    def test_recover_handles_exception(self, redis_queue, mock_redis_client):
        mock_redis_client.xpending_range.side_effect = Exception("Connection lost")
        recovered = redis_queue.recover_stale()
        assert recovered == 0


class TestShutdown:
    """Test shutdown closes Redis connection."""

    def test_shutdown_closes_connection(self, redis_queue, mock_redis_client):
        redis_queue.shutdown()
        mock_redis_client.close.assert_called_once()

    def test_shutdown_handles_error(self, redis_queue, mock_redis_client):
        mock_redis_client.close.side_effect = Exception("Already closed")
        # Should not raise
        redis_queue.shutdown()


class TestRedisUnavailable:
    """Test graceful failure when redis package is not installed."""

    def test_import_error_when_redis_missing(self, mock_store):
        # Remove redis from modules if present and block import
        with patch.dict(sys.modules, {"redis": None}):
            with pytest.raises(ImportError, match="Redis package is required"):
                from importlib import reload

                import stateloom.jobs.redis_queue as rq_mod

                reload(rq_mod)
                rq_mod.RedisJobQueue(mock_store)


class TestConsumerGroupCreation:
    """Test consumer group initialization."""

    def test_creates_group_on_init(self, mock_redis_client):
        redis_mod = MagicMock()
        redis_mod.Redis.from_url.return_value = mock_redis_client

        with patch.dict(sys.modules, {"redis": redis_mod}):
            from stateloom.jobs.redis_queue import RedisJobQueue

            store = MagicMock()
            RedisJobQueue(store, url="redis://localhost:6379")

        mock_redis_client.xgroup_create.assert_called_once_with(
            "stateloom:jobs", "stateloom-workers", id="0", mkstream=True
        )

    def test_ignores_busygroup_error(self, mock_redis_client):
        mock_redis_client.xgroup_create.side_effect = Exception(
            "BUSYGROUP Consumer Group name already exists"
        )
        redis_mod = MagicMock()
        redis_mod.Redis.from_url.return_value = mock_redis_client

        with patch.dict(sys.modules, {"redis": redis_mod}):
            from stateloom.jobs.redis_queue import RedisJobQueue

            store = MagicMock()
            # Should not raise
            RedisJobQueue(store, url="redis://localhost:6379")

    def test_raises_non_busygroup_error(self, mock_redis_client):
        mock_redis_client.xgroup_create.side_effect = Exception("Connection refused")
        redis_mod = MagicMock()
        redis_mod.Redis.from_url.return_value = mock_redis_client

        with patch.dict(sys.modules, {"redis": redis_mod}):
            from stateloom.jobs.redis_queue import RedisJobQueue

            store = MagicMock()
            with pytest.raises(Exception, match="Connection refused"):
                RedisJobQueue(store, url="redis://localhost:6379")
