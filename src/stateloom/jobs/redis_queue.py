"""Redis Streams-backed job queue for production deployments.

Uses Redis Streams (XADD/XREADGROUP/XACK) for queue semantics with
consumer groups and crash recovery. The Store is used for full job
persistence and dashboard queries — Redis handles ordering and delivery.

Requires the ``redis`` package: ``pip install redis``
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from stateloom.core.job import Job
from stateloom.core.types import JobStatus

if TYPE_CHECKING:
    from stateloom.store.base import Store

logger = logging.getLogger("stateloom.jobs.redis_queue")

_STREAM_KEY = "stateloom:jobs"
_GROUP_NAME = "stateloom-workers"
_STALE_IDLE_MS = 300_000  # 5 minutes


def _consumer_name() -> str:
    """Generate a unique consumer name from hostname + PID."""
    hostname = os.uname().nodename
    pid = os.getpid()
    return f"{hostname}-{pid}"


class RedisJobQueue:
    """Redis Streams-backed job queue.

    Dual persistence: Redis for queue ordering/delivery, Store for
    durability and dashboard queries.
    """

    def __init__(self, store: Store, url: str = "redis://localhost:6379") -> None:
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis package is required for RedisJobQueue. "
                "Install with: pip install stateloom[redis]"
            )

        self._store = store
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._consumer = _consumer_name()

        # Message ID tracking: job_id -> stream message ID
        self._msg_ids: dict[str, str] = {}

        # Ensure consumer group exists
        self._ensure_group()

    def _ensure_group(self) -> None:
        """Create the consumer group if it doesn't exist."""
        try:
            self._redis.xgroup_create(_STREAM_KEY, _GROUP_NAME, id="0", mkstream=True)
        except Exception as e:
            # BUSYGROUP = group already exists, which is fine
            if "BUSYGROUP" not in str(e):
                raise

    def enqueue(self, job: Job) -> None:
        """Add a job to the queue."""
        job.status = JobStatus.PENDING
        self._store.save_job(job)
        self._redis.xadd(_STREAM_KEY, {"job_id": job.id})

    def dequeue(self, max_count: int) -> list[Job]:
        """Fetch up to max_count ready-to-process jobs from the stream."""
        results = self._redis.xreadgroup(
            _GROUP_NAME,
            self._consumer,
            {_STREAM_KEY: ">"},
            count=max_count,
            block=100,
        )

        jobs: list[Job] = []
        if not results:
            return jobs

        for _stream_name, messages in results:
            for msg_id, fields in messages:
                job_id = fields.get("job_id", "")
                if not job_id:
                    # ACK and skip malformed entries
                    self._redis.xack(_STREAM_KEY, _GROUP_NAME, msg_id)
                    continue

                job = self._store.get_job(job_id)
                if job is None or job.status != JobStatus.PENDING:
                    # Job was cancelled or already processed — ACK and skip
                    self._redis.xack(_STREAM_KEY, _GROUP_NAME, msg_id)
                    continue

                self._msg_ids[job_id] = msg_id
                jobs.append(job)

        return jobs

    def mark_running(self, job: Job) -> None:
        """ACK the stream message and update job status to RUNNING."""
        msg_id = self._msg_ids.pop(job.id, None)
        if msg_id:
            self._redis.xack(_STREAM_KEY, _GROUP_NAME, msg_id)

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        self._store.save_job(job)

    def mark_completed(self, job: Job) -> None:
        """Record job completion in the store."""
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        self._store.save_job(job)

    def mark_failed(self, job: Job) -> None:
        """Record job failure in the store."""
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        self._store.save_job(job)

    def mark_dead(self, job: Job) -> None:
        """Move a terminally failed job to the dead letter queue."""
        job.status = JobStatus.DEAD
        job.completed_at = datetime.now(timezone.utc)
        self._store.save_job(job)

    def requeue(self, job: Job) -> None:
        """Put a failed job back on the queue for retry.

        If the job has exhausted retries, moves it to the DLQ instead.
        """
        job.retry_count += 1
        if job.retry_count >= job.max_retries:
            self.mark_dead(job)
            return
        job.status = JobStatus.PENDING
        job.started_at = None
        self._store.save_job(job)
        # Fresh stream entry for the retry
        self._redis.xadd(_STREAM_KEY, {"job_id": job.id})

    def list_dead(self, limit: int = 100) -> list[Job]:
        """List jobs in the dead letter queue."""
        return self._store.list_jobs(status="dead", limit=limit)

    def requeue_dead(self, job_id: str) -> bool:
        """Manually requeue a dead job for retry."""
        job = self._store.get_job(job_id)
        if job is None or job.status != JobStatus.DEAD:
            return False
        job.status = JobStatus.PENDING
        job.retry_count = 0
        job.started_at = None
        job.completed_at = None
        job.error = None
        self._store.save_job(job)
        self._redis.xadd(_STREAM_KEY, {"job_id": job.id})
        return True

    def recover_stale(self) -> int:
        """Recover stale jobs that have been pending in the consumer group too long.

        Uses XPENDING + XCLAIM to reclaim messages idle > 5 minutes,
        then resets their status to PENDING in the store.
        """
        recovered = 0
        try:
            # Get pending entries across all consumers
            pending = self._redis.xpending_range(
                _STREAM_KEY, _GROUP_NAME, min="-", max="+", count=100
            )
            if not pending:
                return 0

            stale_ids = [
                entry["message_id"]
                for entry in pending
                if entry.get("time_since_delivered", 0) >= _STALE_IDLE_MS
            ]

            if not stale_ids:
                return 0

            # Claim stale messages
            claimed = self._redis.xclaim(
                _STREAM_KEY,
                _GROUP_NAME,
                self._consumer,
                min_idle_time=_STALE_IDLE_MS,
                message_ids=stale_ids,
            )

            for msg_id, fields in claimed:
                job_id = fields.get("job_id", "")
                if not job_id:
                    self._redis.xack(_STREAM_KEY, _GROUP_NAME, msg_id)
                    continue

                job = self._store.get_job(job_id)
                if job is None:
                    self._redis.xack(_STREAM_KEY, _GROUP_NAME, msg_id)
                    continue

                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.PENDING
                    job.started_at = None
                    self._store.save_job(job)
                    recovered += 1
                    logger.info("Recovered stale job %s -> pending", job.id)

                # ACK the old message and re-enqueue
                self._redis.xack(_STREAM_KEY, _GROUP_NAME, msg_id)
                if job.status == JobStatus.PENDING:
                    self._redis.xadd(_STREAM_KEY, {"job_id": job.id})

        except Exception:
            logger.debug("Failed to recover stale jobs", exc_info=True)

        return recovered

    def shutdown(self) -> None:
        """Close the Redis connection."""
        try:
            self._redis.close()
        except Exception:
            logger.debug("Error closing Redis connection", exc_info=True)
