"""Pluggable job queue abstraction.

The default InProcessJobQueue polls SQLite/MemoryStore. For production
at scale, implement the JobQueue protocol with Kafka, Redis Streams, SQS,
RabbitMQ, or any message broker — the JobProcessor doesn't care where
jobs come from, only that it can dequeue and acknowledge them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from stateloom.core.job import Job
from stateloom.core.types import JobStatus

if TYPE_CHECKING:
    from stateloom.store.base import Store

logger = logging.getLogger("stateloom.jobs.queue")


@runtime_checkable
class JobQueue(Protocol):
    """Pluggable job queue abstraction.

    Example future implementations:
        - KafkaJobQueue: consume from a Kafka topic, produce on enqueue
        - RedisJobQueue: use Redis Streams (XADD/XREADGROUP/XACK)
        - SQSJobQueue: AWS SQS with visibility timeout as acknowledgment
    """

    def enqueue(self, job: Job) -> None:
        """Add a job to the queue. Called by submit_job()."""
        ...

    def dequeue(self, max_count: int) -> list[Job]:
        """Fetch up to max_count ready-to-process jobs.

        Implementations should return jobs in FIFO order.
        For broker-backed queues, this is the consumer poll/receive.
        """
        ...

    def mark_running(self, job: Job) -> None:
        """Acknowledge that a job has been picked up and is now executing.

        For broker-backed queues, this is the ACK/commit.
        """
        ...

    def mark_completed(self, job: Job) -> None:
        """Record job completion with result."""
        ...

    def mark_failed(self, job: Job) -> None:
        """Record job failure."""
        ...

    def mark_dead(self, job: Job) -> None:
        """Move a terminally failed job to the dead letter queue."""
        ...

    def requeue(self, job: Job) -> None:
        """Put a failed job back on the queue for retry.

        If the job has exhausted all retries, moves it to the DLQ.
        """
        ...

    def list_dead(self, limit: int = 100) -> list[Job]:
        """List jobs in the dead letter queue."""
        ...

    def requeue_dead(self, job_id: str) -> bool:
        """Manually requeue a dead job for retry."""
        ...

    def recover_stale(self) -> int:
        """Reset stuck RUNNING jobs to PENDING on startup (crash recovery).

        Returns the number of jobs recovered.
        """
        ...

    def shutdown(self) -> None:
        """Clean up queue resources (close connections, etc.)."""
        ...


class InProcessJobQueue:
    """Default job queue backed by the StateLoom Store (SQLite/Memory).

    Keeps the existing polling behavior — dequeue queries the store for
    PENDING jobs, and state transitions are persisted via store.save_job().
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def enqueue(self, job: Job) -> None:
        job.status = JobStatus.PENDING
        self._store.save_job(job)

    def dequeue(self, max_count: int) -> list[Job]:
        jobs = self._store.list_jobs(status="pending", limit=max_count * 3)
        now = datetime.now(timezone.utc)

        # Filter out expired jobs (TTL enforcement)
        valid: list[Job] = []
        for job in jobs:
            if job.ttl_seconds > 0:
                age = (now - job.created_at).total_seconds()
                if age > job.ttl_seconds:
                    job.status = JobStatus.FAILED
                    job.error = "TTL expired"
                    job.completed_at = now
                    self._store.save_job(job)
                    logger.info("Job %s expired (TTL %ds)", job.id, job.ttl_seconds)
                    continue
            valid.append(job)

        # Sort by priority DESC (highest first), then created_at ASC (FIFO)
        valid.sort(key=lambda j: (-j.priority, j.created_at))

        return valid[:max_count]

    def mark_running(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        self._store.save_job(job)

    def mark_completed(self, job: Job) -> None:
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        self._store.save_job(job)

    def mark_failed(self, job: Job) -> None:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        self._store.save_job(job)

    def mark_dead(self, job: Job) -> None:
        """Move a terminally failed job to the dead letter queue."""
        job.status = JobStatus.DEAD
        job.completed_at = datetime.now(timezone.utc)
        self._store.save_job(job)
        logger.warning("Job %s moved to DLQ after %d retries", job.id, job.retry_count)

    def requeue(self, job: Job) -> None:
        job.retry_count += 1
        if job.retry_count >= job.max_retries:
            self.mark_dead(job)
            return
        job.status = JobStatus.PENDING
        job.started_at = None
        self._store.save_job(job)

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
        job.error = ""
        self._store.save_job(job)
        logger.info("Requeued dead job %s", job_id)
        return True

    def recover_stale(self) -> int:
        try:
            running_jobs = self._store.list_jobs(status="running", limit=1000)
            for job in running_jobs:
                job.status = JobStatus.PENDING
                job.started_at = None
                self._store.save_job(job)
                logger.info("Recovered stale job %s -> pending", job.id)
            return len(running_jobs)
        except Exception:
            logger.debug("Failed to recover stale jobs", exc_info=True)
            return 0

    def shutdown(self) -> None:
        pass
