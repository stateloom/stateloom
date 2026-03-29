"""Tests for Job persistence in MemoryStore and SQLiteStore."""

from datetime import datetime, timedelta, timezone

from stateloom.core.job import Job
from stateloom.core.types import JobStatus
from stateloom.store.memory_store import MemoryStore
from stateloom.store.sqlite_store import SQLiteStore

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_job(**overrides) -> Job:
    defaults = dict(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "hello"}],
        session_id="sess_1",
    )
    defaults.update(overrides)
    return Job(**defaults)


# ===========================================================================
# MemoryStore tests
# ===========================================================================


def test_memory_save_and_get_job():
    store = MemoryStore()
    job = _make_job()
    store.save_job(job)

    retrieved = store.get_job(job.id)
    assert retrieved is not None
    assert retrieved.id == job.id
    assert retrieved.provider == "openai"
    assert retrieved.model == "gpt-4"
    assert retrieved.session_id == "sess_1"
    assert retrieved.status == JobStatus.PENDING
    assert retrieved.messages == [{"role": "user", "content": "hello"}]


def test_memory_get_nonexistent_job():
    store = MemoryStore()
    assert store.get_job("nonexistent_id") is None


def test_memory_list_jobs():
    store = MemoryStore()
    job1 = _make_job(model="gpt-4")
    job2 = _make_job(model="gpt-3.5-turbo")
    job3 = _make_job(model="claude-3-opus")
    store.save_job(job1)
    store.save_job(job2)
    store.save_job(job3)

    jobs = store.list_jobs()
    assert len(jobs) == 3
    ids = {j.id for j in jobs}
    assert job1.id in ids
    assert job2.id in ids
    assert job3.id in ids


def test_memory_list_jobs_by_status():
    store = MemoryStore()
    job_pending = _make_job()
    job_running = _make_job()
    job_running.status = JobStatus.RUNNING
    job_completed = _make_job()
    job_completed.status = JobStatus.COMPLETED

    store.save_job(job_pending)
    store.save_job(job_running)
    store.save_job(job_completed)

    pending = store.list_jobs(status="pending")
    assert len(pending) == 1
    assert pending[0].id == job_pending.id

    running = store.list_jobs(status="running")
    assert len(running) == 1
    assert running[0].id == job_running.id

    completed = store.list_jobs(status="completed")
    assert len(completed) == 1
    assert completed[0].id == job_completed.id


def test_memory_list_jobs_by_session():
    store = MemoryStore()
    job_a = _make_job(session_id="sess_a")
    job_b = _make_job(session_id="sess_b")
    job_a2 = _make_job(session_id="sess_a")

    store.save_job(job_a)
    store.save_job(job_b)
    store.save_job(job_a2)

    sess_a_jobs = store.list_jobs(session_id="sess_a")
    assert len(sess_a_jobs) == 2
    assert all(j.session_id == "sess_a" for j in sess_a_jobs)

    sess_b_jobs = store.list_jobs(session_id="sess_b")
    assert len(sess_b_jobs) == 1
    assert sess_b_jobs[0].id == job_b.id


def test_memory_delete_job():
    store = MemoryStore()
    job = _make_job()
    store.save_job(job)

    assert store.delete_job(job.id) is True
    assert store.get_job(job.id) is None


def test_memory_delete_nonexistent():
    store = MemoryStore()
    assert store.delete_job("does_not_exist") is False


def test_memory_job_stats():
    store = MemoryStore()
    now = datetime.now(timezone.utc)

    job1 = _make_job()
    job1.status = JobStatus.COMPLETED
    job1.started_at = now - timedelta(seconds=2)
    job1.completed_at = now

    job2 = _make_job()
    job2.status = JobStatus.COMPLETED
    job2.started_at = now - timedelta(seconds=4)
    job2.completed_at = now

    job3 = _make_job()
    job3.status = JobStatus.FAILED

    store.save_job(job1)
    store.save_job(job2)
    store.save_job(job3)

    stats = store.get_job_stats()
    assert stats["total"] == 3
    assert stats["by_status"]["completed"] == 2
    assert stats["by_status"]["failed"] == 1
    # avg of 2000ms and 4000ms = 3000ms
    assert abs(stats["avg_processing_time_ms"] - 3000.0) < 50


# ===========================================================================
# SQLiteStore tests
# ===========================================================================


def test_sqlite_save_and_get_job(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    job = _make_job()
    store.save_job(job)

    retrieved = store.get_job(job.id)
    assert retrieved is not None
    assert retrieved.id == job.id
    assert retrieved.provider == "openai"
    assert retrieved.model == "gpt-4"
    assert retrieved.session_id == "sess_1"
    assert retrieved.status == JobStatus.PENDING
    assert retrieved.messages == [{"role": "user", "content": "hello"}]


def test_sqlite_list_jobs_by_status(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))

    job_pending = _make_job()
    job_running = _make_job()
    job_running.status = JobStatus.RUNNING
    job_completed = _make_job()
    job_completed.status = JobStatus.COMPLETED

    store.save_job(job_pending)
    store.save_job(job_running)
    store.save_job(job_completed)

    pending = store.list_jobs(status="pending")
    assert len(pending) == 1
    assert pending[0].id == job_pending.id

    running = store.list_jobs(status="running")
    assert len(running) == 1
    assert running[0].id == job_running.id

    completed = store.list_jobs(status="completed")
    assert len(completed) == 1
    assert completed[0].id == job_completed.id


def test_sqlite_delete_job(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    job = _make_job()
    store.save_job(job)

    assert store.delete_job(job.id) is True
    assert store.get_job(job.id) is None
    # Deleting again should return False
    assert store.delete_job(job.id) is False


def test_sqlite_job_stats(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    now = datetime.now(timezone.utc)

    job1 = _make_job()
    job1.status = JobStatus.COMPLETED
    job1.started_at = now - timedelta(seconds=2)
    job1.completed_at = now

    job2 = _make_job()
    job2.status = JobStatus.COMPLETED
    job2.started_at = now - timedelta(seconds=4)
    job2.completed_at = now

    job3 = _make_job()
    job3.status = JobStatus.FAILED

    store.save_job(job1)
    store.save_job(job2)
    store.save_job(job3)

    stats = store.get_job_stats()
    assert stats["total"] == 3
    assert stats["by_status"]["completed"] == 2
    assert stats["by_status"]["failed"] == 1
    # avg of 2000ms and 4000ms = 3000ms
    assert abs(stats["avg_processing_time_ms"] - 3000.0) < 50
