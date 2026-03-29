"""Tests for the Job dataclass and helpers."""

from stateloom.core.job import Job, _new_job_id
from stateloom.core.types import JobStatus

# --- _new_job_id uniqueness ---


def test_new_job_id_starts_with_prefix():
    jid = _new_job_id()
    assert jid.startswith("job_")


def test_new_job_id_length():
    jid = _new_job_id()
    # "job_" (4 chars) + 12 hex chars = 16 total
    assert len(jid) == 16


def test_new_job_id_uniqueness():
    ids = [_new_job_id() for _ in range(200)]
    assert len(set(ids)) == 200


# --- Job creation with defaults ---


def test_job_default_id_has_prefix():
    job = Job()
    assert job.id.startswith("job_")


def test_job_default_status_is_pending():
    job = Job()
    assert job.status == JobStatus.PENDING


def test_job_default_string_fields_are_empty():
    job = Job()
    assert job.session_id == ""
    assert job.org_id == ""
    assert job.team_id == ""
    assert job.provider == ""
    assert job.model == ""
    assert job.webhook_url == ""
    assert job.webhook_secret == ""
    assert job.error == ""
    assert job.error_code == ""
    assert job.webhook_status == ""
    assert job.webhook_last_error == ""


def test_job_default_collection_fields_are_empty():
    job = Job()
    assert job.messages == []
    assert job.request_kwargs == {}
    assert job.metadata == {}


def test_job_default_nullable_fields_are_none():
    job = Job()
    assert job.result is None
    assert job.started_at is None
    assert job.completed_at is None


def test_job_default_numeric_fields():
    job = Job()
    assert job.retry_count == 0
    assert job.max_retries == 3
    assert job.ttl_seconds == 3600
    assert job.webhook_attempts == 0


def test_job_default_created_at_is_set():
    job = Job()
    assert job.created_at is not None


# --- JobStatus serialization ---


def test_job_status_serializes_as_string():
    assert JobStatus.PENDING == "pending"
    assert JobStatus.RUNNING == "running"
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.FAILED == "failed"
    assert JobStatus.CANCELLED == "cancelled"


def test_job_status_from_string():
    assert JobStatus("pending") == JobStatus.PENDING
    assert JobStatus("running") == JobStatus.RUNNING
    assert JobStatus("completed") == JobStatus.COMPLETED
    assert JobStatus("failed") == JobStatus.FAILED
    assert JobStatus("cancelled") == JobStatus.CANCELLED


def test_job_status_is_str_subclass():
    # (str, Enum) pattern means each member is an instance of str
    for member in JobStatus:
        assert isinstance(member, str)


# --- Job with all fields set ---


def test_job_with_all_custom_fields():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    job = Job(
        id="job_custom12345",
        session_id="sess-1",
        org_id="org-1",
        team_id="team-1",
        status=JobStatus.RUNNING,
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "hello"}],
        request_kwargs={"temperature": 0.7},
        webhook_url="https://example.com/hook",
        webhook_secret="secret-abc",
        result={"text": "response"},
        error="some error",
        error_code="TIMEOUT",
        created_at=now,
        started_at=now,
        completed_at=now,
        retry_count=2,
        max_retries=5,
        ttl_seconds=7200,
        metadata={"key": "value"},
        webhook_status="delivered",
        webhook_attempts=3,
        webhook_last_error="connection refused",
    )
    assert job.id == "job_custom12345"
    assert job.session_id == "sess-1"
    assert job.org_id == "org-1"
    assert job.team_id == "team-1"
    assert job.status == JobStatus.RUNNING
    assert job.provider == "anthropic"
    assert job.model == "claude-sonnet-4-20250514"
    assert job.messages == [{"role": "user", "content": "hello"}]
    assert job.request_kwargs == {"temperature": 0.7}
    assert job.webhook_url == "https://example.com/hook"
    assert job.webhook_secret == "secret-abc"
    assert job.result == {"text": "response"}
    assert job.error == "some error"
    assert job.error_code == "TIMEOUT"
    assert job.created_at == now
    assert job.started_at == now
    assert job.completed_at == now
    assert job.retry_count == 2
    assert job.max_retries == 5
    assert job.ttl_seconds == 7200
    assert job.metadata == {"key": "value"}
    assert job.webhook_status == "delivered"
    assert job.webhook_attempts == 3
    assert job.webhook_last_error == "connection refused"


# --- Job status transitions ---


def test_job_status_pending_to_running():
    job = Job()
    assert job.status == JobStatus.PENDING
    job.status = JobStatus.RUNNING
    assert job.status == JobStatus.RUNNING


def test_job_status_running_to_completed():
    job = Job(status=JobStatus.RUNNING)
    job.status = JobStatus.COMPLETED
    assert job.status == JobStatus.COMPLETED


def test_job_status_running_to_failed():
    job = Job(status=JobStatus.RUNNING)
    job.status = JobStatus.FAILED
    assert job.status == JobStatus.FAILED


def test_job_status_full_lifecycle():
    job = Job()
    assert job.status == JobStatus.PENDING

    job.status = JobStatus.RUNNING
    assert job.status == JobStatus.RUNNING

    job.status = JobStatus.COMPLETED
    assert job.status == JobStatus.COMPLETED


def test_job_status_pending_to_cancelled():
    job = Job()
    job.status = JobStatus.CANCELLED
    assert job.status == JobStatus.CANCELLED


# --- Mutable default independence ---


def test_independent_messages_lists():
    """Ensure mutable defaults are independent between instances."""
    job1 = Job()
    job2 = Job()
    job1.messages.append({"role": "user", "content": "hello"})
    assert len(job2.messages) == 0


def test_independent_metadata_dicts():
    """Ensure metadata dicts are independent between instances."""
    job1 = Job()
    job2 = Job()
    job1.metadata["key"] = "value"
    assert "key" not in job2.metadata


def test_independent_request_kwargs_dicts():
    """Ensure request_kwargs dicts are independent between instances."""
    job1 = Job()
    job2 = Job()
    job1.request_kwargs["temperature"] = 0.5
    assert "temperature" not in job2.request_kwargs
