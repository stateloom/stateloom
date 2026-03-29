"""Job model for async job-based API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stateloom.core.types import JobStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:12]


class Job(BaseModel):
    """Represents an async LLM job submitted via the job-based API."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_job_id)
    session_id: str = ""
    org_id: str = ""
    team_id: str = ""
    status: JobStatus = JobStatus.PENDING

    # Request params
    provider: str = ""
    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    request_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Webhook
    webhook_url: str = ""
    webhook_secret: str = ""

    # Result / Error
    result: dict[str, Any] | None = None
    error: str = ""
    error_code: str = ""

    # Timestamps
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Priority (higher = processed first)
    priority: int = Field(default=0, ge=0)

    # Retry
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)

    # TTL
    ttl_seconds: int = Field(default=3600, ge=0)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Webhook delivery tracking
    webhook_status: str = ""
    webhook_attempts: int = Field(default=0, ge=0)
    webhook_last_error: str = ""
