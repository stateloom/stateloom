"""Core data models for the experiment framework."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from stateloom.core.types import AssignmentStrategy, ExperimentStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class VariantConfig(BaseModel):
    """Configuration overrides for one experiment variant."""

    model_config = ConfigDict(extra="ignore")

    name: str
    weight: float = 1.0
    model: str | None = None
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_version_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VariantConfig:
        return cls.model_validate(data)


class Experiment(BaseModel):
    """A named experiment with variants and lifecycle management."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    id: str = Field(default_factory=_new_id)
    name: str = ""
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    strategy: AssignmentStrategy = AssignmentStrategy.RANDOM
    variants: list[VariantConfig] = Field(default_factory=list)
    assignment_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def start(self) -> None:
        """Transition to RUNNING status."""
        with self._lock:
            if self.status not in (ExperimentStatus.DRAFT, ExperimentStatus.PAUSED):
                raise ValueError(f"Cannot start experiment in '{self.status.value}' status")
            self.status = ExperimentStatus.RUNNING
            self.updated_at = _utcnow()

    def pause(self) -> None:
        """Transition to PAUSED status."""
        with self._lock:
            if self.status != ExperimentStatus.RUNNING:
                raise ValueError(f"Cannot pause experiment in '{self.status.value}' status")
            self.status = ExperimentStatus.PAUSED
            self.updated_at = _utcnow()

    def conclude(self) -> None:
        """Transition to CONCLUDED status."""
        with self._lock:
            if self.status not in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
                raise ValueError(f"Cannot conclude experiment in '{self.status.value}' status")
            self.status = ExperimentStatus.CONCLUDED
            self.updated_at = _utcnow()

    def record_assignment(self, variant_name: str) -> None:
        """Increment assignment count for a variant (thread-safe)."""
        with self._lock:
            self.assignment_counts[variant_name] = self.assignment_counts.get(variant_name, 0) + 1
            self.updated_at = _utcnow()


class ExperimentAssignment(BaseModel):
    """Links a session to an experiment variant.

    Stores a full snapshot of the VariantConfig at assignment time so that
    later changes to the experiment don't affect existing sessions.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: str
    experiment_id: str
    variant_name: str
    variant_config: dict[str, Any] = Field(default_factory=dict)
    assigned_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def create(
        cls,
        session_id: str,
        experiment_id: str,
        variant: VariantConfig,
    ) -> ExperimentAssignment:
        return cls(
            session_id=session_id,
            experiment_id=experiment_id,
            variant_name=variant.name,
            variant_config=variant.to_dict(),
        )


class SessionFeedback(BaseModel):
    """User feedback for a session."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    rating: str
    score: float | None = None
    comment: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
