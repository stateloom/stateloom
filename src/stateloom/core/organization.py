"""Organization and Team models for multi-tenant hierarchy."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from stateloom.core.config import ComplianceProfile, PIIRule
from stateloom.core.types import OrgStatus, TeamStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_org_id() -> str:
    return "org-" + uuid.uuid4().hex[:10]


def _new_team_id() -> str:
    return "team-" + uuid.uuid4().hex[:10]


class Organization(BaseModel):
    """An organization — the top-level billing unit in the hierarchy."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    id: str = Field(default_factory=_new_org_id)
    name: str = ""
    status: OrgStatus = OrgStatus.ACTIVE
    created_at: datetime = Field(default_factory=_utcnow)

    # Budget
    budget: float | None = None

    # Cost accumulators (updated via callbacks)
    total_cost: float = Field(default=0.0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    # PII rules (org-level floor)
    pii_rules: list[PIIRule] = Field(default_factory=list)

    # Compliance
    compliance_profile: ComplianceProfile | None = None

    # Arbitrary metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Thread safety for accumulators
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def add_cost(self, cost: float, tokens: int = 0) -> None:
        """Add cost and token counts to org accumulators (thread-safe)."""
        with self._lock:
            self.total_cost += cost
            self.total_tokens += tokens


class Team(BaseModel):
    """A team — groups sessions by project/team within an organization."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    id: str = Field(default_factory=_new_team_id)
    org_id: str = ""
    name: str = ""
    status: TeamStatus = TeamStatus.ACTIVE
    created_at: datetime = Field(default_factory=_utcnow)

    # Budget
    budget: float | None = None

    # Cost accumulators
    total_cost: float = Field(default=0.0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    # Compliance
    compliance_profile: ComplianceProfile | None = None

    # Arbitrary metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Rate limiting (configured via dashboard, not config file)
    rate_limit_tps: float | None = None
    rate_limit_priority: int = Field(default=0, ge=0)
    rate_limit_max_queue: int = Field(default=100, ge=0)
    rate_limit_queue_timeout: float = Field(default=30.0, ge=0)

    # Thread safety for accumulators
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def add_cost(self, cost: float, tokens: int = 0) -> None:
        """Add cost and token counts to team accumulators (thread-safe)."""
        with self._lock:
            self.total_cost += cost
            self.total_tokens += tokens
