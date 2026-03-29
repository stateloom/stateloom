"""User and role models for StateLoom authentication."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stateloom.core.types import Role


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_user_id() -> str:
    return "usr-" + uuid.uuid4().hex[:12]


def _new_utr_id() -> str:
    return "utr-" + uuid.uuid4().hex[:12]


class User(BaseModel):
    """A user account in the StateLoom system."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_user_id)
    email: str
    display_name: str = ""
    password_hash: str = ""
    email_verified: bool = False
    org_id: str = ""
    org_role: Role | None = None
    oidc_provider_id: str = ""
    oidc_subject: str = ""
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
    last_login: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserTeamRole(BaseModel):
    """A user's role assignment within a specific team."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_utr_id)
    user_id: str
    team_id: str
    role: Role
    granted_at: datetime = Field(default_factory=_utcnow)
    granted_by: str = ""
