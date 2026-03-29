"""OIDC provider configuration model."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _new_oidc_id() -> str:
    return "oidc-" + secrets.token_hex(6)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OIDCProvider(BaseModel):
    """An OIDC identity provider configuration."""

    id: str = Field(default_factory=_new_oidc_id)
    name: str = ""
    issuer_url: str
    client_id: str
    client_secret_encrypted: str = ""
    scopes: str = "openid email profile"
    group_claim: str = ""
    group_role_mapping: dict[str, dict[str, str]] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
