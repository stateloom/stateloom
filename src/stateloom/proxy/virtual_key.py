"""Virtual API key model for proxy authentication."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VirtualKey(BaseModel):
    """A virtual API key that maps to a team/org for proxy authentication."""

    model_config = ConfigDict(extra="ignore")

    id: str
    key_hash: str
    key_preview: str
    team_id: str
    org_id: str
    name: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    revoked: bool = False
    scopes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Per-key model access control
    allowed_models: list[str] = Field(default_factory=list)  # glob patterns

    # Per-key budget cap
    budget_limit: float | None = None
    budget_spent: float = Field(default=0.0, ge=0)

    # Per-key rate limiting (overrides team limits when set)
    rate_limit_tps: float | None = None
    rate_limit_max_queue: int = Field(default=100, ge=0)
    rate_limit_queue_timeout: float = Field(default=30.0, ge=0)

    # Agent access control
    agent_ids: list[str] = Field(default_factory=list)  # if non-empty, restricts to these agents

    # Billing mode: "api" (per-token cost) or "subscription" (flat-rate, cost=$0)
    billing_mode: str = "api"


def generate_virtual_key(prefix: str = "ag") -> tuple[str, str]:
    """Generate a virtual API key.

    Returns:
        (full_key, sha256_hash). Key format: {prefix}-{random_hex}.
    """
    random_part = secrets.token_hex(24)
    full_key = f"{prefix}-{random_part}"
    key_hash = hash_key(full_key)
    return full_key, key_hash


def hash_key(key: str) -> str:
    """SHA256 hash of a key."""
    return hashlib.sha256(key.encode()).hexdigest()


def make_key_preview(key: str) -> str:
    """Create a safe preview of a key: first 6 chars...last 4 chars."""
    if len(key) <= 12:
        return key[:4] + "..."
    return key[:6] + "..." + key[-4:]


def make_virtual_key_id() -> str:
    """Generate a virtual key ID."""
    return "vk-" + uuid.uuid4().hex[:12]
