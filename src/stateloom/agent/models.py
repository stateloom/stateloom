"""Agent and AgentVersion models."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stateloom.core.types import AgentStatus

# Slug format: 3–64 chars, lowercase alphanumeric + internal hyphens,
# no leading/trailing hyphen.  The 1,62 repetition count in the middle
# group allows total length 3–64 (1 lead + 1–62 middle + 1 trail).
_SLUG_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$")


def validate_slug(slug: str) -> bool:
    """Validate an agent slug.

    Args:
        slug: The slug string to validate.

    Returns:
        True if the slug matches the required format (3–64 characters,
        lowercase alphanumeric with internal hyphens, no leading/trailing
        hyphen), False otherwise.
    """
    return bool(_SLUG_PATTERN.match(slug))


def _make_agent_id() -> str:
    return "agt-" + uuid.uuid4().hex[:12]


def _make_agent_version_id() -> str:
    return "agv-" + uuid.uuid4().hex[:12]


class AgentVersion(BaseModel):
    """An immutable snapshot of an agent's configuration.

    Versions are never mutated after creation.  "Updating" an agent means
    creating a new ``AgentVersion`` with an auto-incremented
    ``version_number``.  Rolling back means pointing
    ``Agent.active_version_id`` at an older version.

    ``request_overrides`` is a dict of extra kwargs merged into every
    request routed through this agent version.  Agent-side values win on
    key conflicts with client-supplied kwargs.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_make_agent_version_id)
    agent_id: str = ""
    # Auto-incremented by store.get_next_version_number(agent_id).
    version_number: int = Field(default=1, ge=1)
    model: str = ""
    system_prompt: str = ""
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    compliance_profile_json: str = ""
    budget_per_session: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""


class Agent(BaseModel):
    """A managed agent definition, scoped to a team.

    Lifecycle states (``AgentStatus``):

    - ``ACTIVE`` — serving requests normally.
    - ``PAUSED`` — temporarily disabled; proxy returns HTTP 403.
    - ``ARCHIVED`` — soft-deleted; proxy returns HTTP 410.

    Slug uniqueness is enforced within a team (not globally).
    Virtual keys can restrict access to specific agents via the
    ``VirtualKey.agent_ids`` list; an empty list means unrestricted.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_make_agent_id)
    slug: str = ""
    team_id: str = ""
    org_id: str = ""
    name: str = ""
    description: str = ""
    # See AgentStatus for HTTP response codes per state.
    status: AgentStatus = AgentStatus.ACTIVE
    # Points to the live AgentVersion; set via activate_agent_version().
    active_version_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
