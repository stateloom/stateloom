"""Role-permission mapping and authorization checks."""

from __future__ import annotations

from enum import Enum

from stateloom.core.types import Role


class Permission(str, Enum):
    SESSIONS_READ = "sessions:read"
    SESSIONS_CANCEL = "sessions:cancel"
    SESSIONS_EXPORT = "sessions:export"
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"
    AGENTS_READ = "agents:read"
    AGENTS_WRITE = "agents:write"
    AGENTS_DELETE = "agents:delete"
    EXPERIMENTS_READ = "experiments:read"
    EXPERIMENTS_WRITE = "experiments:write"
    VIRTUAL_KEYS_READ = "virtual_keys:read"
    VIRTUAL_KEYS_WRITE = "virtual_keys:write"
    VIRTUAL_KEYS_REVOKE = "virtual_keys:revoke"
    ORGS_READ = "orgs:read"
    ORGS_WRITE = "orgs:write"
    TEAMS_READ = "teams:read"
    TEAMS_WRITE = "teams:write"
    ADMIN_LOCKS = "admin:locks"
    ADMIN_USERS = "admin:users"
    ADMIN_OIDC = "admin:oidc"
    KILL_SWITCH_READ = "kill_switch:read"
    KILL_SWITCH_WRITE = "kill_switch:write"
    BLAST_RADIUS_READ = "blast_radius:read"
    BLAST_RADIUS_WRITE = "blast_radius:write"
    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    COMPLIANCE_READ = "compliance:read"
    COMPLIANCE_WRITE = "compliance:write"
    PURGE = "purge"


_ALL_PERMISSIONS = frozenset(Permission)

_READ_PERMISSIONS = frozenset(p for p in Permission if p.value.endswith(":read"))

_TEAM_ADMIN_PERMISSIONS = frozenset(
    {
        Permission.SESSIONS_READ,
        Permission.SESSIONS_CANCEL,
        Permission.SESSIONS_EXPORT,
        Permission.CONFIG_READ,
        Permission.AGENTS_READ,
        Permission.AGENTS_WRITE,
        Permission.AGENTS_DELETE,
        Permission.EXPERIMENTS_READ,
        Permission.EXPERIMENTS_WRITE,
        Permission.VIRTUAL_KEYS_READ,
        Permission.VIRTUAL_KEYS_WRITE,
        Permission.VIRTUAL_KEYS_REVOKE,
        Permission.TEAMS_READ,
        Permission.TEAMS_WRITE,
        Permission.KILL_SWITCH_READ,
        Permission.BLAST_RADIUS_READ,
        Permission.BLAST_RADIUS_WRITE,
        Permission.JOBS_READ,
        Permission.JOBS_WRITE,
        Permission.COMPLIANCE_READ,
    }
)

_TEAM_EDITOR_PERMISSIONS = frozenset(
    {
        Permission.SESSIONS_READ,
        Permission.CONFIG_READ,
        Permission.AGENTS_READ,
        Permission.AGENTS_WRITE,
        Permission.EXPERIMENTS_READ,
        Permission.EXPERIMENTS_WRITE,
        Permission.VIRTUAL_KEYS_READ,
        Permission.TEAMS_READ,
        Permission.KILL_SWITCH_READ,
        Permission.BLAST_RADIUS_READ,
        Permission.JOBS_READ,
        Permission.JOBS_WRITE,
        Permission.COMPLIANCE_READ,
    }
)

_TEAM_VIEWER_PERMISSIONS = frozenset(
    {
        Permission.SESSIONS_READ,
        Permission.CONFIG_READ,
        Permission.AGENTS_READ,
        Permission.EXPERIMENTS_READ,
        Permission.VIRTUAL_KEYS_READ,
        Permission.TEAMS_READ,
        Permission.KILL_SWITCH_READ,
        Permission.BLAST_RADIUS_READ,
        Permission.JOBS_READ,
        Permission.COMPLIANCE_READ,
    }
)

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ORG_ADMIN: _ALL_PERMISSIONS,
    Role.ORG_AUDITOR: _READ_PERMISSIONS,
    Role.TEAM_ADMIN: _TEAM_ADMIN_PERMISSIONS,
    Role.TEAM_EDITOR: _TEAM_EDITOR_PERMISSIONS,
    Role.TEAM_VIEWER: _TEAM_VIEWER_PERMISSIONS,
}


def _is_org_role(role: Role) -> bool:
    return role in (Role.ORG_ADMIN, Role.ORG_AUDITOR)


def resolve_permissions(
    user: object,
    team_roles: list[object],
    team_id: str | None = None,
) -> frozenset[Permission]:
    """Resolve the effective permission set for a user.

    Args:
        user: User object with ``org_role`` attribute.
        team_roles: List of UserTeamRole objects with ``team_id`` and ``role`` attrs.
        team_id: If given, only include team-level permissions for this team.

    Returns:
        Combined permission set from org-level + team-level roles.
    """
    perms: set[Permission] = set()

    org_role = getattr(user, "org_role", None)
    if org_role is not None:
        perms |= ROLE_PERMISSIONS.get(org_role, frozenset())

    for tr in team_roles:
        tr_team_id = getattr(tr, "team_id", "")
        tr_role = getattr(tr, "role", None)
        if team_id is not None and tr_team_id != team_id:
            continue
        if tr_role is not None:
            perms |= ROLE_PERMISSIONS.get(tr_role, frozenset())

    return frozenset(perms)


def check_permission(
    user: object,
    team_roles: list[object],
    required: Permission,
    team_id: str | None = None,
) -> bool:
    """Check if a user has a specific permission.

    Args:
        user: User object with ``org_role`` attribute.
        team_roles: List of UserTeamRole objects.
        required: The permission to check.
        team_id: If given, scope the check to this team.

    Returns:
        True if the user has the required permission.
    """
    return required in resolve_permissions(user, team_roles, team_id)
