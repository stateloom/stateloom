"""Tests for role-permission mapping and authorization checks."""

from stateloom.auth.models import User, UserTeamRole
from stateloom.auth.permissions import (
    ROLE_PERMISSIONS,
    Permission,
    check_permission,
    resolve_permissions,
)
from stateloom.core.types import Role


def test_all_roles_have_permission_sets():
    for role in Role:
        assert role in ROLE_PERMISSIONS
        assert isinstance(ROLE_PERMISSIONS[role], frozenset)


def test_org_admin_has_all_permissions():
    perms = ROLE_PERMISSIONS[Role.ORG_ADMIN]
    for p in Permission:
        assert p in perms, f"org_admin missing {p}"


def test_org_auditor_has_only_read_permissions():
    perms = ROLE_PERMISSIONS[Role.ORG_AUDITOR]
    for p in perms:
        assert p.value.endswith(":read"), f"org_auditor has non-read permission: {p}"


def test_team_viewer_has_only_read_permissions():
    perms = ROLE_PERMISSIONS[Role.TEAM_VIEWER]
    for p in perms:
        assert p.value.endswith(":read"), f"team_viewer has non-read permission: {p}"


def test_team_editor_has_write_for_agents():
    perms = ROLE_PERMISSIONS[Role.TEAM_EDITOR]
    assert Permission.AGENTS_WRITE in perms
    assert Permission.EXPERIMENTS_WRITE in perms


def test_team_editor_no_admin():
    perms = ROLE_PERMISSIONS[Role.TEAM_EDITOR]
    assert Permission.ADMIN_USERS not in perms
    assert Permission.ADMIN_OIDC not in perms
    assert Permission.ADMIN_LOCKS not in perms


def test_team_admin_no_org_admin():
    perms = ROLE_PERMISSIONS[Role.TEAM_ADMIN]
    assert Permission.ADMIN_OIDC not in perms
    assert Permission.ORGS_WRITE not in perms
    assert Permission.ADMIN_USERS not in perms


def test_team_admin_has_team_crud():
    perms = ROLE_PERMISSIONS[Role.TEAM_ADMIN]
    assert Permission.AGENTS_WRITE in perms
    assert Permission.AGENTS_DELETE in perms
    assert Permission.VIRTUAL_KEYS_WRITE in perms
    assert Permission.VIRTUAL_KEYS_REVOKE in perms


def test_check_permission_org_admin():
    user = User(email="admin@test.com", org_role=Role.ORG_ADMIN)
    assert check_permission(user, [], Permission.ADMIN_USERS) is True
    assert check_permission(user, [], Permission.PURGE) is True


def test_check_permission_org_auditor():
    user = User(email="auditor@test.com", org_role=Role.ORG_AUDITOR)
    assert check_permission(user, [], Permission.SESSIONS_READ) is True
    assert check_permission(user, [], Permission.SESSIONS_CANCEL) is False


def test_check_permission_team_role():
    user = User(email="dev@test.com")
    role = UserTeamRole(
        user_id=user.id,
        team_id="team-1",
        role=Role.TEAM_EDITOR,
    )
    assert check_permission(user, [role], Permission.AGENTS_WRITE) is True
    assert check_permission(user, [role], Permission.ADMIN_USERS) is False


def test_check_permission_team_scoped():
    user = User(email="dev@test.com")
    role = UserTeamRole(
        user_id=user.id,
        team_id="team-1",
        role=Role.TEAM_EDITOR,
    )
    # Scoped to the correct team
    assert check_permission(user, [role], Permission.AGENTS_WRITE, team_id="team-1") is True
    # Scoped to a different team — no access
    assert check_permission(user, [role], Permission.AGENTS_WRITE, team_id="team-2") is False


def test_resolve_permissions_combines_org_and_team():
    user = User(email="user@test.com", org_role=Role.ORG_AUDITOR)
    role = UserTeamRole(
        user_id=user.id,
        team_id="team-1",
        role=Role.TEAM_EDITOR,
    )
    perms = resolve_permissions(user, [role])
    # From org_auditor: read-only
    assert Permission.CONFIG_READ in perms
    # From team_editor: agent writes
    assert Permission.AGENTS_WRITE in perms


def test_resolve_permissions_no_roles():
    user = User(email="nobody@test.com")
    perms = resolve_permissions(user, [])
    assert len(perms) == 0


def test_permission_enum_values():
    assert Permission.SESSIONS_READ == "sessions:read"
    assert Permission.PURGE == "purge"
