"""Tests for FastAPI permission dependencies."""

import pytest

from stateloom.auth.dependencies import require_permission
from stateloom.auth.models import User, UserTeamRole
from stateloom.auth.permissions import Permission
from stateloom.core.types import Role


class FakeRequest:
    def __init__(self, user=None, team_roles=None, path_params=None, query_params=None):
        self.state = type("State", (), {})()
        if user:
            self.state.user = user
        if team_roles:
            self.state.team_roles = team_roles
        else:
            self.state.team_roles = []
        self.path_params = path_params or {}
        self.query_params = query_params or {}


@pytest.mark.asyncio
async def test_require_permission_passes_for_admin():
    user = User(email="admin@test.com", org_role=Role.ORG_ADMIN)
    request = FakeRequest(user=user)
    check = require_permission(Permission.ADMIN_USERS)
    result = await check(request)
    assert result == user


@pytest.mark.asyncio
async def test_require_permission_denies_for_viewer():
    user = User(email="viewer@test.com")
    role = UserTeamRole(user_id=user.id, team_id="team-1", role=Role.TEAM_VIEWER)
    request = FakeRequest(user=user, team_roles=[role])
    check = require_permission(Permission.ADMIN_USERS)
    result = await check(request)
    # Should return JSONResponse with 403
    assert hasattr(result, "status_code")
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_require_permission_denies_no_user():
    request = FakeRequest()
    check = require_permission(Permission.SESSIONS_READ)
    result = await check(request)
    assert hasattr(result, "status_code")
    assert result.status_code == 401


@pytest.mark.asyncio
async def test_require_permission_with_team_scope():
    user = User(email="dev@test.com")
    role = UserTeamRole(user_id=user.id, team_id="team-a", role=Role.TEAM_EDITOR)
    request = FakeRequest(
        user=user,
        team_roles=[role],
        path_params={"team_id": "team-a"},
    )
    check = require_permission(Permission.AGENTS_WRITE, team_id_param="team_id")
    result = await check(request)
    assert result == user


@pytest.mark.asyncio
async def test_require_permission_wrong_team():
    user = User(email="dev@test.com")
    role = UserTeamRole(user_id=user.id, team_id="team-a", role=Role.TEAM_EDITOR)
    request = FakeRequest(
        user=user,
        team_roles=[role],
        path_params={"team_id": "team-b"},
    )
    check = require_permission(Permission.AGENTS_WRITE, team_id_param="team_id")
    result = await check(request)
    assert hasattr(result, "status_code")
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_org_admin_bypasses_team_scope():
    user = User(email="orgadmin@test.com", org_role=Role.ORG_ADMIN)
    request = FakeRequest(
        user=user,
        path_params={"team_id": "any-team"},
    )
    check = require_permission(Permission.AGENTS_WRITE, team_id_param="team_id")
    result = await check(request)
    assert result == user
