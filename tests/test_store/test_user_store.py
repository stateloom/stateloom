"""Tests for user/role/token store methods on both MemoryStore and SQLiteStore."""

import os
import tempfile

import pytest

from stateloom.auth.models import User, UserTeamRole
from stateloom.core.types import Role
from stateloom.store.memory_store import MemoryStore
from stateloom.store.sqlite_store import SQLiteStore


@pytest.fixture
def memory_store():
    return MemoryStore()


@pytest.fixture
def sqlite_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(path=path, auto_migrate=False)
    yield store
    store.close()
    os.unlink(path)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, memory_store, sqlite_store):
    if request.param == "memory":
        return memory_store
    return sqlite_store


class TestUserCRUD:
    def test_save_and_get_user(self, store):
        user = User(email="alice@example.com", org_id="org-1", org_role=Role.ORG_ADMIN)
        store.save_user(user)
        retrieved = store.get_user(user.id)
        assert retrieved is not None
        assert retrieved.email == "alice@example.com"
        assert retrieved.org_role == Role.ORG_ADMIN

    def test_get_user_not_found(self, store):
        assert store.get_user("usr-nonexistent") is None

    def test_get_user_by_email(self, store):
        user = User(email="Bob@Example.Com", org_id="org-1")
        store.save_user(user)
        # Case-insensitive lookup
        retrieved = store.get_user_by_email("bob@example.com")
        assert retrieved is not None
        assert retrieved.id == user.id

    def test_get_user_by_email_not_found(self, store):
        assert store.get_user_by_email("nobody@test.com") is None

    def test_get_user_by_oidc(self, store):
        user = User(
            email="oidc@test.com",
            oidc_provider_id="oidc-google",
            oidc_subject="sub-123",
        )
        store.save_user(user)
        retrieved = store.get_user_by_oidc("oidc-google", "sub-123")
        assert retrieved is not None
        assert retrieved.id == user.id

    def test_get_user_by_oidc_not_found(self, store):
        assert store.get_user_by_oidc("oidc-none", "sub-none") is None

    def test_list_users(self, store):
        user1 = User(email="a@test.com", org_id="org-1")
        user2 = User(email="b@test.com", org_id="org-2")
        store.save_user(user1)
        store.save_user(user2)
        users = store.list_users()
        assert len(users) >= 2

    def test_list_users_filtered_by_org(self, store):
        user1 = User(email="x@test.com", org_id="org-filter")
        user2 = User(email="y@test.com", org_id="org-other")
        store.save_user(user1)
        store.save_user(user2)
        users = store.list_users(org_id="org-filter")
        assert len(users) == 1
        assert users[0].org_id == "org-filter"

    def test_list_users_pagination(self, store):
        for i in range(5):
            store.save_user(User(email=f"page{i}@test.com"))
        page = store.list_users(limit=2, offset=0)
        assert len(page) == 2
        page2 = store.list_users(limit=2, offset=2)
        assert len(page2) == 2

    def test_delete_user_soft(self, store):
        user = User(email="delete@test.com")
        store.save_user(user)
        assert store.delete_user(user.id) is True
        retrieved = store.get_user(user.id)
        assert retrieved is not None
        assert retrieved.is_active is False

    def test_delete_user_not_found(self, store):
        assert store.delete_user("usr-ghost") is False

    def test_update_user(self, store):
        user = User(email="update@test.com", display_name="Before")
        store.save_user(user)
        user.display_name = "After"
        store.save_user(user)
        retrieved = store.get_user(user.id)
        assert retrieved.display_name == "After"


class TestUserTeamRoleCRUD:
    def test_save_and_get_roles(self, store):
        role = UserTeamRole(
            user_id="usr-111",
            team_id="team-222",
            role=Role.TEAM_EDITOR,
        )
        store.save_user_team_role(role)
        roles = store.get_user_team_roles("usr-111")
        assert len(roles) == 1
        assert roles[0].role == Role.TEAM_EDITOR

    def test_get_team_members(self, store):
        role1 = UserTeamRole(user_id="usr-a", team_id="team-x", role=Role.TEAM_ADMIN)
        role2 = UserTeamRole(user_id="usr-b", team_id="team-x", role=Role.TEAM_VIEWER)
        store.save_user_team_role(role1)
        store.save_user_team_role(role2)
        members = store.get_team_members("team-x")
        assert len(members) == 2

    def test_delete_role(self, store):
        role = UserTeamRole(user_id="usr-c", team_id="team-y", role=Role.TEAM_EDITOR)
        store.save_user_team_role(role)
        assert store.delete_user_team_role(role.id) is True
        assert store.get_user_team_roles("usr-c") == []

    def test_delete_role_not_found(self, store):
        assert store.delete_user_team_role("utr-ghost") is False

    def test_unique_user_team_constraint(self, store):
        """Saving a role with the same (user_id, team_id) should upsert."""
        role1 = UserTeamRole(
            user_id="usr-dup",
            team_id="team-dup",
            role=Role.TEAM_VIEWER,
        )
        store.save_user_team_role(role1)
        role2 = UserTeamRole(
            user_id="usr-dup",
            team_id="team-dup",
            role=Role.TEAM_ADMIN,
        )
        store.save_user_team_role(role2)
        roles = store.get_user_team_roles("usr-dup")
        assert len(roles) == 1
        assert roles[0].role == Role.TEAM_ADMIN


class TestRefreshTokenCRUD:
    def test_save_and_get_token(self, store):
        store.save_refresh_token("hash123", "usr-1", "2025-01-01T00:00:00Z")
        token = store.get_refresh_token("hash123")
        assert token is not None
        assert token["user_id"] == "usr-1"
        assert token["revoked"] is False

    def test_get_token_not_found(self, store):
        assert store.get_refresh_token("nonexistent") is None

    def test_revoke_token(self, store):
        store.save_refresh_token("hash456", "usr-2", "2025-01-01T00:00:00Z")
        store.revoke_refresh_token("hash456")
        token = store.get_refresh_token("hash456")
        assert token["revoked"] is True

    def test_revoke_all_tokens(self, store):
        store.save_refresh_token("h1", "usr-3", "2025-01-01T00:00:00Z")
        store.save_refresh_token("h2", "usr-3", "2025-01-01T00:00:00Z")
        store.save_refresh_token("h3", "usr-other", "2025-01-01T00:00:00Z")
        count = store.revoke_all_refresh_tokens("usr-3")
        assert count == 2
        assert store.get_refresh_token("h1")["revoked"] is True
        assert store.get_refresh_token("h3")["revoked"] is False
