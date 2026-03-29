"""Tests for IaC admin bootstrap via env vars."""

import os

import pytest

from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


class TestBootstrapAdmin:
    def test_bootstrap_creates_admin(self, monkeypatch, store):
        """Admin user is created when env vars are set."""
        monkeypatch.setenv("STATELOOM_ADMIN_EMAIL", "admin@corp.com")
        monkeypatch.setenv("STATELOOM_ADMIN_PASSWORD", "securepass123")

        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(store_backend="memory")
        gate = Gate(config)
        gate.store = store
        gate._load_hierarchy()
        from stateloom.ee.setup import _bootstrap_admin

        _bootstrap_admin(gate)

        user = store.get_user_by_email("admin@corp.com")
        assert user is not None
        assert user.email == "admin@corp.com"
        assert user.org_role is not None
        assert user.org_role.value == "org_admin"
        assert user.email_verified is True
        assert user.is_active is True

    def test_bootstrap_with_password_hash(self, monkeypatch, store):
        """Admin user is created with pre-hashed password."""
        from stateloom.auth.password import hash_password

        pw_hash = hash_password("myhash")
        monkeypatch.setenv("STATELOOM_ADMIN_EMAIL", "admin2@corp.com")
        monkeypatch.setenv("STATELOOM_ADMIN_PASSWORD_HASH", pw_hash)

        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(store_backend="memory")
        gate = Gate(config)
        gate.store = store
        gate._load_hierarchy()
        from stateloom.ee.setup import _bootstrap_admin

        _bootstrap_admin(gate)

        user = store.get_user_by_email("admin2@corp.com")
        assert user is not None
        assert user.password_hash == pw_hash

    def test_bootstrap_skips_existing_user(self, monkeypatch, store):
        """Bootstrap doesn't overwrite an existing user."""
        from stateloom.auth.models import User
        from stateloom.core.types import Role

        existing = User(
            email="admin@corp.com",
            password_hash="old-hash",
            org_role=Role.ORG_AUDITOR,
        )
        store.save_user(existing)

        monkeypatch.setenv("STATELOOM_ADMIN_EMAIL", "admin@corp.com")
        monkeypatch.setenv("STATELOOM_ADMIN_PASSWORD", "newpass")

        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(store_backend="memory")
        gate = Gate(config)
        gate.store = store
        gate._load_hierarchy()
        from stateloom.ee.setup import _bootstrap_admin

        _bootstrap_admin(gate)

        user = store.get_user_by_email("admin@corp.com")
        assert user.password_hash == "old-hash"
        assert user.org_role == Role.ORG_AUDITOR

    def test_bootstrap_skipped_without_env(self, monkeypatch, store):
        """No bootstrap when env vars are absent."""
        monkeypatch.delenv("STATELOOM_ADMIN_EMAIL", raising=False)
        monkeypatch.delenv("STATELOOM_ADMIN_PASSWORD", raising=False)
        monkeypatch.delenv("STATELOOM_ADMIN_PASSWORD_HASH", raising=False)

        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(store_backend="memory")
        gate = Gate(config)
        gate.store = store
        gate._load_hierarchy()
        from stateloom.ee.setup import _bootstrap_admin

        _bootstrap_admin(gate)

        users = store.list_users()
        assert len(users) == 0

    def test_bootstrap_email_without_password(self, monkeypatch, store):
        """Warning logged when email set but no password."""
        monkeypatch.setenv("STATELOOM_ADMIN_EMAIL", "admin@corp.com")
        monkeypatch.delenv("STATELOOM_ADMIN_PASSWORD", raising=False)
        monkeypatch.delenv("STATELOOM_ADMIN_PASSWORD_HASH", raising=False)

        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(store_backend="memory")
        gate = Gate(config)
        gate.store = store
        gate._load_hierarchy()
        from stateloom.ee.setup import _bootstrap_admin

        _bootstrap_admin(gate)

        users = store.list_users()
        assert len(users) == 0

    def test_bootstrap_creates_default_org(self, monkeypatch, store):
        """Bootstrap creates Default Organization if none exist."""
        monkeypatch.setenv("STATELOOM_ADMIN_EMAIL", "admin@corp.com")
        monkeypatch.setenv("STATELOOM_ADMIN_PASSWORD", "securepass123")

        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(store_backend="memory")
        gate = Gate(config)
        gate.store = store
        gate._load_hierarchy()
        from stateloom.ee.setup import _bootstrap_admin

        _bootstrap_admin(gate)

        orgs = store.list_organizations()
        assert len(orgs) >= 1
        assert any(o.name == "Default Organization" for o in orgs)

        user = store.get_user_by_email("admin@corp.com")
        assert user.org_id == orgs[0].id

    def test_bootstrap_uses_existing_org(self, monkeypatch, store):
        """Bootstrap uses existing org rather than creating a new one."""
        from stateloom.core.organization import Organization

        org = Organization(name="My Corp")
        store.save_organization(org)

        monkeypatch.setenv("STATELOOM_ADMIN_EMAIL", "admin@corp.com")
        monkeypatch.setenv("STATELOOM_ADMIN_PASSWORD", "securepass123")

        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(store_backend="memory")
        gate = Gate(config)
        gate.store = store
        gate._load_hierarchy()
        from stateloom.ee.setup import _bootstrap_admin

        _bootstrap_admin(gate)

        user = store.get_user_by_email("admin@corp.com")
        assert user.org_id == org.id

        # Should not create a second org
        orgs = store.list_organizations()
        assert len(orgs) == 1
