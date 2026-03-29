"""Tests for admin-locked settings (enterprise config protection)."""

from __future__ import annotations

import json
import tempfile

import pytest

from stateloom.core.errors import StateLoomConfigLockedError
from stateloom.store.memory_store import MemoryStore
from stateloom.store.sqlite_store import SQLiteStore

# ── Store-level tests (both backends) ──────────────────────────────────────


class TestMemoryStoreAdminLocks:
    def test_save_and_get_lock(self):
        store = MemoryStore()
        store.save_admin_lock("kill_switch_active", "true", locked_by="admin", reason="safety")
        lock = store.get_admin_lock("kill_switch_active")
        assert lock is not None
        assert lock["setting"] == "kill_switch_active"
        assert lock["value"] == "true"
        assert lock["locked_by"] == "admin"
        assert lock["reason"] == "safety"
        assert "locked_at" in lock

    def test_get_nonexistent_lock_returns_none(self):
        store = MemoryStore()
        assert store.get_admin_lock("nonexistent") is None

    def test_list_locks(self):
        store = MemoryStore()
        store.save_admin_lock("pii_enabled", "true")
        store.save_admin_lock("blast_radius_enabled", "true")
        locks = store.list_admin_locks()
        assert len(locks) == 2
        settings = [l["setting"] for l in locks]
        assert "blast_radius_enabled" in settings
        assert "pii_enabled" in settings

    def test_list_locks_empty(self):
        store = MemoryStore()
        assert store.list_admin_locks() == []

    def test_delete_lock(self):
        store = MemoryStore()
        store.save_admin_lock("kill_switch_active", "true")
        assert store.get_admin_lock("kill_switch_active") is not None
        store.delete_admin_lock("kill_switch_active")
        assert store.get_admin_lock("kill_switch_active") is None

    def test_delete_nonexistent_lock_no_error(self):
        store = MemoryStore()
        store.delete_admin_lock("nonexistent")  # should not raise

    def test_upsert_lock(self):
        store = MemoryStore()
        store.save_admin_lock("pii_enabled", "true", reason="v1")
        store.save_admin_lock("pii_enabled", "false", reason="v2")
        lock = store.get_admin_lock("pii_enabled")
        assert lock is not None
        assert lock["value"] == "false"
        assert lock["reason"] == "v2"


class TestSQLiteStoreAdminLocks:
    def test_save_and_get_lock(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.save_admin_lock("kill_switch_active", "true", locked_by="admin", reason="safety")
        lock = store.get_admin_lock("kill_switch_active")
        assert lock is not None
        assert lock["setting"] == "kill_switch_active"
        assert lock["value"] == "true"
        assert lock["locked_by"] == "admin"
        assert lock["reason"] == "safety"
        assert "locked_at" in lock

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        assert store.get_admin_lock("nonexistent") is None

    def test_list_locks(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.save_admin_lock("pii_enabled", "true")
        store.save_admin_lock("blast_radius_enabled", "true")
        locks = store.list_admin_locks()
        assert len(locks) == 2

    def test_delete_lock(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.save_admin_lock("kill_switch_active", "true")
        store.delete_admin_lock("kill_switch_active")
        assert store.get_admin_lock("kill_switch_active") is None

    def test_upsert_lock(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.save_admin_lock("pii_enabled", "true", reason="v1")
        store.save_admin_lock("pii_enabled", "false", reason="v2")
        lock = store.get_admin_lock("pii_enabled")
        assert lock is not None
        assert lock["value"] == "false"
        assert lock["reason"] == "v2"

    def test_persistence_across_reopens(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store1 = SQLiteStore(db_path)
        store1.save_admin_lock("pii_enabled", "true", reason="persisted")

        store2 = SQLiteStore(db_path)
        lock = store2.get_admin_lock("pii_enabled")
        assert lock is not None
        assert lock["value"] == "true"
        assert lock["reason"] == "persisted"


# ── Gate-level tests ───────────────────────────────────────────────────────


class TestGateAdminLocks:
    def _make_gate(self):
        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(
            store_backend="memory",
            dashboard=False,
            console_output=False,
            auto_patch=False,
        )
        gate = Gate(config)
        return gate

    def test_lock_setting(self):
        gate = self._make_gate()
        lock = gate.lock_setting("kill_switch_active", True, reason="safety")
        assert lock["setting"] == "kill_switch_active"
        assert json.loads(lock["value"]) is True
        assert lock["reason"] == "safety"

    def test_lock_setting_invalid_name_raises(self):
        gate = self._make_gate()
        with pytest.raises(Exception, match="Unknown config setting"):
            gate.lock_setting("nonexistent_setting", "value")

    def test_lock_setting_uses_current_value_when_none(self):
        gate = self._make_gate()
        gate.config.pii_enabled = True
        lock = gate.lock_setting("pii_enabled")
        assert json.loads(lock["value"]) is True

    def test_unlock_setting(self):
        gate = self._make_gate()
        gate.lock_setting("kill_switch_active", True)
        assert gate.unlock_setting("kill_switch_active") is True
        assert gate.list_locked_settings() == []

    def test_unlock_nonexistent_returns_false(self):
        gate = self._make_gate()
        assert gate.unlock_setting("kill_switch_active") is False

    def test_list_locked_settings(self):
        gate = self._make_gate()
        gate.lock_setting("pii_enabled", True)
        gate.lock_setting("blast_radius_enabled", True)
        locks = gate.list_locked_settings()
        assert len(locks) == 2
        settings = [l["setting"] for l in locks]
        assert "pii_enabled" in settings
        assert "blast_radius_enabled" in settings

    def test_check_locked_settings_no_conflict(self):
        gate = self._make_gate()
        gate.lock_setting("pii_enabled", True)
        # Passing the same value as the lock should not raise
        gate.check_locked_settings({"pii_enabled": True})

    def test_check_locked_settings_conflict_raises(self):
        gate = self._make_gate()
        gate.lock_setting("pii_enabled", True, reason="compliance")
        with pytest.raises(StateLoomConfigLockedError) as exc_info:
            gate.check_locked_settings({"pii_enabled": False})
        assert exc_info.value.setting == "pii_enabled"
        assert exc_info.value.locked_value is True
        assert exc_info.value.reason == "compliance"

    def test_check_locked_settings_unrelated_field_ok(self):
        gate = self._make_gate()
        gate.lock_setting("pii_enabled", True)
        # A different field should not raise
        gate.check_locked_settings({"kill_switch_active": False})


# ── init() enforcement tests ──────────────────────────────────────────────


class TestInitEnforcement:
    def test_init_with_conflicting_lock_raises(self, tmp_path):
        """init() with a conflicting locked setting should raise."""
        import stateloom

        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)
        store.save_admin_lock("pii_enabled", json.dumps(True), reason="compliance")

        # Ensure clean state
        if stateloom._gate is not None:
            stateloom.shutdown()

        with pytest.raises(StateLoomConfigLockedError) as exc_info:
            stateloom.init(
                auto_patch=False,
                dashboard=False,
                console_output=False,
                pii=False,  # pii=False → config_kwargs has pii_enabled=False
                store_backend="sqlite",
                store_path=db_path,
            )
        assert exc_info.value.setting == "pii_enabled"

        # Clean up partial init
        if stateloom._gate is not None:
            stateloom.shutdown()

    def test_init_with_matching_lock_succeeds(self, tmp_path):
        """init() with a matching locked setting value should succeed."""
        import stateloom

        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)
        store.save_admin_lock("pii_enabled", json.dumps(True), reason="compliance")

        if stateloom._gate is not None:
            stateloom.shutdown()

        try:
            gate = stateloom.init(
                auto_patch=False,
                dashboard=False,
                console_output=False,
                pii=True,  # matches locked value
                store_backend="sqlite",
                store_path=db_path,
            )
            assert gate is not None
        finally:
            stateloom.shutdown()


# ── Dashboard API tests ───────────────────────────────────────────────────


class TestDashboardAdminLockAPI:
    def _make_client(self):
        from fastapi.testclient import TestClient

        from stateloom.core.config import StateLoomConfig
        from stateloom.dashboard.api import create_api_router
        from stateloom.gate import Gate

        config = StateLoomConfig(
            store_backend="memory",
            dashboard=False,
            console_output=False,
            auto_patch=False,
        )
        gate = Gate(config)
        gate._setup_middleware()

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(create_api_router(gate), prefix="/api/v1")
        return TestClient(app), gate

    def test_create_lock(self):
        client, gate = self._make_client()
        resp = client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "kill_switch_active",
                "value": True,
                "reason": "emergency stop",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["setting"] == "kill_switch_active"
        assert json.loads(data["value"]) is True
        assert data["reason"] == "emergency stop"

    def test_create_lock_invalid_setting(self):
        client, gate = self._make_client()
        resp = client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "nonexistent_field",
                "value": True,
            },
        )
        assert resp.status_code == 400

    def test_create_lock_value_none_uses_current(self):
        client, gate = self._make_client()
        gate.config.pii_enabled = True
        resp = client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "pii_enabled",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert json.loads(data["value"]) is True

    def test_list_locks(self):
        client, gate = self._make_client()
        client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "pii_enabled",
                "value": True,
            },
        )
        client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "blast_radius_enabled",
                "value": True,
            },
        )
        resp = client.get("/api/v1/admin/locks")
        assert resp.status_code == 200
        locks = resp.json()["locks"]
        assert len(locks) == 2

    def test_delete_lock(self):
        client, gate = self._make_client()
        client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "kill_switch_active",
                "value": True,
            },
        )
        resp = client.delete("/api/v1/admin/locks/kill_switch_active")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unlocked"

        # Verify it's gone
        resp = client.get("/api/v1/admin/locks")
        assert resp.json()["locks"] == []

    def test_delete_nonexistent_lock_returns_404(self):
        client, gate = self._make_client()
        resp = client.delete("/api/v1/admin/locks/nonexistent")
        assert resp.status_code == 404

    def test_patch_config_with_locked_setting_returns_403(self):
        client, gate = self._make_client()
        # Lock blast_radius_enabled to True
        client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "blast_radius_enabled",
                "value": True,
            },
        )
        # Try to change it via PATCH /config
        resp = client.patch(
            "/api/v1/config",
            json={
                "blast_radius_enabled": False,
            },
        )
        assert resp.status_code == 403
        assert "admin-locked" in resp.json()["detail"]

    def test_patch_config_with_matching_locked_value_succeeds(self):
        client, gate = self._make_client()
        client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "blast_radius_enabled",
                "value": True,
            },
        )
        # Setting to the same value should be allowed
        resp = client.patch(
            "/api/v1/config",
            json={
                "blast_radius_enabled": True,
            },
        )
        assert resp.status_code == 200

    def test_patch_config_unlocked_setting_succeeds(self):
        client, gate = self._make_client()
        # Lock one setting, change a different one
        client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "pii_enabled",
                "value": True,
            },
        )
        resp = client.patch(
            "/api/v1/config",
            json={
                "kill_switch_active": True,
            },
        )
        assert resp.status_code == 200

    def test_config_includes_locked_list(self):
        client, gate = self._make_client()
        client.post(
            "/api/v1/admin/locks",
            json={
                "setting": "pii_enabled",
                "value": True,
            },
        )
        resp = client.get("/api/v1/config")
        assert resp.status_code == 200
        config = resp.json()
        assert "_locked" in config
        assert "pii_enabled" in config["_locked"]


# ── Public API tests ──────────────────────────────────────────────────────


class TestPublicAPI:
    def test_lock_unlock_list(self):
        import stateloom

        if stateloom._gate is not None:
            stateloom.shutdown()

        try:
            stateloom.init(
                auto_patch=False,
                dashboard=False,
                console_output=False,
                store_backend="memory",
            )

            lock = stateloom.lock_setting("kill_switch_active", True, reason="test")
            assert lock["setting"] == "kill_switch_active"

            locks = stateloom.list_locked_settings()
            assert len(locks) == 1
            assert locks[0]["setting"] == "kill_switch_active"

            removed = stateloom.unlock_setting("kill_switch_active")
            assert removed is True

            assert stateloom.list_locked_settings() == []
        finally:
            stateloom.shutdown()


# ── Error class tests ─────────────────────────────────────────────────────


class TestStateLoomConfigLockedError:
    def test_error_attributes(self):
        err = StateLoomConfigLockedError("pii_enabled", True, "compliance")
        assert err.setting == "pii_enabled"
        assert err.locked_value is True
        assert err.reason == "compliance"
        assert "CONFIG_LOCKED" in str(err)
        assert "pii_enabled" in str(err)

    def test_error_default_reason(self):
        err = StateLoomConfigLockedError("pii_enabled", True)
        assert "admin has locked" in str(err).lower()

    def test_error_is_stateloom_error(self):
        from stateloom.core.errors import StateLoomError

        err = StateLoomConfigLockedError("x", True)
        assert isinstance(err, StateLoomError)
