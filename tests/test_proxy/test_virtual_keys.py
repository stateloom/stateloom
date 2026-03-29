"""Tests for virtual key creation, hashing, storage, and revocation."""

from __future__ import annotations

import pytest

from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    hash_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore


class TestGenerateVirtualKey:
    def test_returns_tuple(self):
        full_key, key_hash = generate_virtual_key()
        assert isinstance(full_key, str)
        assert isinstance(key_hash, str)

    def test_key_starts_with_prefix(self):
        full_key, _ = generate_virtual_key(prefix="ag")
        assert full_key.startswith("ag-")

    def test_custom_prefix(self):
        full_key, _ = generate_virtual_key(prefix="test")
        assert full_key.startswith("test-")

    def test_hash_is_sha256(self):
        _, key_hash = generate_virtual_key()
        assert len(key_hash) == 64  # SHA256 hex

    def test_hash_matches(self):
        full_key, key_hash = generate_virtual_key()
        assert hash_key(full_key) == key_hash

    def test_uniqueness(self):
        keys = {generate_virtual_key()[0] for _ in range(10)}
        assert len(keys) == 10


class TestHashKey:
    def test_deterministic(self):
        assert hash_key("test-123") == hash_key("test-123")

    def test_different_keys_different_hashes(self):
        assert hash_key("key-1") != hash_key("key-2")


class TestMakeKeyPreview:
    def test_long_key(self):
        preview = make_key_preview("ag-abcdef123456789xyz")
        assert preview.startswith("ag-abc")
        assert preview.endswith("xyz")
        assert "..." in preview

    def test_short_key(self):
        preview = make_key_preview("abc")
        assert preview == "abc..."


class TestMakeVirtualKeyId:
    def test_format(self):
        vid = make_virtual_key_id()
        assert vid.startswith("vk-")
        assert len(vid) == 15  # "vk-" + 12 hex chars


class TestMemoryStoreVirtualKeys:
    @pytest.fixture
    def store(self):
        return MemoryStore()

    def _make_vk(self, **overrides) -> VirtualKey:
        full_key, key_hash = generate_virtual_key()
        defaults = {
            "id": make_virtual_key_id(),
            "key_hash": key_hash,
            "key_preview": make_key_preview(full_key),
            "team_id": "team-1",
            "org_id": "org-1",
            "name": "test-key",
        }
        defaults.update(overrides)
        return VirtualKey(**defaults)

    def test_save_and_get_by_hash(self, store):
        vk = self._make_vk()
        store.save_virtual_key(vk)
        result = store.get_virtual_key_by_hash(vk.key_hash)
        assert result is not None
        assert result.id == vk.id
        assert result.team_id == vk.team_id

    def test_get_by_hash_not_found(self, store):
        assert store.get_virtual_key_by_hash("nonexistent") is None

    def test_list_all(self, store):
        vk1 = self._make_vk(team_id="team-1")
        vk2 = self._make_vk(team_id="team-2")
        store.save_virtual_key(vk1)
        store.save_virtual_key(vk2)
        keys = store.list_virtual_keys()
        assert len(keys) == 2

    def test_list_by_team(self, store):
        vk1 = self._make_vk(team_id="team-1")
        vk2 = self._make_vk(team_id="team-2")
        store.save_virtual_key(vk1)
        store.save_virtual_key(vk2)
        keys = store.list_virtual_keys(team_id="team-1")
        assert len(keys) == 1
        assert keys[0].team_id == "team-1"

    def test_revoke(self, store):
        vk = self._make_vk()
        store.save_virtual_key(vk)
        result = store.revoke_virtual_key(vk.id)
        assert result is True
        # Verify revoked
        fetched = store.get_virtual_key_by_hash(vk.key_hash)
        assert fetched is not None
        assert fetched.revoked is True

    def test_revoke_nonexistent(self, store):
        assert store.revoke_virtual_key("nonexistent") is False

    def test_revoke_already_revoked(self, store):
        vk = self._make_vk()
        store.save_virtual_key(vk)
        store.revoke_virtual_key(vk.id)
        # Second revoke returns False
        assert store.revoke_virtual_key(vk.id) is False


class TestSQLiteStoreVirtualKeys:
    @pytest.fixture
    def store(self, tmp_path):
        from stateloom.store.sqlite_store import SQLiteStore

        return SQLiteStore(str(tmp_path / "test.db"))

    def _make_vk(self, **overrides) -> VirtualKey:
        full_key, key_hash = generate_virtual_key()
        defaults = {
            "id": make_virtual_key_id(),
            "key_hash": key_hash,
            "key_preview": make_key_preview(full_key),
            "team_id": "team-1",
            "org_id": "org-1",
            "name": "test-key",
        }
        defaults.update(overrides)
        return VirtualKey(**defaults)

    def test_save_and_get_by_hash(self, store):
        vk = self._make_vk()
        store.save_virtual_key(vk)
        result = store.get_virtual_key_by_hash(vk.key_hash)
        assert result is not None
        assert result.id == vk.id
        assert result.team_id == vk.team_id
        assert result.org_id == vk.org_id

    def test_get_by_hash_not_found(self, store):
        assert store.get_virtual_key_by_hash("nonexistent") is None

    def test_list_all(self, store):
        vk1 = self._make_vk(team_id="team-1")
        vk2 = self._make_vk(team_id="team-2")
        store.save_virtual_key(vk1)
        store.save_virtual_key(vk2)
        keys = store.list_virtual_keys()
        assert len(keys) == 2

    def test_list_by_team(self, store):
        vk1 = self._make_vk(team_id="team-1")
        vk2 = self._make_vk(team_id="team-2")
        store.save_virtual_key(vk1)
        store.save_virtual_key(vk2)
        keys = store.list_virtual_keys(team_id="team-1")
        assert len(keys) == 1
        assert keys[0].team_id == "team-1"

    def test_revoke(self, store):
        vk = self._make_vk()
        store.save_virtual_key(vk)
        result = store.revoke_virtual_key(vk.id)
        assert result is True
        fetched = store.get_virtual_key_by_hash(vk.key_hash)
        assert fetched is not None
        assert fetched.revoked is True

    def test_revoke_nonexistent(self, store):
        assert store.revoke_virtual_key("nonexistent") is False

    def test_scopes_and_metadata_persist(self, store):
        vk = self._make_vk()
        vk.scopes = ["read", "write"]
        vk.metadata = {"env": "prod"}
        store.save_virtual_key(vk)
        fetched = store.get_virtual_key_by_hash(vk.key_hash)
        assert fetched.scopes == ["read", "write"]
        assert fetched.metadata == {"env": "prod"}
