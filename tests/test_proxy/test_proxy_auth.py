"""Tests for proxy authentication and provider key resolution."""

from __future__ import annotations

import os
import unittest.mock
from unittest.mock import MagicMock, patch

from stateloom.proxy.auth import ProxyAuth
from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore


def _make_gate(store=None):
    """Create a mock Gate with the required attributes."""
    gate = MagicMock()
    gate.store = store or MemoryStore()
    gate.config = MagicMock()
    gate.config.provider_api_key_openai = ""
    gate.config.provider_api_key_anthropic = ""
    gate.config.provider_api_key_google = ""
    gate._secret_vault = None
    return gate


def _make_vk(**overrides) -> tuple[str, VirtualKey]:
    """Create a virtual key and return (full_key, VirtualKey)."""
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
    return full_key, VirtualKey(**defaults)


class TestProxyAuthAuthenticate:
    def test_valid_key(self):
        store = MemoryStore()
        full_key, vk = _make_vk()
        store.save_virtual_key(vk)

        auth = ProxyAuth(_make_gate(store))
        result = auth.authenticate(f"Bearer {full_key}")
        assert result is not None
        assert result.id == vk.id

    def test_invalid_key(self):
        store = MemoryStore()
        auth = ProxyAuth(_make_gate(store))
        result = auth.authenticate("Bearer invalid-key-123")
        assert result is None

    def test_missing_bearer_prefix(self):
        store = MemoryStore()
        full_key, vk = _make_vk()
        store.save_virtual_key(vk)

        auth = ProxyAuth(_make_gate(store))
        result = auth.authenticate(full_key)
        assert result is None

    def test_empty_authorization(self):
        auth = ProxyAuth(_make_gate())
        assert auth.authenticate("") is None

    def test_revoked_key_rejected(self):
        store = MemoryStore()
        full_key, vk = _make_vk()
        store.save_virtual_key(vk)
        store.revoke_virtual_key(vk.id)

        auth = ProxyAuth(_make_gate(store))
        result = auth.authenticate(f"Bearer {full_key}")
        assert result is None

    def test_caching(self):
        store = MemoryStore()
        full_key, vk = _make_vk()
        store.save_virtual_key(vk)

        auth = ProxyAuth(_make_gate(store))
        # First call hits the store
        result1 = auth.authenticate(f"Bearer {full_key}")
        assert result1 is not None

        # Second call should use cache (even if store is empty, cache has it)
        with patch.object(store, "get_virtual_key_by_hash", return_value=None):
            result2 = auth.authenticate(f"Bearer {full_key}")
            assert result2 is not None
            assert result2.id == vk.id

    def test_invalidate_cache(self):
        store = MemoryStore()
        full_key, vk = _make_vk()
        store.save_virtual_key(vk)

        auth = ProxyAuth(_make_gate(store))
        auth.authenticate(f"Bearer {full_key}")

        # Invalidate and revoke
        auth.invalidate_cache(vk.key_hash)
        store.revoke_virtual_key(vk.id)

        result = auth.authenticate(f"Bearer {full_key}")
        assert result is None


class TestProxyAuthProviderKeys:
    def test_global_config_fallback(self):
        gate = _make_gate()
        gate.config.provider_api_key_openai = "sk-global-openai"
        gate.config.provider_api_key_anthropic = "sk-global-anthropic"
        gate.config.provider_api_key_google = "sk-global-google"

        auth = ProxyAuth(gate)
        _, vk = _make_vk(org_id="org-1")
        keys = auth.get_provider_keys(vk)
        assert keys["openai"] == "sk-global-openai"
        assert keys["anthropic"] == "sk-global-anthropic"
        assert keys["google"] == "sk-global-google"

    def test_org_secret_takes_priority(self):
        store = MemoryStore()
        store.save_secret("org:org-1:provider_key_openai", "sk-org-openai")

        gate = _make_gate(store)
        gate.config.provider_api_key_openai = "sk-global-openai"

        auth = ProxyAuth(gate)
        _, vk = _make_vk(org_id="org-1")
        keys = auth.get_provider_keys(vk)
        assert keys["openai"] == "sk-org-openai"

    def test_empty_keys_not_included(self):
        gate = _make_gate()
        gate.config.provider_api_key_openai = ""
        gate.config.provider_api_key_anthropic = ""
        gate.config.provider_api_key_google = ""

        auth = ProxyAuth(gate)
        _, vk = _make_vk(org_id="org-1")
        env_patch = {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GOOGLE_API_KEY": "",
        }
        with unittest.mock.patch.dict(os.environ, env_patch, clear=False):
            # Remove the keys entirely so env fallback returns nothing
            for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
                os.environ.pop(k, None)
            keys = auth.get_provider_keys(vk)
        assert len(keys) == 0
