"""Tests for virtual key scope enforcement (model access + budget limits)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stateloom.proxy.auth import ProxyAuth
from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore


def _make_gate(store=None):
    gate = MagicMock()
    gate.store = store or MemoryStore()
    gate.config = MagicMock()
    gate.config.provider_api_key_openai = ""
    gate.config.provider_api_key_anthropic = ""
    gate.config.provider_api_key_google = ""
    return gate


def _make_vk(**overrides) -> tuple[str, VirtualKey]:
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


class TestCheckModelAccess:
    """ProxyAuth.check_model_access() — glob matching on allowed_models."""

    def test_empty_allowed_models_allows_all(self):
        _, vk = _make_vk(allowed_models=[])
        assert ProxyAuth.check_model_access(vk, "gpt-4o") is True
        assert ProxyAuth.check_model_access(vk, "claude-sonnet-4-20250514") is True

    def test_exact_match(self):
        _, vk = _make_vk(allowed_models=["gpt-4o", "gpt-4o-mini"])
        assert ProxyAuth.check_model_access(vk, "gpt-4o") is True
        assert ProxyAuth.check_model_access(vk, "gpt-4o-mini") is True
        assert ProxyAuth.check_model_access(vk, "gpt-3.5-turbo") is False

    def test_glob_pattern(self):
        _, vk = _make_vk(allowed_models=["gpt-*"])
        assert ProxyAuth.check_model_access(vk, "gpt-4o") is True
        assert ProxyAuth.check_model_access(vk, "gpt-3.5-turbo") is True
        assert ProxyAuth.check_model_access(vk, "claude-sonnet-4-20250514") is False

    def test_multiple_glob_patterns(self):
        _, vk = _make_vk(allowed_models=["gpt-*", "claude-*"])
        assert ProxyAuth.check_model_access(vk, "gpt-4o") is True
        assert ProxyAuth.check_model_access(vk, "claude-3-5-sonnet-20241022") is True
        assert ProxyAuth.check_model_access(vk, "gemini-1.5-pro") is False

    def test_wildcard_allows_all(self):
        _, vk = _make_vk(allowed_models=["*"])
        assert ProxyAuth.check_model_access(vk, "gpt-4o") is True
        assert ProxyAuth.check_model_access(vk, "anything") is True


class TestCheckBudget:
    """ProxyAuth.check_budget() — per-key budget enforcement."""

    def test_no_budget_limit_allows_all(self):
        _, vk = _make_vk(budget_limit=None, budget_spent=0.0)
        assert ProxyAuth.check_budget(vk) is True
        assert ProxyAuth.check_budget(vk, cost=100.0) is True

    def test_within_budget(self):
        _, vk = _make_vk(budget_limit=10.0, budget_spent=5.0)
        assert ProxyAuth.check_budget(vk) is True

    def test_budget_exactly_at_limit(self):
        _, vk = _make_vk(budget_limit=10.0, budget_spent=10.0)
        assert ProxyAuth.check_budget(vk) is True  # equal is still allowed

    def test_budget_exceeded(self):
        _, vk = _make_vk(budget_limit=10.0, budget_spent=10.01)
        assert ProxyAuth.check_budget(vk) is False

    def test_budget_exceeded_with_cost(self):
        _, vk = _make_vk(budget_limit=10.0, budget_spent=8.0)
        assert ProxyAuth.check_budget(vk, cost=2.0) is True
        assert ProxyAuth.check_budget(vk, cost=2.01) is False

    def test_zero_budget_allows_zero_cost(self):
        _, vk = _make_vk(budget_limit=0.0, budget_spent=0.0)
        assert ProxyAuth.check_budget(vk) is True

    def test_zero_budget_blocks_any_cost(self):
        _, vk = _make_vk(budget_limit=0.0, budget_spent=0.0)
        assert ProxyAuth.check_budget(vk, cost=0.01) is False


class TestVirtualKeyFields:
    """VirtualKey dataclass has expected scope fields."""

    def test_default_allowed_models_is_empty(self):
        _, vk = _make_vk()
        assert vk.allowed_models == []

    def test_default_budget_limit_is_none(self):
        _, vk = _make_vk()
        assert vk.budget_limit is None
        assert vk.budget_spent == 0.0

    def test_custom_allowed_models(self):
        _, vk = _make_vk(allowed_models=["gpt-*", "claude-*"])
        assert vk.allowed_models == ["gpt-*", "claude-*"]

    def test_custom_budget_limit(self):
        _, vk = _make_vk(budget_limit=50.0, budget_spent=12.5)
        assert vk.budget_limit == 50.0
        assert vk.budget_spent == 12.5


class TestVirtualKeyPersistence:
    """Virtual key scopes survive save/load via MemoryStore."""

    def test_save_and_retrieve_by_hash(self):
        store = MemoryStore()
        full_key, vk = _make_vk(
            allowed_models=["gpt-*"],
            budget_limit=100.0,
            budget_spent=25.0,
        )
        store.save_virtual_key(vk)

        loaded = store.get_virtual_key_by_hash(vk.key_hash)
        assert loaded is not None
        assert loaded.allowed_models == ["gpt-*"]
        assert loaded.budget_limit == 100.0
        assert loaded.budget_spent == 25.0
