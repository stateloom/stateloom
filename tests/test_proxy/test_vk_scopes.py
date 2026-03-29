"""Tests for VK scope enforcement on proxy endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    gate.config.proxy_require_virtual_key = True
    gate.config.proxy_upstream_openai = "https://api.openai.com"
    gate.config.proxy_upstream_anthropic = "https://api.anthropic.com"
    gate.config.proxy_upstream_gemini = "https://generativelanguage.googleapis.com"
    gate.config.proxy_timeout = 600.0
    gate.config.provider_api_key_openai = ""
    gate.config.provider_api_key_anthropic = ""
    gate.config.provider_api_key_google = ""
    gate.config.kill_switch_active = False
    gate.config.kill_switch_rules = []
    gate.config.blast_radius_enabled = False
    gate.pricing = MagicMock()
    gate.pricing.calculate_cost = MagicMock(return_value=0.001)
    gate._metrics_collector = None
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


# ── ProxyAuth.check_scope unit tests ──────────────────────────────────


class TestCheckScope:
    """ProxyAuth.check_scope() — VK scope enforcement."""

    def test_empty_scopes_allows_all(self):
        _, vk = _make_vk(scopes=[])
        assert ProxyAuth.check_scope(vk, "chat") is True
        assert ProxyAuth.check_scope(vk, "messages") is True
        assert ProxyAuth.check_scope(vk, "responses") is True
        assert ProxyAuth.check_scope(vk, "generate") is True
        assert ProxyAuth.check_scope(vk, "agents") is True

    def test_scoped_key_allows_matching_scope(self):
        _, vk = _make_vk(scopes=["chat", "messages"])
        assert ProxyAuth.check_scope(vk, "chat") is True
        assert ProxyAuth.check_scope(vk, "messages") is True

    def test_scoped_key_denies_non_matching_scope(self):
        _, vk = _make_vk(scopes=["chat"])
        assert ProxyAuth.check_scope(vk, "messages") is False
        assert ProxyAuth.check_scope(vk, "responses") is False
        assert ProxyAuth.check_scope(vk, "generate") is False
        assert ProxyAuth.check_scope(vk, "agents") is False

    def test_all_proxy_scopes(self):
        from stateloom.proxy.auth import PROXY_SCOPES

        _, vk = _make_vk(scopes=list(PROXY_SCOPES))
        for scope in PROXY_SCOPES:
            assert ProxyAuth.check_scope(vk, scope) is True

    def test_single_scope_key(self):
        _, vk = _make_vk(scopes=["responses"])
        assert ProxyAuth.check_scope(vk, "responses") is True
        assert ProxyAuth.check_scope(vk, "chat") is False


# ── Router endpoint scope enforcement tests ───────────────────────────


class TestRouterScopeEnforcement:
    """Scope enforcement on /v1/chat/completions."""

    def test_chat_scope_denied(self):
        """Key with scopes that exclude 'chat' gets 403."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["messages"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.router import create_proxy_router

        router = create_proxy_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1")

        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {full_key}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "scope_denied"

    def test_chat_scope_allowed(self):
        """Key with 'chat' scope passes scope check (may fail later on upstream)."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["chat"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.router import create_proxy_router

        router = create_proxy_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1")

        client = TestClient(app)
        # Should not get scope_denied (may get a different error since no upstream)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {full_key}"},
        )
        # Not a scope error
        if resp.status_code == 403:
            assert resp.json()["error"]["code"] != "scope_denied"

    def test_empty_scopes_allowed(self):
        """Key with empty scopes passes scope check."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=[])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.router import create_proxy_router

        router = create_proxy_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1")

        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {full_key}"},
        )
        # Not a scope error
        if resp.status_code == 403:
            assert resp.json()["error"]["code"] != "scope_denied"


class TestAnthropicScopeEnforcement:
    """Scope enforcement on /v1/messages."""

    def test_messages_scope_denied(self):
        """Key with scopes excluding 'messages' gets 403."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["chat"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.anthropic_native import create_anthropic_router

        router = create_anthropic_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1")

        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-api-key": full_key},
        )
        assert resp.status_code == 403
        # Anthropic uses {type: "error", error: {type: ..., message: ...}} format
        data = resp.json()
        assert (
            "messages" in data["error"]["message"].lower()
            or "scope" in data["error"]["message"].lower()
        )

    def test_messages_scope_allowed(self):
        """Key with 'messages' scope passes scope check."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["messages"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.anthropic_native import create_anthropic_router

        router = create_anthropic_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1")

        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-api-key": full_key},
        )
        # Should not be a scope-related 403
        if resp.status_code == 403:
            assert "scope" not in resp.json()["error"].get("message", "").lower()


class TestGeminiScopeEnforcement:
    """Scope enforcement on /v1beta/models/{model}:generateContent."""

    def test_generate_scope_denied(self):
        """Key with scopes excluding 'generate' gets 403."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["chat"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.gemini_native import create_gemini_router

        router = create_gemini_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1beta")

        client = TestClient(app)
        resp = client.post(
            "/v1beta/models/gemini-1.5-pro:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            },
            headers={"x-goog-api-key": full_key},
        )
        assert resp.status_code == 403
        # Gemini uses {error: {code: 403, message: ..., status: ...}} format
        data = resp.json()
        assert "scope" in data["error"]["message"].lower()

    def test_generate_scope_allowed(self):
        """Key with 'generate' scope passes scope check."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["generate"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.gemini_native import create_gemini_router

        router = create_gemini_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1beta")

        client = TestClient(app)
        resp = client.post(
            "/v1beta/models/gemini-1.5-pro:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            },
            headers={"x-goog-api-key": full_key},
        )
        # Should not be a scope-related 403
        if resp.status_code == 403:
            assert "scope" not in resp.json()["error"].get("message", "").lower()


class TestResponsesScopeEnforcement:
    """Scope enforcement on /v1/responses."""

    def test_responses_scope_denied(self):
        """Key with scopes excluding 'responses' gets 403."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["chat"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.responses import create_responses_router

        router = create_responses_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1")

        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={"model": "gpt-4o", "input": "hi"},
            headers={"Authorization": f"Bearer {full_key}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "scope_denied"

    def test_responses_scope_allowed(self):
        """Key with 'responses' scope passes scope check."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["responses"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.responses import create_responses_router

        router = create_responses_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1")

        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={"model": "gpt-4o", "input": "hi"},
            headers={"Authorization": f"Bearer {full_key}"},
        )
        if resp.status_code == 403:
            assert resp.json()["error"]["code"] != "scope_denied"


class TestAgentsScopeEnforcement:
    """Scope enforcement on /v1/agents/{ref}/chat/completions."""

    def test_agents_scope_denied(self):
        """Key with scopes excluding 'agents' gets 403."""
        store = MemoryStore()
        full_key, vk = _make_vk(scopes=["chat"])
        store.save_virtual_key(vk)
        gate = _make_gate(store)

        from stateloom.proxy.router import create_proxy_router

        router = create_proxy_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/v1")

        client = TestClient(app)
        resp = client.post(
            "/v1/agents/test-agent/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {full_key}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "scope_denied"
