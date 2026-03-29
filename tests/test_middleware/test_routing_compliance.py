"""Tests for compliance-aware routing and shadow blocking."""

from __future__ import annotations

import pytest

from stateloom.core.config import ComplianceProfile, StateLoomConfig
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext


def _make_ctx(
    org_id: str = "",
    team_id: str = "",
) -> MiddlewareContext:
    return MiddlewareContext(
        session=Session(id="test-session", org_id=org_id, team_id=team_id),
        config=StateLoomConfig(console_output=False),
        provider="openai",
        model="gpt-4",
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )


class TestAutoRouterCompliance:
    """AutoRouter respects block_local_routing from compliance profile."""

    async def test_no_compliance_normal_routing(self):
        """Without compliance_fn, routing proceeds normally (to call_next)."""
        from stateloom.middleware.auto_router import AutoRouterMiddleware
        from stateloom.store.memory_store import MemoryStore

        config = StateLoomConfig(
            console_output=False,
            auto_route_enabled=True,
            local_model_enabled=True,
            local_model_default="llama3.2",
            auto_route_model="llama3.2",
        )
        store = MemoryStore()
        mw = AutoRouterMiddleware(config, store, compliance_fn=None)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"

    async def test_compliance_blocks_routing(self):
        """compliance_fn returning block_local_routing=True skips routing."""
        from stateloom.middleware.auto_router import AutoRouterMiddleware
        from stateloom.store.memory_store import MemoryStore

        config = StateLoomConfig(
            console_output=False,
            auto_route_enabled=True,
            local_model_enabled=True,
            local_model_default="llama3.2",
            auto_route_model="llama3.2",
        )
        store = MemoryStore()

        profile = ComplianceProfile(standard="gdpr", block_local_routing=True)

        def compliance_fn(org_id, team_id):
            return profile

        mw = AutoRouterMiddleware(config, store, compliance_fn=compliance_fn)
        ctx = _make_ctx(org_id="org-1")

        call_next_called = False

        async def call_next(c):
            nonlocal call_next_called
            call_next_called = True
            return "cloud-response"

        result = await mw.process(ctx, call_next)
        assert result == "cloud-response"
        assert call_next_called

    async def test_compliance_allows_routing_when_not_blocked(self):
        from stateloom.middleware.auto_router import AutoRouterMiddleware
        from stateloom.store.memory_store import MemoryStore

        config = StateLoomConfig(
            console_output=False,
            auto_route_enabled=True,
            local_model_enabled=True,
            local_model_default="llama3.2",
            auto_route_model="llama3.2",
        )
        store = MemoryStore()

        profile = ComplianceProfile(standard="ccpa", block_local_routing=False)

        def compliance_fn(org_id, team_id):
            return profile

        mw = AutoRouterMiddleware(config, store, compliance_fn=compliance_fn)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"


class TestShadowCompliance:
    """ShadowMiddleware respects block_shadow from compliance profile."""

    async def test_compliance_blocks_shadow(self):
        from stateloom.middleware.shadow import ShadowMiddleware
        from stateloom.store.memory_store import MemoryStore

        config = StateLoomConfig(
            console_output=False,
            shadow_enabled=True,
            shadow_model="llama3.2",
            local_model_enabled=True,
            local_model_default="llama3.2",
        )
        store = MemoryStore()

        profile = ComplianceProfile(standard="hipaa", block_shadow=True)

        def compliance_fn(org_id, team_id):
            return profile

        mw = ShadowMiddleware(config, store, compliance_fn=compliance_fn)
        ctx = _make_ctx(org_id="org-1")

        call_next_called = False

        async def call_next(c):
            nonlocal call_next_called
            call_next_called = True
            return "cloud-response"

        result = await mw.process(ctx, call_next)
        assert result == "cloud-response"
        assert call_next_called

    async def test_no_compliance_shadow_proceeds(self):
        """Without compliance_fn, shadow middleware proceeds normally."""
        from stateloom.middleware.shadow import ShadowMiddleware
        from stateloom.store.memory_store import MemoryStore

        config = StateLoomConfig(
            console_output=False,
            shadow_enabled=True,
            shadow_model="llama3.2",
            local_model_enabled=True,
            local_model_default="llama3.2",
        )
        store = MemoryStore()
        mw = ShadowMiddleware(config, store, compliance_fn=None)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"

    async def test_compliance_allows_shadow_when_not_blocked(self):
        from stateloom.middleware.shadow import ShadowMiddleware
        from stateloom.store.memory_store import MemoryStore

        config = StateLoomConfig(
            console_output=False,
            shadow_enabled=True,
            shadow_model="llama3.2",
            local_model_enabled=True,
            local_model_default="llama3.2",
        )
        store = MemoryStore()

        profile = ComplianceProfile(standard="ccpa", block_shadow=False)

        def compliance_fn(org_id, team_id):
            return profile

        mw = ShadowMiddleware(config, store, compliance_fn=compliance_fn)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"
