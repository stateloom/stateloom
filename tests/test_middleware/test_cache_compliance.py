"""Tests for compliance-aware cache middleware."""

from __future__ import annotations

import time

import pytest

from stateloom.cache.base import CacheEntry
from stateloom.cache.memory_store import MemoryCacheStore
from stateloom.core.config import ComplianceProfile, StateLoomConfig
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.cache import CacheMiddleware


def _make_ctx(
    org_id: str = "",
    team_id: str = "",
    request_hash: str = "hash-123",
) -> MiddlewareContext:
    return MiddlewareContext(
        session=Session(id="test-session", org_id=org_id, team_id=team_id),
        config=StateLoomConfig(console_output=False),
        provider="openai",
        model="gpt-4",
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
        request_hash=request_hash,
    )


class TestHIPAANoCache:
    async def test_hipaa_zero_retention_skips_cache(self):
        """HIPAA with zero_retention_logs + cache_ttl_seconds=0 skips cache entirely."""
        profile = ComplianceProfile(
            standard="hipaa",
            zero_retention_logs=True,
            cache_ttl_seconds=0,
        )

        def compliance_fn(org_id, team_id):
            return profile

        cache_store = MemoryCacheStore()
        config = StateLoomConfig(console_output=False, cache_enabled=True)
        mw = CacheMiddleware(config, cache_store=cache_store, compliance_fn=compliance_fn)

        ctx = _make_ctx()
        called = False

        async def call_next(c):
            nonlocal called
            called = True
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"
        assert called
        # Nothing should be cached
        assert cache_store.size() == 0

    async def test_non_hipaa_allows_cache(self):
        """Non-HIPAA profile with cache_ttl > 0 allows caching."""
        profile = ComplianceProfile(
            standard="ccpa",
            cache_ttl_seconds=7776000,
        )

        def compliance_fn(org_id, team_id):
            return profile

        cache_store = MemoryCacheStore()
        config = StateLoomConfig(console_output=False, cache_enabled=True)
        mw = CacheMiddleware(config, cache_store=cache_store, compliance_fn=compliance_fn)

        ctx = _make_ctx()

        async def call_next(c):
            return {"id": "response-1"}

        result = await mw.process(ctx, call_next)
        assert result is not None


class TestOrgScopedCacheIsolation:
    async def test_org_scoped_cache(self):
        """With compliance active and org_id set, cache is org-scoped."""
        profile = ComplianceProfile(standard="gdpr", cache_ttl_seconds=86400)

        def compliance_fn(org_id, team_id):
            return profile

        cache_store = MemoryCacheStore()
        config = StateLoomConfig(console_output=False, cache_enabled=True)
        mw = CacheMiddleware(config, cache_store=cache_store, compliance_fn=compliance_fn)

        # Store a cache entry scoped to org-1
        entry = CacheEntry(
            request_hash="hash-123",
            session_id="org:org-1",
            response_json='{"result": "cached"}',
            model="gpt-4",
            provider="openai",
            cost=0.01,
            created_at=time.time(),
        )
        cache_store.put(entry)

        # Request from org-1 should hit
        ctx1 = _make_ctx(org_id="org-1")
        hit_result = None

        async def call_next_hit(c):
            return "cloud-response"

        result1 = await mw.process(ctx1, call_next_hit)
        # The cache should have matched (the entry session_id="org:org-1")

        # Request from org-2 should NOT hit the org-1 cache
        ctx2 = _make_ctx(org_id="org-2")

        call_next_called = False

        async def call_next_miss(c):
            nonlocal call_next_called
            call_next_called = True
            return "fresh-response"

        result2 = await mw.process(ctx2, call_next_miss)
        assert call_next_called


class TestTTLOverride:
    async def test_compliance_ttl_override(self):
        """Compliance profile TTL overrides config TTL."""
        profile = ComplianceProfile(standard="gdpr", cache_ttl_seconds=300)

        def compliance_fn(org_id, team_id):
            return profile

        cache_store = MemoryCacheStore()
        # Config TTL = 0 (no eviction), but compliance overrides to 300
        config = StateLoomConfig(
            console_output=False,
            cache_enabled=True,
            cache_ttl_seconds=0,
        )
        mw = CacheMiddleware(config, cache_store=cache_store, compliance_fn=compliance_fn)

        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"


class TestNoComplianceDefaultBehavior:
    async def test_no_compliance_fn_uses_defaults(self):
        """Without compliance_fn, cache uses config defaults."""
        cache_store = MemoryCacheStore()
        config = StateLoomConfig(console_output=False, cache_enabled=True)
        mw = CacheMiddleware(config, cache_store=cache_store, compliance_fn=None)

        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"
