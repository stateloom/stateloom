"""Tests for the two-tier cache middleware."""

from __future__ import annotations

import time

import pytest
from stateloom.cache.base import CacheEntry
from stateloom.cache.memory_store import MemoryCacheStore
from stateloom.core.config import StateLoomConfig
from stateloom.core.event import CacheHitEvent
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.cache import CacheMiddleware


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "budget_per_session": None,
        "budget_global": None,
        "dashboard": False,
        "pii_enabled": False,
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(
    request_hash: str = "abcdef1234567890",
    model: str = "gpt-4",
    session_id: str = "sess-1",
    provider: str = "openai",
) -> MiddlewareContext:
    session = Session(id=session_id, name="test")
    return MiddlewareContext(
        session=session,
        config=_make_config(),
        provider=provider,
        model=model,
        request_kwargs={"model": model, "messages": [{"role": "user", "content": "hello"}]},
        request_hash=request_hash,
    )


def _make_pipeline_next(llm_response=None):
    """Create a call_next that simulates pipeline terminal behavior.

    Returns (call_next_fn, call_counter) where call_counter tracks LLM calls.
    """
    counter = {"llm_calls": 0}
    response = llm_response or {"id": "resp-1", "object": "chat.completion", "choices": []}

    async def call_next(c):
        if c.skip_call and c.cached_response is not None:
            return c.cached_response
        counter["llm_calls"] += 1
        return response

    return call_next, counter


class TestCacheMiddleware:
    @pytest.fixture
    def store(self):
        return MemoryCacheStore(max_size=100)

    @pytest.fixture
    def config(self):
        return _make_config(cache_scope="global")

    async def test_cache_miss_then_hit(self, store, config):
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, counter = _make_pipeline_next()

        # First call → miss
        ctx = _make_ctx()
        await middleware.process(ctx, call_next)
        assert counter["llm_calls"] == 1
        assert store.size() == 1

        # Second call with same hash → hit
        ctx2 = _make_ctx()
        await middleware.process(ctx2, call_next)
        assert counter["llm_calls"] == 1  # Not incremented — cache hit
        assert ctx2.skip_call is True

        # Verify CacheHitEvent
        hit_events = [e for e in ctx2.events if isinstance(e, CacheHitEvent)]
        assert len(hit_events) == 1
        assert hit_events[0].match_type == "exact"
        assert hit_events[0].similarity_score is None

    async def test_session_scope_isolation(self):
        config = _make_config(cache_scope="session")
        store = MemoryCacheStore(max_size=100)
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, counter = _make_pipeline_next()

        # Session 1 cache
        ctx1 = _make_ctx(session_id="sess-1")
        await middleware.process(ctx1, call_next)
        assert counter["llm_calls"] == 1

        # Session 2 same hash → should miss (session-scoped)
        ctx2 = _make_ctx(session_id="sess-2")
        await middleware.process(ctx2, call_next)
        assert counter["llm_calls"] == 2  # Miss — called LLM

    async def test_global_scope_cross_session(self, store, config):
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, counter = _make_pipeline_next()

        # Session 1
        ctx1 = _make_ctx(session_id="sess-1")
        await middleware.process(ctx1, call_next)
        assert counter["llm_calls"] == 1

        # Session 2 same hash → should hit (global scope)
        ctx2 = _make_ctx(session_id="sess-2")
        await middleware.process(ctx2, call_next)
        assert counter["llm_calls"] == 1  # Hit — used cache

    async def test_no_cache_on_streaming(self, store, config):
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, _ = _make_pipeline_next()
        ctx = _make_ctx()
        ctx.is_streaming = True

        await middleware.process(ctx, call_next)
        assert store.size() == 0  # Not cached

    async def test_streaming_skips_cache_lookup(self, store, config):
        """Streaming requests must not get cache hits from prior non-streaming."""
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, counter = _make_pipeline_next()

        # Populate cache with a non-streaming request
        ctx1 = _make_ctx()
        await middleware.process(ctx1, call_next)
        assert counter["llm_calls"] == 1
        assert store.size() == 1

        # Streaming request with same hash — should NOT get a cache hit
        ctx2 = _make_ctx()
        ctx2.is_streaming = True
        await middleware.process(ctx2, call_next)
        assert counter["llm_calls"] == 2  # Made a fresh call, no cache hit
        assert ctx2.skip_call is False

    async def test_no_cache_on_empty_hash(self, store, config):
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, counter = _make_pipeline_next()
        ctx = _make_ctx(request_hash="")

        await middleware.process(ctx, call_next)
        assert counter["llm_calls"] == 1
        assert store.size() == 0

    async def test_ttl_expiration(self):
        config = _make_config(cache_ttl_seconds=10)
        store = MemoryCacheStore(max_size=100)
        middleware = CacheMiddleware(config, cache_store=store)

        # Pre-populate with an expired entry
        entry = CacheEntry(
            request_hash="abcdef1234567890",
            session_id="sess-1",
            response_json='{"id":"resp-1"}',
            model="gpt-4",
            provider="openai",
            cost=0.01,
            created_at=time.time() - 20,  # 20 seconds ago, TTL is 10
        )
        store.put(entry)

        ctx = _make_ctx()
        call_next, counter = _make_pipeline_next()
        await middleware.process(ctx, call_next)
        # Expired entry should have been evicted, so it's a miss
        assert counter["llm_calls"] == 1

    async def test_fallback_to_memory_store(self):
        config = _make_config()
        # No store provided → falls back to MemoryCacheStore
        middleware = CacheMiddleware(config)
        assert isinstance(middleware._store, MemoryCacheStore)

    async def test_cache_hit_updates_session_stats(self, store, config):
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, _ = _make_pipeline_next()

        ctx1 = _make_ctx()
        await middleware.process(ctx1, call_next)

        ctx2 = _make_ctx()
        await middleware.process(ctx2, call_next)

        assert ctx2.session.cache_hits == 1

    async def test_cache_hit_event_fields(self, store, config):
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, _ = _make_pipeline_next()

        # Populate cache
        ctx1 = _make_ctx()
        await middleware.process(ctx1, call_next)

        # Hit
        ctx2 = _make_ctx()
        await middleware.process(ctx2, call_next)

        hit = [e for e in ctx2.events if isinstance(e, CacheHitEvent)][0]
        assert hit.match_type == "exact"
        assert hit.matched_hash == "abcdef1234567890"
        assert hit.original_model == "gpt-4"
        assert hit.request_hash == "abcdef1234567890"

    async def test_error_response_not_cached_anthropic(self, store, config):
        """Anthropic-style error dicts must not be stored in cache."""
        error_resp = {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        }
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, _ = _make_pipeline_next(llm_response=error_resp)

        ctx = _make_ctx()
        result = await middleware.process(ctx, call_next)
        assert result == error_resp
        assert store.size() == 0  # Error NOT cached

    async def test_error_response_not_cached_openai(self, store, config):
        """OpenAI-style error dicts must not be stored in cache."""
        error_resp = {
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": 429}
        }
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, _ = _make_pipeline_next(llm_response=error_resp)

        ctx = _make_ctx()
        await middleware.process(ctx, call_next)
        assert store.size() == 0

    async def test_error_response_not_cached_upstream_sentinel(self, store, config):
        """Passthrough proxy upstream error sentinel must not be cached."""
        error_resp = {"_upstream_error": True, "status_code": 529}
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, _ = _make_pipeline_next(llm_response=error_resp)

        ctx = _make_ctx()
        await middleware.process(ctx, call_next)
        assert store.size() == 0

    async def test_success_response_still_cached(self, store, config):
        """Normal success responses should still be cached."""
        success_resp = {
            "id": "resp-1",
            "object": "chat.completion",
            "choices": [{"message": {"content": "hi"}}],
        }
        middleware = CacheMiddleware(config, cache_store=store)
        call_next, _ = _make_pipeline_next(llm_response=success_resp)

        ctx = _make_ctx()
        await middleware.process(ctx, call_next)
        assert store.size() == 1
