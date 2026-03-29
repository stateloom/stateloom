"""Tests for shared embedding cache across middleware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.cache import CacheMiddleware


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "console_output": False,
        "cache_enabled": True,
        "cache_scope": "global",
        "cache_similarity_threshold": 0.85,
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(**overrides) -> MiddlewareContext:
    config = _make_config()
    defaults = {
        "session": Session(id="test-session"),
        "config": config,
        "provider": "openai",
        "model": "gpt-4o",
        "request_kwargs": {"messages": [{"role": "user", "content": "hello"}]},
        "request_hash": "abc123",
    }
    defaults.update(overrides)
    return MiddlewareContext(**defaults)


class TestEmbeddingCacheField:
    """MiddlewareContext._embedding_cache exists and is usable."""

    def test_embedding_cache_initialized_empty(self):
        ctx = _make_ctx()
        assert ctx._embedding_cache == {}

    def test_embedding_cache_is_dict(self):
        ctx = _make_ctx()
        ctx._embedding_cache["key"] = [1.0, 2.0, 3.0]
        assert ctx._embedding_cache["key"] == [1.0, 2.0, 3.0]


class TestCacheMiddlewareEmbeddingReuse:
    """Cache middleware uses shared embedding cache to avoid double computation."""

    async def test_embedding_computed_once_on_cache_miss(self):
        """On a cache miss, embed_request is called once (cached for storage)."""
        mock_semantic = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3]
        mock_semantic.embed_request.return_value = mock_embedding
        mock_semantic.search.return_value = (None, 0.0)

        config = _make_config()
        mw = CacheMiddleware(config, semantic_matcher=mock_semantic)

        ctx = _make_ctx()

        async def call_next(c):
            return {"choices": [{"message": {"content": "response"}}]}

        result = await mw.process(ctx, call_next)

        # embed_request should be called only ONCE (search call uses cache for store)
        assert mock_semantic.embed_request.call_count == 1

        # Embedding should be in the shared cache
        assert ctx._embedding_cache[f"embed:{ctx.request_hash}"] == mock_embedding

    async def test_precomputed_embedding_reused_for_storage(self):
        """If embedding is already in cache, it's reused for FAISS storage."""
        mock_semantic = MagicMock()
        precomputed = [0.5, 0.6, 0.7]
        mock_semantic.search.return_value = (None, 0.0)

        config = _make_config()
        mw = CacheMiddleware(config, semantic_matcher=mock_semantic)

        ctx = _make_ctx()
        # Pre-populate the embedding cache
        ctx._embedding_cache[f"embed:{ctx.request_hash}"] = precomputed

        async def call_next(c):
            return {"choices": [{"message": {"content": "response"}}]}

        await mw.process(ctx, call_next)

        # embed_request should NOT have been called (embedding was cached)
        mock_semantic.embed_request.assert_not_called()

    async def test_no_semantic_matcher_no_cache_interaction(self):
        """Without semantic matcher, embedding cache is not touched."""
        config = _make_config()
        mw = CacheMiddleware(config, semantic_matcher=None)

        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        assert ctx._embedding_cache == {}

    async def test_separate_contexts_have_separate_caches(self):
        """Each request context has its own embedding cache."""
        ctx1 = _make_ctx(request_hash="hash1")
        ctx2 = _make_ctx(request_hash="hash2")

        ctx1._embedding_cache["embed:hash1"] = [1.0]
        assert ctx2._embedding_cache == {}
