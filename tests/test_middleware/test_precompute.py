"""Tests for the parallel pre-computation middleware."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.session import Session
from stateloom.middleware.auto_router import AutoRouterMiddleware
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.precompute import PrecomputeMiddleware, _extract_last_user_text
from stateloom.store.memory_store import MemoryStore


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "auto_route_enabled": True,
        "auto_route_model": "llama3.2",
        "local_model_enabled": True,
        "local_model_default": "llama3.2",
        "console_output": False,
        "parallel_precompute": True,
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(
    messages: list | None = None,
    **kwargs,
) -> MiddlewareContext:
    session = Session(id="test-session")
    if messages is None:
        messages = [{"role": "user", "content": "hello world"}]
    return MiddlewareContext(
        session=session,
        config=_make_config(),
        provider="openai",
        model="gpt-4",
        request_kwargs={"messages": messages},
        auto_route_eligible=True,
        **kwargs,
    )


class TestExtractLastUserText:
    """Tests for the standalone text extraction helper."""

    def test_simple_message(self):
        kwargs = {"messages": [{"role": "user", "content": "hello"}]}
        assert _extract_last_user_text(kwargs) == "hello"

    def test_multipart_content(self):
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "image_url", "url": "http://img.png"},
                        {"type": "text", "text": "second"},
                    ],
                }
            ]
        }
        assert _extract_last_user_text(kwargs) == "first second"

    def test_no_user_message(self):
        kwargs = {"messages": [{"role": "system", "content": "be helpful"}]}
        assert _extract_last_user_text(kwargs) == ""

    def test_empty_messages(self):
        assert _extract_last_user_text({}) == ""
        assert _extract_last_user_text({"messages": []}) == ""

    def test_last_user_message_used(self):
        kwargs = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
        }
        assert _extract_last_user_text(kwargs) == "second"


class TestPrecomputeMiddleware:
    """Tests for the PrecomputeMiddleware."""

    @pytest.fixture
    def mock_classifier(self):
        classifier = MagicMock()
        classifier.classify.return_value = 0.25
        return classifier

    async def test_launches_scoring_in_background(self, mock_classifier):
        """PrecomputeMiddleware should set _precomputed_complexity_future on ctx."""
        middleware = PrecomputeMiddleware(mock_classifier)
        ctx = _make_ctx()

        call_next_called = False

        async def call_next(c):
            nonlocal call_next_called
            call_next_called = True
            # Future should be set by now
            assert c._precomputed_complexity_future is not None
            return "response"

        result = await middleware.process(ctx, call_next)
        assert result == "response"
        assert call_next_called

    async def test_future_resolves_to_score(self, mock_classifier):
        """The future should resolve to the classifier's score."""
        middleware = PrecomputeMiddleware(mock_classifier)
        ctx = _make_ctx()

        async def call_next(c):
            # Await the future
            score = await c._precomputed_complexity_future
            assert score == 0.25
            return "response"

        await middleware.process(ctx, call_next)
        mock_classifier.classify.assert_called_once_with("hello world")

    async def test_no_classifier_passthrough(self):
        """With no classifier, middleware should be a no-op passthrough."""
        middleware = PrecomputeMiddleware(None)
        ctx = _make_ctx()

        async def call_next(c):
            assert c._precomputed_complexity_future is None
            return "response"

        result = await middleware.process(ctx, call_next)
        assert result == "response"

    async def test_no_user_message_passthrough(self, mock_classifier):
        """With no user messages, middleware should be a no-op passthrough."""
        middleware = PrecomputeMiddleware(mock_classifier)
        ctx = _make_ctx(messages=[{"role": "system", "content": "be helpful"}])

        async def call_next(c):
            assert c._precomputed_complexity_future is None
            return "response"

        result = await middleware.process(ctx, call_next)
        assert result == "response"
        mock_classifier.classify.assert_not_called()

    async def test_classifier_error_failopen(self):
        """Classifier errors in the executor should not break the pipeline."""
        classifier = MagicMock()
        classifier.classify.side_effect = RuntimeError("model crashed")
        middleware = PrecomputeMiddleware(classifier)
        ctx = _make_ctx()

        async def call_next(c):
            # Future is set, but will raise when awaited
            assert c._precomputed_complexity_future is not None
            return "response"

        result = await middleware.process(ctx, call_next)
        assert result == "response"


class TestAutoRouterConsumesPrecomputed:
    """Tests that AutoRouter correctly uses pre-computed scores."""

    async def test_precomputed_score_consumed(self):
        """AutoRouter should use _precomputed_complexity_score when available."""
        config = _make_config(auto_route_semantic_enabled=False)
        store = MemoryStore()
        middleware = AutoRouterMiddleware(config, store)

        ctx = _make_ctx()
        ctx._precomputed_complexity_score = 0.10  # Low complexity → route local

        # Mock Ollama availability and chat
        middleware._ollama_available = True
        middleware._ollama_check_time = float("inf")

        decision = middleware._should_route_local(ctx)
        # Score should be used (0.10 < 0.15 threshold)
        assert decision.route_local is True
        assert decision.complexity_score == 0.10

    async def test_precomputed_score_high_blocks_routing(self):
        """High pre-computed score should prevent local routing."""
        config = _make_config(auto_route_semantic_enabled=False)
        store = MemoryStore()
        middleware = AutoRouterMiddleware(config, store)

        ctx = _make_ctx()
        ctx._precomputed_complexity_score = 0.75  # High complexity → cloud

        middleware._ollama_available = True
        middleware._ollama_check_time = float("inf")

        decision = middleware._should_route_local(ctx)
        assert decision.route_local is False

    async def test_fallback_to_inline_without_precomputed(self):
        """Without pre-computed score, AutoRouter should compute its own."""
        config = _make_config(auto_route_semantic_enabled=False)
        store = MemoryStore()
        middleware = AutoRouterMiddleware(config, store)

        ctx = _make_ctx()
        # No precomputed score set (default None)
        assert ctx._precomputed_complexity_score is None

        middleware._ollama_available = True
        middleware._ollama_check_time = float("inf")

        decision = middleware._should_route_local(ctx)
        # Should still work via heuristic fallback
        assert decision.complexity_score >= 0.0

    async def test_future_resolved_in_process(self):
        """AutoRouter.process() should resolve the pre-computed future."""
        config = _make_config(auto_route_semantic_enabled=False)
        store = MemoryStore()
        middleware = AutoRouterMiddleware(config, store)

        ctx = _make_ctx()

        # Create a future that resolves to a score
        loop = asyncio.get_event_loop()
        ctx._precomputed_complexity_future = loop.run_in_executor(None, lambda: 0.2)

        cloud_called = False

        async def call_next(c):
            nonlocal cloud_called
            cloud_called = True
            return "cloud_response"

        # Mock Ollama as unavailable so we skip the local route attempt
        middleware._ollama_available = False
        middleware._ollama_check_time = float("inf")

        await middleware.process(ctx, call_next)
        # Future should have been resolved
        assert ctx._precomputed_complexity_score == 0.2
        assert cloud_called

    async def test_future_error_failopen(self):
        """If the pre-computed future raises, AutoRouter should fall back gracefully."""
        config = _make_config(auto_route_semantic_enabled=False)
        store = MemoryStore()
        middleware = AutoRouterMiddleware(config, store)

        ctx = _make_ctx()

        # Create a future that raises
        async def failing_future():
            raise RuntimeError("scoring failed")

        ctx._precomputed_complexity_future = asyncio.ensure_future(failing_future())

        async def call_next(c):
            return "cloud_response"

        middleware._ollama_available = False
        middleware._ollama_check_time = float("inf")

        # Should not raise
        result = await middleware.process(ctx, call_next)
        assert result == "cloud_response"
        # Score should remain None (failed)
        assert ctx._precomputed_complexity_score is None

    async def test_cache_hit_skips_routing(self):
        """When skip_call is set (cache hit), AutoRouter should skip routing."""
        config = _make_config(auto_route_semantic_enabled=False)
        store = MemoryStore()
        middleware = AutoRouterMiddleware(config, store)

        ctx = _make_ctx(skip_call=True)
        ctx._precomputed_complexity_score = 0.1

        decision = middleware._should_route_local(ctx)
        assert decision.route_local is False
        assert decision.reason == "skip_call set (cache hit)"


class TestPrecomputeContextFields:
    """Tests that MiddlewareContext fields work correctly."""

    def test_default_values(self):
        """Pre-computed fields should default to None."""
        ctx = _make_ctx()
        assert ctx._precomputed_complexity_score is None
        assert ctx._precomputed_complexity_future is None

    def test_fields_not_in_init(self):
        """Pre-computed fields should not be in __init__ params."""
        # This should work without providing the pre-computed fields
        session = Session(id="test")
        ctx = MiddlewareContext(
            session=session,
            config=_make_config(),
        )
        assert ctx._precomputed_complexity_score is None
        assert ctx._precomputed_complexity_future is None

    def test_fields_not_in_repr(self):
        """Pre-computed fields should not appear in repr."""
        ctx = _make_ctx()
        ctx._precomputed_complexity_score = 0.5
        r = repr(ctx)
        assert "_precomputed" not in r


class TestConfigFlag:
    """Tests for the parallel_precompute config flag."""

    def test_default_enabled(self):
        config = StateLoomConfig()
        assert config.parallel_precompute is True

    def test_can_disable(self):
        config = StateLoomConfig(parallel_precompute=False)
        assert config.parallel_precompute is False
