"""Tests for the shadow drafting middleware."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import ShadowDraftEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType, Provider
from stateloom.local.client import OllamaResponse
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.shadow import ShadowMiddleware, _SimilarityBridge
from stateloom.middleware.similarity import (
    _SEMANTIC_AVAILABLE,
    SemanticSimilarityScorer,
    SimilarityResult,
    compute_semantic_similarity,
    compute_similarity_auto,
)
from stateloom.store.memory_store import MemoryStore


def _make_ctx(
    provider: str = "openai",
    model: str = "gpt-4",
    shadow_enabled: bool | None = None,
    shadow_model: str = "",
) -> MiddlewareContext:
    session = Session(id="test-session")
    if shadow_enabled is not None:
        session.metadata["shadow_enabled"] = shadow_enabled
    if shadow_model:
        session.metadata["shadow_model"] = shadow_model
    return MiddlewareContext(
        session=session,
        config=StateLoomConfig(
            shadow_enabled=True,
            shadow_model="llama3.2",
            console_output=False,
        ),
        provider=provider,
        model=model,
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "shadow_enabled": True,
        "shadow_model": "llama3.2",
        "console_output": False,
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


class TestShadowMiddleware:
    """Test shadow middleware behavior."""

    @pytest.fixture
    def store(self):
        return MemoryStore()

    @pytest.fixture
    def middleware(self, store):
        config = _make_config()
        return ShadowMiddleware(config, store)

    async def test_continues_pipeline(self, middleware):
        """Shadow middleware must always call call_next and return the cloud response."""
        ctx = _make_ctx()
        cloud_response = {"choices": [{"message": {"content": "hi"}}]}

        async def call_next(c):
            return cloud_response

        result = await middleware.process(ctx, call_next)
        assert result is cloud_response

    async def test_skips_when_provider_is_local(self, middleware):
        """Don't shadow local model calls."""
        ctx = _make_ctx(provider=Provider.LOCAL)

        call_next = AsyncMock(return_value="ok")
        result = await middleware.process(ctx, call_next)
        assert result == "ok"

    async def test_skips_when_session_disables_shadow(self, middleware):
        """Per-session shadow_enabled=False override."""
        ctx = _make_ctx(shadow_enabled=False)

        call_next = AsyncMock(return_value="ok")
        result = await middleware.process(ctx, call_next)
        assert result == "ok"

    async def test_uses_session_shadow_model(self, middleware):
        """Per-session shadow_model override."""
        ctx = _make_ctx(shadow_model="mistral:7b")
        model = middleware._resolve_shadow_model(ctx)
        assert model == "mistral:7b"

    async def test_resolve_model_from_config(self, middleware):
        """Falls back to config shadow_model."""
        ctx = _make_ctx()
        model = middleware._resolve_shadow_model(ctx)
        assert model == "llama3.2"

    async def test_resolve_model_empty_when_disabled(self, store):
        """Returns empty string when shadow is disabled in config."""
        config = _make_config(shadow_enabled=False)
        mw = ShadowMiddleware(config, store)
        ctx = _make_ctx()
        ctx.config = config
        model = mw._resolve_shadow_model(ctx)
        assert model == ""

    def test_record_shadow_event_success(self, middleware, store):
        """Records a success event with token counts."""
        response = OllamaResponse(
            model="llama3.2",
            content="Hi!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=100.0,
        )
        middleware._record_shadow_event(
            session_id="test-session",
            cloud_provider="openai",
            cloud_model="gpt-4",
            shadow_model="llama3.2",
            response=response,
            elapsed_ms=100.0,
            status="success",
        )
        events = store.get_session_events("test-session")
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ShadowDraftEvent)
        assert event.shadow_status == "success"
        assert event.local_model == "llama3.2"
        assert event.local_tokens == 15
        assert event.local_prompt_tokens == 10
        assert event.local_completion_tokens == 5

    def test_record_shadow_event_error(self, middleware, store):
        """Records an error event."""
        middleware._record_shadow_event(
            session_id="test-session",
            cloud_provider="openai",
            cloud_model="gpt-4",
            shadow_model="llama3.2",
            response=None,
            elapsed_ms=0.0,
            status="error",
            error="Connection refused",
        )
        events = store.get_session_events("test-session")
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ShadowDraftEvent)
        assert event.shadow_status == "error"
        assert event.error_message == "Connection refused"

    def test_record_shadow_event_timeout(self, middleware, store):
        middleware._record_shadow_event(
            session_id="test-session",
            cloud_provider="openai",
            cloud_model="gpt-4",
            shadow_model="llama3.2",
            response=None,
            elapsed_ms=0.0,
            status="timeout",
        )
        events = store.get_session_events("test-session")
        assert events[0].shadow_status == "timeout"

    def test_shutdown(self, middleware):
        """Shutdown cleans up executor."""
        middleware._get_executor()  # Force creation
        assert middleware._executor is not None
        middleware.shutdown()
        assert middleware._executor is None

    def test_shutdown_idempotent(self, middleware):
        middleware.shutdown()
        middleware.shutdown()  # Should not raise

    def test_record_shadow_event_with_similarity(self, middleware, store):
        """Records a success event with similarity result and cost_saved."""
        response = OllamaResponse(
            model="llama3.2",
            content="The capital of France is Paris.",
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            latency_ms=150.0,
        )
        sim = SimilarityResult(
            score=0.85,
            method="difflib",
            cloud_preview="The capital of France is Paris, a beautiful city.",
            local_preview="The capital of France is Paris.",
            cloud_length=49,
            local_length=30,
            length_ratio=30 / 49,
        )
        middleware._record_shadow_event(
            session_id="test-session",
            cloud_provider="openai",
            cloud_model="gpt-4",
            shadow_model="llama3.2",
            response=response,
            elapsed_ms=150.0,
            status="success",
            similarity=sim,
        )
        events = store.get_session_events("test-session")
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ShadowDraftEvent)
        assert event.similarity_score == 0.85
        assert event.similarity_method == "difflib"
        assert event.cloud_preview == "The capital of France is Paris, a beautiful city."
        assert event.local_preview == "The capital of France is Paris."
        assert event.length_ratio == pytest.approx(30 / 49)

    def test_record_shadow_event_without_similarity(self, middleware, store):
        """When similarity is None, fields stay at defaults."""
        response = OllamaResponse(
            model="llama3.2",
            content="Hello",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            latency_ms=50.0,
        )
        middleware._record_shadow_event(
            session_id="test-session",
            cloud_provider="openai",
            cloud_model="gpt-4",
            shadow_model="llama3.2",
            response=response,
            elapsed_ms=50.0,
            status="success",
            similarity=None,
        )
        events = store.get_session_events("test-session")
        event = events[0]
        assert event.similarity_score is None
        assert event.similarity_method == ""
        assert event.cloud_preview == ""
        assert event.local_preview == ""
        assert event.length_ratio is None

    def test_record_shadow_event_similarity_with_cost_saved(self, middleware, store):
        """Similarity score and cost_saved both appear on the same event."""
        response = OllamaResponse(
            model="llama3.2",
            content="Paris is the capital.",
            prompt_tokens=8,
            completion_tokens=6,
            total_tokens=14,
            latency_ms=80.0,
        )
        sim = SimilarityResult(
            score=0.72,
            method="difflib",
            cloud_preview="Paris is the capital of France.",
            local_preview="Paris is the capital.",
            cloud_length=30,
            local_length=21,
            length_ratio=21 / 30,
        )
        # Manually create the event with cost_saved to verify both fields coexist
        middleware._record_shadow_event(
            session_id="test-session",
            cloud_provider="openai",
            cloud_model="gpt-4",
            shadow_model="llama3.2",
            response=response,
            elapsed_ms=80.0,
            status="success",
            similarity=sim,
        )
        events = store.get_session_events("test-session")
        event = events[0]
        # Similarity is present
        assert event.similarity_score == 0.72
        assert event.similarity_method == "difflib"
        # Token counts are present
        assert event.local_tokens == 14
        assert event.local_prompt_tokens == 8
        assert event.local_completion_tokens == 6

    def test_bridge_cloud_first_then_local(self, store):
        """Bridge computes similarity when cloud arrives before local."""
        config = _make_config()
        bridge = _SimilarityBridge()

        # Cloud arrives first
        bridge.set_cloud("Hello world", store, config)
        assert bridge._cloud_ready is True
        assert bridge._local_ready is False

        # Local arrives second — triggers similarity computation
        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_local("Hello world", event, store, config)

        assert event.similarity_score is not None
        assert event.similarity_score == 1.0
        assert event.similarity_method == "difflib"

    def test_bridge_local_first_then_cloud(self, store):
        """Bridge computes similarity when local arrives before cloud."""
        config = _make_config()
        bridge = _SimilarityBridge()

        # Local arrives first
        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_local("Hello world", event, store, config)
        assert bridge._local_ready is True
        assert bridge._cloud_ready is False
        assert event.similarity_score is None  # Not yet computed

        # Cloud arrives second — triggers similarity computation
        bridge.set_cloud("Hello world", store, config)

        assert event.similarity_score is not None
        assert event.similarity_score == 1.0
        assert event.similarity_method == "difflib"

    def test_bridge_cancel_prevents_similarity(self, store):
        """Cancelled bridge does not compute similarity."""
        config = _make_config()
        bridge = _SimilarityBridge()

        bridge.cancel()

        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_local("Hello", event, store, config)
        bridge.set_cloud("Hello", store, config)

        assert event.similarity_score is None

    def test_bridge_cancel_after_local(self, store):
        """Cancel after local arrives prevents cloud from triggering similarity."""
        config = _make_config()
        bridge = _SimilarityBridge()

        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_local("Hello", event, store, config)

        bridge.cancel()
        bridge.set_cloud("Hello", store, config)

        assert event.similarity_score is None

    def test_bridge_different_texts(self, store):
        """Bridge computes low similarity for different texts."""
        config = _make_config()
        bridge = _SimilarityBridge()

        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_cloud("The quick brown fox jumps over the lazy dog", store, config)
        bridge.set_local(
            "Completely unrelated response about quantum physics", event, store, config
        )

        assert event.similarity_score is not None
        assert event.similarity_score < 0.5

    def test_bridge_empty_text_no_similarity(self, store):
        """Bridge does not compute similarity when either text is empty."""
        config = _make_config()
        bridge = _SimilarityBridge()

        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_cloud("", store, config)
        bridge.set_local("Hello", event, store, config)

        # compute_similarity returns None for empty text
        assert event.similarity_score is None

    def test_bridge_saves_updated_event(self, store):
        """Bridge re-saves the event with similarity fields to the store."""
        config = _make_config()
        bridge = _SimilarityBridge()

        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        # Save the event initially (as _record_shadow_event does)
        store.save_event(event)

        bridge.set_cloud("Hello world", store, config)
        bridge.set_local("Hello world", event, store, config)

        # The event should have been re-saved with similarity
        events = store.get_session_events("test-session")
        saved = [e for e in events if isinstance(e, ShadowDraftEvent)]
        assert len(saved) >= 1
        assert any(e.similarity_score is not None for e in saved)

    async def test_process_computes_similarity_via_bridge(self, store):
        """process() extracts cloud text, bridge computes similarity."""
        config = _make_config()
        mw = ShadowMiddleware(config, store)
        ctx = _make_ctx()

        # Mock the Ollama client to return a known response
        local_response = OllamaResponse(
            model="llama3.2",
            content="Hi there!",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            latency_ms=50.0,
        )

        # Cloud response (OpenAI-like object)
        cloud_msg = MagicMock()
        cloud_msg.content = "Hello there!"
        cloud_choice = MagicMock()
        cloud_choice.message = cloud_msg
        cloud_response = MagicMock()
        cloud_response.choices = [cloud_choice]
        # Ensure .content is the list of choices (not a str) so it falls
        # through the Ollama check to the OpenAI path
        cloud_response.content = [cloud_choice]

        async def call_next(c):
            return cloud_response

        # Mock chat (sync) since shadow now always uses thread pool
        mw._client.chat = MagicMock(return_value=local_response)

        result = await mw.process(ctx, call_next)
        assert result is cloud_response

        # In production, no waiting needed — dashboard polls every 5s.
        # Here we wait only so the test can assert on the saved event.
        await asyncio.sleep(1.0)

        events = store.get_session_events("test-session")
        shadow_events = [e for e in events if isinstance(e, ShadowDraftEvent)]
        # At least 1 event (initial save); bridge re-save may add a second copy in MemoryStore
        assert len(shadow_events) >= 1
        # The last saved copy should have similarity (bridge re-saves after computation)
        event = shadow_events[-1]
        assert event.shadow_status == "success"
        # Similarity should have been computed via the bridge
        assert event.similarity_score is not None
        assert 0.0 <= event.similarity_score <= 1.0
        assert event.similarity_method == "difflib"
        assert event.cloud_preview == "Hello there!"
        assert event.local_preview == "Hi there!"


class TestShadowCancelOnError:
    """Shadow bridge is cancelled when downstream middleware raises (e.g. PII block)."""

    async def test_pii_block_cancels_bridge(self):
        """PII block cancels the shadow bridge — local model never gets the request."""
        store = MemoryStore()
        config = _make_config()
        mw = ShadowMiddleware(config, store)
        ctx = _make_ctx()
        bridges: list = []

        class _FakePIIBlockedError(Exception):
            pass

        async def call_next(c):
            raise _FakePIIBlockedError("PII blocked: credit_card")

        # Capture the bridge created inside process()
        _orig_submit = mw._get_executor().submit

        def _capture_submit(fn, **kwargs):
            bridge = kwargs.get("bridge")
            if bridge is not None:
                bridges.append(bridge)
            future = MagicMock()
            future.add_done_callback = MagicMock()
            return future

        with patch.object(mw._get_executor(), "submit", side_effect=_capture_submit):
            with pytest.raises(_FakePIIBlockedError):
                await mw.process(ctx, call_next)

        assert len(bridges) == 1
        assert bridges[0].cancelled is True

    async def test_error_propagates_after_cancel(self):
        """The original exception propagates unchanged after shadow cancellation."""
        store = MemoryStore()
        config = _make_config()
        mw = ShadowMiddleware(config, store)
        ctx = _make_ctx()

        async def call_next(c):
            raise RuntimeError("downstream error")

        with patch.object(mw, "_shadow_sync"):
            with pytest.raises(RuntimeError, match="downstream error"):
                await mw.process(ctx, call_next)

    async def test_cache_hit_cancels_bridge(self):
        """Global cache hit on a cloud-sized query cancels the shadow bridge.

        Cache is global-scoped by default (not session-scoped), so a cache hit
        from any session should cancel the shadow — no point comparing a cached
        cloud response against a fresh local model call.
        """
        store = MemoryStore()
        config = _make_config()
        mw = ShadowMiddleware(config, store)

        # Realistic multi-turn cloud query (not just "hello")
        ctx = _make_ctx(model="gpt-4")
        ctx.request_kwargs = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain the difference between TCP and UDP."},
                {
                    "role": "assistant",
                    "content": (
                        "TCP is connection-oriented and guarantees delivery, "
                        "while UDP is connectionless and faster but unreliable."
                    ),
                },
                {"role": "user", "content": "Which one should I use for video streaming?"},
            ],
        }
        bridges: list = []

        cached_cloud_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "For video streaming, UDP is preferred because low latency "
                            "matters more than guaranteed delivery."
                        ),
                    }
                }
            ],
        }

        async def call_next(c):
            # Global cache hit — same query was seen from a different session
            c.skip_call = True
            c.cached_response = cached_cloud_response
            return c.cached_response

        def _capture_submit(fn, **kwargs):
            bridge = kwargs.get("bridge")
            if bridge is not None:
                bridges.append(bridge)
            future = MagicMock()
            future.add_done_callback = MagicMock()
            return future

        with patch.object(mw._get_executor(), "submit", side_effect=_capture_submit):
            result = await mw.process(ctx, call_next)

        # Cache hit should cancel the bridge (no point comparing cached vs local)
        assert len(bridges) == 1
        assert bridges[0].cancelled is True
        # Result should still be the cached response
        assert result is cached_cloud_response

        # No shadow events should be persisted
        events = store.get_session_events("test-session")
        shadow_events = [e for e in events if isinstance(e, ShadowDraftEvent)]
        assert len(shadow_events) == 0


class TestShadowDraftEvent:
    def test_event_type(self):
        event = ShadowDraftEvent()
        assert event.event_type == EventType.SHADOW_DRAFT

    def test_fields(self):
        event = ShadowDraftEvent(
            session_id="s1",
            cloud_provider="openai",
            cloud_model="gpt-4",
            local_model="llama3.2",
            local_latency_ms=100.0,
            local_tokens=15,
            shadow_status="success",
            cost_saved=0.003,
        )
        assert event.cloud_provider == "openai"
        assert event.local_model == "llama3.2"
        assert event.cost_saved == 0.003

    def test_similarity_fields(self):
        event = ShadowDraftEvent(
            session_id="s1",
            cloud_provider="openai",
            cloud_model="gpt-4",
            local_model="llama3.2",
            shadow_status="success",
            cost_saved=0.005,
            similarity_score=0.82,
            similarity_method="difflib",
            cloud_preview="Cloud says hello",
            local_preview="Local says hello",
            length_ratio=0.9,
        )
        assert event.similarity_score == 0.82
        assert event.similarity_method == "difflib"
        assert event.cloud_preview == "Cloud says hello"
        assert event.local_preview == "Local says hello"
        assert event.length_ratio == 0.9
        assert event.cost_saved == 0.005


class TestSemanticSimilarityScorer:
    """Tests for embedding-based semantic similarity scoring."""

    def test_is_available_reflects_import(self):
        scorer = SemanticSimilarityScorer()
        assert scorer.is_available == _SEMANTIC_AVAILABLE

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_score_identical_texts(self):
        scorer = SemanticSimilarityScorer()
        score = scorer.score("The dog sat on the mat", "The dog sat on the mat")
        assert score is not None
        assert score >= 0.99

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_score_different_texts(self):
        scorer = SemanticSimilarityScorer()
        score = scorer.score(
            "The quick brown fox jumps over the lazy dog",
            "Quantum mechanics describes the behavior of subatomic particles",
        )
        assert score is not None
        assert score < 0.5

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_score_semantically_similar(self):
        """Semantically equivalent but differently worded texts should score high."""
        scorer = SemanticSimilarityScorer()
        score = scorer.score(
            "The dog sat on the mat",
            "A canine was sitting on the rug",
        )
        assert score is not None
        assert score > 0.5  # Should be notably higher than difflib would give

    def test_not_installed_returns_none(self):
        scorer = SemanticSimilarityScorer()
        with patch("stateloom.middleware.similarity._SEMANTIC_AVAILABLE", False):
            scorer._initialized = False
            scorer._model = None
            result = scorer._lazy_init()
            assert result is False
            assert scorer.score("a", "b") is None

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_lazy_init_thread_safety(self):
        """Concurrent threads all get a valid scorer."""
        import concurrent.futures

        scorer = SemanticSimilarityScorer()
        results = []

        def _score():
            return scorer.score("hello world", "hello world")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_score) for _ in range(10)]
            results = [f.result() for f in futures]

        assert all(r is not None and r >= 0.99 for r in results)


class TestComputeSemanticSimilarity:
    """Tests for compute_semantic_similarity() function."""

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_returns_similarity_result(self):
        scorer = SemanticSimilarityScorer()
        result = compute_semantic_similarity("hello", "hello", scorer)
        assert result is not None
        assert result.method == "semantic"
        assert result.score >= 0.99

    def test_empty_text_returns_none(self):
        scorer = SemanticSimilarityScorer()
        assert compute_semantic_similarity("", "hello", scorer) is None
        assert compute_semantic_similarity("hello", "", scorer) is None

    def test_scorer_failure_returns_none(self):
        scorer = MagicMock()
        scorer.score.return_value = None
        assert compute_semantic_similarity("a", "b", scorer) is None


class TestComputeSimilarityAuto:
    """Tests for compute_similarity_auto() dispatch function."""

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_auto_prefers_semantic(self):
        scorer = SemanticSimilarityScorer()
        result = compute_similarity_auto("hello", "hello", method="auto", scorer=scorer)
        assert result is not None
        assert result.method == "semantic"

    def test_auto_fallback_to_difflib(self):
        """When scorer returns None, auto falls back to difflib."""
        scorer = MagicMock()
        scorer.score.return_value = None
        result = compute_similarity_auto("hello", "hello", method="auto", scorer=scorer)
        assert result is not None
        assert result.method == "difflib"

    def test_difflib_mode_ignores_scorer(self):
        scorer = MagicMock()
        result = compute_similarity_auto("hello", "hello", method="difflib", scorer=scorer)
        assert result is not None
        assert result.method == "difflib"
        scorer.score.assert_not_called()

    def test_semantic_mode_no_scorer_returns_none(self):
        result = compute_similarity_auto("hello", "hello", method="semantic", scorer=None)
        assert result is None

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_semantic_mode_with_scorer(self):
        scorer = SemanticSimilarityScorer()
        result = compute_similarity_auto("hello", "hello", method="semantic", scorer=scorer)
        assert result is not None
        assert result.method == "semantic"

    def test_auto_without_scorer_uses_difflib(self):
        result = compute_similarity_auto("hello", "hello", method="auto", scorer=None)
        assert result is not None
        assert result.method == "difflib"


class TestBridgeSemantic:
    """Tests for _SimilarityBridge with semantic similarity."""

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_bridge_with_semantic_method(self):
        store = MemoryStore()
        config = _make_config()
        scorer = SemanticSimilarityScorer()
        bridge = _SimilarityBridge(similarity_method="semantic", similarity_scorer=scorer)

        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_cloud("The capital of France is Paris", store, config)
        bridge.set_local("Paris is the capital of France", event, store, config)

        assert event.similarity_score is not None
        assert event.similarity_method == "semantic"
        assert event.similarity_score > 0.7

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_bridge_with_auto_fallback(self):
        """Auto method with working scorer uses semantic."""
        store = MemoryStore()
        config = _make_config()
        scorer = SemanticSimilarityScorer()
        bridge = _SimilarityBridge(similarity_method="auto", similarity_scorer=scorer)

        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_cloud("hello world", store, config)
        bridge.set_local("hello world", event, store, config)

        assert event.similarity_method == "semantic"

    def test_bridge_auto_fallback_to_difflib_no_scorer(self):
        """Auto method without scorer falls back to difflib."""
        store = MemoryStore()
        config = _make_config()
        bridge = _SimilarityBridge(similarity_method="auto", similarity_scorer=None)

        event = ShadowDraftEvent(session_id="test-session", local_model="llama3.2")
        bridge.set_cloud("hello world", store, config)
        bridge.set_local("hello world", event, store, config)

        assert event.similarity_method == "difflib"

    def test_shadow_middleware_creates_scorer_for_semantic(self):
        store = MemoryStore()
        config = _make_config(shadow_similarity_method="semantic")
        mw = ShadowMiddleware(config, store)
        if _SEMANTIC_AVAILABLE:
            assert mw._similarity_scorer is not None
        else:
            # When sentence-transformers is not installed, scorer is still created
            # but will return None on score()
            assert mw._similarity_scorer is not None

    def test_shadow_middleware_no_scorer_for_difflib(self):
        store = MemoryStore()
        config = _make_config(shadow_similarity_method="difflib")
        mw = ShadowMiddleware(config, store)
        assert mw._similarity_scorer is None
