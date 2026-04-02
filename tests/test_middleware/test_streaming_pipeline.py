"""Tests for streaming-aware middleware pipeline and StreamChunkInfo."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomKillSwitchError,
)
from stateloom.core.event import LLMCallEvent
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext, StreamChunkInfo
from stateloom.middleware.pipeline import Pipeline

# ── StreamChunkInfo unit tests ──


class TestStreamChunkInfo:
    def test_defaults(self):
        info = StreamChunkInfo()
        assert info.text_delta == ""
        assert info.finish_reason is None
        assert info.tool_call_delta is None
        assert info.prompt_tokens == 0
        assert info.completion_tokens == 0
        assert info.has_usage is False

    def test_with_values(self):
        info = StreamChunkInfo(
            text_delta="Hello",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            has_usage=True,
        )
        assert info.text_delta == "Hello"
        assert info.finish_reason == "stop"
        assert info.prompt_tokens == 10
        assert info.completion_tokens == 5
        assert info.has_usage is True

    def test_tool_call_delta(self):
        info = StreamChunkInfo(
            tool_call_delta={"id": "call_1", "type": "function"},
        )
        assert info.tool_call_delta == {"id": "call_1", "type": "function"}


# ── MiddlewareContext streaming fields ──


class TestMiddlewareContextStreamingFields:
    def test_on_stream_complete_default_empty(self):
        ctx = MiddlewareContext(
            session=Session(id="test"),
            config=StateLoomConfig(console_output=False),
        )
        assert ctx._on_stream_complete == []

    def test_stream_error_default_none(self):
        ctx = MiddlewareContext(
            session=Session(id="test"),
            config=StateLoomConfig(console_output=False),
        )
        assert ctx._stream_error is None

    def test_on_stream_complete_independent_per_instance(self):
        ctx1 = MiddlewareContext(
            session=Session(id="t1"),
            config=StateLoomConfig(console_output=False),
        )
        ctx2 = MiddlewareContext(
            session=Session(id="t2"),
            config=StateLoomConfig(console_output=False),
        )
        ctx1._on_stream_complete.append(lambda: None)
        assert len(ctx1._on_stream_complete) == 1
        assert len(ctx2._on_stream_complete) == 0


# ── Pipeline streaming entry points ──


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {"console_output": False}
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_session(**overrides) -> Session:
    defaults = {"id": "test-session"}
    defaults.update(overrides)
    return Session(**defaults)


class TestPipelineStreaming:
    def test_execute_streaming_sync_returns_ctx(self):
        """Streaming sync entry point should return a MiddlewareContext."""
        pipeline = Pipeline()
        ctx = pipeline.execute_streaming_sync(
            provider="openai",
            method="chat.completions.create",
            model="gpt-4o",
            request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
            session=_make_session(),
            config=_make_config(),
        )
        assert isinstance(ctx, MiddlewareContext)
        assert ctx.is_streaming is True
        assert ctx.provider == "openai"
        assert ctx.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_execute_streaming_async_returns_ctx(self):
        """Streaming async entry point should return a MiddlewareContext."""
        pipeline = Pipeline()
        ctx = await pipeline.execute_streaming_async(
            provider="anthropic",
            method="messages.create",
            model="claude-3-5-sonnet",
            request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
            session=_make_session(),
            config=_make_config(),
        )
        assert isinstance(ctx, MiddlewareContext)
        assert ctx.is_streaming is True

    def test_streaming_noop_terminal(self):
        """Terminal should return None (no LLM call)."""
        pipeline = Pipeline()
        ctx = pipeline.execute_streaming_sync(
            provider="openai",
            method="chat.completions.create",
            model="gpt-4o",
            request_kwargs={},
            session=_make_session(),
            config=_make_config(),
        )
        assert ctx.response is None
        assert ctx.skip_call is False

    def test_streaming_cache_hit(self):
        """Pre-call middleware can short-circuit with cached response."""

        class CacheMW:
            async def process(self, ctx, call_next):
                ctx.skip_call = True
                ctx.cached_response = {"cached": True}
                return await call_next(ctx)

        pipeline = Pipeline([CacheMW()])
        ctx = pipeline.execute_streaming_sync(
            provider="openai",
            method="chat.completions.create",
            model="gpt-4o",
            request_kwargs={},
            session=_make_session(),
            config=_make_config(),
        )
        assert ctx.skip_call is True
        assert ctx.cached_response == {"cached": True}

    def test_streaming_precall_middleware_runs(self):
        """Pre-call middleware should execute before the terminal."""
        calls = []

        class PreCallMW:
            async def process(self, ctx, call_next):
                calls.append("pre")
                result = await call_next(ctx)
                calls.append("post")
                return result

        pipeline = Pipeline([PreCallMW()])
        pipeline.execute_streaming_sync(
            provider="openai",
            method="chat.completions.create",
            model="gpt-4o",
            request_kwargs={},
            session=_make_session(),
            config=_make_config(),
        )
        assert calls == ["pre", "post"]

    def test_streaming_kill_switch_blocks(self):
        """Kill switch middleware should raise before streaming starts."""

        class KillSwitchMW:
            async def process(self, ctx, call_next):
                raise StateLoomKillSwitchError("test kill switch")

        pipeline = Pipeline([KillSwitchMW()])
        with pytest.raises(StateLoomKillSwitchError):
            pipeline.execute_streaming_sync(
                provider="openai",
                method="chat.completions.create",
                model="gpt-4o",
                request_kwargs={},
                session=_make_session(),
                config=_make_config(),
            )


# ── Middleware callback registration tests ──


class TestMiddlewareStreamingCallbacks:
    """Verify each middleware registers callbacks when ctx.is_streaming=True."""

    def _make_streaming_ctx(self, **overrides) -> MiddlewareContext:
        defaults = dict(
            session=_make_session(),
            config=_make_config(),
            provider="openai",
            model="gpt-4o",
            is_streaming=True,
        )
        defaults.update(overrides)
        return MiddlewareContext(**defaults)

    @pytest.mark.asyncio
    async def test_latency_tracker_registers_callback(self):
        from stateloom.middleware.latency_tracker import LatencyTracker

        tracker = LatencyTracker()
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await tracker.process(ctx, noop)
        assert len(ctx._on_stream_complete) == 1

        # Fire callback and verify latency was set
        ctx._on_stream_complete[0]()
        assert ctx.latency_ms > 0

    @pytest.mark.asyncio
    async def test_latency_tracker_inline_for_non_streaming(self):
        from stateloom.middleware.latency_tracker import LatencyTracker

        tracker = LatencyTracker()
        ctx = MiddlewareContext(
            session=_make_session(),
            config=_make_config(),
            provider="openai",
            model="gpt-4o",
            is_streaming=False,
        )

        async def noop(c):
            return None

        await tracker.process(ctx, noop)
        assert len(ctx._on_stream_complete) == 0
        assert ctx.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_cost_tracker_registers_callback(self):
        from stateloom.middleware.cost_tracker import CostTracker
        from stateloom.pricing.registry import PricingRegistry

        pricing = PricingRegistry()
        tracker = CostTracker(pricing=pricing)
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await tracker.process(ctx, noop)
        assert len(ctx._on_stream_complete) == 1

    @pytest.mark.asyncio
    async def test_cost_tracker_callback_creates_event(self):
        from stateloom.middleware.cost_tracker import CostTracker
        from stateloom.pricing.registry import PricingRegistry

        pricing = PricingRegistry()
        tracker = CostTracker(pricing=pricing)
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await tracker.process(ctx, noop)

        # Simulate stream completion with tokens
        ctx.prompt_tokens = 100
        ctx.completion_tokens = 50
        ctx._on_stream_complete[0]()

        assert len(ctx.events) == 1
        event = ctx.events[0]
        assert isinstance(event, LLMCallEvent)
        assert event.is_streaming is True
        assert event.prompt_tokens == 100
        assert event.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_console_output_registers_callback(self):
        from stateloom.export.console import ConsoleOutput

        output = ConsoleOutput(config=_make_config())
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await output.process(ctx, noop)
        assert len(ctx._on_stream_complete) == 1

    @pytest.mark.asyncio
    async def test_event_recorder_registers_callback(self):
        from stateloom.middleware.event_recorder import EventRecorder
        from stateloom.store.memory_store import MemoryStore

        store = MemoryStore()
        recorder = EventRecorder(store=store)
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await recorder.process(ctx, noop)
        assert len(ctx._on_stream_complete) == 1

    @pytest.mark.asyncio
    async def test_timeout_checker_registers_callback(self):
        from stateloom.middleware.timeout_checker import TimeoutCheckerMiddleware

        checker = TimeoutCheckerMiddleware()
        session = _make_session()
        session.timeout = 60.0
        session.heartbeat()
        ctx = self._make_streaming_ctx(session=session)

        async def noop(c):
            return None

        await checker.process(ctx, noop)
        assert len(ctx._on_stream_complete) == 1

    @pytest.mark.asyncio
    async def test_blast_radius_registers_callback(self):
        from stateloom.middleware.blast_radius import BlastRadiusMiddleware

        config = _make_config(blast_radius_enabled=True)
        middleware = BlastRadiusMiddleware(config=config)
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await middleware.process(ctx, noop)
        assert len(ctx._on_stream_complete) == 1

    @pytest.mark.asyncio
    async def test_blast_radius_callback_resets_on_success(self):
        from stateloom.middleware.blast_radius import BlastRadiusMiddleware

        config = _make_config(blast_radius_enabled=True)
        middleware = BlastRadiusMiddleware(config=config)

        # Record some failures first
        session = _make_session()
        ctx = self._make_streaming_ctx(session=session)
        middleware._failure_counts[session.id] = 2

        async def noop(c):
            return None

        await middleware.process(ctx, noop)

        # Fire callback (no error = success)
        ctx._on_stream_complete[0]()
        assert session.id not in middleware._failure_counts

    @pytest.mark.asyncio
    async def test_blast_radius_callback_records_failure(self):
        from stateloom.middleware.blast_radius import BlastRadiusMiddleware

        config = _make_config(blast_radius_enabled=True)
        middleware = BlastRadiusMiddleware(config=config)
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await middleware.process(ctx, noop)

        # Simulate stream error
        ctx._stream_error = RuntimeError("provider error")
        ctx._on_stream_complete[0]()

        assert middleware._failure_counts.get(ctx.session.id, 0) == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_registers_callback(self):
        from stateloom.middleware.circuit_breaker import (
            ProviderCircuitBreakerMiddleware,
        )

        config = _make_config(circuit_breaker_enabled=True)
        middleware = ProviderCircuitBreakerMiddleware(config=config)
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await middleware.process(ctx, noop)
        assert len(ctx._on_stream_complete) == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_callback_success(self):
        from stateloom.middleware.circuit_breaker import (
            ProviderCircuitBreakerMiddleware,
        )

        config = _make_config(circuit_breaker_enabled=True)
        middleware = ProviderCircuitBreakerMiddleware(config=config)
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await middleware.process(ctx, noop)

        # No error — circuit should stay closed
        ctx._on_stream_complete[0]()
        circuit = middleware._get_circuit("openai")
        assert circuit.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_callback_failure(self):
        from stateloom.middleware.circuit_breaker import (
            ProviderCircuitBreakerMiddleware,
        )

        config = _make_config(
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=3,
        )
        middleware = ProviderCircuitBreakerMiddleware(config=config)
        ctx = self._make_streaming_ctx()

        async def noop(c):
            return None

        await middleware.process(ctx, noop)

        # Simulate a provider error
        ctx._stream_error = RuntimeError("provider error")
        ctx._on_stream_complete[0]()

        circuit = middleware._get_circuit("openai")
        assert circuit.get_failure_count(300) == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_streaming_defers_release(self):
        from stateloom.core.organization import Team
        from stateloom.middleware.rate_limiter import RateLimiterMiddleware

        team = Team(id="team-1", name="TestTeam", rate_limit_tps=10.0)
        middleware = RateLimiterMiddleware(
            team_lookup=lambda tid: team if tid == "team-1" else None,
        )
        session = _make_session(team_id="team-1")
        ctx = self._make_streaming_ctx(session=session)

        async def noop(c):
            return None

        await middleware.process(ctx, noop)

        # Verify callback registered (bucket release deferred)
        bucket_callbacks = [cb for cb in ctx._on_stream_complete]
        assert len(bucket_callbacks) >= 1

        bucket = middleware._get_or_create_bucket(team)
        active_before = bucket.active_count

        # Fire callback — should release the bucket
        for cb in ctx._on_stream_complete:
            cb()

        assert bucket.active_count == active_before - 1


# ── Shadow skip streaming test ──


class TestShadowStreaming:
    @pytest.mark.asyncio
    async def test_shadow_skips_streaming(self):
        from stateloom.middleware.shadow import ShadowMiddleware
        from stateloom.store.memory_store import MemoryStore

        config = _make_config(shadow_enabled=True, shadow_model="llama3.2")
        store = MemoryStore()
        middleware = ShadowMiddleware(config=config, store=store)

        ctx = MiddlewareContext(
            session=_make_session(),
            config=config,
            provider="openai",
            model="gpt-4o",
            is_streaming=True,
        )

        calls = []

        async def call_next(c):
            calls.append("next")
            return None

        await middleware.process(ctx, call_next)
        assert calls == ["next"]
        # No shadow thread should have been launched
        assert len(ctx._on_stream_complete) == 0


# ── extract_chunk_info tests ──


class TestExtractChunkInfo:
    def test_openai_extract_chunk_info(self):
        from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter()

        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello", tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.text_delta == "Hello"
        assert info.finish_reason is None
        assert info.has_usage is False

    def test_openai_extract_chunk_with_usage(self):
        from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter()

        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.finish_reason == "stop"
        assert info.prompt_tokens == 10
        assert info.completion_tokens == 5
        assert info.has_usage is True

    def test_anthropic_extract_text_delta(self):
        from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter()

        chunk = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="World"),
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.text_delta == "World"

    def test_anthropic_extract_message_start(self):
        from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter()

        chunk = SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=50),
            ),
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.prompt_tokens == 50
        assert info.has_usage is True

    def test_anthropic_extract_message_delta(self):
        from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter()

        chunk = SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(output_tokens=20),
            delta=SimpleNamespace(stop_reason="end_turn"),
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.completion_tokens == 20
        assert info.finish_reason == "end_turn"
        assert info.has_usage is True

    def test_gemini_extract_chunk_info(self):
        from stateloom.intercept.adapters.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter()

        chunk = SimpleNamespace(
            text="Hi there",
            usage_metadata=SimpleNamespace(
                prompt_token_count=15,
                candidates_token_count=3,
            ),
            candidates=None,
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.text_delta == "Hi there"
        assert info.prompt_tokens == 15
        assert info.completion_tokens == 3
        assert info.has_usage is True

    def test_mistral_extract_chunk_info(self):
        from stateloom.intercept.adapters.mistral_adapter import MistralAdapter

        adapter = MistralAdapter()

        chunk = SimpleNamespace(
            data=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Bonjour"),
                        finish_reason=None,
                    ),
                ],
                usage=None,
            ),
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.text_delta == "Bonjour"

    def test_base_adapter_extract_chunk_info(self):
        from stateloom.intercept.provider_adapter import BaseProviderAdapter

        adapter = BaseProviderAdapter()
        info = adapter.extract_chunk_info(object())
        assert info.text_delta == ""
        assert info.has_usage is False

    def test_litellm_extract_chunk_info(self):
        from stateloom.intercept.adapters.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter()

        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello"),
                    finish_reason=None,
                ),
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.text_delta == "Hello"
        assert info.prompt_tokens == 5
        assert info.completion_tokens == 2
        assert info.has_usage is True


# ── Callback ordering test ──


class TestCallbackOrdering:
    """Verify callbacks fire in registration order (chain unwind order)."""

    @pytest.mark.asyncio
    async def test_callback_order(self):
        order = []

        class MW1:
            async def process(self, ctx, call_next):
                result = await call_next(ctx)
                if ctx.is_streaming:
                    ctx._on_stream_complete.append(lambda: order.append("MW1"))
                return result

        class MW2:
            async def process(self, ctx, call_next):
                result = await call_next(ctx)
                if ctx.is_streaming:
                    ctx._on_stream_complete.append(lambda: order.append("MW2"))
                return result

        class MW3:
            async def process(self, ctx, call_next):
                result = await call_next(ctx)
                if ctx.is_streaming:
                    ctx._on_stream_complete.append(lambda: order.append("MW3"))
                return result

        pipeline = Pipeline([MW1(), MW2(), MW3()])
        ctx = await pipeline.execute_streaming_async(
            provider="openai",
            method="chat.completions.create",
            model="gpt-4o",
            request_kwargs={},
            session=_make_session(),
            config=_make_config(),
        )

        # Fire callbacks
        for cb in ctx._on_stream_complete:
            cb()

        # Callbacks registered in chain unwind order: MW3 → MW2 → MW1
        assert order == ["MW3", "MW2", "MW1"]


# ── Error handling tests ──


class TestStreamErrorHandling:
    def test_stream_error_captured_in_ctx(self):
        """When a stream errors, ctx._stream_error should be set."""
        from stateloom.intercept.generic_interceptor import _wrap_stream_sync

        gate = MagicMock()
        adapter = MagicMock()
        adapter.extract_stream_tokens.side_effect = lambda c, a: a
        session = MagicMock()
        session.id = "test"
        ctx = MiddlewareContext(
            session=Session(id="test"),
            config=_make_config(),
            is_streaming=True,
        )

        def failing_stream():
            yield "chunk1"
            raise RuntimeError("mid-stream failure")

        gen = _wrap_stream_sync(gate, adapter, failing_stream(), session, "gpt-4o", 1, {}, ctx=ctx)

        # Consume until error
        chunks = []
        with pytest.raises(RuntimeError, match="mid-stream failure"):
            for chunk in gen:
                chunks.append(chunk)

        assert chunks == ["chunk1"]
        assert ctx._stream_error is not None
        assert isinstance(ctx._stream_error, RuntimeError)

    def test_callbacks_fire_on_error(self):
        """Callbacks should fire even when the stream errors."""
        from stateloom.intercept.generic_interceptor import _wrap_stream_sync

        gate = MagicMock()
        adapter = MagicMock()
        adapter.extract_stream_tokens.side_effect = lambda c, a: a
        session = MagicMock()
        session.id = "test"
        ctx = MiddlewareContext(
            session=Session(id="test"),
            config=_make_config(),
            is_streaming=True,
        )
        fired = []
        ctx._on_stream_complete.append(lambda: fired.append(True))

        def failing_stream():
            raise RuntimeError("immediate failure")
            yield  # noqa: F401

        gen = _wrap_stream_sync(gate, adapter, failing_stream(), session, "gpt-4o", 1, {}, ctx=ctx)

        with pytest.raises(RuntimeError):
            list(gen)

        assert fired == [True]

    def test_legacy_path_when_ctx_is_none(self):
        """When ctx=None, legacy direct-to-store behavior should work."""
        from stateloom.intercept.generic_interceptor import _wrap_stream_sync

        gate = MagicMock()
        gate.pricing.calculate_cost.return_value = 0.01

        adapter = MagicMock()
        adapter.name = "openai"
        adapter.extract_stream_tokens.side_effect = lambda c, a: a

        session = MagicMock()
        session.id = "test"

        def simple_stream():
            yield "chunk1"
            yield "chunk2"

        gen = _wrap_stream_sync(gate, adapter, simple_stream(), session, "gpt-4o", 1, {}, ctx=None)
        chunks = list(gen)

        assert chunks == ["chunk1", "chunk2"]
        # Legacy path saves event directly to store
        gate.store.save_event.assert_called_once()
        event = gate.store.save_event.call_args[0][0]
        assert isinstance(event, LLMCallEvent)
        assert event.is_streaming is True


# ── Integration test: full pipeline with streaming ──


class TestStreamingIntegration:
    def test_full_pipeline_streaming(self):
        """Integration test: pipeline pre-call runs, callbacks fire after stream."""
        from stateloom.middleware.cost_tracker import CostTracker
        from stateloom.middleware.event_recorder import EventRecorder
        from stateloom.middleware.latency_tracker import LatencyTracker
        from stateloom.pricing.registry import PricingRegistry
        from stateloom.store.memory_store import MemoryStore

        store = MemoryStore()
        pricing = PricingRegistry()

        pipeline = Pipeline(
            [
                LatencyTracker(),
                CostTracker(pricing=pricing),
                EventRecorder(store=store),
            ]
        )

        ctx = pipeline.execute_streaming_sync(
            provider="openai",
            method="chat.completions.create",
            model="gpt-4o",
            request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
            session=_make_session(),
            config=_make_config(),
        )

        assert ctx.is_streaming is True
        assert len(ctx._on_stream_complete) == 3  # latency, cost, event_recorder

        # Simulate stream completion with token data
        ctx.prompt_tokens = 50
        ctx.completion_tokens = 25

        for cb in ctx._on_stream_complete:
            cb()

        # Verify latency was tracked
        assert ctx.latency_ms > 0

        # Verify LLMCallEvent was created
        llm_events = [e for e in ctx.events if isinstance(e, LLMCallEvent)]
        assert len(llm_events) == 1
        assert llm_events[0].prompt_tokens == 50
        assert llm_events[0].completion_tokens == 25
        assert llm_events[0].is_streaming is True
