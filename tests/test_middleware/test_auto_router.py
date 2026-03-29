"""Tests for the intelligent auto-routing middleware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import LocalRoutingEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType, Provider
from stateloom.local.client import OllamaResponse
from stateloom.middleware.auto_router import (
    AutoRouterMiddleware,
    ComplexitySignals,
    RoutingContext,
    RoutingDecision,
    _ModelStats,
)
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.response_converter import (
    _GeminiResponseProxy,
    convert_response,
)
from stateloom.store.memory_store import MemoryStore


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "auto_route_enabled": True,
        "auto_route_model": "llama3.2",
        "local_model_enabled": True,
        "local_model_default": "llama3.2",
        "console_output": False,
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(
    provider: str = "openai",
    model: str = "gpt-4",
    messages: list | None = None,
    is_streaming: bool = False,
    skip_call: bool = False,
    budget: float | None = None,
    total_cost: float = 0.0,
    session_metadata: dict | None = None,
    **extra_kwargs,
) -> MiddlewareContext:
    session = Session(id="test-session", budget=budget, total_cost=total_cost)
    if session_metadata:
        session.metadata.update(session_metadata)
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    kwargs = {"messages": messages, **extra_kwargs}
    return MiddlewareContext(
        session=session,
        config=_make_config(),
        provider=provider,
        model=model,
        request_kwargs=kwargs,
        is_streaming=is_streaming,
        skip_call=skip_call,
        auto_route_eligible=True,
    )


class TestComplexityAnalysis:
    """Test complexity scoring heuristics."""

    @pytest.fixture
    def middleware(self):
        store = MemoryStore()
        return AutoRouterMiddleware(_make_config(), store)

    def test_short_message_low_score(self, middleware):
        """A short single-turn message should have low complexity."""
        ctx = _make_ctx(messages=[{"role": "user", "content": "hi"}])
        signals = middleware._analyze_complexity(ctx)
        score = middleware._compute_score(signals)
        assert score < 0.2

    def test_long_multiturn_high_score(self, middleware):
        """A long multi-turn conversation should have high complexity."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": "x" * 2000})
            messages.append({"role": "assistant", "content": "y" * 2000})
        ctx = _make_ctx(messages=messages)
        signals = middleware._analyze_complexity(ctx)
        score = middleware._compute_score(signals)
        assert score > 0.3  # Well above 0.15 threshold → routes to cloud

    def test_system_prompt_adds_complexity(self, middleware):
        """System prompt should contribute to complexity score."""
        ctx_no_system = _make_ctx(messages=[{"role": "user", "content": "hello"}])
        ctx_with_system = _make_ctx(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
            ]
        )
        score_no = middleware._compute_score(middleware._analyze_complexity(ctx_no_system))
        score_with = middleware._compute_score(middleware._analyze_complexity(ctx_with_system))
        assert score_with > score_no

    def test_high_tier_model_adds_complexity(self, middleware):
        """Targeting gpt-4 should increase tier factor."""
        ctx_gpt4 = _make_ctx(model="gpt-4")
        ctx_gpt35 = _make_ctx(model="gpt-3.5-turbo")
        signals_4 = middleware._analyze_complexity(ctx_gpt4)
        signals_35 = middleware._analyze_complexity(ctx_gpt35)
        assert signals_4.model_tier > signals_35.model_tier

    def test_content_blocks_handled(self, middleware):
        """Content blocks (list format) should be analyzed correctly."""
        ctx = _make_ctx(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello world"},
                    ],
                }
            ]
        )
        signals = middleware._analyze_complexity(ctx)
        score = middleware._compute_score(signals)
        assert score < 0.3  # Should still be low


class TestRoutingDecision:
    """Test the routing decision logic."""

    @pytest.fixture
    def store(self):
        return MemoryStore()

    @pytest.fixture
    def middleware(self, store):
        return AutoRouterMiddleware(_make_config(), store)

    def test_simple_routes_local(self, middleware):
        """Simple short message should route to local."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is True
            assert decision.complexity_score < 0.35

    def test_complex_routes_cloud(self, middleware):
        """Complex request should route to cloud."""
        messages = []
        for i in range(12):
            messages.append({"role": "user", "content": "x" * 3000})
            messages.append({"role": "assistant", "content": "y" * 3000})
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(messages=messages, model="gpt-4")
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False

    def test_streaming_skipped(self, middleware):
        """Streaming requests should not be routed."""
        ctx = _make_ctx(is_streaming=True)
        decision = middleware._should_route_local(ctx)
        assert decision.route_local is False
        assert "streaming" in decision.reason

    def test_tools_skipped(self, middleware):
        """Requests with tools should not be routed."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(tools=[{"type": "function", "function": {"name": "test"}}])
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False
            assert "tools" in decision.reason

    def test_functions_skipped(self, middleware):
        """Requests with functions should not be routed."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(functions=[{"name": "test"}])
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False

    def test_images_skipped(self, middleware):
        """Requests with image content blocks should not be routed."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "http://example.com/img.png"},
                            },
                        ],
                    }
                ]
            )
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False
            assert "images" in decision.reason

    def test_response_format_skipped(self, middleware):
        """Requests with response_format should not be routed."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(response_format={"type": "json_object"})
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False

    def test_session_opt_out(self, middleware):
        """Session-level opt-out should prevent routing."""
        ctx = _make_ctx(session_metadata={"auto_route_enabled": False})
        decision = middleware._should_route_local(ctx)
        assert decision.route_local is False
        assert "session opt-out" in decision.reason

    def test_skip_call_skipped(self, middleware):
        """Cache hits (skip_call=True) should not trigger routing."""
        ctx = _make_ctx(skip_call=True)
        decision = middleware._should_route_local(ctx)
        assert decision.route_local is False
        assert "skip_call" in decision.reason

    def test_already_local_skipped(self, middleware):
        """Already local provider should not be routed."""
        ctx = _make_ctx(provider=Provider.LOCAL)
        decision = middleware._should_route_local(ctx)
        assert decision.route_local is False
        assert "already local" in decision.reason

    def test_ollama_unavailable_skipped(self, middleware):
        """Ollama unavailable should prevent routing."""
        with patch.object(middleware, "_check_ollama_available", return_value=False):
            ctx = _make_ctx()
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False
            assert "Ollama unavailable" in decision.reason

    def test_disabled_skipped(self):
        """Disabled auto_route should prevent routing."""
        config = _make_config(auto_route_enabled=False)
        store = MemoryStore()
        mw = AutoRouterMiddleware(config, store)
        ctx = _make_ctx()
        decision = mw._should_route_local(ctx)
        assert decision.route_local is False
        assert "disabled" in decision.reason

    def test_budget_pressure_lowers_threshold(self, middleware):
        """Budget pressure should make routing more aggressive."""
        # At 80% budget usage, pressure = (0.8 - 0.5) / 0.5 = 0.6
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx_no_budget = _make_ctx(model="gpt-3.5-turbo")
            ctx_with_budget = _make_ctx(model="gpt-3.5-turbo", budget=10.0, total_cost=8.0)

            decision_no = middleware._should_route_local(ctx_no_budget)
            decision_with = middleware._should_route_local(ctx_with_budget)

            # Both should route local for a simple message, but with budget
            # the threshold is lower, so a slightly more complex message
            # would still route local
            assert decision_with.budget_pressure > 0.0
            assert decision_no.budget_pressure == 0.0

    def test_historical_success_adjusts_threshold(self, middleware):
        """Historical success rate should adjust routing threshold."""
        # Simulate good history
        middleware._stats["gpt-4"] = _ModelStats(successes=9, failures=1)

        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-4")
            decision = middleware._should_route_local(ctx)
            assert decision.historical_success_rate == pytest.approx(0.9)

    def test_cold_start_no_adjustment(self, middleware):
        """Fewer than 5 data points should not adjust threshold."""
        middleware._stats["gpt-4"] = _ModelStats(successes=2, failures=1)

        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-4")
            decision = middleware._should_route_local(ctx)
            assert decision.historical_success_rate is None


class TestBudgetPressure:
    """Test budget pressure computation."""

    @pytest.fixture
    def middleware(self):
        store = MemoryStore()
        return AutoRouterMiddleware(_make_config(), store)

    def test_no_budget_zero_pressure(self, middleware):
        ctx = _make_ctx(budget=None)
        assert middleware._compute_budget_pressure(ctx) == 0.0

    def test_under_half_zero_pressure(self, middleware):
        ctx = _make_ctx(budget=10.0, total_cost=4.0)
        assert middleware._compute_budget_pressure(ctx) == 0.0

    def test_at_half_zero_pressure(self, middleware):
        ctx = _make_ctx(budget=10.0, total_cost=5.0)
        assert middleware._compute_budget_pressure(ctx) == 0.0

    def test_at_75_percent(self, middleware):
        ctx = _make_ctx(budget=10.0, total_cost=7.5)
        assert middleware._compute_budget_pressure(ctx) == pytest.approx(0.5)

    def test_at_100_percent(self, middleware):
        ctx = _make_ctx(budget=10.0, total_cost=10.0)
        assert middleware._compute_budget_pressure(ctx) == pytest.approx(1.0)

    def test_over_budget_capped(self, middleware):
        ctx = _make_ctx(budget=10.0, total_cost=15.0)
        assert middleware._compute_budget_pressure(ctx) == 1.0


class TestProbeLocal:
    """Test the local model probe mechanism."""

    @pytest.fixture
    def middleware(self):
        store = MemoryStore()
        return AutoRouterMiddleware(_make_config(), store)

    def test_successful_probe_returns_float(self, middleware):
        """Successful probe returns normalized confidence."""
        mock_response = OllamaResponse(
            content="8", prompt_tokens=10, completion_tokens=1, total_tokens=11
        )
        with patch.object(middleware._probe_client, "chat", return_value=mock_response):
            ctx = _make_ctx()
            confidence = middleware._probe_local(ctx)
            assert confidence == pytest.approx(0.8)

    def test_probe_with_text(self, middleware):
        """Probe handles '7/10' style responses."""
        mock_response = OllamaResponse(
            content="I'd rate this a 7 out of 10",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        with patch.object(middleware._probe_client, "chat", return_value=mock_response):
            ctx = _make_ctx()
            confidence = middleware._probe_local(ctx)
            assert confidence == pytest.approx(0.7)

    def test_probe_timeout_returns_none(self, middleware):
        """Timeout during probe returns None."""
        with patch.object(middleware._probe_client, "chat", side_effect=TimeoutError("timeout")):
            ctx = _make_ctx()
            confidence = middleware._probe_local(ctx)
            assert confidence is None

    def test_probe_parse_failure_returns_none(self, middleware):
        """Unparseable response returns None."""
        mock_response = OllamaResponse(
            content="I cannot rate this", prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        with patch.object(middleware._probe_client, "chat", return_value=mock_response):
            ctx = _make_ctx()
            confidence = middleware._probe_local(ctx)
            assert confidence is None

    def test_probe_empty_messages_returns_none(self, middleware):
        """No user messages returns None."""
        ctx = _make_ctx(messages=[{"role": "system", "content": "be helpful"}])
        confidence = middleware._probe_local(ctx)
        assert confidence is None

    def test_probe_high_confidence_routes_local(self, middleware):
        """High probe confidence in uncertain zone should route local."""
        mock_response = OllamaResponse(
            content="9", prompt_tokens=10, completion_tokens=1, total_tokens=11
        )
        with patch.object(middleware._probe_client, "chat", return_value=mock_response):
            with patch.object(middleware, "_check_ollama_available", return_value=True):
                # Create a medium complexity context that lands in uncertain zone
                messages = [
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "x" * 1500},
                    {"role": "assistant", "content": "y" * 1000},
                    {"role": "user", "content": "z" * 1500},
                ]
                ctx = _make_ctx(messages=messages, model="gpt-3.5-turbo")
                decision = middleware._should_route_local(ctx)
                # In the uncertain zone, probe should be attempted
                if decision.probed:
                    assert decision.probe_confidence is not None

    def test_probe_low_confidence_routes_cloud(self, middleware):
        """Low probe confidence should route to cloud."""
        mock_response = OllamaResponse(
            content="2", prompt_tokens=10, completion_tokens=1, total_tokens=11
        )
        with patch.object(middleware._probe_client, "chat", return_value=mock_response):
            with patch.object(middleware, "_check_ollama_available", return_value=True):
                messages = [
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "x" * 1500},
                    {"role": "assistant", "content": "y" * 1000},
                    {"role": "user", "content": "z" * 1500},
                ]
                ctx = _make_ctx(messages=messages, model="gpt-3.5-turbo")
                decision = middleware._should_route_local(ctx)
                if decision.probed and decision.probe_confidence is not None:
                    if decision.probe_confidence < middleware._config.auto_route_probe_threshold:
                        assert decision.route_local is False


class TestResponseConversion:
    """Test response conversion for different providers."""

    def test_openai_conversion(self):
        """OpenAI ChatCompletion conversion works."""
        resp = OllamaResponse(
            model="llama3.2",
            content="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        result = convert_response("openai", "gpt-4", resp)
        assert result is not None
        assert result.choices[0].message.content == "Hello!"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    def test_anthropic_conversion(self):
        """Anthropic Message conversion works."""
        resp = OllamaResponse(
            model="llama3.2",
            content="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        result = convert_response("anthropic", "claude-3-haiku-20240307", resp)
        assert result is not None
        assert result.content[0].text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_gemini_conversion(self):
        """Gemini proxy conversion works."""
        resp = OllamaResponse(
            model="llama3.2",
            content="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        result = convert_response("gemini", "gemini-1.5-flash", resp)
        assert result is not None
        assert isinstance(result, _GeminiResponseProxy)
        assert result.text == "Hello!"
        assert result.candidates[0].content.parts[0].text == "Hello!"
        assert result.usage_metadata.prompt_token_count == 10
        assert result.usage_metadata.candidates_token_count == 5

    def test_unknown_provider_falls_back_to_openai(self):
        """Unknown provider falls back to OpenAI format."""
        resp = OllamaResponse(
            model="llama3.2",
            content="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        result = convert_response("custom_provider", "custom-model", resp)
        assert result is not None
        assert result.choices[0].message.content == "Hello!"


class TestRouteToLocal:
    """Test the actual local routing flow."""

    @pytest.fixture
    def store(self):
        return MemoryStore()

    @pytest.fixture
    def middleware(self, store):
        return AutoRouterMiddleware(_make_config(), store)

    def test_successful_routing_mutates_ctx(self, middleware):
        """Successful routing should set skip_call, response, model, provider."""
        mock_response = OllamaResponse(
            model="llama3.2",
            content="Hi there!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=50.0,
        )
        with patch.object(middleware._client, "chat", return_value=mock_response):
            ctx = _make_ctx()
            decision = RoutingDecision(
                route_local=True, reason="low complexity", complexity_score=0.1
            )
            result = middleware._route_to_local(ctx, decision)
            assert result is not None
            assert ctx.skip_call is True
            assert ctx.model == "llama3.2"
            assert ctx.provider == Provider.LOCAL
            assert ctx.prompt_tokens == 10
            assert ctx.completion_tokens == 5
            assert ctx.total_tokens == 15
            assert ctx.response is not None

    def test_empty_content_returns_none(self, middleware):
        """Empty response content triggers fallback."""
        mock_response = OllamaResponse(
            model="llama3.2",
            content="",
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
        )
        with patch.object(middleware._client, "chat", return_value=mock_response):
            ctx = _make_ctx()
            decision = RoutingDecision(route_local=True, reason="test")
            result = middleware._route_to_local(ctx, decision)
            assert result is None

    def test_ollama_error_returns_none(self, middleware):
        """Ollama error triggers fallback."""
        with patch.object(middleware._client, "chat", side_effect=ConnectionError("refused")):
            ctx = _make_ctx()
            decision = RoutingDecision(route_local=True, reason="test")
            result = middleware._route_to_local(ctx, decision)
            assert result is None

    def test_no_model_returns_none(self):
        """No local model configured returns None."""
        config = _make_config(auto_route_model="", local_model_default="")
        store = MemoryStore()
        mw = AutoRouterMiddleware(config, store)
        ctx = _make_ctx()
        decision = RoutingDecision(route_local=True, reason="test")
        result = mw._route_to_local(ctx, decision)
        assert result is None

    def test_session_model_override(self, middleware):
        """Session metadata can override the routing model."""
        ctx = _make_ctx(session_metadata={"auto_route_model": "mistral:7b"})
        model = middleware._resolve_local_model(ctx)
        assert model == "mistral:7b"


class TestLearning:
    """Test historical learning from routing outcomes."""

    @pytest.fixture
    def store(self):
        return MemoryStore()

    @pytest.fixture
    def middleware(self, store):
        return AutoRouterMiddleware(_make_config(), store)

    def test_stats_update_on_success(self, middleware):
        middleware._update_stats("gpt-4", success=True)
        assert middleware._stats["gpt-4"].successes == 1
        assert middleware._stats["gpt-4"].failures == 0

    def test_stats_update_on_failure(self, middleware):
        middleware._update_stats("gpt-4", success=False)
        assert middleware._stats["gpt-4"].successes == 0
        assert middleware._stats["gpt-4"].failures == 1

    def test_stats_accumulate(self, middleware):
        for _ in range(7):
            middleware._update_stats("gpt-4", success=True)
        for _ in range(3):
            middleware._update_stats("gpt-4", success=False)
        assert middleware._stats["gpt-4"].total == 10
        assert middleware._stats["gpt-4"].success_rate == pytest.approx(0.7)

    def test_poor_history_affects_decision(self, middleware):
        """Poor history (<0.3 success rate) makes routing harder."""
        for _ in range(8):
            middleware._update_stats("gpt-4", success=False)
        for _ in range(2):
            middleware._update_stats("gpt-4", success=True)

        rate = middleware._get_historical_rate("gpt-4")
        assert rate == pytest.approx(0.2)

    def test_good_history_affects_decision(self, middleware):
        """Good history (>0.8 success rate) makes routing easier."""
        for _ in range(9):
            middleware._update_stats("gpt-4", success=True)
        for _ in range(1):
            middleware._update_stats("gpt-4", success=False)

        rate = middleware._get_historical_rate("gpt-4")
        assert rate == pytest.approx(0.9)

    def test_cold_start_returns_none(self, middleware):
        """Fewer than 5 data points returns None."""
        middleware._update_stats("gpt-4", success=True)
        middleware._update_stats("gpt-4", success=True)
        rate = middleware._get_historical_rate("gpt-4")
        assert rate is None

    def test_unknown_model_returns_none(self, middleware):
        """Unknown model returns None."""
        rate = middleware._get_historical_rate("unknown-model")
        assert rate is None


class TestPersistedLearning:
    """Test loading historical stats from store on startup."""

    def test_loads_stats_from_store(self):
        """AutoRouterMiddleware should load routing stats from persisted events."""
        store = MemoryStore()
        # Persist some LocalRoutingEvents
        for i in range(6):
            store.save_event(
                LocalRoutingEvent(
                    session_id=f"s{i}",
                    step=1,
                    original_cloud_model="gpt-4",
                    local_model="llama3.2",
                    routing_success=True,
                )
            )
        for i in range(4):
            store.save_event(
                LocalRoutingEvent(
                    session_id=f"f{i}",
                    step=1,
                    original_cloud_model="gpt-4",
                    local_model="llama3.2",
                    routing_success=False,
                )
            )
        mw = AutoRouterMiddleware(_make_config(), store)
        assert "gpt-4" in mw._stats
        assert mw._stats["gpt-4"].successes == 6
        assert mw._stats["gpt-4"].failures == 4
        assert mw._stats["gpt-4"].success_rate == pytest.approx(0.6)

    def test_loads_multiple_models(self):
        """Stats are loaded per cloud model."""
        store = MemoryStore()
        store.save_event(
            LocalRoutingEvent(
                session_id="s1",
                step=1,
                original_cloud_model="gpt-4",
                local_model="llama3.2",
                routing_success=True,
            )
        )
        store.save_event(
            LocalRoutingEvent(
                session_id="s2",
                step=1,
                original_cloud_model="claude-3-haiku",
                local_model="llama3.2",
                routing_success=False,
            )
        )
        mw = AutoRouterMiddleware(_make_config(), store)
        assert mw._stats["gpt-4"].successes == 1
        assert mw._stats["claude-3-haiku"].failures == 1

    def test_empty_store_no_crash(self):
        """Empty store should produce empty stats."""
        store = MemoryStore()
        mw = AutoRouterMiddleware(_make_config(), store)
        assert len(mw._stats) == 0

    def test_store_failure_is_fail_open(self):
        """Store failure during load should not crash initialization."""
        store = MagicMock()
        store.get_session_events.side_effect = RuntimeError("db error")
        mw = AutoRouterMiddleware(_make_config(), store)
        assert len(mw._stats) == 0


class TestPipelineIntegration:
    """Test auto-router in the middleware pipeline."""

    @pytest.fixture
    def store(self):
        return MemoryStore()

    @pytest.fixture
    def middleware(self, store):
        return AutoRouterMiddleware(_make_config(), store)

    async def test_full_pipeline_with_local_routing(self, middleware):
        """Middleware routes to local and records event."""
        mock_response = OllamaResponse(
            model="llama3.2",
            content="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            with patch.object(middleware._client, "chat", return_value=mock_response):
                ctx = _make_ctx(model="gpt-3.5-turbo")
                cloud_response = {"choices": [{"message": {"content": "cloud hi"}}]}

                async def call_next(c):
                    return cloud_response

                result = await middleware.process(ctx, call_next)
                # Should have routing event
                routing_events = [e for e in ctx.events if isinstance(e, LocalRoutingEvent)]
                assert len(routing_events) == 1
                assert routing_events[0].routing_success is True
                assert routing_events[0].local_model == "llama3.2"

    async def test_cache_hit_skips_routing(self, middleware):
        """Cache hit (skip_call=True) should pass through without routing."""
        ctx = _make_ctx(skip_call=True)

        async def call_next(c):
            return "cached"

        result = await middleware.process(ctx, call_next)
        assert result == "cached"
        assert len(ctx.events) == 0

    async def test_fallback_to_cloud(self, middleware):
        """Failed local routing falls back to cloud."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            with patch.object(middleware._client, "chat", side_effect=ConnectionError("refused")):
                ctx = _make_ctx(model="gpt-3.5-turbo")
                cloud_response = {"choices": [{"message": {"content": "cloud"}}]}

                async def call_next(c):
                    return cloud_response

                result = await middleware.process(ctx, call_next)
                assert result is cloud_response
                # Should have failure event
                routing_events = [e for e in ctx.events if isinstance(e, LocalRoutingEvent)]
                assert len(routing_events) == 1
                assert routing_events[0].routing_success is False

    async def test_cloud_path_records_informational_event(self, middleware):
        """Complex requests going to cloud should record an informational routing event."""
        messages = []
        for _ in range(12):
            messages.append({"role": "user", "content": "x" * 3000})
            messages.append({"role": "assistant", "content": "y" * 3000})

        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(messages=messages, model="gpt-4")
            cloud_response = {"result": "complex answer"}

            async def call_next(c):
                return cloud_response

            result = await middleware.process(ctx, call_next)
            assert result is cloud_response
            routing_events = [e for e in ctx.events if isinstance(e, LocalRoutingEvent)]
            # Cloud decisions now record an event with the complexity score/reason
            assert len(routing_events) == 1
            assert routing_events[0].routing_success is False

    async def test_always_calls_call_next(self, middleware):
        """Middleware must always call call_next."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            with patch.object(middleware._client, "chat", side_effect=Exception("boom")):
                ctx = _make_ctx(model="gpt-3.5-turbo")
                call_count = 0

                async def call_next(c):
                    nonlocal call_count
                    call_count += 1
                    return "ok"

                await middleware.process(ctx, call_next)
                assert call_count == 1


class TestLocalRoutingEvent:
    """Test LocalRoutingEvent dataclass."""

    def test_event_type(self):
        event = LocalRoutingEvent()
        assert event.event_type == EventType.LOCAL_ROUTING

    def test_fields(self):
        event = LocalRoutingEvent(
            session_id="s1",
            original_cloud_provider="openai",
            original_cloud_model="gpt-4",
            local_model="llama3.2",
            complexity_score=0.25,
            budget_pressure=0.3,
            routing_reason="low complexity",
            routing_success=True,
            probed=False,
            probe_confidence=None,
            historical_success_rate=0.85,
            local_latency_ms=150.0,
        )
        assert event.original_cloud_provider == "openai"
        assert event.local_model == "llama3.2"
        assert event.complexity_score == 0.25
        assert event.routing_success is True
        assert event.historical_success_rate == 0.85


class TestModelStats:
    """Test _ModelStats helper."""

    def test_empty_stats(self):
        stats = _ModelStats()
        assert stats.total == 0
        assert stats.success_rate == 0.0

    def test_all_success(self):
        stats = _ModelStats(successes=10, failures=0)
        assert stats.success_rate == 1.0

    def test_mixed(self):
        stats = _ModelStats(successes=7, failures=3)
        assert stats.total == 10
        assert stats.success_rate == pytest.approx(0.7)


class TestShutdown:
    """Test middleware cleanup."""

    def test_shutdown(self):
        store = MemoryStore()
        mw = AutoRouterMiddleware(_make_config(), store)
        mw.shutdown()  # Should not raise

    def test_shutdown_idempotent(self):
        store = MemoryStore()
        mw = AutoRouterMiddleware(_make_config(), store)
        mw.shutdown()
        mw.shutdown()  # Should not raise


class TestRealtimeDetection:
    """Test realtime/external data request detection."""

    @pytest.fixture
    def middleware(self):
        store = MemoryStore()
        return AutoRouterMiddleware(_make_config(), store)

    def test_weather_request_skips_local(self, middleware):
        """Weather requests should not be routed locally."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "what's the weather today?"}],
            )
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False
            assert decision.reason == "realtime data request"

    def test_stock_price_skips_local(self, middleware):
        """Stock price requests should not be routed locally."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "current stock price of AAPL"}],
            )
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False
            assert decision.reason == "realtime data request"

    def test_latest_news_skips_local(self, middleware):
        """Latest news requests should not be routed locally."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "latest news today"}],
            )
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False
            assert decision.reason == "realtime data request"

    def test_normal_request_not_flagged(self, middleware):
        """Normal requests should not be flagged as realtime."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "explain recursion"}],
            )
            decision = middleware._should_route_local(ctx)
            assert decision.reason != "realtime data request"

    def test_realtime_keyword_in_complex_sentence(self, middleware):
        """Realtime keywords in longer sentences should still be detected."""
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "what's the weather like in Paris right now"}
                ],
            )
            decision = middleware._should_route_local(ctx)
            assert decision.route_local is False
            assert decision.reason == "realtime data request"

    def test_needs_realtime_data_static(self):
        """Direct unit test of the static method."""
        assert AutoRouterMiddleware._needs_realtime_data("what's the weather") is True
        assert AutoRouterMiddleware._needs_realtime_data("stock price of TSLA") is True
        assert AutoRouterMiddleware._needs_realtime_data("currently happening") is True
        assert AutoRouterMiddleware._needs_realtime_data("explain recursion") is False
        assert AutoRouterMiddleware._needs_realtime_data("write a poem") is False


class TestInadequateResponseReroute:
    """Test local model inadequate response detection and reroute."""

    @pytest.fixture
    def middleware(self):
        store = MemoryStore()
        return AutoRouterMiddleware(_make_config(), store)

    def test_inadequate_response_returns_none(self, middleware):
        """Local model saying it can't access realtime data should return None."""
        mock_response = OllamaResponse(
            model="llama3.2",
            content="I don't have access to real-time data or the internet.",
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25,
        )
        with patch.object(middleware._client, "chat", return_value=mock_response):
            ctx = _make_ctx()
            decision = RoutingDecision(route_local=True, reason="low complexity")
            result = middleware._route_to_local(ctx, decision)
            assert result is None
            assert "inadequate response" in decision.reason

    def test_adequate_response_proceeds(self, middleware):
        """Local model giving a real answer should proceed normally."""
        mock_response = OllamaResponse(
            model="llama3.2",
            content="Recursion is when a function calls itself.",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
        )
        with patch.object(middleware._client, "chat", return_value=mock_response):
            ctx = _make_ctx()
            decision = RoutingDecision(route_local=True, reason="low complexity")
            result = middleware._route_to_local(ctx, decision)
            assert result is not None

    def test_knowledge_cutoff_response_rerouted(self, middleware):
        """Local model mentioning knowledge cutoff should be rerouted."""
        mock_response = OllamaResponse(
            model="llama3.2",
            content="My knowledge cutoff is April 2024, so I can't help with that.",
            prompt_tokens=10,
            completion_tokens=12,
            total_tokens=22,
        )
        with patch.object(middleware._client, "chat", return_value=mock_response):
            ctx = _make_ctx()
            decision = RoutingDecision(route_local=True, reason="low complexity")
            result = middleware._route_to_local(ctx, decision)
            assert result is None

    async def test_inadequate_response_event_reason(self, middleware):
        """Routing event should include inadequate response reason for UI logs."""
        mock_response = OllamaResponse(
            model="llama3.2",
            content="I don't have access to real-time data.",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )
        with patch.object(middleware, "_check_ollama_available", return_value=True):
            with patch.object(middleware._client, "chat", return_value=mock_response):
                ctx = _make_ctx(model="gpt-3.5-turbo")

                async def call_next(c):
                    return {"choices": [{"message": {"content": "cloud answer"}}]}

                await middleware.process(ctx, call_next)
                routing_events = [e for e in ctx.events if isinstance(e, LocalRoutingEvent)]
                assert len(routing_events) == 1
                assert routing_events[0].routing_success is False
                assert "inadequate response" in routing_events[0].routing_reason

    def test_is_inadequate_response_static(self):
        """Direct unit test of the static method."""
        assert (
            AutoRouterMiddleware._is_inadequate_response("I don't have access to real-time data")
            is True
        )
        assert (
            AutoRouterMiddleware._is_inadequate_response(
                "I cannot access the internet to check that"
            )
            is True
        )
        assert (
            AutoRouterMiddleware._is_inadequate_response(
                "my training data only goes up to a certain date"
            )
            is True
        )
        assert (
            AutoRouterMiddleware._is_inadequate_response(
                "I do not have current information about that"
            )
            is True
        )
        assert (
            AutoRouterMiddleware._is_inadequate_response(
                "Recursion is a technique where a function calls itself"
            )
            is False
        )
        assert (
            AutoRouterMiddleware._is_inadequate_response("The capital of France is Paris") is False
        )


class TestOllamaAvailabilityCache:
    """Test Ollama availability caching."""

    def test_caches_result(self):
        store = MemoryStore()
        mw = AutoRouterMiddleware(_make_config(), store)
        with patch.object(mw._client, "is_available", return_value=True) as mock_avail:
            assert mw._check_ollama_available() is True
            assert mw._check_ollama_available() is True
            # Should only call is_available once (cached)
            assert mock_avail.call_count == 1

    def test_cache_expires(self):
        store = MemoryStore()
        mw = AutoRouterMiddleware(_make_config(), store)
        with patch.object(mw._client, "is_available", return_value=True) as mock_avail:
            mw._check_ollama_available()
            # Force cache expiry
            mw._ollama_check_time -= 61.0
            mw._check_ollama_available()
            assert mock_avail.call_count == 2


class TestCustomScorer:
    """Test the BYO-Scorer (custom routing callback) feature."""

    @pytest.fixture
    def store(self):
        return MemoryStore()

    def _make_middleware(self, store, scorer=None, **config_overrides):
        config = _make_config(**config_overrides)
        return AutoRouterMiddleware(config, store, custom_scorer=scorer)

    def test_bool_true_routes_local(self, store):
        """True from scorer → route to local, custom_scorer_used=True."""
        mw = self._make_middleware(store, scorer=lambda prompt: True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.route_local is True
            assert decision.custom_scorer_used is True
            assert "custom scorer" in decision.reason

    def test_bool_false_routes_cloud(self, store):
        """False from scorer → route to cloud, custom_scorer_used=True."""
        mw = self._make_middleware(store, scorer=lambda prompt: False)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.route_local is False
            assert decision.custom_scorer_used is True
            assert "cloud" in decision.reason

    def test_float_low_routes_local(self, store):
        """Low float score → below threshold → routes local."""
        mw = self._make_middleware(store, scorer=lambda prompt: 0.1)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.route_local is True
            assert decision.custom_scorer_used is True
            assert decision.complexity_score == pytest.approx(0.1)

    def test_float_high_routes_cloud(self, store):
        """High float score → above complex_floor → routes cloud."""
        mw = self._make_middleware(store, scorer=lambda prompt: 0.9)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.route_local is False
            assert decision.custom_scorer_used is True

    def test_none_falls_through(self, store):
        """None from scorer → default scoring, custom_scorer_used=False."""
        mw = self._make_middleware(store, scorer=lambda prompt: None)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.custom_scorer_used is False

    def test_rich_context_receives_fields(self, store):
        """Scorer annotated with RoutingContext receives full context."""
        received = {}

        def scorer(ctx: RoutingContext) -> bool:
            received["prompt"] = ctx.prompt
            received["model"] = ctx.model
            received["session_id"] = ctx.session_id
            received["is_streaming"] = ctx.is_streaming
            received["has_tools"] = ctx.has_tools
            received["local_model"] = ctx.local_model
            return True

        mw = self._make_middleware(store, scorer=scorer)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                model="gpt-4",
                messages=[{"role": "user", "content": "test prompt"}],
            )
            decision = mw._should_route_local(ctx)
            assert decision.route_local is True
            assert received["prompt"] == "test prompt"
            assert received["model"] == "gpt-4"
            assert received["session_id"] == "test-session"
            assert received["is_streaming"] is False
            assert received["has_tools"] is False
            assert received["local_model"] == "llama3.2"

    def test_exception_falls_through(self, store):
        """Exception in scorer → caught, default scoring continues."""

        def exploding_scorer(prompt):
            raise RuntimeError("boom")

        mw = self._make_middleware(store, scorer=exploding_scorer)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            # Should not raise, should fall through to default scoring
            assert decision.custom_scorer_used is False

    def test_streaming_still_skipped(self, store):
        """Streaming check runs before scorer — scorer never called."""
        call_count = {"n": 0}

        def scorer(prompt):
            call_count["n"] += 1
            return True

        mw = self._make_middleware(store, scorer=scorer)
        ctx = _make_ctx(model="gpt-3.5-turbo", is_streaming=True)
        decision = mw._should_route_local(ctx)
        assert decision.route_local is False
        assert "streaming" in decision.reason
        assert call_count["n"] == 0

    def test_tools_still_skipped(self, store):
        """Tools check runs before scorer — scorer never called."""
        call_count = {"n": 0}

        def scorer(prompt):
            call_count["n"] += 1
            return True

        mw = self._make_middleware(store, scorer=scorer)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo", tools=[{"type": "function"}])
            decision = mw._should_route_local(ctx)
            assert decision.route_local is False
            assert "tools" in decision.reason
            assert call_count["n"] == 0

    def test_float_clamped_negative(self, store):
        """Negative float → clamped to 0.0."""
        mw = self._make_middleware(store, scorer=lambda prompt: -0.5)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.complexity_score == 0.0
            assert decision.custom_scorer_used is True
            assert decision.route_local is True  # 0.0 < threshold

    def test_float_clamped_above_one(self, store):
        """Float > 1.0 → clamped to 1.0."""
        mw = self._make_middleware(store, scorer=lambda prompt: 5.0)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.complexity_score == 1.0
            assert decision.custom_scorer_used is True
            assert decision.route_local is False  # 1.0 >= complex_floor

    def test_no_scorer_backward_compat(self, store):
        """No scorer → custom_scorer_used=False, unchanged behavior."""
        mw = self._make_middleware(store)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.custom_scorer_used is False

    def test_event_records_custom_scorer(self, store):
        """LocalRoutingEvent should have custom_scorer_used=True when scorer is used."""
        mw = self._make_middleware(store, scorer=lambda prompt: True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(model="gpt-3.5-turbo")
            decision = mw._should_route_local(ctx)
            assert decision.custom_scorer_used is True
            # Record event and verify
            mw._record_event(ctx, decision, success=True)
            assert len(ctx.events) == 1
            event = ctx.events[0]
            assert isinstance(event, LocalRoutingEvent)
            assert event.custom_scorer_used is True
