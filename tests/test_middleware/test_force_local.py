"""Tests for force-local routing mode."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import LocalRoutingEvent
from stateloom.core.session import Session
from stateloom.core.types import Provider
from stateloom.local.client import OllamaResponse
from stateloom.middleware.auto_router import AutoRouterMiddleware
from stateloom.middleware.base import MiddlewareContext
from stateloom.store.memory_store import MemoryStore


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "auto_route_enabled": True,
        "auto_route_force_local": False,
        "auto_route_model": "llama3.2",
        "local_model_enabled": True,
        "local_model_default": "llama3.2",
        "console_output": False,
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(
    config: StateLoomConfig | None = None,
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
        config=config or _make_config(),
        provider=provider,
        model=model,
        request_kwargs=kwargs,
        is_streaming=is_streaming,
        skip_call=skip_call,
        auto_route_eligible=True,
    )


class TestForceLocalDecision:
    """Test the force-local routing decision path."""

    @pytest.fixture
    def store(self):
        return MemoryStore()

    def _make_mw(self, store, **config_overrides):
        config = _make_config(**config_overrides)
        return AutoRouterMiddleware(config, store)

    def test_passthrough_when_force_local_off(self, store):
        """Normal complexity routing when force_local is off."""
        mw = self._make_mw(store)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            # A complex message should not force-route to local
            messages = [{"role": "user", "content": "x" * 3000}] * 12
            ctx = _make_ctx(messages=messages, model="gpt-4")
            decision = mw._should_route_local(ctx)
            # Should use normal complexity analysis, not force-local
            assert "force-local" not in decision.reason

    def test_force_local_routes_simple_request(self, store):
        """Force-local routes a simple message to local."""
        mw = self._make_mw(store, auto_route_force_local=True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(config=mw._config)
            decision = mw._should_route_local(ctx)
            assert decision.route_local is True
            assert "force-local mode" in decision.reason

    def test_force_local_routes_long_message(self, store):
        """Force-local routes even long messages to local (skips complexity)."""
        mw = self._make_mw(store, auto_route_force_local=True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            messages = [{"role": "user", "content": "x" * 10000}]
            ctx = _make_ctx(config=mw._config, messages=messages)
            decision = mw._should_route_local(ctx)
            assert decision.route_local is True
            assert "force-local mode" in decision.reason

    def test_force_local_fallback_streaming(self, store):
        """Force-local falls back to cloud for streaming."""
        mw = self._make_mw(store, auto_route_force_local=True)
        ctx = _make_ctx(config=mw._config, is_streaming=True)
        decision = mw._should_route_local(ctx)
        assert decision.route_local is False
        assert "streaming" in decision.reason

    def test_force_local_fallback_tools(self, store):
        """Force-local falls back to cloud when tools are present."""
        mw = self._make_mw(store, auto_route_force_local=True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                config=mw._config,
                tools=[{"type": "function", "function": {"name": "test"}}],
            )
            decision = mw._should_route_local(ctx)
            assert decision.route_local is False
            assert "tools" in decision.reason

    def test_force_local_fallback_response_format(self, store):
        """Force-local falls back to cloud for response_format."""
        mw = self._make_mw(store, auto_route_force_local=True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                config=mw._config,
                response_format={"type": "json_object"},
            )
            decision = mw._should_route_local(ctx)
            assert decision.route_local is False
            assert "response_format" in decision.reason

    def test_force_local_fallback_images(self, store):
        """Force-local falls back to cloud for image content."""
        mw = self._make_mw(store, auto_route_force_local=True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                config=mw._config,
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
                ],
            )
            decision = mw._should_route_local(ctx)
            assert decision.route_local is False
            assert "images" in decision.reason

    def test_force_local_fallback_ollama_unavailable(self, store):
        """Force-local falls back when Ollama is unavailable."""
        mw = self._make_mw(store, auto_route_force_local=True)
        with patch.object(mw, "_check_ollama_available", return_value=False):
            ctx = _make_ctx(config=mw._config)
            decision = mw._should_route_local(ctx)
            assert decision.route_local is False
            assert "Ollama unavailable" in decision.reason

    def test_force_local_ignores_session_opt_out(self, store):
        """Force-local overrides session opt-out."""
        mw = self._make_mw(store, auto_route_force_local=True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(
                config=mw._config,
                session_metadata={"auto_route_enabled": False},
            )
            decision = mw._should_route_local(ctx)
            assert decision.route_local is True
            assert "force-local mode" in decision.reason

    def test_force_local_skips_complexity_analysis(self, store):
        """Force-local sets complexity_score to 0.0."""
        mw = self._make_mw(store, auto_route_force_local=True)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            ctx = _make_ctx(config=mw._config)
            decision = mw._should_route_local(ctx)
            assert decision.route_local is True
            assert decision.complexity_score == 0.0
            assert decision.budget_pressure == 0.0

    def test_config_change_takes_effect_per_request(self, store):
        """Toggling force_local on/off between calls works."""
        config = _make_config(auto_route_force_local=False)
        mw = AutoRouterMiddleware(config, store)
        with patch.object(mw, "_check_ollama_available", return_value=True):
            # Initially off — complex request goes to cloud
            messages = [{"role": "user", "content": "x" * 3000}] * 12
            ctx = _make_ctx(config=config, messages=messages, model="gpt-4")
            decision = mw._should_route_local(ctx)
            assert "force-local" not in decision.reason

            # Turn on
            config.auto_route_force_local = True
            ctx2 = _make_ctx(config=config, messages=messages, model="gpt-4")
            decision2 = mw._should_route_local(ctx2)
            assert decision2.route_local is True
            assert "force-local mode" in decision2.reason

            # Turn off
            config.auto_route_force_local = False
            ctx3 = _make_ctx(config=config, messages=messages, model="gpt-4")
            decision3 = mw._should_route_local(ctx3)
            assert "force-local" not in decision3.reason


class TestForceLocalPipeline:
    """Test force-local through the full middleware process()."""

    @pytest.fixture
    def store(self):
        return MemoryStore()

    async def test_force_local_with_call_failure_falls_back(self, store):
        """When force-local is on but local call fails, fall back to cloud."""
        config = _make_config(auto_route_force_local=True)
        mw = AutoRouterMiddleware(config, store)

        with patch.object(mw, "_check_ollama_available", return_value=True):
            with patch.object(mw._client, "chat", side_effect=ConnectionError("refused")):
                ctx = _make_ctx(config=config, model="gpt-3.5-turbo")
                cloud_response = {"choices": [{"message": {"content": "from cloud"}}]}

                async def call_next(c):
                    return cloud_response

                await mw.process(ctx, call_next)
                # Should fall back to cloud
                # Should have a failure routing event
                routing_events = [e for e in ctx.events if isinstance(e, LocalRoutingEvent)]
                assert len(routing_events) == 1
                assert routing_events[0].routing_success is False

    async def test_force_local_successful_route(self, store):
        """When force-local is on and local call succeeds, route locally."""
        config = _make_config(auto_route_force_local=True)
        mw = AutoRouterMiddleware(config, store)
        mock_response = OllamaResponse(
            model="llama3.2",
            content="Hello from local!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        with patch.object(mw, "_check_ollama_available", return_value=True):
            with patch.object(mw._client, "chat", return_value=mock_response):
                ctx = _make_ctx(config=config, model="gpt-4")

                async def call_next(c):
                    return "cloud"

                await mw.process(ctx, call_next)
                routing_events = [e for e in ctx.events if isinstance(e, LocalRoutingEvent)]
                assert len(routing_events) == 1
                assert routing_events[0].routing_success is True
                assert ctx.provider == Provider.LOCAL


class TestResolveLocalModel:
    """Test that _resolve_local_model returns the current config value."""

    def test_resolve_reflects_config_change(self):
        """After hot-swap changes config, _resolve_local_model returns new model."""
        config = _make_config(local_model_default="old-model", auto_route_model="")
        store = MemoryStore()
        mw = AutoRouterMiddleware(config, store)
        ctx = _make_ctx(config=config)
        assert mw._resolve_local_model(ctx) == "old-model"

        # Simulate hot-swap
        config.local_model_default = "new-model"
        assert mw._resolve_local_model(ctx) == "new-model"
