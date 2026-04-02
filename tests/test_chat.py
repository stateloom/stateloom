"""Tests for the unified stateloom.Client and stateloom.chat() API."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stateloom
from stateloom.chat import (
    ChatResponse,
    Client,
    _default_client,
    _extract_content,
    _resolve_provider,
    achat,
    chat,
)
from stateloom.core.config import StateLoomConfig
from stateloom.core.session import Session
from stateloom.intercept.unpatch import PatchRecord, _patch_registry, get_original
from stateloom.middleware.auto_router import AutoRouterMiddleware, RoutingDecision
from stateloom.middleware.base import MiddlewareContext
from stateloom.store.memory_store import MemoryStore

# --- Helpers ---


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "auto_patch": False,
        "dashboard": False,
        "console_output": False,
        "store_backend": "memory",
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(
    provider: str = "openai",
    model: str = "gpt-4",
    messages: list | None = None,
    auto_route_eligible: bool = False,
    **overrides,
) -> MiddlewareContext:
    session = Session(id="test-session")
    config = _make_config()
    return MiddlewareContext(
        session=session,
        config=config,
        provider=provider,
        method="chat",
        model=model,
        request_kwargs={"messages": messages or [{"role": "user", "content": "hi"}]},
        auto_route_eligible=auto_route_eligible,
        **overrides,
    )


def _make_openai_response(content: str = "Hello!") -> MagicMock:
    """Create a mock OpenAI ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    return response


class _SimpleNamespace:
    """Simple attribute holder that doesn't auto-create attributes like MagicMock."""

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_anthropic_response(content: str = "Hello!") -> _SimpleNamespace:
    """Create a mock Anthropic Message response."""
    text_block = _SimpleNamespace(text=content, type="text")
    return _SimpleNamespace(content=[text_block])


def _make_gemini_response(content: str = "Hello!") -> _SimpleNamespace:
    """Create a mock Gemini response."""
    return _SimpleNamespace(text=content)


# =============================================================================
# 1. Provider resolution
# =============================================================================


class TestProviderResolution:
    def test_gpt_models(self):
        assert _resolve_provider("gpt-4") == "openai"
        assert _resolve_provider("gpt-4-turbo") == "openai"
        assert _resolve_provider("gpt-3.5-turbo") == "openai"
        assert _resolve_provider("gpt-4o") == "openai"
        assert _resolve_provider("gpt-4o-mini") == "openai"

    def test_o_series_models(self):
        assert _resolve_provider("o1-preview") == "openai"
        assert _resolve_provider("o3-mini") == "openai"
        assert _resolve_provider("o4-mini") == "openai"

    def test_chatgpt_models(self):
        assert _resolve_provider("chatgpt-4o-latest") == "openai"

    def test_claude_models(self):
        assert _resolve_provider("claude-3-opus") == "anthropic"
        assert _resolve_provider("claude-3-5-sonnet") == "anthropic"
        assert _resolve_provider("claude-opus-4") == "anthropic"
        assert _resolve_provider("claude-haiku-4") == "anthropic"

    def test_gemini_models(self):
        assert _resolve_provider("gemini-1.5-pro") == "gemini"
        assert _resolve_provider("gemini-2.0-flash") == "gemini"

    def test_unknown_defaults_to_openai(self):
        assert _resolve_provider("some-random-model") == "openai"


# =============================================================================
# 2. ChatResponse metadata
# =============================================================================


class TestChatResponse:
    def test_fields(self):
        resp = ChatResponse(
            content="Hello",
            model="gpt-4",
            provider="openai",
            raw={"id": "test"},
            _stateloom={"actual_model": "gpt-4", "routed_local": False},
        )
        assert resp.content == "Hello"
        assert resp.model == "gpt-4"
        assert resp.provider == "openai"
        assert resp.raw == {"id": "test"}
        assert resp._stateloom["actual_model"] == "gpt-4"
        assert resp._stateloom["routed_local"] is False

    def test_default_stateloom_is_empty_dict(self):
        resp = ChatResponse(content="Hi", model="gpt-4", provider="openai")
        assert resp._stateloom == {}


# =============================================================================
# 3. Content extraction
# =============================================================================


class TestContentExtraction:
    def test_openai_response(self):
        resp = _make_openai_response("Hello from OpenAI")
        assert _extract_content(resp, "openai") == "Hello from OpenAI"

    def test_anthropic_response(self):
        resp = _make_anthropic_response("Hello from Anthropic")
        assert _extract_content(resp, "anthropic") == "Hello from Anthropic"

    def test_gemini_response(self):
        resp = _make_gemini_response("Hello from Gemini")
        assert _extract_content(resp, "gemini") == "Hello from Gemini"

    def test_none_response(self):
        assert _extract_content(None, "openai") == ""

    def test_dict_response(self):
        resp = {"choices": [{"message": {"content": "dict content"}}]}
        assert _extract_content(resp, "openai") == "dict content"

    def test_empty_choices(self):
        resp = types.SimpleNamespace(choices=[], text=None, content=None)
        assert _extract_content(resp, "openai") == ""


# =============================================================================
# 4. get_original() lookup
# =============================================================================


class TestGetOriginal:
    def test_returns_original_when_patched(self):
        class FakeClass:
            pass

        def original_fn() -> str:
            return "original"

        _patch_registry.append(
            PatchRecord(target=FakeClass, method_name="create", original=original_fn)
        )
        try:
            result = get_original(FakeClass, "create")
            assert result is original_fn
        finally:
            _patch_registry.pop()

    def test_returns_none_when_not_patched(self):
        class UnpatchedClass:
            pass

        assert get_original(UnpatchedClass, "create") is None

    def test_returns_none_for_wrong_method(self):
        class FakeClass:
            pass

        def original_fn() -> str:
            return "original"

        _patch_registry.append(
            PatchRecord(target=FakeClass, method_name="create", original=original_fn)
        )
        try:
            assert get_original(FakeClass, "delete") is None
        finally:
            _patch_registry.pop()


# =============================================================================
# 5. Session lifecycle — context manager
# =============================================================================


class TestSessionLifecycleContextManager:
    def test_session_starts_on_enter_ends_on_exit(self):
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        client = Client(session_id="cm-test")
        with client:
            assert client.session is not None
            assert client.session.id == "cm-test"
        assert client.session is None

    def test_session_accumulates_across_calls(self):
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        mock_resp = _make_openai_response("test")

        with Client(session_id="accum-test") as client:
            # Verify session is created
            assert client.session is not None
            session_id = client.session.id
            assert session_id == "accum-test"


# =============================================================================
# 6. Session lifecycle — standalone
# =============================================================================


class TestSessionLifecycleStandalone:
    def test_session_starts_on_first_chat(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        client = Client(session_id="standalone-test")
        # Session is None before any chat call
        assert client.session is None
        # _ensure_session creates the session
        client._ensure_session()
        assert client.session is not None
        assert client.session.id == "standalone-test"
        client.close()
        assert client.session is None

    def test_close_is_idempotent(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        client = Client(session_id="close-test")
        client._ensure_session()
        client.close()
        # Second close should not raise
        client.close()
        assert client.session is None


# =============================================================================
# 7. Session access
# =============================================================================


class TestSessionAccess:
    def test_session_property(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        with Client(session_id="access-test", budget=10.0) as client:
            assert client.session is not None
            assert client.session.budget == 10.0


# =============================================================================
# 8. Sync chat() end-to-end (mocked pipeline)
# =============================================================================


class TestSyncChatE2E:
    def test_chat_runs_pipeline(self):
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        mock_resp = _make_openai_response("Pipeline response")

        # Mock the pipeline.execute to return our response and set ctx fields
        original_execute = gate.pipeline.execute

        async def mock_execute(ctx, llm_call):
            ctx.response = mock_resp
            return mock_resp

        gate.pipeline.execute = mock_execute

        with Client(session_id="e2e-sync") as client:
            result = client.chat(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert isinstance(result, ChatResponse)
            assert result.content == "Pipeline response"
            assert result.model == "gpt-4"
            assert result.provider == "openai"
            assert result.raw is mock_resp
            assert result._stateloom["actual_model"] == "gpt-4"
            assert result._stateloom["session_id"] == "e2e-sync"


# =============================================================================
# 9. Async achat() end-to-end (mocked pipeline)
# =============================================================================


class TestAsyncChatE2E:
    async def test_achat_runs_pipeline(self):
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        mock_resp = _make_openai_response("Async pipeline response")

        async def mock_execute(ctx, llm_call):
            ctx.response = mock_resp
            return mock_resp

        gate.pipeline.execute = mock_execute

        async with Client(session_id="e2e-async") as client:
            result = await client.achat(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert isinstance(result, ChatResponse)
            assert result.content == "Async pipeline response"
            assert result.model == "gpt-4"
            assert result.provider == "openai"
            assert result._stateloom["session_id"] == "e2e-async"


# =============================================================================
# 10. Auto-route eligibility — only when model is omitted
# =============================================================================


class TestAutoRouteEligibility:
    def test_explicit_model_not_eligible(self):
        """Explicit model= means no auto-routing."""
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        captured_ctx = {}

        async def capture_execute(ctx, llm_call):
            captured_ctx["auto_route_eligible"] = ctx.auto_route_eligible
            ctx.response = _make_openai_response("test")
            return ctx.response

        gate.pipeline.execute = capture_execute

        with Client(session_id="explicit-model-test") as client:
            client.chat(
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert captured_ctx["auto_route_eligible"] is False

    def test_omitted_model_is_eligible(self):
        """No model= uses default_model and enables auto-routing."""
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
            default_model="gpt-4o",
        )

        captured_ctx = {}

        async def capture_execute(ctx, llm_call):
            captured_ctx["auto_route_eligible"] = ctx.auto_route_eligible
            captured_ctx["model"] = ctx.model
            ctx.response = _make_openai_response("test")
            return ctx.response

        gate.pipeline.execute = capture_execute

        with Client(session_id="omitted-model-test") as client:
            client.chat(
                messages=[{"role": "user", "content": "hi"}],
            )

        assert captured_ctx["auto_route_eligible"] is True
        assert captured_ctx["model"] == "gpt-4o"

    def test_no_model_no_default_raises(self):
        """No model= and no default_model configured raises an error."""
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        with Client(session_id="no-model-test") as client:
            with pytest.raises(stateloom.StateLoomError, match="No model specified"):
                client.chat(
                    messages=[{"role": "user", "content": "hi"}],
                )


# =============================================================================
# 11. SDK calls NOT eligible — interceptor uses auto_route_eligible=False
# =============================================================================


class TestSDKCallsNotEligible:
    def test_pipeline_default_not_eligible(self):
        """execute_sync without auto_route_eligible defaults to False."""
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        captured_ctx = {}

        original_execute = gate.pipeline.execute

        async def capture_execute(ctx, llm_call):
            captured_ctx["auto_route_eligible"] = ctx.auto_route_eligible
            return _make_openai_response("test")

        gate.pipeline.execute = capture_execute

        session = Session(id="sdk-test")
        # Simulate what the interceptor does — no auto_route_eligible
        gate.pipeline.execute_sync(
            provider="openai",
            method="chat.completions.create",
            model="gpt-4",
            request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
            session=session,
            config=gate.config,
            llm_call=lambda: _make_openai_response("test"),
        )

        assert captured_ctx["auto_route_eligible"] is False

    def test_auto_router_rejects_non_eligible(self):
        """AutoRouter._should_route_local returns early for non-eligible calls."""
        store = MemoryStore()
        config = _make_config(
            auto_route_enabled=True,
            local_model_enabled=True,
            local_model_default="llama3.2",
        )

        router = AutoRouterMiddleware(config, store)
        ctx = _make_ctx(auto_route_eligible=False)

        decision = router._should_route_local(ctx)
        assert decision.route_local is False
        assert "not eligible" in decision.reason

    def test_auto_router_accepts_eligible(self):
        """AutoRouter._should_route_local proceeds past eligibility for eligible calls."""
        store = MemoryStore()
        config = _make_config(
            auto_route_enabled=True,
            local_model_enabled=True,
            local_model_default="llama3.2",
        )

        router = AutoRouterMiddleware(config, store)
        ctx = _make_ctx(auto_route_eligible=True)

        # Patch Ollama availability to avoid actual network calls
        router._ollama_available = False
        router._ollama_check_time = float("inf")

        decision = router._should_route_local(ctx)
        # Should get past the eligibility check (rejected for another reason)
        assert "not eligible" not in decision.reason


# =============================================================================
# 12. Default client singleton
# =============================================================================


class TestDefaultClientSingleton:
    def test_module_level_chat_creates_default(self):
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        async def mock_execute(ctx, llm_call):
            ctx.response = _make_openai_response("default client")
            return ctx.response

        gate.pipeline.execute = mock_execute

        # Access the module via sys.modules since stateloom.chat (attribute)
        # is shadowed by the chat() function import in __init__.py
        chat_mod = sys.modules["stateloom.chat"]

        # Reset default client
        chat_mod._default_client = None

        result = chat_mod.chat(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(result, ChatResponse)
        assert result.content == "default client"
        # Default client was created
        assert chat_mod._default_client is not None

        # Calling again reuses the same client
        first_client = chat_mod._default_client
        chat_mod.chat(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi again"}],
        )
        assert chat_mod._default_client is first_client


# =============================================================================
# 13. Shutdown cleanup
# =============================================================================


class TestShutdownCleanup:
    def test_shutdown_resets_default_client(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        chat_mod = sys.modules["stateloom.chat"]
        chat_mod._default_client = Client()

        stateloom.shutdown()

        assert chat_mod._default_client is None


# =============================================================================
# 14. Request format conversion — Anthropic
# =============================================================================


class TestAnthropicConversion:
    def test_system_messages_extracted(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        client = Client()
        gate = stateloom.get_gate()
        client._gate = gate

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        kwargs, _ = client._prepare_anthropic(gate, "claude-3-opus", messages)
        assert kwargs["system"] == "You are helpful."
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"

    def test_max_tokens_default(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        client = Client()
        gate = stateloom.get_gate()
        client._gate = gate

        messages = [{"role": "user", "content": "Hi"}]
        kwargs, _ = client._prepare_anthropic(gate, "claude-3-opus", messages)
        assert kwargs["max_tokens"] == 4096

    def test_max_tokens_override(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        client = Client()
        gate = stateloom.get_gate()
        client._gate = gate

        messages = [{"role": "user", "content": "Hi"}]
        kwargs, _ = client._prepare_anthropic(gate, "claude-3-opus", messages, max_tokens=1024)
        assert kwargs["max_tokens"] == 1024

    def test_multiple_system_messages_joined(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        client = Client()
        gate = stateloom.get_gate()
        client._gate = gate

        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hi"},
        ]
        kwargs, _ = client._prepare_anthropic(gate, "claude-3-opus", messages)
        assert "Rule 1" in kwargs["system"]
        assert "Rule 2" in kwargs["system"]


# =============================================================================
# 15. Request format conversion — Gemini
# =============================================================================


class TestGeminiConversion:
    def test_messages_converted_to_contents(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        client = Client()
        gate = stateloom.get_gate()
        client._gate = gate

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        kwargs, _ = client._prepare_gemini(gate, "gemini-1.5-pro", messages)
        # System message extracted
        assert kwargs.get("system_instruction") == "Be helpful"
        # Contents should have 2 entries (user + assistant, no system)
        assert len(kwargs["contents"]) == 2
        assert kwargs["contents"][0]["role"] == "user"
        assert kwargs["contents"][1]["role"] == "model"  # assistant -> model


# =============================================================================
# 16. _build_response metadata
# =============================================================================


class TestBuildResponse:
    def test_metadata_populated(self):
        ctx = _make_ctx(provider="openai", model="gpt-4")
        ctx.latency_ms = 150.0
        ctx.prompt_tokens = 100
        ctx.completion_tokens = 50

        mock_resp = _make_openai_response("Hello")
        result = Client._build_response(mock_resp, "gpt-4", "openai", ctx)

        assert result.content == "Hello"
        assert result._stateloom["actual_model"] == "gpt-4"
        assert result._stateloom["actual_provider"] == "openai"
        assert result._stateloom["routed_local"] is False
        assert result._stateloom["latency_ms"] == 150.0
        assert result._stateloom["prompt_tokens"] == 100
        assert result._stateloom["completion_tokens"] == 50
        assert result._stateloom["session_id"] == "test-session"

    def test_routed_local_detected(self):
        ctx = _make_ctx(provider="local", model="llama3.2")

        mock_resp = _make_openai_response("Local response")
        result = Client._build_response(mock_resp, "gpt-4", "openai", ctx)

        assert result._stateloom["routed_local"] is True
        assert result._stateloom["actual_model"] == "llama3.2"
        assert result._stateloom["actual_provider"] == "local"

    def test_cached_flag(self):
        ctx = _make_ctx()
        ctx.skip_call = True
        ctx.cached_response = {"cached": True}

        mock_resp = _make_openai_response("Cached")
        result = Client._build_response(mock_resp, "gpt-4", "openai", ctx)

        assert result._stateloom["cached"] is True


# =============================================================================
# 17. Dynamic provider resolution via adapter registry
# =============================================================================


class TestDynamicProviderResolution:
    """After init(), the adapter registry resolves Mistral/Cohere models correctly."""

    def test_mistral_resolves_after_init(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )
        assert _resolve_provider("mistral-large-latest") == "mistral"
        assert _resolve_provider("codestral-latest") == "mistral"
        assert _resolve_provider("pixtral-large-latest") == "mistral"

    def test_cohere_resolves_after_init(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )
        assert _resolve_provider("command-r-plus") == "cohere"
        assert _resolve_provider("c4ai-aya-expanse") == "cohere"

    def test_existing_providers_still_resolve(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )
        assert _resolve_provider("gpt-4") == "openai"
        assert _resolve_provider("claude-3-opus") == "anthropic"
        assert _resolve_provider("gemini-1.5-pro") == "gemini"

    def test_unknown_still_defaults_to_openai(self):
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )
        assert _resolve_provider("some-unknown-model") == "openai"


# =============================================================================
# 18. Adapter prepare_chat produces correct kwargs
# =============================================================================


class TestAdapterPrepareChat:
    """Adapter prepare_chat methods produce correct request_kwargs."""

    def test_openai_adapter_prepare_chat(self):
        from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter()
        msgs = [{"role": "user", "content": "hi"}]
        kwargs, llm_call = adapter.prepare_chat(model="gpt-4", messages=msgs)
        assert kwargs["model"] == "gpt-4"
        assert kwargs["messages"] == msgs
        assert callable(llm_call)

    def test_anthropic_adapter_prepare_chat_extracts_system(self):
        from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        kwargs, llm_call = adapter.prepare_chat(model="claude-3-opus", messages=msgs)
        assert kwargs["system"] == "You are helpful."
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["max_tokens"] == 4096

    def test_anthropic_adapter_max_tokens_override(self):
        from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter()
        msgs = [{"role": "user", "content": "Hi"}]
        kwargs, _ = adapter.prepare_chat(model="claude-3-opus", messages=msgs, max_tokens=1024)
        assert kwargs["max_tokens"] == 1024

    def test_gemini_adapter_prepare_chat_converts_messages(self):
        from stateloom.intercept.adapters.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter()
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        kwargs, llm_call = adapter.prepare_chat(model="gemini-1.5-pro", messages=msgs)
        assert kwargs.get("system_instruction") == "Be helpful"
        assert len(kwargs["contents"]) == 2
        assert kwargs["contents"][0]["role"] == "user"
        assert kwargs["contents"][1]["role"] == "model"

    def test_mistral_adapter_prepare_chat(self):
        from stateloom.intercept.adapters.mistral_adapter import MistralAdapter

        adapter = MistralAdapter()
        msgs = [{"role": "user", "content": "hi"}]
        kwargs, llm_call = adapter.prepare_chat(model="mistral-large-latest", messages=msgs)
        assert kwargs["model"] == "mistral-large-latest"
        assert kwargs["messages"] == msgs
        assert callable(llm_call)

    def test_cohere_adapter_prepare_chat(self):
        from stateloom.intercept.adapters.cohere_adapter import CohereAdapter

        adapter = CohereAdapter()
        msgs = [{"role": "user", "content": "hi"}]
        kwargs, llm_call = adapter.prepare_chat(model="command-r-plus", messages=msgs)
        assert kwargs["model"] == "command-r-plus"
        assert kwargs["messages"] == msgs
        assert callable(llm_call)


# =============================================================================
# 19. Client routes through adapter prepare_chat
# =============================================================================


class TestClientDynamicRouting:
    """Client._prepare_call delegates to adapter.prepare_chat when available."""

    def test_mistral_routes_through_adapter(self):
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        captured_ctx = {}

        async def capture_execute(ctx, llm_call):
            captured_ctx["provider"] = ctx.provider
            captured_ctx["model"] = ctx.model
            captured_ctx["base_url"] = ctx.provider_base_url
            ctx.response = _make_openai_response("mistral response")
            return ctx.response

        gate.pipeline.execute = capture_execute

        with Client(session_id="mistral-route-test") as client:
            result = client.chat(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert captured_ctx["provider"] == "mistral"
        assert captured_ctx["model"] == "mistral-large-latest"
        assert captured_ctx["base_url"] == "https://api.mistral.ai/v1"
        assert result.provider == "mistral"

    def test_cohere_routes_through_adapter(self):
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )

        captured_ctx = {}

        async def capture_execute(ctx, llm_call):
            captured_ctx["provider"] = ctx.provider
            captured_ctx["model"] = ctx.model
            captured_ctx["base_url"] = ctx.provider_base_url
            ctx.response = _make_openai_response("cohere response")
            return ctx.response

        gate.pipeline.execute = capture_execute

        with Client(session_id="cohere-route-test") as client:
            result = client.chat(
                model="command-r-plus",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert captured_ctx["provider"] == "cohere"
        assert captured_ctx["model"] == "command-r-plus"
        assert captured_ctx["base_url"] == "https://api.cohere.com/v2"
        assert result.provider == "cohere"
