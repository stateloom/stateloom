"""Tests for the LangChain callback handler (all mocked, no real LangChain)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

import stateloom
from stateloom.core.event import LLMCallEvent, ToolCallEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gate():
    g = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )
    yield g
    stateloom.shutdown()


@pytest.fixture
def gate_auto_patch():
    g = stateloom.init(
        auto_patch=True,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )
    yield g
    stateloom.shutdown()


def _make_handler(gate, *, tools_only=None):
    """Import handler with langchain_core mocked."""
    mock_base = type("BaseCallbackHandler", (), {"__init__": lambda self: None})
    mock_callbacks = MagicMock()
    mock_callbacks.BaseCallbackHandler = mock_base

    mock_outputs = MagicMock()
    mock_outputs.LLMResult = object

    with patch.dict(
        "sys.modules",
        {
            "langchain_core": MagicMock(),
            "langchain_core.callbacks": mock_callbacks,
            "langchain_core.outputs": mock_outputs,
        },
    ):
        # Force reimport to pick up mocked modules
        import importlib

        import stateloom.ext.langchain as mod

        importlib.reload(mod)
        return mod.StateLoomCallbackHandler(gate=gate, tools_only=tools_only)


def _make_serialized(
    model: str = "gpt-4o",
    provider_id: str = "openai",
) -> dict:
    return {
        "id": ["langchain", "chat_models", provider_id, "ChatOpenAI"],
        "kwargs": {"model_name": model},
    }


def _make_llm_result(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    style: str = "openai",
) -> SimpleNamespace:
    if style == "openai":
        return SimpleNamespace(
            llm_output={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            }
        )
    elif style == "anthropic":
        return SimpleNamespace(
            llm_output={
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                }
            }
        )
    else:
        return SimpleNamespace(llm_output={})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLangChainHandler:
    def test_handler_raises_without_langchain(self):
        """ImportError with install instructions when langchain_core missing."""
        with patch.dict(
            "sys.modules",
            {
                "langchain_core": None,
                "langchain_core.callbacks": None,
                "langchain_core.outputs": None,
            },
        ):
            import importlib

            import stateloom.ext.langchain as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="stateloom\\[langchain\\]"):
                mod.StateLoomCallbackHandler()

    def test_handler_instantiation(self, gate):
        """Constructs with explicit gate reference."""
        handler = _make_handler(gate)
        assert handler._gate_ref is gate

    def test_on_llm_end_records_event(self, gate):
        """Full LLM lifecycle: chat_model_start -> llm_end -> LLMCallEvent saved."""
        handler = _make_handler(gate)
        run_id = uuid4()

        with gate.session("test-llm") as session:
            handler.on_chat_model_start(
                _make_serialized("gpt-4o", "openai"),
                messages=[[]],
                run_id=run_id,
            )
            handler.on_llm_end(
                _make_llm_result(100, 50, "openai"),
                run_id=run_id,
            )

        events = gate.store.get_session_events(session.id, event_type="llm_call")
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, LLMCallEvent)
        assert ev.model == "gpt-4o"
        assert ev.provider == "openai"
        assert ev.prompt_tokens == 100
        assert ev.completion_tokens == 50
        assert ev.total_tokens == 150
        assert ev.cost >= 0
        assert ev.latency_ms > 0
        assert ev.is_streaming is False

    def test_on_tool_end_records_event(self, gate):
        """Full tool lifecycle: tool_start -> tool_end -> ToolCallEvent saved."""
        handler = _make_handler(gate)
        run_id = uuid4()

        with gate.session("test-tool") as session:
            handler.on_tool_start(
                {"name": "web_search"},
                input_str="query",
                run_id=run_id,
            )
            handler.on_tool_end(
                "search results",
                run_id=run_id,
            )

        events = gate.store.get_session_events(session.id, event_type="tool_call")
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, ToolCallEvent)
        assert ev.tool_name == "web_search"
        assert ev.latency_ms > 0

    def test_noop_without_gate(self):
        """Graceful no-op when gate not initialized."""
        mock_base = type("BaseCallbackHandler", (), {"__init__": lambda self: None})
        mock_callbacks = MagicMock()
        mock_callbacks.BaseCallbackHandler = mock_base
        mock_outputs = MagicMock()
        mock_outputs.LLMResult = object

        with patch.dict(
            "sys.modules",
            {
                "langchain_core": MagicMock(),
                "langchain_core.callbacks": mock_callbacks,
                "langchain_core.outputs": mock_outputs,
            },
        ):
            import importlib

            import stateloom.ext.langchain as mod

            importlib.reload(mod)
            # Pass gate=None and no global gate initialized
            handler = mod.StateLoomCallbackHandler(gate=None)
            handler._gate_ref = None  # ensure no gate

            # Patch get_gate to raise (simulating no init)
            with patch("stateloom.get_gate", side_effect=Exception("not init")):
                # Should not raise
                handler.on_llm_start(_make_serialized(), prompts=["hi"], run_id=uuid4())

    def test_on_llm_error_cleans_state(self, gate):
        """Error callback removes run from _llm_runs."""
        handler = _make_handler(gate)
        run_id = uuid4()

        with gate.session("test-err"):
            handler.on_chat_model_start(
                _make_serialized(),
                messages=[[]],
                run_id=run_id,
            )
            assert run_id in handler._llm_runs

            handler.on_llm_error(
                RuntimeError("boom"),
                run_id=run_id,
            )
            assert run_id not in handler._llm_runs

    def test_streaming_detection(self, gate):
        """on_llm_new_token sets is_streaming=True on the event."""
        handler = _make_handler(gate)
        run_id = uuid4()

        with gate.session("test-stream") as session:
            handler.on_chat_model_start(
                _make_serialized(),
                messages=[[]],
                run_id=run_id,
            )
            handler.on_llm_new_token("Hello", run_id=run_id)
            handler.on_llm_new_token(" world", run_id=run_id)
            handler.on_llm_end(
                _make_llm_result(50, 25, "openai"),
                run_id=run_id,
            )

        events = gate.store.get_session_events(session.id, event_type="llm_call")
        assert len(events) == 1
        assert events[0].is_streaming is True

    def test_anthropic_token_extraction(self, gate):
        """Parses Anthropic's usage.input_tokens / output_tokens format."""
        handler = _make_handler(gate)
        run_id = uuid4()

        with gate.session("test-anthropic") as session:
            handler.on_chat_model_start(
                _make_serialized("claude-opus-4-6", "anthropic"),
                messages=[[]],
                run_id=run_id,
            )
            handler.on_llm_end(
                _make_llm_result(200, 80, "anthropic"),
                run_id=run_id,
            )

        events = gate.store.get_session_events(session.id, event_type="llm_call")
        assert len(events) == 1
        ev = events[0]
        assert ev.prompt_tokens == 200
        assert ev.completion_tokens == 80
        assert ev.provider == "anthropic"

    def test_session_cost_accumulation(self, gate):
        """Multiple LLM calls correctly accumulate cost/tokens on session."""
        handler = _make_handler(gate)

        with gate.session("test-accum") as session:
            for _ in range(3):
                run_id = uuid4()
                handler.on_chat_model_start(
                    _make_serialized("gpt-4o", "openai"),
                    messages=[[]],
                    run_id=run_id,
                )
                handler.on_llm_end(
                    _make_llm_result(100, 50, "openai"),
                    run_id=run_id,
                )

        assert session.total_prompt_tokens == 300
        assert session.total_completion_tokens == 150
        assert session.total_tokens == 450
        # call_count is incremented by session.add_cost (once per call)
        assert session.call_count == 3

    def test_orphan_on_llm_end(self, gate):
        """on_llm_end for unknown run_id doesn't crash."""
        handler = _make_handler(gate)
        # No matching start — should silently return
        handler.on_llm_end(
            _make_llm_result(10, 5, "openai"),
            run_id=uuid4(),
        )

    def test_extract_model_and_provider(self):
        """Unit test the helper for various serialized dict shapes."""
        import importlib

        import stateloom.ext.langchain as mod

        # Need to ensure the helper is available regardless of langchain
        fn = mod._extract_model_and_provider

        # Standard OpenAI
        model, provider = fn(
            {
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                "kwargs": {"model_name": "gpt-4o"},
            }
        )
        assert model == "gpt-4o"
        assert provider == "openai"

        # Anthropic with "model" key
        model, provider = fn(
            {
                "id": ["langchain", "chat_models", "anthropic", "ChatAnthropic"],
                "kwargs": {"model": "claude-opus-4-6"},
            }
        )
        assert model == "claude-opus-4-6"
        assert provider == "anthropic"

        # Google GenAI
        model, provider = fn(
            {
                "id": ["langchain", "chat_models", "google_genai", "ChatGoogleGenerativeAI"],
                "kwargs": {"model": "gemini-pro"},
            }
        )
        assert model == "gemini-pro"
        assert provider == "gemini"

        # Fallback to invocation_params
        model, provider = fn(
            {
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                "kwargs": {"invocation_params": {"model_name": "gpt-3.5-turbo"}},
            }
        )
        assert model == "gpt-3.5-turbo"
        assert provider == "openai"

        # Empty / unknown
        model, provider = fn({"id": [], "kwargs": {}})
        assert model == ""
        assert provider == "unknown"

    def test_langchain_cache_hit_records_zero_cost(self, gate):
        """LangChain cache hit (same prompt twice) records two events — second has cost=0.

        When LangChain caching is enabled, the cached response preserves
        llm_output with original token counts but strips usage_metadata to None.
        The handler must detect this and record $0 cost for the cached call.
        """
        handler = _make_handler(gate)
        serialized = _make_serialized("gpt-4o", "openai")
        prompt_messages = [[{"role": "user", "content": "What is 2+2?"}]]

        with gate.session("test-lc-cache") as session:
            # --- Call 1: real LLM call (tokens + cost) ---
            run1 = uuid4()
            handler.on_chat_model_start(
                serialized, messages=prompt_messages, run_id=run1,
            )
            handler.on_llm_end(
                _make_llm_result(100, 50, "openai"),
                run_id=run1,
            )

            # --- Call 2: LangChain cache hit ---
            # Reproduces the real bug: llm_output has original token counts
            # (not empty!) but usage_metadata is stripped to None by the cache
            # layer.  Without proper detection, cost would be re-calculated
            # from the preserved llm_output tokens.
            run2 = uuid4()
            handler.on_chat_model_start(
                serialized, messages=prompt_messages, run_id=run2,
            )
            cached_msg = SimpleNamespace(usage_metadata=None)
            cached_gen = SimpleNamespace(message=cached_msg)
            cached_result = SimpleNamespace(
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                    }
                },
                generations=[[cached_gen]],
            )
            handler.on_llm_end(cached_result, run_id=run2)

        events = gate.store.get_session_events(session.id, event_type="llm_call")
        assert len(events) == 2, "Both calls must be recorded, no dedup"

        real, cached = events[0], events[1]
        # Real call has tokens and cost
        assert real.prompt_tokens == 100
        assert real.completion_tokens == 50
        assert real.cost > 0

        # Cached call has zero tokens and zero cost
        assert cached.prompt_tokens == 0
        assert cached.completion_tokens == 0
        assert cached.cost == 0.0

    def test_langchain_cache_hit_total_cost_zero(self, gate):
        """Cache hit detected via usage_metadata.total_cost == 0 (legacy signal)."""
        handler = _make_handler(gate)
        serialized = _make_serialized("gpt-4o", "openai")

        with gate.session("test-lc-cache-tc0") as session:
            run_id = uuid4()
            handler.on_chat_model_start(
                serialized, messages=[[]], run_id=run_id,
            )
            cached_msg = SimpleNamespace(usage_metadata={"total_cost": 0})
            cached_gen = SimpleNamespace(message=cached_msg)
            cached_result = SimpleNamespace(
                llm_output={"token_usage": {"prompt_tokens": 80, "completion_tokens": 40}},
                generations=[[cached_gen]],
            )
            handler.on_llm_end(cached_result, run_id=run_id)

        events = gate.store.get_session_events(session.id, event_type="llm_call")
        assert len(events) == 1
        assert events[0].cost == 0.0
        assert events[0].prompt_tokens == 0

    def test_langchain_cache_hit_latency_fallback(self, gate):
        """Cache hit detected via low latency when response structure looks fresh.

        Some LangChain versions preserve the original llm_output and
        usage_metadata on cached responses.  The latency fallback (< 100ms)
        catches these — real API calls need network + generation time.
        """
        handler = _make_handler(gate)
        serialized = _make_serialized("claude-haiku-4-5-20251001", "anthropic")

        with gate.session("test-lc-cache-latency") as session:
            # --- Real call ---
            run1 = uuid4()
            handler.on_chat_model_start(
                serialized, messages=[[]], run_id=run1,
            )
            # Simulate ~1s latency by backdating start_time
            handler._llm_runs[run1].start_time -= 1.0
            handler.on_llm_end(
                _make_llm_result(50, 30, "anthropic"),
                run_id=run1,
            )

            # --- Cache hit: llm_output AND usage_metadata fully preserved ---
            # Only the latency (< 100ms) reveals it's cached
            run2 = uuid4()
            handler.on_chat_model_start(
                serialized, messages=[[]], run_id=run2,
            )
            # Don't backdate — latency will be ~0ms (cache speed)
            cached_msg = SimpleNamespace(
                usage_metadata={"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
            )
            cached_gen = SimpleNamespace(message=cached_msg)
            cached_result = SimpleNamespace(
                llm_output={"usage": {"input_tokens": 50, "output_tokens": 30}},
                generations=[[cached_gen]],
            )
            handler.on_llm_end(cached_result, run_id=run2)

        events = gate.store.get_session_events(session.id, event_type="llm_call")
        assert len(events) == 2

        real, cached = events[0], events[1]
        assert real.cost > 0
        assert real.prompt_tokens == 50

        # Latency fallback: response looks fresh but completed too fast
        assert cached.cost == 0.0
        assert cached.prompt_tokens == 0

    def test_tools_only_skips_llm_events(self, gate):
        """tools_only=True records no LLMCallEvent, still records ToolCallEvent."""
        handler = _make_handler(gate, tools_only=True)

        with gate.session("test-tools-only") as session:
            # LLM call — should be skipped
            llm_run = uuid4()
            handler.on_chat_model_start(
                _make_serialized("gpt-4o", "openai"),
                messages=[[]],
                run_id=llm_run,
            )
            handler.on_llm_end(
                _make_llm_result(100, 50, "openai"),
                run_id=llm_run,
            )

            # Tool call — should be recorded
            tool_run = uuid4()
            handler.on_tool_start(
                {"name": "web_search"},
                input_str="query",
                run_id=tool_run,
            )
            handler.on_tool_end("results", run_id=tool_run)

        llm_events = gate.store.get_session_events(session.id, event_type="llm_call")
        tool_events = gate.store.get_session_events(session.id, event_type="tool_call")
        assert len(llm_events) == 0
        assert len(tool_events) == 1
        assert tool_events[0].tool_name == "web_search"

    def test_tools_only_false_records_llm_events(self, gate):
        """Explicit tools_only=False preserves current behavior (records both)."""
        handler = _make_handler(gate, tools_only=False)

        with gate.session("test-tools-only-false") as session:
            run_id = uuid4()
            handler.on_chat_model_start(
                _make_serialized("gpt-4o", "openai"),
                messages=[[]],
                run_id=run_id,
            )
            handler.on_llm_end(
                _make_llm_result(100, 50, "openai"),
                run_id=run_id,
            )

        events = gate.store.get_session_events(session.id, event_type="llm_call")
        assert len(events) == 1

    def test_tools_only_auto_detects_auto_patch_true(self, gate_auto_patch):
        """tools_only=None + auto_patch=True -> tools only (skips LLM events)."""
        handler = _make_handler(gate_auto_patch, tools_only=None)
        assert handler._effective_tools_only is True

        with gate_auto_patch.session("test-auto-detect-true") as session:
            run_id = uuid4()
            handler.on_chat_model_start(
                _make_serialized("gpt-4o", "openai"),
                messages=[[]],
                run_id=run_id,
            )
            handler.on_llm_end(
                _make_llm_result(100, 50, "openai"),
                run_id=run_id,
            )

        events = gate_auto_patch.store.get_session_events(
            session.id, event_type="llm_call"
        )
        assert len(events) == 0

    def test_tools_only_auto_detects_auto_patch_false(self, gate):
        """tools_only=None + auto_patch=False -> records both."""
        handler = _make_handler(gate, tools_only=None)
        assert handler._effective_tools_only is False

        with gate.session("test-auto-detect-false") as session:
            run_id = uuid4()
            handler.on_chat_model_start(
                _make_serialized("gpt-4o", "openai"),
                messages=[[]],
                run_id=run_id,
            )
            handler.on_llm_end(
                _make_llm_result(100, 50, "openai"),
                run_id=run_id,
            )

        events = gate.store.get_session_events(session.id, event_type="llm_call")
        assert len(events) == 1

    def test_tools_only_no_step_inflation(self, gate):
        """Step counter not incremented by LLM callbacks in tools_only mode."""
        handler = _make_handler(gate, tools_only=True)

        with gate.session("test-no-step-inflation") as session:
            # LLM calls should not increment step counter
            for _ in range(3):
                run_id = uuid4()
                handler.on_chat_model_start(
                    _make_serialized("gpt-4o", "openai"),
                    messages=[[]],
                    run_id=run_id,
                )
                handler.on_llm_end(
                    _make_llm_result(100, 50, "openai"),
                    run_id=run_id,
                )

            # No _llm_runs state should be stored
            assert len(handler._llm_runs) == 0

            # Step counter should still be at 0 (no increments)
            assert session.step_counter == 0

    def test_langchain_callback_factory_forwards_tools_only(self, gate):
        """stateloom.langchain_callback(tools_only=True) forwards the parameter."""
        handler = stateloom.langchain_callback(gate=gate, tools_only=True)
        assert handler._tools_only is True

        handler_none = stateloom.langchain_callback(gate=gate, tools_only=None)
        assert handler_none._tools_only is None

        handler_false = stateloom.langchain_callback(gate=gate, tools_only=False)
        assert handler_false._tools_only is False

    def test_framework_context_set_in_tools_only_mode(self, gate):
        """on_llm_start sets the framework ContextVar even when tools_only=True."""
        from stateloom.core.context import get_framework_context, clear_framework_context

        handler = _make_handler(gate, tools_only=True)
        run_id = uuid4()

        with gate.session("test-fw-ctx-tools-only"):
            handler.on_llm_start(
                _make_serialized("gpt-4o", "openai"),
                prompts=["hello"],
                run_id=run_id,
            )
            ctx = get_framework_context()
            assert "langchain" in ctx
            assert ctx["langchain"]["run_id"] == str(run_id)
            assert ctx["langchain"]["model"] == "gpt-4o"
            assert ctx["langchain"]["provider"] == "openai"

            # Clean up
            clear_framework_context()

    def test_framework_context_set_in_on_chat_model_start(self, gate):
        """on_chat_model_start sets the framework ContextVar even when tools_only=True."""
        from stateloom.core.context import get_framework_context, clear_framework_context

        handler = _make_handler(gate, tools_only=True)
        run_id = uuid4()

        with gate.session("test-fw-ctx-chat-model"):
            handler.on_chat_model_start(
                _make_serialized("claude-opus-4-6", "anthropic"),
                messages=[[]],
                run_id=run_id,
            )
            ctx = get_framework_context()
            assert "langchain" in ctx
            assert ctx["langchain"]["run_id"] == str(run_id)
            assert ctx["langchain"]["model"] == "claude-opus-4-6"
            assert ctx["langchain"]["provider"] == "anthropic"

            # Clean up
            clear_framework_context()

    def test_framework_context_cleared_on_llm_end(self, gate):
        """on_llm_end clears the framework ContextVar."""
        from stateloom.core.context import get_framework_context, set_framework_context

        handler = _make_handler(gate)
        run_id = uuid4()

        with gate.session("test-fw-ctx-clear-end"):
            handler.on_chat_model_start(
                _make_serialized("gpt-4o", "openai"),
                messages=[[]],
                run_id=run_id,
            )
            # Verify it was set
            assert get_framework_context().get("langchain") is not None

            handler.on_llm_end(
                _make_llm_result(100, 50, "openai"),
                run_id=run_id,
            )
            # Verify it was cleared
            assert get_framework_context() == {}

    def test_framework_context_cleared_on_llm_error(self, gate):
        """on_llm_error clears the framework ContextVar."""
        from stateloom.core.context import get_framework_context, set_framework_context

        handler = _make_handler(gate)
        run_id = uuid4()

        with gate.session("test-fw-ctx-clear-err"):
            handler.on_chat_model_start(
                _make_serialized("gpt-4o", "openai"),
                messages=[[]],
                run_id=run_id,
            )
            assert get_framework_context().get("langchain") is not None

            handler.on_llm_error(
                RuntimeError("boom"),
                run_id=run_id,
            )
            assert get_framework_context() == {}
