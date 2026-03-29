"""Tests for tool-continuation detection in CostTracker."""

from __future__ import annotations

import types

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import LLMCallEvent
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.cost_tracker import CostTracker, _extract_tool_preview, _is_tool_continuation
from stateloom.pricing.registry import PricingRegistry


# --- Unit tests for _is_tool_continuation ---


class TestIsToolContinuation:
    def test_openai_tool_result_message(self):
        """OpenAI format: last message role=tool indicates tool results."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt\nfile2.txt"},
        ]
        assert _is_tool_continuation(messages) is True

    def test_anthropic_tool_result_block(self):
        """Anthropic format: last message is user with tool_result content blocks."""
        messages = [
            {"role": "user", "content": "What files are here?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "result"},
                ],
            },
        ]
        assert _is_tool_continuation(messages) is True

    def test_normal_user_message_not_continuation(self):
        """Plain user messages are not tool continuations."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        assert _is_tool_continuation(messages) is False

    def test_empty_messages(self):
        assert _is_tool_continuation([]) is False

    def test_assistant_only_not_continuation(self):
        messages = [{"role": "assistant", "content": "I can help with that."}]
        assert _is_tool_continuation(messages) is False

    def test_mixed_content_blocks_without_tool_result(self):
        """User message with text content blocks but no tool_result."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the weather?"},
                    {"type": "image", "source": {"data": "..."}},
                ],
            },
        ]
        assert _is_tool_continuation(messages) is False

    def test_multiple_tool_results(self):
        """Multiple OpenAI tool results — last message is still role=tool."""
        messages = [
            {"role": "user", "content": "Run both commands"},
            {"role": "assistant", "content": None},
            {"role": "tool", "tool_call_id": "tc1", "content": "output1"},
            {"role": "tool", "tool_call_id": "tc2", "content": "output2"},
        ]
        assert _is_tool_continuation(messages) is True

    def test_string_content_not_misdetected(self):
        """String content in user messages should not trigger detection."""
        messages = [
            {"role": "user", "content": "Please use the tool_result for analysis"},
        ]
        assert _is_tool_continuation(messages) is False

    # --- Multi-turn tests: old tool results in history, new user prompt at end ---

    def test_openai_new_user_prompt_after_tool_history(self):
        """New user message after prior tool results — NOT a continuation.

        This is the core bug: scanning all messages would return True because
        of the old role=tool message, but only the last message matters.
        """
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt\nfile2.txt"},
            {"role": "assistant", "content": "Here are the files: file1.txt, file2.txt"},
            {"role": "user", "content": "Now check the demo directory"},
        ]
        assert _is_tool_continuation(messages) is False

    def test_anthropic_new_user_prompt_after_tool_history(self):
        """Anthropic format: new user text after prior tool_result blocks — NOT a continuation."""
        messages = [
            {"role": "user", "content": "What files are here?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "result"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "I found some files."}]},
            {"role": "user", "content": "Great, now summarize them"},
        ]
        assert _is_tool_continuation(messages) is False

    def test_multi_turn_with_multiple_tool_rounds_then_user(self):
        """Several tool rounds in history, then a fresh user prompt."""
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Search for errors"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "error.log contents..."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc2"}]},
            {"role": "tool", "tool_call_id": "tc2", "content": "more errors..."},
            {"role": "assistant", "content": "Found 3 errors in the logs."},
            {"role": "user", "content": "Fix the first error"},
        ]
        assert _is_tool_continuation(messages) is False

    def test_multi_turn_still_detects_actual_continuation(self):
        """Even with prior tool history, a new tool result at the end IS a continuation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt"},
            {"role": "assistant", "content": "Found file1.txt. Let me check more."},
            {"role": "user", "content": "Check the demo dir too"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc2"}]},
            {"role": "tool", "tool_call_id": "tc2", "content": "demo/app.py"},
        ]
        assert _is_tool_continuation(messages) is True

    def test_anthropic_multi_turn_new_tool_result_at_end(self):
        """Anthropic format: new tool_result at end after prior history IS a continuation."""
        messages = [
            {"role": "user", "content": "Analyze code"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "old result"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            {"role": "user", "content": "Now run the tests"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu2"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu2", "content": "new result"},
                ],
            },
        ]
        assert _is_tool_continuation(messages) is True

    def test_last_message_is_system_not_continuation(self):
        """Edge case: last message is system role (unusual but possible)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "system", "content": "Updated instructions"},
        ]
        assert _is_tool_continuation(messages) is False

    # --- Gemini-translated messages (functionResponse → role="tool") ---

    def test_gemini_translated_function_response(self):
        """Gemini functionResponse translated to role=tool is detected."""
        # After Gemini proxy translation, functionResponse becomes role="tool"
        messages = [
            {"role": "user", "content": "check the codebase"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "codebase_investigator", "type": "function",
                 "function": {"name": "codebase_investigator", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": '{"output": "found 3 issues"}',
             "tool_call_id": "codebase_investigator"},
        ]
        assert _is_tool_continuation(messages) is True

    def test_gemini_translated_new_prompt_after_function_history(self):
        """Gemini: new user prompt after prior function calls — NOT a continuation."""
        messages = [
            {"role": "user", "content": "check the codebase"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "search", "type": "function",
                 "function": {"name": "search", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": '{"result": "ok"}', "tool_call_id": "search"},
            {"role": "assistant", "content": "I found no issues."},
            {"role": "user", "content": "now fix the bug"},
        ]
        assert _is_tool_continuation(messages) is False


# --- Integration test: CostTracker sets is_tool_continuation on event ---


def _make_ctx(messages: list[dict]) -> MiddlewareContext:
    session = Session(id="tc-test")
    config = StateLoomConfig()
    ctx = MiddlewareContext(session=session, config=config)
    ctx.model = "gpt-4o"
    ctx.provider = "openai"
    ctx.prompt_tokens = 100
    ctx.completion_tokens = 50
    ctx.request_kwargs = {"messages": messages}
    ctx.response = types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="done"))],
    )
    return ctx


@pytest.mark.asyncio
async def test_cost_tracker_sets_tool_continuation_true():
    """CostTracker should mark events with is_tool_continuation when tool results present."""
    pricing = PricingRegistry()
    pricing.register("gpt-4o", input_per_token=0.00001, output_per_token=0.00002)
    tracker = CostTracker(pricing)

    messages = [
        {"role": "user", "content": "List files"},
        {"role": "assistant", "content": None},
        {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt"},
    ]
    ctx = _make_ctx(messages)

    async def call_next(c):
        return c.response

    await tracker.process(ctx, call_next)

    llm_events = [e for e in ctx.events if isinstance(e, LLMCallEvent)]
    assert len(llm_events) == 1
    assert llm_events[0].is_tool_continuation is True


@pytest.mark.asyncio
async def test_cost_tracker_sets_tool_continuation_false():
    """CostTracker should not mark normal messages as tool continuations."""
    pricing = PricingRegistry()
    pricing.register("gpt-4o", input_per_token=0.00001, output_per_token=0.00002)
    tracker = CostTracker(pricing)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    ctx = _make_ctx(messages)

    async def call_next(c):
        return c.response

    await tracker.process(ctx, call_next)

    llm_events = [e for e in ctx.events if isinstance(e, LLMCallEvent)]
    assert len(llm_events) == 1
    assert llm_events[0].is_tool_continuation is False


@pytest.mark.asyncio
async def test_cost_tracker_multi_turn_new_prompt_not_continuation():
    """Multi-turn: old tool results in history + new user prompt → NOT continuation."""
    pricing = PricingRegistry()
    pricing.register("gpt-4o", input_per_token=0.00001, output_per_token=0.00002)
    tracker = CostTracker(pricing)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "List files"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt"},
        {"role": "assistant", "content": "Found file1.txt"},
        {"role": "user", "content": "Now check the demo directory"},
    ]
    ctx = _make_ctx(messages)

    async def call_next(c):
        return c.response

    await tracker.process(ctx, call_next)

    llm_events = [e for e in ctx.events if isinstance(e, LLMCallEvent)]
    assert len(llm_events) == 1
    assert llm_events[0].is_tool_continuation is False
    assert "demo directory" in llm_events[0].prompt_preview


# --- Tests for _extract_tool_preview ---


class TestExtractToolPreview:
    def test_openai_tool_calls_with_reasoning(self):
        """Shows tool name + assistant reasoning text."""
        messages = [
            {"role": "user", "content": "check the code"},
            {
                "role": "assistant",
                "content": "I will read the main file.",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "print('hi')"},
        ]
        preview = _extract_tool_preview(messages)
        assert "read_file" in preview
        assert "I will read the main file" in preview

    def test_multiple_tool_calls(self):
        """Multiple tool calls shown as comma-separated names."""
        messages = [
            {"role": "user", "content": "analyze everything"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "read_file"}},
                    {"id": "tc2", "type": "function", "function": {"name": "list_dir"}},
                    {"id": "tc3", "type": "function", "function": {"name": "search"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "..."},
            {"role": "tool", "tool_call_id": "tc2", "content": "..."},
            {"role": "tool", "tool_call_id": "tc3", "content": "..."},
        ]
        preview = _extract_tool_preview(messages)
        assert "read_file" in preview
        assert "list_dir" in preview
        assert "search" in preview

    def test_no_assistant_message_returns_empty(self):
        """No assistant message → empty preview."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        preview = _extract_tool_preview(messages)
        assert preview == ""

    def test_anthropic_tool_use_blocks(self):
        """Anthropic tool_use content blocks detected."""
        messages = [
            {"role": "user", "content": "check"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {"type": "tool_use", "id": "tu1", "name": "grep_search", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "found"}],
            },
        ]
        preview = _extract_tool_preview(messages)
        assert "grep_search" in preview
        assert "Let me search" in preview

    def test_tool_name_only_when_no_text(self):
        """Tool names shown even when assistant has no text content."""
        messages = [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "execute_cmd"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        ]
        preview = _extract_tool_preview(messages)
        assert preview == "execute_cmd"

    def test_preview_truncated_at_50_chars(self):
        """Long previews are truncated to 50 chars + ellipsis."""
        messages = [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": "I will now examine the entire codebase structure and identify potential improvements",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "read_file"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "..."},
        ]
        preview = _extract_tool_preview(messages)
        assert len(preview) == 53  # 50 + "..."
        assert preview.endswith("...")

    def test_gemini_translated_tool_calls(self):
        """Gemini functionCall translated to OpenAI format shows tool names."""
        # After code_assist translation: functionCall → tool_calls
        messages = [
            {"role": "user", "content": "check the codebase"},
            {
                "role": "assistant",
                "content": "I will explore the project root.",
                "tool_calls": [
                    {"id": "list_directory", "type": "function",
                     "function": {"name": "list_directory", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": '{"files": ["main.py"]}', "tool_call_id": "list_directory"},
        ]
        preview = _extract_tool_preview(messages)
        assert "list_directory" in preview
        assert "explore the project root" in preview


@pytest.mark.asyncio
async def test_cost_tracker_tool_continuation_uses_tool_preview():
    """CostTracker uses tool preview (not user prompt) for tool continuations."""
    pricing = PricingRegistry()
    pricing.register("gpt-4o", input_per_token=0.00001, output_per_token=0.00002)
    tracker = CostTracker(pricing)

    messages = [
        {"role": "user", "content": "check the codebase"},
        {
            "role": "assistant",
            "content": "I will read the main file.",
            "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read_file"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": "print('hello')"},
    ]
    ctx = _make_ctx(messages)

    async def call_next(c):
        return c.response

    await tracker.process(ctx, call_next)

    llm_events = [e for e in ctx.events if isinstance(e, LLMCallEvent)]
    assert len(llm_events) == 1
    assert llm_events[0].is_tool_continuation is True
    # Preview should show tool info, NOT the user prompt
    assert "read_file" in llm_events[0].prompt_preview
    assert "check the codebase" not in llm_events[0].prompt_preview
