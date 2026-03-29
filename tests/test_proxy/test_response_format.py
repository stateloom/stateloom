"""Tests for response format conversion to OpenAI ChatCompletion format."""

from __future__ import annotations

from dataclasses import dataclass, field

from stateloom.proxy.response_format import (
    to_openai_completion_dict,
    to_openai_done_event,
    to_openai_sse_event,
)

# --- Mock response objects ---


@dataclass
class MockOpenAIUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class MockOpenAIMessage:
    role: str = "assistant"
    content: str = "Hello from OpenAI"
    tool_calls: list | None = None


@dataclass
class MockOpenAIChoice:
    index: int = 0
    message: MockOpenAIMessage = field(default_factory=MockOpenAIMessage)
    finish_reason: str = "stop"


@dataclass
class MockOpenAIResponse:
    choices: list[MockOpenAIChoice] = field(default_factory=lambda: [MockOpenAIChoice()])
    usage: MockOpenAIUsage = field(default_factory=MockOpenAIUsage)
    model: str = "gpt-4"


@dataclass
class MockAnthropicTextBlock:
    type: str = "text"
    text: str = "Hello from Anthropic"


@dataclass
class MockAnthropicToolBlock:
    type: str = "tool_use"
    id: str = "call_123"
    name: str = "get_weather"
    input: dict = field(default_factory=lambda: {"city": "SF"})


@dataclass
class MockAnthropicUsage:
    input_tokens: int = 15
    output_tokens: int = 25


@dataclass
class MockAnthropicMessage:
    content: list = field(default_factory=lambda: [MockAnthropicTextBlock()])
    stop_reason: str = "end_turn"
    usage: MockAnthropicUsage = field(default_factory=MockAnthropicUsage)
    model: str = "claude-3-opus"


@dataclass
class MockGeminiPart:
    text: str = "Hello from Gemini"


@dataclass
class MockGeminiContent:
    parts: list[MockGeminiPart] = field(default_factory=lambda: [MockGeminiPart()])


@dataclass
class MockGeminiCandidate:
    content: MockGeminiContent = field(default_factory=MockGeminiContent)


@dataclass
class MockGeminiUsageMetadata:
    prompt_token_count: int = 12
    candidates_token_count: int = 18


@dataclass
class MockGeminiResponse:
    text: str = "Hello from Gemini"
    candidates: list[MockGeminiCandidate] = field(default_factory=lambda: [MockGeminiCandidate()])
    usage_metadata: MockGeminiUsageMetadata = field(default_factory=MockGeminiUsageMetadata)


# --- Tests ---


class TestOpenAIResponseConversion:
    def test_basic_conversion(self):
        response = MockOpenAIResponse()
        result = to_openai_completion_dict(response, "openai", "gpt-4", "req-1")
        assert result["id"] == "req-1"
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-4"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello from OpenAI"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20

    def test_auto_generates_request_id(self):
        response = MockOpenAIResponse()
        result = to_openai_completion_dict(response, "openai", "gpt-4")
        assert result["id"].startswith("chatcmpl-")


class TestAnthropicResponseConversion:
    def test_basic_conversion(self):
        response = MockAnthropicMessage()
        result = to_openai_completion_dict(response, "anthropic", "claude-3-opus", "req-2")
        assert result["id"] == "req-2"
        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-3-opus"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello from Anthropic"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 25

    def test_tool_use_conversion(self):
        response = MockAnthropicMessage(
            content=[MockAnthropicToolBlock()],
            stop_reason="tool_use",
        )
        result = to_openai_completion_dict(response, "anthropic", "claude-3-opus", "req-3")
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["function"]["name"] == "get_weather"

    def test_max_tokens_stop_reason(self):
        response = MockAnthropicMessage(stop_reason="max_tokens")
        result = to_openai_completion_dict(response, "anthropic", "claude-3-opus", "req-4")
        assert result["choices"][0]["finish_reason"] == "length"

    def test_mixed_content_blocks(self):
        response = MockAnthropicMessage(
            content=[MockAnthropicTextBlock(text="Here's the answer"), MockAnthropicToolBlock()],
            stop_reason="tool_use",
        )
        result = to_openai_completion_dict(response, "anthropic", "claude-3-opus", "req-5")
        # Content should be the first text block
        assert result["choices"][0]["message"]["content"] == "Here's the answer"
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1


class TestGeminiResponseConversion:
    def test_basic_conversion(self):
        response = MockGeminiResponse()
        result = to_openai_completion_dict(response, "gemini", "gemini-pro", "req-6")
        assert result["id"] == "req-6"
        assert result["object"] == "chat.completion"
        assert result["model"] == "gemini-pro"
        assert result["choices"][0]["message"]["content"] == "Hello from Gemini"
        assert result["usage"]["prompt_tokens"] == 12
        assert result["usage"]["completion_tokens"] == 18


class TestDictResponseConversion:
    def test_already_openai_format(self):
        data = {
            "id": "existing-id",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = to_openai_completion_dict(data, "openai", "gpt-4", "req-7")
        # The dict already has 'choices' and 'object', so it's recognized as OpenAI format.
        # The original id is preserved via setdefault (already present).
        assert result["object"] == "chat.completion"
        choices = result.get("choices", [])
        assert len(choices) == 1
        assert choices[0]["message"]["content"] == "Hi"

    def test_simple_dict(self):
        data = {"content": "Simple response"}
        result = to_openai_completion_dict(data, "openai", "gpt-4", "req-8")
        assert result["choices"][0]["message"]["content"] == "Simple response"


class TestNoneResponse:
    def test_none_returns_empty(self):
        result = to_openai_completion_dict(None, "openai", "gpt-4", "req-9")
        assert result["choices"][0]["message"]["content"] == ""


class TestSSEEvents:
    def test_sse_event_format(self):
        chunk = {"id": "1", "choices": []}
        result = to_openai_sse_event(chunk)
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_done_event(self):
        result = to_openai_done_event()
        assert result == "data: [DONE]\n\n"
