"""Tests for adapter response methods: extract_content, modify_response_text, to_openai_dict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter
from stateloom.intercept.adapters.cohere_adapter import CohereAdapter
from stateloom.intercept.adapters.gemini_adapter import GeminiAdapter
from stateloom.intercept.adapters.gemini_genai_adapter import GeminiGenaiAdapter
from stateloom.intercept.adapters.mistral_adapter import MistralAdapter
from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter
from stateloom.local.adapter import OllamaAdapter
from stateloom.local.client import OllamaResponse

# ---------------------------------------------------------------------------
# Mock response objects
# ---------------------------------------------------------------------------


@dataclass
class MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class MockMessage:
    role: str = "assistant"
    content: str = "Hello from OpenAI"
    tool_calls: list | None = None


@dataclass
class MockChoice:
    index: int = 0
    message: MockMessage = field(default_factory=MockMessage)
    finish_reason: str = "stop"


@dataclass
class MockOpenAIResponse:
    choices: list[MockChoice] = field(default_factory=lambda: [MockChoice()])
    usage: MockUsage = field(default_factory=MockUsage)
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
    total_token_count: int = 30


@dataclass
class MockGeminiResponse:
    text: str = "Hello from Gemini"
    candidates: list[MockGeminiCandidate] = field(default_factory=lambda: [MockGeminiCandidate()])
    usage_metadata: MockGeminiUsageMetadata = field(default_factory=MockGeminiUsageMetadata)


# Mistral wraps responses in a .data attribute
@dataclass
class MockMistralResponse:
    choices: list[MockChoice] = field(
        default_factory=lambda: [MockChoice(message=MockMessage(content="Hello from Mistral"))]
    )
    usage: MockUsage = field(default_factory=MockUsage)
    model: str = "mistral-large"


@dataclass
class MockCohereTextBlock:
    text: str = "Hello from Cohere"


@dataclass
class MockCohereMessage:
    content: list = field(default_factory=lambda: [MockCohereTextBlock()])


@dataclass
class MockCohereTokens:
    input_tokens: int = 8
    output_tokens: int = 12


@dataclass
class MockCohereUsageInner:
    tokens: MockCohereTokens = field(default_factory=MockCohereTokens)


@dataclass
class MockCohereResponse:
    message: MockCohereMessage = field(default_factory=MockCohereMessage)
    usage: MockCohereUsageInner = field(default_factory=MockCohereUsageInner)


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class TestOpenAIExtractContent:
    def test_basic(self):
        adapter = OpenAIAdapter()
        resp = MockOpenAIResponse()
        assert adapter.extract_content(resp) == "Hello from OpenAI"

    def test_empty_choices(self):
        adapter = OpenAIAdapter()
        resp = MockOpenAIResponse(choices=[])
        assert adapter.extract_content(resp) == ""

    def test_none_content(self):
        adapter = OpenAIAdapter()
        resp = MockOpenAIResponse(choices=[MockChoice(message=MockMessage(content=None))])
        assert adapter.extract_content(resp) == ""


class TestOpenAIModifyResponseText:
    def test_modifies_content(self):
        adapter = OpenAIAdapter()
        resp = MockOpenAIResponse()
        adapter.modify_response_text(resp, str.upper)
        assert resp.choices[0].message.content == "HELLO FROM OPENAI"

    def test_multiple_choices(self):
        adapter = OpenAIAdapter()
        resp = MockOpenAIResponse(
            choices=[
                MockChoice(message=MockMessage(content="First")),
                MockChoice(index=1, message=MockMessage(content="Second")),
            ]
        )
        adapter.modify_response_text(resp, str.upper)
        assert resp.choices[0].message.content == "FIRST"
        assert resp.choices[1].message.content == "SECOND"


class TestOpenAIToDict:
    def test_basic_conversion(self):
        adapter = OpenAIAdapter()
        resp = MockOpenAIResponse()
        result = adapter.to_openai_dict(resp, "gpt-4", "req-1")
        assert result["id"] == "req-1"
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello from OpenAI"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20

    def test_model_dump_passthrough(self):
        """When response has model_dump, use it directly."""
        adapter = OpenAIAdapter()

        class DumpableResponse:
            model = "gpt-4"

            def model_dump(self):
                return {"id": "orig", "object": "chat.completion", "choices": []}

        result = adapter.to_openai_dict(DumpableResponse(), "gpt-4", "req-2")
        assert result["id"] == "req-2"  # request_id overrides


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


class TestAnthropicExtractContent:
    def test_text_block(self):
        adapter = AnthropicAdapter()
        resp = MockAnthropicMessage()
        assert adapter.extract_content(resp) == "Hello from Anthropic"

    def test_tool_only(self):
        adapter = AnthropicAdapter()
        resp = MockAnthropicMessage(content=[MockAnthropicToolBlock()])
        assert adapter.extract_content(resp) == ""

    def test_mixed_blocks(self):
        adapter = AnthropicAdapter()
        resp = MockAnthropicMessage(
            content=[MockAnthropicToolBlock(), MockAnthropicTextBlock(text="After tool")]
        )
        assert adapter.extract_content(resp) == "After tool"


class TestAnthropicModifyResponseText:
    def test_modifies_text_blocks(self):
        adapter = AnthropicAdapter()
        resp = MockAnthropicMessage()
        adapter.modify_response_text(resp, str.upper)
        assert resp.content[0].text == "HELLO FROM ANTHROPIC"

    def test_skips_non_text_blocks(self):
        adapter = AnthropicAdapter()
        tool_block = MockAnthropicToolBlock()
        text_block = MockAnthropicTextBlock(text="Modify me")
        resp = MockAnthropicMessage(content=[tool_block, text_block])
        adapter.modify_response_text(resp, str.upper)
        assert text_block.text == "MODIFY ME"
        # tool block unchanged (has no .text that's a string to modify)
        assert tool_block.name == "get_weather"


class TestAnthropicToDict:
    def test_basic_conversion(self):
        adapter = AnthropicAdapter()
        resp = MockAnthropicMessage()
        result = adapter.to_openai_dict(resp, "claude-3-opus", "req-3")
        assert result["id"] == "req-3"
        assert result["choices"][0]["message"]["content"] == "Hello from Anthropic"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 25

    def test_tool_use_conversion(self):
        adapter = AnthropicAdapter()
        resp = MockAnthropicMessage(
            content=[MockAnthropicToolBlock()],
            stop_reason="tool_use",
        )
        result = adapter.to_openai_dict(resp, "claude-3-opus", "req-4")
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["function"]["name"] == "get_weather"

    def test_max_tokens_stop_reason(self):
        adapter = AnthropicAdapter()
        resp = MockAnthropicMessage(stop_reason="max_tokens")
        result = adapter.to_openai_dict(resp, "claude-3-opus", "req-5")
        assert result["choices"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------


class TestGeminiExtractContent:
    def test_text_property(self):
        adapter = GeminiAdapter()
        resp = MockGeminiResponse()
        assert adapter.extract_content(resp) == "Hello from Gemini"

    def test_text_raises_fallback_to_candidates(self):
        adapter = GeminiAdapter()

        class _Resp:
            @property
            def text(self):
                raise ValueError("Multiple candidates")

            candidates = [
                MockGeminiCandidate(
                    content=MockGeminiContent(parts=[MockGeminiPart(text="Fallback")])
                )
            ]

        assert adapter.extract_content(_Resp()) == "Fallback"

    def test_empty_candidates(self):
        adapter = GeminiAdapter()

        class _Resp:
            @property
            def text(self):
                raise ValueError("No content")

            candidates = []

        assert adapter.extract_content(_Resp()) == ""


class TestGeminiModifyResponseText:
    def test_modifies_parts(self):
        adapter = GeminiAdapter()
        resp = MockGeminiResponse()
        adapter.modify_response_text(resp, str.upper)
        assert resp.candidates[0].content.parts[0].text == "HELLO FROM GEMINI"


class TestGeminiToDict:
    def test_basic_conversion(self):
        adapter = GeminiAdapter()
        resp = MockGeminiResponse()
        result = adapter.to_openai_dict(resp, "gemini-pro", "req-6")
        assert result["id"] == "req-6"
        assert result["choices"][0]["message"]["content"] == "Hello from Gemini"
        assert result["usage"]["prompt_tokens"] == 12
        assert result["usage"]["completion_tokens"] == 18


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------


class TestOllamaExtractContent:
    def test_ollama_response(self):
        adapter = OllamaAdapter()
        resp = OllamaResponse(
            model="llama3.2",
            content="Hello local",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            latency_ms=50.0,
        )
        assert adapter.extract_content(resp) == "Hello local"

    def test_generic_content_attr(self):
        adapter = OllamaAdapter()

        class _Resp:
            content = "Generic content"

        assert adapter.extract_content(_Resp()) == "Generic content"

    def test_empty_content(self):
        adapter = OllamaAdapter()
        resp = OllamaResponse(content="")
        assert adapter.extract_content(resp) == ""


class TestOllamaModifyResponseText:
    def test_modifies_ollama_response(self):
        adapter = OllamaAdapter()
        resp = OllamaResponse(content="Original text")
        adapter.modify_response_text(resp, str.upper)
        assert resp.content == "ORIGINAL TEXT"


class TestOllamaToDict:
    def test_basic_conversion(self):
        adapter = OllamaAdapter()
        resp = OllamaResponse(
            model="llama3.2",
            content="Hello local",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            latency_ms=50.0,
        )
        result = adapter.to_openai_dict(resp, "llama3.2", "req-7")
        assert result["id"] == "req-7"
        assert result["choices"][0]["message"]["content"] == "Hello local"
        assert result["usage"]["prompt_tokens"] == 5
        assert result["usage"]["completion_tokens"] == 3


# ---------------------------------------------------------------------------
# Mistral adapter
# ---------------------------------------------------------------------------


class TestMistralExtractContent:
    def test_basic(self):
        adapter = MistralAdapter()
        resp = MockMistralResponse()
        assert adapter.extract_content(resp) == "Hello from Mistral"

    def test_wrapped_in_data(self):
        """Mistral sometimes wraps responses in a .data attribute."""
        adapter = MistralAdapter()

        class _Wrapped:
            data = MockMistralResponse()

        assert adapter.extract_content(_Wrapped()) == "Hello from Mistral"


class TestMistralToDict:
    def test_basic_conversion(self):
        adapter = MistralAdapter()
        resp = MockMistralResponse()
        result = adapter.to_openai_dict(resp, "mistral-large", "req-8")
        assert result["id"] == "req-8"
        assert result["choices"][0]["message"]["content"] == "Hello from Mistral"
        assert result["usage"]["prompt_tokens"] == 10


# ---------------------------------------------------------------------------
# Cohere adapter
# ---------------------------------------------------------------------------


class TestCohereExtractContent:
    def test_basic(self):
        adapter = CohereAdapter()
        resp = MockCohereResponse()
        assert adapter.extract_content(resp) == "Hello from Cohere"


class TestCohereModifyResponseText:
    def test_modifies_content_blocks(self):
        adapter = CohereAdapter()
        resp = MockCohereResponse()
        adapter.modify_response_text(resp, str.upper)
        assert resp.message.content[0].text == "HELLO FROM COHERE"


class TestCohereToDict:
    def test_basic_conversion(self):
        adapter = CohereAdapter()
        resp = MockCohereResponse()
        result = adapter.to_openai_dict(resp, "command-r-plus", "req-9")
        assert result["id"] == "req-9"
        assert result["choices"][0]["message"]["content"] == "Hello from Cohere"
        assert result["usage"]["prompt_tokens"] == 8
        assert result["usage"]["completion_tokens"] == 12


# ---------------------------------------------------------------------------
# Cross-adapter: None / broken responses
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize(
        "adapter_cls",
        [
            OpenAIAdapter,
            AnthropicAdapter,
            GeminiAdapter,
            GeminiGenaiAdapter,
            OllamaAdapter,
            MistralAdapter,
            CohereAdapter,
        ],
    )
    def test_extract_content_none_response(self, adapter_cls):
        """All adapters should return '' for None."""
        adapter = adapter_cls()
        assert adapter.extract_content(None) == ""

    @pytest.mark.parametrize(
        "adapter_cls",
        [
            OpenAIAdapter,
            AnthropicAdapter,
            GeminiAdapter,
            GeminiGenaiAdapter,
            OllamaAdapter,
            MistralAdapter,
            CohereAdapter,
        ],
    )
    def test_modify_response_text_none_response(self, adapter_cls):
        """All adapters should silently handle None."""
        adapter = adapter_cls()
        adapter.modify_response_text(None, str.upper)  # should not raise

    @pytest.mark.parametrize(
        "adapter_cls",
        [
            OpenAIAdapter,
            AnthropicAdapter,
            GeminiAdapter,
            GeminiGenaiAdapter,
            OllamaAdapter,
            MistralAdapter,
            CohereAdapter,
        ],
    )
    def test_extract_content_unrelated_object(self, adapter_cls):
        """All adapters return '' for unrelated objects."""
        adapter = adapter_cls()
        assert adapter.extract_content(object()) == ""


# ---------------------------------------------------------------------------
# GeminiGenai adapter — mock objects for function calling
# ---------------------------------------------------------------------------


@dataclass
class MockGenaiFC:
    """Mock Gemini function_call attribute on a part."""

    name: str = "get_weather"
    args: dict = field(default_factory=lambda: {"city": "SF"})


@dataclass
class MockGenaiPartText:
    """Mock Gemini part with text only."""

    text: str = "Hello from GenAI"
    function_call: MockGenaiFC | None = None


@dataclass
class MockGenaiPartFC:
    """Mock Gemini part with function_call only."""

    text: str = ""
    function_call: MockGenaiFC = field(default_factory=MockGenaiFC)


@dataclass
class MockGenaiContent:
    parts: list = field(default_factory=lambda: [MockGenaiPartText()])


@dataclass
class MockGenaiCandidate:
    content: MockGenaiContent = field(default_factory=MockGenaiContent)


@dataclass
class MockGenaiUsageMeta:
    prompt_token_count: int = 10
    candidates_token_count: int = 15
    total_token_count: int = 25


@dataclass
class MockGenaiResponse:
    text: str = "Hello from GenAI"
    candidates: list[MockGenaiCandidate] = field(default_factory=lambda: [MockGenaiCandidate()])
    usage_metadata: MockGenaiUsageMeta = field(default_factory=MockGenaiUsageMeta)


@dataclass
class MockGenaiResponseFC:
    """Response with a function call part (no text)."""

    text: str = ""
    candidates: list[MockGenaiCandidate] = field(
        default_factory=lambda: [
            MockGenaiCandidate(content=MockGenaiContent(parts=[MockGenaiPartFC()]))
        ]
    )
    usage_metadata: MockGenaiUsageMeta = field(default_factory=MockGenaiUsageMeta)


@dataclass
class MockGenaiResponseMixed:
    """Response with both text and function call parts."""

    text: str = "Checking weather"
    candidates: list[MockGenaiCandidate] = field(
        default_factory=lambda: [
            MockGenaiCandidate(
                content=MockGenaiContent(
                    parts=[
                        MockGenaiPartText(text="Checking weather"),
                        MockGenaiPartFC(),
                    ]
                )
            )
        ]
    )
    usage_metadata: MockGenaiUsageMeta = field(default_factory=MockGenaiUsageMeta)


@dataclass
class MockGenaiResponseMultiFC:
    """Response with multiple function calls."""

    text: str = ""
    candidates: list[MockGenaiCandidate] = field(
        default_factory=lambda: [
            MockGenaiCandidate(
                content=MockGenaiContent(
                    parts=[
                        MockGenaiPartFC(
                            function_call=MockGenaiFC(name="get_weather", args={"city": "SF"})
                        ),
                        MockGenaiPartFC(
                            function_call=MockGenaiFC(name="get_time", args={"timezone": "PST"})
                        ),
                    ]
                )
            )
        ]
    )
    usage_metadata: MockGenaiUsageMeta = field(default_factory=MockGenaiUsageMeta)


# ---------------------------------------------------------------------------
# GeminiGenai adapter — extract_content
# ---------------------------------------------------------------------------


class TestGeminiGenaiExtractContent:
    def test_text(self):
        adapter = GeminiGenaiAdapter()
        resp = MockGenaiResponse()
        assert adapter.extract_content(resp) == "Hello from GenAI"

    def test_function_call_only(self):
        adapter = GeminiGenaiAdapter()
        resp = MockGenaiResponseFC()
        assert adapter.extract_content(resp) == ""

    def test_mixed_text_and_function_call(self):
        adapter = GeminiGenaiAdapter()
        resp = MockGenaiResponseMixed()
        assert adapter.extract_content(resp) == "Checking weather"


# ---------------------------------------------------------------------------
# GeminiGenai adapter — to_openai_dict
# ---------------------------------------------------------------------------


class TestGeminiGenaiToDict:
    def test_text_response(self):
        adapter = GeminiGenaiAdapter()
        resp = MockGenaiResponse()
        result = adapter.to_openai_dict(resp, "gemini-2.0-flash", "req-genai-1")
        assert result["id"] == "req-genai-1"
        assert result["model"] == "gemini-2.0-flash"
        assert result["choices"][0]["message"]["content"] == "Hello from GenAI"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in result["choices"][0]["message"]
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 15

    def test_single_tool_call(self):
        adapter = GeminiGenaiAdapter()
        resp = MockGenaiResponseFC()
        result = adapter.to_openai_dict(resp, "gemini-2.0-flash", "req-genai-2")
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tcs = result["choices"][0]["message"]["tool_calls"]
        assert len(tcs) == 1
        assert tcs[0]["type"] == "function"
        assert tcs[0]["function"]["name"] == "get_weather"
        assert tcs[0]["function"]["arguments"] == '{"city": "SF"}'
        assert tcs[0]["id"].startswith("call_")

    def test_multiple_tool_calls(self):
        adapter = GeminiGenaiAdapter()
        resp = MockGenaiResponseMultiFC()
        result = adapter.to_openai_dict(resp, "gemini-2.0-flash", "req-genai-3")
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tcs = result["choices"][0]["message"]["tool_calls"]
        assert len(tcs) == 2
        assert tcs[0]["function"]["name"] == "get_weather"
        assert tcs[1]["function"]["name"] == "get_time"


# ---------------------------------------------------------------------------
# GeminiGenai adapter — edge cases
# ---------------------------------------------------------------------------


class TestGeminiGenaiEdgeCases:
    def test_none_response(self):
        adapter = GeminiGenaiAdapter()
        assert adapter.extract_content(None) == ""

    def test_unrelated_object(self):
        adapter = GeminiGenaiAdapter()
        assert adapter.extract_content(object()) == ""

    def test_modify_response_text_none(self):
        adapter = GeminiGenaiAdapter()
        adapter.modify_response_text(None, str.upper)  # should not raise

    def test_to_dict_with_no_usage(self):
        """Response without usage_metadata should default to 0 tokens."""
        adapter = GeminiGenaiAdapter()

        @dataclass
        class _Resp:
            text: str = "hello"
            candidates: list = field(default_factory=lambda: [MockGenaiCandidate()])

        resp = _Resp()
        result = adapter.to_openai_dict(resp, "gemini-pro", "req-x")
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0
