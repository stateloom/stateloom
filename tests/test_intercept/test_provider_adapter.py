"""Tests for ProviderAdapter protocol, BaseProviderAdapter, and PatchTarget."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from stateloom.intercept.provider_adapter import (
    BaseProviderAdapter,
    PatchTarget,
    ProviderAdapter,
    TokenFieldMap,
)


class TestPatchTarget:
    def test_creation(self):
        target = PatchTarget(
            target_class=object,
            method_name="foo",
            is_async=True,
            description="test target",
        )
        assert target.target_class is object
        assert target.method_name == "foo"
        assert target.is_async is True
        assert target.description == "test target"

    def test_defaults(self):
        target = PatchTarget(target_class=str, method_name="upper")
        assert target.is_async is False
        assert target.description == ""


class TestBaseProviderAdapterDefaults:
    """Verify that BaseProviderAdapter provides sensible defaults."""

    def _make_adapter(self) -> BaseProviderAdapter:
        class TestAdapter(BaseProviderAdapter):
            @property
            def name(self) -> str:
                return "test"

            @property
            def method_label(self) -> str:
                return "test.call"

        return TestAdapter()

    def test_extract_model_from_kwargs(self):
        adapter = self._make_adapter()
        assert adapter.extract_model(None, (), {"model": "gpt-4"}) == "gpt-4"

    def test_extract_model_default(self):
        adapter = self._make_adapter()
        assert adapter.extract_model(None, (), {}) == "unknown"

    def test_is_streaming_true(self):
        adapter = self._make_adapter()
        assert adapter.is_streaming({"stream": True}) is True

    def test_is_streaming_false(self):
        adapter = self._make_adapter()
        assert adapter.is_streaming({}) is False

    def test_extract_tokens_returns_zeros(self):
        adapter = self._make_adapter()
        assert adapter.extract_tokens(SimpleNamespace()) == (0, 0, 0)

    def test_extract_stream_tokens_noop(self):
        adapter = self._make_adapter()
        acc = {"prompt_tokens": 5, "completion_tokens": 10}
        result = adapter.extract_stream_tokens(SimpleNamespace(), acc)
        assert result == {"prompt_tokens": 5, "completion_tokens": 10}

    def test_apply_system_prompt_openai_style_insert(self):
        adapter = self._make_adapter()
        kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}
        adapter.apply_system_prompt(kwargs, "You are helpful.")
        assert kwargs["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert kwargs["messages"][1] == {"role": "user", "content": "hello"}

    def test_apply_system_prompt_openai_style_replace(self):
        adapter = self._make_adapter()
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "old"},
                {"role": "user", "content": "hello"},
            ]
        }
        adapter.apply_system_prompt(kwargs, "new")
        assert kwargs["messages"][0]["content"] == "new"
        assert len(kwargs["messages"]) == 2

    def test_apply_system_prompt_empty_messages(self):
        adapter = self._make_adapter()
        kwargs: dict[str, Any] = {}
        adapter.apply_system_prompt(kwargs, "sys")
        assert kwargs["messages"] == [{"role": "system", "content": "sys"}]

    def test_get_patch_targets_empty(self):
        adapter = self._make_adapter()
        assert adapter.get_patch_targets() == []

    def test_get_instance_targets_empty(self):
        adapter = self._make_adapter()
        assert adapter.get_instance_targets(None) == []

    def test_model_patterns_empty(self):
        adapter = self._make_adapter()
        assert adapter.model_patterns == []

    def test_default_base_url_empty(self):
        adapter = self._make_adapter()
        assert adapter.default_base_url == ""

    def test_prepare_chat_raises_not_implemented(self):
        adapter = self._make_adapter()
        with pytest.raises(NotImplementedError):
            adapter.prepare_chat(model="test", messages=[])


class TestProtocolRuntimeCheck:
    """Verify that the Protocol runtime_checkable works with duck-typed classes."""

    def test_duck_typed_class_passes(self):
        class DuckAdapter:
            @property
            def name(self) -> str:
                return "duck"

            @property
            def method_label(self) -> str:
                return "quack"

            def get_patch_targets(self):
                return []

            def extract_model(self, instance, args, kwargs):
                return "duck-model"

            def extract_tokens(self, response):
                return (0, 0, 0)

            def is_streaming(self, kwargs):
                return False

            def extract_stream_tokens(self, chunk, accumulated):
                return accumulated

            def extract_chunk_info(self, chunk):
                from stateloom.middleware.base import StreamChunkInfo

                return StreamChunkInfo()

            def apply_system_prompt(self, kwargs, prompt):
                pass

            def get_instance_targets(self, client):
                return []

            def normalize_request(self, args, kwargs):
                return kwargs

            def extract_base_url(self, instance):
                return ""

            def extract_content(self, response):
                return ""

            def modify_response_text(self, response, modifier):
                pass

            def to_openai_dict(self, response, model, request_id):
                return {}

            def modify_chunk_text(self, chunk, new_text):
                return chunk

            @property
            def model_patterns(self):
                return []

            @property
            def default_base_url(self):
                return ""

            def confidence_instruction(self):
                return ""

            def extract_confidence(self, text):
                return None

            def prepare_chat(self, *, model, messages, provider_keys=None, **kwargs):
                raise NotImplementedError

        assert isinstance(DuckAdapter(), ProviderAdapter)

    def test_base_adapter_subclass_passes(self):
        class MyAdapter(BaseProviderAdapter):
            @property
            def name(self):
                return "my"

            @property
            def method_label(self):
                return "my.call"

        assert isinstance(MyAdapter(), ProviderAdapter)


class TestExtractTokensFromFields:
    """Test _extract_tokens_from_fields with all field-map shapes."""

    def _make_adapter(self) -> BaseProviderAdapter:
        class _A(BaseProviderAdapter):
            @property
            def name(self) -> str:
                return "test"

            @property
            def method_label(self) -> str:
                return "test.call"

        return _A()

    @pytest.mark.parametrize(
        "fields, response, expected",
        [
            # OpenAI / LiteLLM / Mistral (defaults)
            (
                TokenFieldMap(),
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
                ),
                (10, 20, 30),
            ),
            # Anthropic (input_tokens/output_tokens, total computed)
            (
                TokenFieldMap(
                    input_field="input_tokens",
                    output_field="output_tokens",
                    total_field="",
                ),
                SimpleNamespace(usage=SimpleNamespace(input_tokens=15, output_tokens=25)),
                (15, 25, 40),
            ),
            # Gemini (usage_metadata, custom field names)
            (
                TokenFieldMap(
                    usage_attr="usage_metadata",
                    input_field="prompt_token_count",
                    output_field="candidates_token_count",
                    total_field="total_token_count",
                ),
                SimpleNamespace(
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=5,
                        candidates_token_count=12,
                        total_token_count=17,
                    )
                ),
                (5, 12, 17),
            ),
            # Cohere (nested usage.tokens.*)
            (
                TokenFieldMap(
                    input_field="input_tokens",
                    output_field="output_tokens",
                    total_field="",
                    nested_attr="tokens",
                ),
                SimpleNamespace(
                    usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=8, output_tokens=16))
                ),
                (8, 16, 24),
            ),
            # Missing usage → zeros
            (
                TokenFieldMap(),
                SimpleNamespace(),
                (0, 0, 0),
            ),
            # None usage → zeros
            (
                TokenFieldMap(),
                SimpleNamespace(usage=None),
                (0, 0, 0),
            ),
            # Nested attr missing → zeros
            (
                TokenFieldMap(nested_attr="tokens"),
                SimpleNamespace(usage=SimpleNamespace()),
                (0, 0, 0),
            ),
        ],
        ids=[
            "openai_defaults",
            "anthropic_computed_total",
            "gemini_custom_attrs",
            "cohere_nested",
            "missing_usage",
            "none_usage",
            "missing_nested_attr",
        ],
    )
    def test_field_map_extraction(self, fields, response, expected):
        adapter = self._make_adapter()
        assert adapter._extract_tokens_from_fields(response, fields) == expected


class TestUnwrapClient:
    """Test _unwrap_client bounded unwrapping."""

    def _make_adapter(self) -> BaseProviderAdapter:
        class _A(BaseProviderAdapter):
            @property
            def name(self) -> str:
                return "test"

            @property
            def method_label(self) -> str:
                return "test.call"

        return _A()

    def test_normal_unwrap_chain(self):
        """Unwraps a depth-2 _client chain."""
        inner = SimpleNamespace(api_key="sk-inner")
        middle = SimpleNamespace(_client=inner)
        outer = SimpleNamespace(_client=middle)
        result = self._make_adapter()._unwrap_client(outer)
        assert result is inner
        assert result.api_key == "sk-inner"

    def test_single_level_no_client(self):
        """Returns the same object when no _client attribute."""
        obj = SimpleNamespace(api_key="sk-direct")
        result = self._make_adapter()._unwrap_client(obj)
        assert result is obj

    def test_circular_reference_stops(self):
        """Circular reference stops at _MAX_UNWRAP_DEPTH without hanging."""
        from stateloom.intercept.provider_adapter import _MAX_UNWRAP_DEPTH

        a = SimpleNamespace()
        b = SimpleNamespace(_client=a)
        a._client = b  # circular
        result = self._make_adapter()._unwrap_client(a)
        # Should terminate — the result will be one of a or b
        assert result is a or result is b
