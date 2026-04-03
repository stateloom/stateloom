"""Tests for the google-genai adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.types import Provider
from stateloom.intercept.adapters.genai_adapter import GenaiAdapter


class TestGenaiAdapterUnit:
    """Unit tests for GenaiAdapter methods (no google-genai install needed)."""

    @pytest.fixture
    def adapter(self) -> GenaiAdapter:
        return GenaiAdapter()

    def test_name(self, adapter: GenaiAdapter) -> None:
        assert adapter.name == Provider.GEMINI
        assert adapter.name == "gemini"

    def test_method_label(self, adapter: GenaiAdapter) -> None:
        assert adapter.method_label == "generate_content"

    def test_get_patch_targets_empty_without_sdk(self, adapter: GenaiAdapter) -> None:
        """Returns empty list when google-genai is not installed."""
        with patch.dict("sys.modules", {"google.genai": None, "google.genai.models": None}):
            assert adapter.get_patch_targets() == []

    def test_extract_model_from_kwargs(self, adapter: GenaiAdapter) -> None:
        assert adapter.extract_model(None, (), {"model": "gemini-2.0-flash"}) == "gemini-2.0-flash"

    def test_extract_model_default(self, adapter: GenaiAdapter) -> None:
        assert adapter.extract_model(None, (), {}) == "unknown"

    def test_extract_model_non_string(self, adapter: GenaiAdapter) -> None:
        """Non-string model values are coerced to str."""
        assert adapter.extract_model(None, (), {"model": 42}) == "42"

    def test_is_streaming_always_false(self, adapter: GenaiAdapter) -> None:
        """Streaming is determined by method, not kwargs."""
        assert adapter.is_streaming({"stream": True}) is False
        assert adapter.is_streaming({}) is False

    def test_extract_tokens(self, adapter: GenaiAdapter) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
            )
        )
        assert adapter.extract_tokens(response) == (100, 50, 150)

    def test_extract_tokens_no_usage(self, adapter: GenaiAdapter) -> None:
        assert adapter.extract_tokens(SimpleNamespace()) == (0, 0, 0)

    def test_extract_tokens_none_values(self, adapter: GenaiAdapter) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=None,
                candidates_token_count=None,
                total_token_count=None,
            )
        )
        assert adapter.extract_tokens(response) == (0, 0, 0)

    def test_extract_content_via_text(self, adapter: GenaiAdapter) -> None:
        response = SimpleNamespace(text="Hello world")
        assert adapter.extract_content(response) == "Hello world"

    def test_extract_content_via_candidates(self, adapter: GenaiAdapter) -> None:
        response = SimpleNamespace(
            text=None,
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="fallback content")])
                )
            ],
        )
        assert adapter.extract_content(response) == "fallback content"

    def test_extract_content_empty(self, adapter: GenaiAdapter) -> None:
        response = SimpleNamespace(text="", candidates=[])
        assert adapter.extract_content(response) == ""

    def test_extract_stream_tokens(self, adapter: GenaiAdapter) -> None:
        chunk = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
        )
        acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        result = adapter.extract_stream_tokens(chunk, acc)
        assert result == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_extract_stream_tokens_no_usage(self, adapter: GenaiAdapter) -> None:
        chunk = SimpleNamespace()
        acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        result = adapter.extract_stream_tokens(chunk, acc)
        assert result == {"prompt_tokens": 0, "completion_tokens": 0}

    def test_extract_chunk_info(self, adapter: GenaiAdapter) -> None:
        chunk = SimpleNamespace(
            text="hello",
            usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )
        info = adapter.extract_chunk_info(chunk)
        assert info.text_delta == "hello"
        assert info.prompt_tokens == 5
        assert info.completion_tokens == 3
        assert info.has_usage is True
        assert info.finish_reason == "STOP"

    def test_extract_chunk_info_minimal(self, adapter: GenaiAdapter) -> None:
        chunk = SimpleNamespace()
        info = adapter.extract_chunk_info(chunk)
        assert not info.text_delta
        assert info.has_usage is False

    def test_modify_chunk_text(self, adapter: GenaiAdapter) -> None:
        chunk = SimpleNamespace(
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="original")]))
            ]
        )
        result = adapter.modify_chunk_text(chunk, "replaced")
        assert result.candidates[0].content.parts[0].text == "replaced"

    def test_apply_system_prompt(self, adapter: GenaiAdapter) -> None:
        kwargs: dict[str, str] = {}
        adapter.apply_system_prompt(kwargs, "Be helpful")
        assert kwargs["system_instruction"] == "Be helpful"

    def test_extract_base_url(self, adapter: GenaiAdapter) -> None:
        assert adapter.extract_base_url(None) == "https://generativelanguage.googleapis.com"

    def test_model_patterns(self, adapter: GenaiAdapter) -> None:
        patterns = adapter.model_patterns
        assert len(patterns) == 1
        assert patterns[0].match("gemini-2.0-flash")
        assert not patterns[0].match("gpt-4o")

    def test_default_base_url(self, adapter: GenaiAdapter) -> None:
        assert adapter.default_base_url == "https://generativelanguage.googleapis.com"


class TestGenaiNormalizeRequest:
    """Tests for normalize_request — contents to messages conversion."""

    @pytest.fixture
    def adapter(self) -> GenaiAdapter:
        return GenaiAdapter()

    def test_string_contents(self, adapter: GenaiAdapter) -> None:
        result = adapter.normalize_request((), {"contents": "Hello"})
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert "contents" not in result

    def test_list_of_strings(self, adapter: GenaiAdapter) -> None:
        result = adapter.normalize_request((), {"contents": ["Hello", "World"]})
        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "user", "content": "Hello"}
        assert result["messages"][1] == {"role": "user", "content": "World"}

    def test_content_dicts(self, adapter: GenaiAdapter) -> None:
        contents = [
            {"role": "user", "parts": [{"text": "Hi"}]},
            {"role": "model", "parts": [{"text": "Hello!"}]},
        ]
        result = adapter.normalize_request((), {"contents": contents})
        assert result["messages"] == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

    def test_content_objects(self, adapter: GenaiAdapter) -> None:
        contents = [
            SimpleNamespace(role="user", parts=[SimpleNamespace(text="object msg")]),
        ]
        result = adapter.normalize_request((), {"contents": contents})
        assert result["messages"] == [{"role": "user", "content": "object msg"}]

    def test_no_contents(self, adapter: GenaiAdapter) -> None:
        result = adapter.normalize_request((), {"model": "gemini-2.0-flash"})
        assert "messages" not in result
        assert result["model"] == "gemini-2.0-flash"

    def test_preserves_other_kwargs(self, adapter: GenaiAdapter) -> None:
        result = adapter.normalize_request(
            (), {"contents": "Hi", "model": "gemini-2.0-flash", "config": {"temperature": 0.5}}
        )
        assert result["model"] == "gemini-2.0-flash"
        assert result["config"] == {"temperature": 0.5}
        assert "messages" in result


class TestGenaiRebuildCallArgs:
    """Tests for rebuild_call_args — messages back to contents."""

    @pytest.fixture
    def adapter(self) -> GenaiAdapter:
        return GenaiAdapter()

    def test_rebuild_from_messages(self, adapter: GenaiAdapter) -> None:
        normalized = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        original_kwargs = {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
        args, kwargs = adapter.rebuild_call_args(normalized, (), original_kwargs)
        assert args == ()
        assert len(kwargs["contents"]) == 2
        assert kwargs["contents"][0]["role"] == "user"
        assert kwargs["contents"][0]["parts"] == [{"text": "Hello"}]
        assert kwargs["contents"][1]["role"] == "model"
        assert kwargs["contents"][1]["parts"] == [{"text": "Hi there"}]

    def test_rebuild_skips_system(self, adapter: GenaiAdapter) -> None:
        normalized = {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ]
        }
        _, kwargs = adapter.rebuild_call_args(normalized, (), {"contents": []})
        assert len(kwargs["contents"]) == 1
        assert kwargs["contents"][0]["role"] == "user"

    def test_rebuild_no_messages(self, adapter: GenaiAdapter) -> None:
        """When no messages in normalized_kwargs, return originals unchanged."""
        original_args = ("original",)
        original_kwargs = {"contents": "original"}
        args, kwargs = adapter.rebuild_call_args({}, original_args, original_kwargs)
        assert args == original_args
        assert kwargs == original_kwargs


class TestPatchGenai:
    """Tests for the patch_genai function."""

    def test_patch_genai_returns_empty_without_sdk(self) -> None:
        """patch_genai returns empty list when google-genai is not installed."""
        from stateloom.intercept.adapters.genai_adapter import patch_genai

        gate = MagicMock()
        with patch.dict("sys.modules", {"google.genai": None, "google.genai.models": None}):
            result = patch_genai(gate)
            assert result == []

    def test_patch_genai_patches_methods(self) -> None:
        """patch_genai patches all four methods on Models/AsyncModels."""
        from stateloom.intercept.adapters.genai_adapter import patch_genai
        from stateloom.intercept.unpatch import _patch_registry

        # Create fake Models and AsyncModels classes
        class FakeModels:
            def generate_content(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            def generate_content_stream(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

        class FakeAsyncModels:
            async def generate_content(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def generate_content_stream(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

        fake_genai_models = MagicMock()
        fake_genai_models.Models = FakeModels
        fake_genai_models.AsyncModels = FakeAsyncModels

        fake_genai = MagicMock()

        gate = MagicMock()
        gate.config = MagicMock()
        gate.config.fail_open = True

        initial_count = len(_patch_registry)

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": fake_genai,
                "google.genai.models": fake_genai_models,
            },
        ):
            result = patch_genai(gate)

        assert len(result) == 4
        assert any("Models.generate_content" in r for r in result)
        assert any("Models.generate_content_stream" in r for r in result)
        assert any("AsyncModels.generate_content" in r for r in result)
        assert any("AsyncModels.generate_content_stream" in r for r in result)
        # Check stream descriptions
        assert any("sync+stream" in r for r in result)
        assert any("async+stream" in r for r in result)
        # Patches registered for unpatch
        assert len(_patch_registry) >= initial_count + 4


class TestPatchTargetAlwaysStreaming:
    """Tests for the always_streaming flag on PatchTarget."""

    def test_always_streaming_default_false(self) -> None:
        from stateloom.intercept.provider_adapter import PatchTarget

        target = PatchTarget(target_class=object, method_name="foo")
        assert target.always_streaming is False

    def test_always_streaming_set_true(self) -> None:
        from stateloom.intercept.provider_adapter import PatchTarget

        target = PatchTarget(target_class=object, method_name="foo", always_streaming=True)
        assert target.always_streaming is True

    def test_genai_stream_targets_have_always_streaming(self) -> None:
        """Stream targets on GenaiAdapter set always_streaming=True."""

        # Create fake SDK classes so get_patch_targets() succeeds
        class FakeModels:
            def generate_content(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            def generate_content_stream(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

        class FakeAsyncModels:
            async def generate_content(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def generate_content_stream(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

        fake_genai_models = MagicMock()
        fake_genai_models.Models = FakeModels
        fake_genai_models.AsyncModels = FakeAsyncModels

        adapter = GenaiAdapter()
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": MagicMock(),
                "google.genai.models": fake_genai_models,
            },
        ):
            targets = adapter.get_patch_targets()

        assert len(targets) == 4

        stream_targets = [t for t in targets if "stream" in t.method_name.lower()]
        non_stream_targets = [t for t in targets if "stream" not in t.method_name.lower()]

        assert len(stream_targets) == 2
        assert len(non_stream_targets) == 2

        for t in stream_targets:
            assert t.always_streaming is True, f"{t.description} should be always_streaming"

        for t in non_stream_targets:
            assert t.always_streaming is False, f"{t.description} should not be always_streaming"


class TestToOpenAIDict:
    """Tests for to_openai_dict conversion."""

    @pytest.fixture
    def adapter(self) -> GenaiAdapter:
        return GenaiAdapter()

    def test_basic_conversion(self, adapter: GenaiAdapter) -> None:
        response = SimpleNamespace(
            text="Hello!",
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
            ),
        )
        result = adapter.to_openai_dict(response, "gemini-2.0-flash", "req-123")
        assert result["model"] == "gemini-2.0-flash"
        assert result["id"] == "req-123"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_no_usage_metadata(self, adapter: GenaiAdapter) -> None:
        response = SimpleNamespace(text="Hi")
        result = adapter.to_openai_dict(response, "gemini-2.0-flash", "req-456")
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0
