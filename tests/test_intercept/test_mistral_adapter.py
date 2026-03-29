"""Tests for the Mistral adapter."""

from __future__ import annotations

from types import SimpleNamespace

from stateloom.core.types import Provider
from stateloom.intercept.adapters.mistral_adapter import MistralAdapter


class TestMistralAdapterUnit:
    """Unit tests for MistralAdapter methods (no mistralai install needed)."""

    def _adapter(self):
        return MistralAdapter()

    def test_name(self):
        assert self._adapter().name == Provider.MISTRAL
        assert self._adapter().name == "mistral"

    def test_method_label(self):
        assert self._adapter().method_label == "chat.complete"

    def test_get_patch_targets(self):
        targets = self._adapter().get_patch_targets()
        # Returns [] if mistralai not installed, or 2 targets if it is
        assert len(targets) in (0, 2)

    def test_extract_model_from_kwargs(self):
        adapter = self._adapter()
        assert adapter.extract_model(None, (), {"model": "mistral-large"}) == "mistral-large"

    def test_extract_model_default(self):
        assert self._adapter().extract_model(None, (), {}) == "unknown"

    def test_is_streaming(self):
        adapter = self._adapter()
        assert adapter.is_streaming({"stream": True}) is True
        assert adapter.is_streaming({}) is False

    def test_extract_tokens(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )
        )
        assert self._adapter().extract_tokens(response) == (100, 50, 150)

    def test_extract_tokens_no_usage(self):
        assert self._adapter().extract_tokens(SimpleNamespace()) == (0, 0, 0)

    def test_extract_tokens_none_values(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )
        )
        assert self._adapter().extract_tokens(response) == (0, 0, 0)

    def test_extract_stream_tokens_with_data_wrapper(self):
        """Mistral stream chunks wrap usage in .data."""
        chunk = SimpleNamespace(
            data=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10))
        )
        acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        result = self._adapter().extract_stream_tokens(chunk, acc)
        assert result == {"prompt_tokens": 20, "completion_tokens": 10}

    def test_extract_stream_tokens_no_usage(self):
        chunk = SimpleNamespace(data=SimpleNamespace())
        acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        result = self._adapter().extract_stream_tokens(chunk, acc)
        assert result == {"prompt_tokens": 0, "completion_tokens": 0}

    def test_get_instance_targets_with_chat(self):
        client = SimpleNamespace(chat=SimpleNamespace())
        targets = self._adapter().get_instance_targets(client)
        assert len(targets) == 1
        assert targets[0][1] == "complete"

    def test_get_instance_targets_no_chat(self):
        client = SimpleNamespace()
        assert self._adapter().get_instance_targets(client) == []


class TestProviderEnum:
    def test_mistral_in_provider_enum(self):
        assert Provider.MISTRAL == "mistral"
        assert Provider.MISTRAL.value == "mistral"
