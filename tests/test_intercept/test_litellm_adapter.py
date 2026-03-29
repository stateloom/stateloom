"""Tests for the LiteLLM adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateloom.core.types import Provider
from stateloom.intercept.adapters.litellm_adapter import LiteLLMAdapter


class TestLiteLLMAdapterUnit:
    """Unit tests for LiteLLMAdapter methods (no litellm install needed)."""

    @pytest.fixture
    def adapter(self):
        return LiteLLMAdapter()

    def test_name(self, adapter):
        assert adapter.name == Provider.LITELLM
        assert adapter.name == "litellm"

    def test_method_label(self, adapter):
        assert adapter.method_label == "completion"

    def test_get_patch_targets_empty(self, adapter):
        assert adapter.get_patch_targets() == []

    def test_extract_model_from_kwargs(self, adapter):
        assert adapter.extract_model(None, (), {"model": "gpt-4"}) == "gpt-4"

    def test_extract_model_default(self, adapter):
        assert adapter.extract_model(None, (), {}) == "unknown"

    def test_is_streaming(self, adapter):
        assert adapter.is_streaming({"stream": True}) is True
        assert adapter.is_streaming({"stream": False}) is False
        assert adapter.is_streaming({}) is False

    def test_extract_tokens(self, adapter):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )
        )
        assert adapter.extract_tokens(response) == (100, 50, 150)

    def test_extract_tokens_no_usage(self, adapter):
        assert adapter.extract_tokens(SimpleNamespace()) == (0, 0, 0)

    def test_extract_tokens_none_values(self, adapter):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )
        )
        assert adapter.extract_tokens(response) == (0, 0, 0)

    def test_extract_stream_tokens(self, adapter):
        chunk = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        result = adapter.extract_stream_tokens(chunk, acc)
        assert result == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_extract_stream_tokens_no_usage(self, adapter):
        chunk = SimpleNamespace()
        acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        result = adapter.extract_stream_tokens(chunk, acc)
        assert result == {"prompt_tokens": 0, "completion_tokens": 0}


class TestPatchLiteLLM:
    """Tests for patch_litellm function with mocked litellm module."""

    def test_patch_litellm_no_litellm_installed(self):
        """patch_litellm returns empty list when litellm is not installed."""
        from stateloom.intercept.adapters.litellm_adapter import patch_litellm

        gate = MagicMock()
        # If litellm is not installed, patch_litellm should return []
        with patch.dict("sys.modules", {"litellm": None, "litellm.main": None}):
            result = patch_litellm(gate)
            assert result == []

    def test_patch_litellm_patches_functions(self):
        """patch_litellm patches completion and acompletion on the module."""
        from stateloom.intercept.adapters.litellm_adapter import patch_litellm
        from stateloom.intercept.unpatch import _patch_registry

        # Create a fake litellm module
        fake_litellm = MagicMock()
        fake_litellm.completion = MagicMock(name="original_completion")
        fake_litellm.acompletion = AsyncMock(name="original_acompletion")

        fake_litellm_main = MagicMock()
        fake_litellm_main.completion = fake_litellm.completion
        fake_litellm_main.acompletion = fake_litellm.acompletion

        gate = MagicMock()
        gate.config = MagicMock()
        gate.config.fail_open = True

        initial_count = len(_patch_registry)

        with patch.dict(
            "sys.modules",
            {
                "litellm": fake_litellm,
                "litellm.main": fake_litellm_main,
            },
        ):
            result = patch_litellm(gate)

        assert len(result) == 2
        assert "litellm.completion (sync)" in result
        assert "litellm.acompletion (async)" in result
        # Patches should have been registered for unpatch
        assert len(_patch_registry) > initial_count

    def test_adapter_registered_after_patch(self):
        """patch_litellm registers the adapter in the provider registry."""
        from stateloom.intercept.adapters.litellm_adapter import patch_litellm
        from stateloom.intercept.provider_registry import get_adapter

        fake_litellm = MagicMock()
        fake_litellm.completion = MagicMock()
        fake_litellm.acompletion = AsyncMock()

        fake_litellm_main = MagicMock()
        fake_litellm_main.completion = fake_litellm.completion
        fake_litellm_main.acompletion = fake_litellm.acompletion

        gate = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "litellm": fake_litellm,
                "litellm.main": fake_litellm_main,
            },
        ):
            patch_litellm(gate)

        adapter = get_adapter(Provider.LITELLM)
        assert adapter is not None
        assert adapter.name == "litellm"


class TestProviderEnum:
    def test_litellm_in_provider_enum(self):
        assert Provider.LITELLM == "litellm"
        assert Provider.LITELLM.value == "litellm"
