"""Tests for the provider adapter registry."""

from __future__ import annotations

import re

import pytest

from stateloom.intercept.provider_adapter import BaseProviderAdapter
from stateloom.intercept.provider_registry import (
    clear_adapters,
    get_adapter,
    get_all_adapters,
    register_adapter,
    register_builtin_adapters,
    resolve_provider,
)


class _FakeAdapter(BaseProviderAdapter):
    def __init__(self, adapter_name: str = "fake"):
        self._name = adapter_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def method_label(self) -> str:
        return "fake.call"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure registry is clean before and after each test."""
    clear_adapters()
    yield
    clear_adapters()


class TestRegistryLifecycle:
    def test_register_and_get(self):
        adapter = _FakeAdapter("test-provider")
        register_adapter(adapter)
        assert get_adapter("test-provider") is adapter

    def test_get_unknown_returns_none(self):
        assert get_adapter("nonexistent") is None

    def test_get_all_adapters(self):
        register_adapter(_FakeAdapter("a"))
        register_adapter(_FakeAdapter("b"))
        all_adapters = get_all_adapters()
        assert set(all_adapters.keys()) == {"a", "b"}

    def test_clear_adapters(self):
        register_adapter(_FakeAdapter("x"))
        clear_adapters()
        assert get_adapter("x") is None
        assert get_all_adapters() == {}

    def test_duplicate_registration_overwrites(self):
        adapter1 = _FakeAdapter("dup")
        adapter2 = _FakeAdapter("dup")
        register_adapter(adapter1)
        register_adapter(adapter2)
        assert get_adapter("dup") is adapter2


class TestBuiltinAdapters:
    def test_register_builtin_adapters(self):
        register_builtin_adapters()
        all_adapters = get_all_adapters()
        assert "openai" in all_adapters
        assert "anthropic" in all_adapters
        assert "gemini" in all_adapters

    def test_builtin_does_not_overwrite_custom(self):
        """If a custom adapter is registered with a built-in name, it is preserved."""
        custom = _FakeAdapter("openai")
        register_adapter(custom)
        register_builtin_adapters()
        assert get_adapter("openai") is custom


class TestResolveProvider:
    def test_openai_models(self):
        register_builtin_adapters()
        assert resolve_provider("gpt-4") == "openai"
        assert resolve_provider("gpt-4o-mini") == "openai"
        assert resolve_provider("o1-preview") == "openai"
        assert resolve_provider("chatgpt-4o-latest") == "openai"

    def test_anthropic_models(self):
        register_builtin_adapters()
        assert resolve_provider("claude-3-opus") == "anthropic"
        assert resolve_provider("claude-opus-4") == "anthropic"

    def test_gemini_models(self):
        register_builtin_adapters()
        assert resolve_provider("gemini-1.5-pro") == "gemini"
        assert resolve_provider("gemini-2.0-flash") == "gemini"

    def test_mistral_models(self):
        register_builtin_adapters()
        assert resolve_provider("mistral-large-latest") == "mistral"
        assert resolve_provider("codestral-latest") == "mistral"
        assert resolve_provider("pixtral-large-latest") == "mistral"

    def test_cohere_models(self):
        register_builtin_adapters()
        assert resolve_provider("command-r-plus") == "cohere"
        assert resolve_provider("c4ai-aya-expanse") == "cohere"

    def test_unknown_returns_none(self):
        register_builtin_adapters()
        assert resolve_provider("some-random-model") is None

    def test_empty_registry_returns_none(self):
        assert resolve_provider("gpt-4") is None

    def test_adapter_without_model_patterns_skipped(self):
        """Custom adapters without model_patterns are gracefully skipped."""

        class _NoPatterns(BaseProviderAdapter):
            @property
            def name(self):
                return "nopat"

            @property
            def method_label(self):
                return "nopat.call"

            @property
            def model_patterns(self):
                raise AttributeError

        register_adapter(_NoPatterns())
        assert resolve_provider("nopat-model") is None
