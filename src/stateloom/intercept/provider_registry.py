"""Provider adapter registry — central lookup for all registered providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateloom.intercept.provider_adapter import ProviderAdapter

_adapters: dict[str, ProviderAdapter] = {}


def register_adapter(adapter: ProviderAdapter) -> None:
    """Register a provider adapter (overwrites any existing adapter with the same name)."""
    _adapters[adapter.name] = adapter


def get_adapter(name: str) -> ProviderAdapter | None:
    """Look up a registered adapter by name."""
    return _adapters.get(name)


def get_all_adapters() -> dict[str, ProviderAdapter]:
    """Return all registered adapters."""
    return dict(_adapters)


def clear_adapters() -> None:
    """Remove all registered adapters (for test teardown)."""
    _adapters.clear()


def resolve_provider(model: str) -> str | None:
    """Resolve model name to provider via registered adapter model_patterns.

    Returns adapter name on first match, or None.
    """
    for adapter in _adapters.values():
        try:
            patterns = adapter.model_patterns
        except (AttributeError, NotImplementedError):
            continue
        for pattern in patterns:
            if pattern.match(model):
                return adapter.name
    return None


def register_builtin_adapters() -> None:
    """Register the built-in adapters."""
    from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter
    from stateloom.intercept.adapters.cohere_adapter import CohereAdapter
    from stateloom.intercept.adapters.gemini_adapter import GeminiAdapter
    from stateloom.intercept.adapters.mistral_adapter import MistralAdapter
    from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter
    from stateloom.local.adapter import OllamaAdapter

    for adapter in (
        OpenAIAdapter(),
        AnthropicAdapter(),
        GeminiAdapter(),
        MistralAdapter(),
        CohereAdapter(),
        OllamaAdapter(),
    ):
        if adapter.name not in _adapters:
            register_adapter(adapter)
