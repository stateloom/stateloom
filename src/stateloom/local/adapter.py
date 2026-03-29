"""Ollama provider adapter for StateLoom."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stateloom.core.types import Provider
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget
from stateloom.local.client import OllamaResponse


class OllamaAdapter(BaseProviderAdapter):
    """Provider adapter for Ollama local models.

    Unlike cloud providers, Ollama has no SDK to monkey-patch —
    calls go through OllamaClient directly. This adapter is registered
    so the middleware pipeline can identify LOCAL provider calls.
    """

    @property
    def name(self) -> str:
        return Provider.LOCAL

    @property
    def method_label(self) -> str:
        return "chat"

    def get_patch_targets(self) -> list[PatchTarget]:
        # No SDK to patch — Ollama calls are made via OllamaClient
        return []

    def extract_model(self, instance: Any, args: tuple, kwargs: dict[str, Any]) -> str:
        return kwargs.get("model", "unknown")

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        if isinstance(response, OllamaResponse):
            return (
                response.prompt_tokens,
                response.completion_tokens,
                response.total_tokens,
            )
        return (0, 0, 0)

    def extract_content(self, response: Any) -> str:
        if isinstance(response, OllamaResponse):
            return response.content
        try:
            content = getattr(response, "content", None)
            if isinstance(content, str):
                return content
        except Exception:
            pass
        return ""

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        try:
            if isinstance(response, OllamaResponse) and response.content:
                response.content = modifier(response.content)
            elif hasattr(response, "content") and isinstance(response.content, str):
                response.content = modifier(response.content)
        except Exception:
            pass

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        from stateloom.core.response_helpers import _make_completion

        content = self.extract_content(response)
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(response, OllamaResponse):
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens

        return _make_completion(
            content=content,
            model=model,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def is_streaming(self, kwargs: dict[str, Any]) -> bool:
        return False

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        """OpenAI-style messages list."""
        messages = kwargs.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = prompt
        else:
            messages.insert(0, {"role": "system", "content": prompt})
        kwargs["messages"] = messages
