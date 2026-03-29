"""OpenAI provider adapter."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from stateloom.core.types import Provider
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget, TokenFieldMap

logger = logging.getLogger("stateloom.intercept.adapters.openai")

_TOKEN_FIELDS = TokenFieldMap()


class OpenAIAdapter(BaseProviderAdapter):
    """Adapter for the OpenAI Python SDK."""

    @property
    def name(self) -> str:
        return Provider.OPENAI

    @property
    def method_label(self) -> str:
        return "chat.completions.create"

    def get_patch_targets(self) -> list[PatchTarget]:
        try:
            import openai.resources.chat.completions
        except ImportError:
            return []

        return [
            PatchTarget(
                target_class=openai.resources.chat.completions.Completions,
                method_name="create",
                is_async=False,
                description="openai.Completions.create",
            ),
            PatchTarget(
                target_class=openai.resources.chat.completions.AsyncCompletions,
                method_name="create",
                is_async=True,
                description="openai.AsyncCompletions.create",
            ),
        ]

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        result = self._extract_tokens_from_fields(response, _TOKEN_FIELDS)
        if result == (0, 0, 0):
            usage = getattr(response, "usage", None)
            if usage is None:
                logger.debug("OpenAI token extraction failed: no usage attribute")
        return result

    def extract_content(self, response: Any) -> str:
        try:
            if response.choices:
                msg = getattr(response.choices[0], "message", None)
                if msg:
                    return getattr(msg, "content", "") or ""
        except Exception:
            logger.debug("OpenAI content extraction failed", exc_info=True)
        return ""

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        try:
            for choice in response.choices:
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    if choice.message.content:
                        choice.message.content = modifier(choice.message.content)
        except Exception:
            logger.debug("OpenAI response text modification failed", exc_info=True)

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """Convert OpenAI ChatCompletion — mostly passthrough via model_dump."""
        if hasattr(response, "model_dump"):
            result = response.model_dump()
            result["id"] = request_id
            return result

        # Manual extraction fallback for responses without model_dump
        return self._openai_style_to_dict(response, model, request_id)

    def extract_stream_tokens(self, chunk: Any, accumulated: dict[str, int]) -> dict[str, int]:
        try:
            if hasattr(chunk, "usage") and chunk.usage:
                accumulated["prompt_tokens"] = chunk.usage.prompt_tokens or 0
                accumulated["completion_tokens"] = chunk.usage.completion_tokens or 0
        except AttributeError:
            pass
        return accumulated

    def extract_chunk_info(self, chunk: Any) -> StreamChunkInfo:
        from stateloom.middleware.base import StreamChunkInfo

        info = StreamChunkInfo()
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta and hasattr(delta, "content") and delta.content:
                    info.text_delta = delta.content
                if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
                    tc = delta.tool_calls[0]
                    info.tool_call_delta = {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(tc.function, "name", None),
                            "arguments": getattr(tc.function, "arguments", ""),
                        }
                        if hasattr(tc, "function") and tc.function
                        else None,
                    }
                info.finish_reason = getattr(choice, "finish_reason", None)
            if hasattr(chunk, "usage") and chunk.usage:
                info.prompt_tokens = chunk.usage.prompt_tokens or 0
                info.completion_tokens = chunk.usage.completion_tokens or 0
                info.has_usage = True
        except (AttributeError, IndexError):
            pass
        return info

    def modify_chunk_text(self, chunk: Any, new_text: str) -> Any:
        """Replace the text delta content in an OpenAI stream chunk."""
        try:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                chunk.choices[0].delta.content = new_text
        except (AttributeError, IndexError):
            pass
        return chunk

    def extract_base_url(self, instance: Any) -> str:
        try:
            client = self._unwrap_client(instance)
            base_url = getattr(client, "base_url", None)
            if base_url is not None:
                return str(base_url).rstrip("/")
        except Exception:
            pass
        return "https://api.openai.com/v1"

    def get_instance_targets(self, client: Any) -> list[tuple[Any, str]]:
        try:
            return [(client.chat.completions, "create")]
        except AttributeError:
            return []

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(r"^(gpt-|o1|o3|o4|chatgpt-)")]

    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def prepare_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        provider_keys: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Callable[[], Any]]:
        request_kwargs = {"model": model, "messages": messages, **kwargs}
        keys = provider_keys or {}

        def llm_call() -> Any:
            import openai

            from stateloom.intercept.unpatch import get_original

            ctor_kwargs: dict[str, Any] = {
                # Always use the canonical base URL so that proxy mode
                # doesn't loop back via OPENAI_BASE_URL env var.
                "base_url": "https://api.openai.com/v1",
            }
            if keys.get("openai"):
                ctor_kwargs["api_key"] = keys["openai"]
            client = openai.OpenAI(**ctor_kwargs)
            original = get_original(type(client.chat.completions), "create")
            method = original or client.chat.completions.create
            if original:
                return method(client.chat.completions, **request_kwargs)
            return method(**request_kwargs)

        return request_kwargs, llm_call
