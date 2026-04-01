"""Mistral provider adapter."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.types import Provider

if TYPE_CHECKING:
    from stateloom.middleware.base import StreamChunkInfo
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget, TokenFieldMap

_TOKEN_FIELDS = TokenFieldMap()


class MistralAdapter(BaseProviderAdapter):
    """Adapter for the Mistral Python SDK (mistralai).

    Patches ``Chat.complete`` and ``Chat.complete_async`` so all calls
    flow through StateLoom's middleware pipeline.
    """

    @property
    def name(self) -> str:
        return Provider.MISTRAL

    @property
    def method_label(self) -> str:
        return "chat.complete"

    def get_patch_targets(self) -> list[PatchTarget]:
        try:
            from mistralai.client.chat import Chat
        except ImportError:
            return []

        return [
            PatchTarget(
                target_class=Chat,
                method_name="complete",
                is_async=False,
                description="mistral.Chat.complete",
            ),
            PatchTarget(
                target_class=Chat,
                method_name="complete_async",
                is_async=True,
                description="mistral.Chat.complete_async",
            ),
        ]

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        return self._extract_tokens_from_fields(response, _TOKEN_FIELDS)

    def extract_content(self, response: Any) -> str:
        try:
            data = response.data if hasattr(response, "data") else response
            if hasattr(data, "choices") and data.choices:
                msg = getattr(data.choices[0], "message", None)
                if msg:
                    return getattr(msg, "content", "") or ""
        except Exception:
            pass
        return ""

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        try:
            data = response.data if hasattr(response, "data") else response
            for choice in data.choices:
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    if choice.message.content:
                        choice.message.content = modifier(choice.message.content)
        except Exception:
            pass

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """Mistral uses OpenAI-like format — delegate to shared helper."""
        return self._openai_style_to_dict(response, model, request_id)

    def extract_stream_tokens(self, chunk: Any, accumulated: dict[str, int]) -> dict[str, int]:
        try:
            data = chunk.data if hasattr(chunk, "data") else chunk
            if hasattr(data, "usage") and data.usage:
                accumulated["prompt_tokens"] = data.usage.prompt_tokens or 0
                accumulated["completion_tokens"] = data.usage.completion_tokens or 0
        except AttributeError:
            pass
        return accumulated

    def extract_chunk_info(self, chunk: Any) -> StreamChunkInfo:
        from stateloom.middleware.base import StreamChunkInfo

        info = StreamChunkInfo()
        try:
            data = chunk.data if hasattr(chunk, "data") else chunk
            if hasattr(data, "choices") and data.choices:
                choice = data.choices[0]
                delta = getattr(choice, "delta", None)
                if delta and hasattr(delta, "content") and delta.content:
                    info.text_delta = delta.content
                info.finish_reason = getattr(choice, "finish_reason", None)
            if hasattr(data, "usage") and data.usage:
                info.prompt_tokens = data.usage.prompt_tokens or 0
                info.completion_tokens = data.usage.completion_tokens or 0
                info.has_usage = True
        except (AttributeError, IndexError):
            pass
        return info

    def get_instance_targets(self, client: Any) -> list[tuple[Any, str]]:
        try:
            return [(client.chat, "complete")]
        except AttributeError:
            return []

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(r"^(mistral-|codestral-|pixtral-)")]

    @property
    def default_base_url(self) -> str:
        return "https://api.mistral.ai/v1"

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
            try:
                from mistralai.client import Mistral
            except ImportError:
                try:
                    from mistralai import Mistral  # type: ignore[attr-defined,no-redef]
                except ImportError:
                    raise ImportError(
                        "mistralai package is required for Mistral models. "
                        "Install with: pip install mistralai"
                    )

            from stateloom.intercept.unpatch import get_original

            ctor_kwargs: dict[str, Any] = {}
            if keys.get("mistral"):
                ctor_kwargs["api_key"] = keys["mistral"]
            client = Mistral(**ctor_kwargs)
            original = get_original(type(client.chat), "complete")
            method = original or client.chat.complete
            if original:
                return method(client.chat, **request_kwargs)  # type: ignore[misc,arg-type]
            return method(**request_kwargs)

        return request_kwargs, llm_call
