"""Cohere provider adapter (V2 API)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.types import Provider

if TYPE_CHECKING:
    from stateloom.middleware.base import StreamChunkInfo
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget, TokenFieldMap

_TOKEN_FIELDS = TokenFieldMap(
    input_field="input_tokens",
    output_field="output_tokens",
    total_field="",
    nested_attr="tokens",
)


class CohereAdapter(BaseProviderAdapter):
    """Adapter for the Cohere Python SDK (V2 API).

    Patches ``V2Client.chat`` and ``AsyncV2Client.chat`` so all calls
    flow through StateLoom's middleware pipeline.
    """

    @property
    def name(self) -> str:
        return Provider.COHERE

    @property
    def method_label(self) -> str:
        return "chat"

    def get_patch_targets(self) -> list[PatchTarget]:
        try:
            from cohere.v2.client import AsyncV2Client, V2Client
        except ImportError:
            return []

        return [
            PatchTarget(
                target_class=V2Client,
                method_name="chat",
                is_async=False,
                description="cohere.V2Client.chat",
            ),
            PatchTarget(
                target_class=AsyncV2Client,
                method_name="chat",
                is_async=True,
                description="cohere.AsyncV2Client.chat",
            ),
            PatchTarget(
                target_class=V2Client,
                method_name="chat_stream",
                is_async=False,
                description="cohere.V2Client.chat_stream",
                always_streaming=True,
            ),
            PatchTarget(
                target_class=AsyncV2Client,
                method_name="chat_stream",
                is_async=True,
                description="cohere.AsyncV2Client.chat_stream",
                always_streaming=True,
            ),
        ]

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        return self._extract_tokens_from_fields(response, _TOKEN_FIELDS)

    def extract_stream_tokens(self, chunk: Any, accumulated: dict[str, int]) -> dict[str, int]:
        try:
            if hasattr(chunk, "type") and chunk.type == "message-end":
                usage = chunk.delta.usage
                if usage and hasattr(usage, "tokens") and usage.tokens:
                    accumulated["prompt_tokens"] = int(usage.tokens.input_tokens or 0)
                    accumulated["completion_tokens"] = int(usage.tokens.output_tokens or 0)
        except AttributeError:
            pass
        return accumulated

    def extract_chunk_info(self, chunk: Any) -> StreamChunkInfo:
        from stateloom.middleware.base import StreamChunkInfo

        info = StreamChunkInfo()
        try:
            if hasattr(chunk, "type") and chunk.type == "content-delta":
                delta = chunk.delta
                if hasattr(delta, "message") and delta.message:
                    content = delta.message.content
                    if content and hasattr(content, "text") and content.text:
                        info.text_delta = content.text
            elif hasattr(chunk, "type") and chunk.type == "message-end":
                delta = chunk.delta
                if hasattr(delta, "finish_reason"):
                    info.finish_reason = str(delta.finish_reason)
                if hasattr(delta, "usage") and delta.usage:
                    tokens = getattr(delta.usage, "tokens", None)
                    if tokens:
                        info.prompt_tokens = int(tokens.input_tokens or 0)
                        info.completion_tokens = int(tokens.output_tokens or 0)
                        info.has_usage = True
        except (AttributeError, TypeError):
            pass
        return info

    def modify_chunk_text(self, chunk: Any, new_text: str) -> Any:
        try:
            if hasattr(chunk, "type") and chunk.type == "content-delta":
                if chunk.delta and chunk.delta.message and chunk.delta.message.content:
                    chunk.delta.message.content.text = new_text
        except (AttributeError, TypeError):
            pass
        return chunk

    def extract_content(self, response: Any) -> str:
        try:
            msg = getattr(response, "message", None)
            if msg:
                content_blocks = getattr(msg, "content", None)
                if isinstance(content_blocks, list) and content_blocks:
                    return getattr(content_blocks[0], "text", "") or ""
        except Exception:
            pass
        return ""

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        try:
            msg = getattr(response, "message", None)
            if msg:
                content_blocks = getattr(msg, "content", None)
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if hasattr(block, "text") and block.text:
                            block.text = modifier(block.text)
        except Exception:
            pass

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """Convert Cohere V2 response to OpenAI ChatCompletion dict."""
        from stateloom.core.response_helpers import _make_completion

        content = self.extract_content(response)
        prompt_tokens = 0
        completion_tokens = 0
        try:
            usage = response.usage
            if usage and hasattr(usage, "tokens") and usage.tokens:
                prompt_tokens = int(usage.tokens.input_tokens or 0)
                completion_tokens = int(usage.tokens.output_tokens or 0)
        except Exception:
            pass

        return _make_completion(
            content=content,
            model=model,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        """Cohere V2 uses OpenAI-style messages list."""
        messages = kwargs.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = prompt
        else:
            messages.insert(0, {"role": "system", "content": prompt})
        kwargs["messages"] = messages

    def get_instance_targets(self, client: Any) -> list[tuple[Any, str]]:
        # Check if it's a V2 client or has .v2 accessor
        client_type = type(client).__name__
        if "V2" in client_type:
            return [(client, "chat"), (client, "chat_stream")]
        if hasattr(client, "v2"):
            return [(client.v2, "chat"), (client.v2, "chat_stream")]
        return []

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(r"^(command-|c4ai-)")]

    @property
    def default_base_url(self) -> str:
        return "https://api.cohere.com/v2"

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
                from cohere import ClientV2
            except ImportError:
                raise ImportError(
                    "cohere package is required for Cohere models. Install with: pip install cohere"
                )

            from stateloom.intercept.unpatch import get_original

            ctor_kwargs: dict[str, Any] = {}
            if keys.get("cohere"):
                ctor_kwargs["api_key"] = keys["cohere"]
            client = ClientV2(**ctor_kwargs)
            original = get_original(type(client), "chat")
            method = original or client.chat
            if original:
                return method(client, **request_kwargs)  # type: ignore[arg-type,misc]
            return method(**request_kwargs)

        return request_kwargs, llm_call
