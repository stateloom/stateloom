"""Anthropic provider adapter."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from stateloom.core.types import Provider
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget, TokenFieldMap

_TOKEN_FIELDS = TokenFieldMap(
    input_field="input_tokens",
    output_field="output_tokens",
    total_field="",
)


class AnthropicAdapter(BaseProviderAdapter):
    """Adapter for the Anthropic Python SDK."""

    @property
    def name(self) -> str:
        return Provider.ANTHROPIC

    @property
    def method_label(self) -> str:
        return "messages.create"

    def get_patch_targets(self) -> list[PatchTarget]:
        try:
            import anthropic.resources.messages
        except ImportError:
            return []

        return [
            PatchTarget(
                target_class=anthropic.resources.messages.Messages,
                method_name="create",
                is_async=False,
                description="anthropic.Messages.create",
            ),
            PatchTarget(
                target_class=anthropic.resources.messages.AsyncMessages,
                method_name="create",
                is_async=True,
                description="anthropic.AsyncMessages.create",
            ),
        ]

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        return self._extract_tokens_from_fields(response, _TOKEN_FIELDS)

    def extract_content(self, response: Any) -> str:
        try:
            if hasattr(response, "content") and isinstance(response.content, list):
                # First pass: find text block by type
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        return getattr(block, "text", "") or ""
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
                # Fallback: grab .text from first block that has it
                for block in response.content:
                    text = getattr(block, "text", None)
                    if isinstance(text, str) and text:
                        return text
        except Exception:
            pass
        return ""

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        try:
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    block.text = modifier(block.text)
        except Exception:
            pass

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """Convert Anthropic Message to OpenAI ChatCompletion dict."""
        import json
        import time
        import uuid

        content = ""
        tool_calls: list[dict[str, Any]] = []

        try:
            if hasattr(response, "content") and isinstance(response.content, list):
                for block in response.content:
                    block_type = getattr(block, "type", "")
                    if block_type == "text":
                        content = getattr(block, "text", "")
                    elif block_type == "tool_use":
                        tool_calls.append(
                            {
                                "id": getattr(block, "id", f"call_{uuid.uuid4().hex[:8]}"),
                                "type": "function",
                                "function": {
                                    "name": getattr(block, "name", ""),
                                    "arguments": json.dumps(getattr(block, "input", {})),
                                },
                            }
                        )
        except Exception:
            pass

        stop_reason = getattr(response, "stop_reason", "end_turn")
        if stop_reason == "end_turn":
            finish_reason = "stop"
        elif stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif stop_reason == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        message_dict: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message_dict["tool_calls"] = tool_calls

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": getattr(response, "model", model),
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def extract_stream_tokens(self, chunk: Any, accumulated: dict[str, int]) -> dict[str, int]:
        try:
            if hasattr(chunk, "type"):
                if chunk.type == "message_start" and hasattr(chunk, "message"):
                    usage = getattr(chunk.message, "usage", None)
                    if usage:
                        accumulated["prompt_tokens"] = usage.input_tokens or 0
                elif chunk.type == "message_delta":
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        accumulated["completion_tokens"] = usage.output_tokens or 0
        except AttributeError:
            pass
        return accumulated

    def extract_chunk_info(self, chunk: Any) -> StreamChunkInfo:
        from stateloom.middleware.base import StreamChunkInfo

        info = StreamChunkInfo()
        try:
            chunk_type = getattr(chunk, "type", "")
            if chunk_type == "content_block_delta":
                delta = getattr(chunk, "delta", None)
                if delta:
                    delta_type = getattr(delta, "type", "")
                    if delta_type == "text_delta":
                        info.text_delta = getattr(delta, "text", "")
                    elif delta_type == "input_json_delta":
                        info.tool_call_delta = {
                            "partial_json": getattr(delta, "partial_json", ""),
                        }
            elif chunk_type == "message_start":
                message = getattr(chunk, "message", None)
                if message:
                    usage = getattr(message, "usage", None)
                    if usage:
                        info.prompt_tokens = getattr(usage, "input_tokens", 0) or 0
                        info.has_usage = True
            elif chunk_type == "message_delta":
                usage = getattr(chunk, "usage", None)
                if usage:
                    info.completion_tokens = getattr(usage, "output_tokens", 0) or 0
                    info.has_usage = True
                info.finish_reason = getattr(getattr(chunk, "delta", None), "stop_reason", None)
        except (AttributeError, IndexError):
            pass
        return info

    def modify_chunk_text(self, chunk: Any, new_text: str) -> Any:
        """Replace the text delta in an Anthropic content_block_delta chunk."""
        try:
            if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                chunk.delta.text = new_text
        except (AttributeError, TypeError):
            pass
        return chunk

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        kwargs["system"] = prompt

    def extract_base_url(self, instance: Any) -> str:
        try:
            client = self._unwrap_client(instance)
            base_url = getattr(client, "base_url", None)
            if base_url is not None:
                return str(base_url).rstrip("/")
        except Exception:
            pass
        return "https://api.anthropic.com"

    def get_instance_targets(self, client: Any) -> list[tuple[Any, str]]:
        try:
            return [(client.messages, "create")]
        except AttributeError:
            return []

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(r"^claude-")]

    @property
    def default_base_url(self) -> str:
        return "https://api.anthropic.com"

    def prepare_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        provider_keys: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Callable[[], Any]]:
        system_parts: list[str] = []
        non_system: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
            else:
                non_system.append(msg)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": non_system,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system_parts:
            request_kwargs["system"] = "\n\n".join(system_parts)

        keys = provider_keys or {}

        def llm_call() -> Any:
            import anthropic

            from stateloom.intercept.unpatch import get_original

            ctor_kwargs: dict[str, Any] = {
                # Always use the canonical base URL so that proxy mode
                # doesn't loop back via ANTHROPIC_BASE_URL env var.
                "base_url": "https://api.anthropic.com",
            }
            if keys.get("anthropic"):
                ctor_kwargs["api_key"] = keys["anthropic"]
            client = anthropic.Anthropic(**ctor_kwargs)
            original = get_original(type(client.messages), "create")
            method = original or client.messages.create
            if original:
                return method(client.messages, **request_kwargs)
            return method(**request_kwargs)

        return request_kwargs, llm_call
