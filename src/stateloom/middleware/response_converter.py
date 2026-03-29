"""Convert OllamaResponse to provider-native SDK response objects."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("stateloom.middleware.response_converter")


def convert_response(provider: str, model: str, ollama_response: Any) -> Any | None:
    """Convert an OllamaResponse to a provider-native SDK response object.

    Returns None on failure (triggers cloud fallback).
    """
    try:
        if provider == "openai":
            return _convert_openai(model, ollama_response)
        elif provider == "anthropic":
            return _convert_anthropic(model, ollama_response)
        elif provider == "gemini":
            return _convert_gemini(model, ollama_response)
        else:
            return _convert_openai(model, ollama_response)
    except Exception:
        logger.debug("Response conversion failed for provider=%s", provider, exc_info=True)
        return None


def _convert_openai(model: str, resp: Any) -> Any:
    """Convert to OpenAI ChatCompletion object."""
    from openai.types.chat import ChatCompletion

    content = resp.content if resp.content else ""

    # Build message dict
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    # Handle tool calls from Ollama response (if present)
    tool_calls = getattr(resp, "tool_calls", None)
    if tool_calls:
        message["content"] = content or None
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                },
            }
            for tc in tool_calls
        ]
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    data = {
        "id": f"chatcmpl-local-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": resp.total_tokens,
        },
    }
    return ChatCompletion.model_validate(data)


def _convert_anthropic(model: str, resp: Any) -> Any:
    """Convert to Anthropic Message object."""
    from anthropic.types import Message

    content_blocks: list[dict[str, Any]] = []

    # Handle tool use from Ollama response
    tool_calls = getattr(resp, "tool_calls", None)
    if tool_calls:
        # Add text content if present
        if resp.content:
            content_blocks.append({"type": "text", "text": resp.content})
        # Add tool use blocks
        for tc in tool_calls:
            func = tc.get("function", {})
            # Parse arguments — Anthropic expects a dict, not a JSON string
            args = func.get("arguments", {})
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": func.get("name", ""),
                    "input": args,
                }
            )
        stop_reason = "tool_use"
    else:
        content_blocks.append(
            {
                "type": "text",
                "text": resp.content if resp.content else "",
            }
        )
        stop_reason = "end_turn"

    data = {
        "id": f"msg_local_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": resp.prompt_tokens,
            "output_tokens": resp.completion_tokens,
        },
    }
    return Message.model_validate(data)


class _GeminiResponseProxy:
    """Lightweight proxy mimicking GenerateContentResponse interface."""

    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int) -> None:
        self._content = content
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    @property
    def text(self) -> str:
        return self._content

    @property
    def candidates(self) -> list:
        return [_GeminiCandidate(self._content)]

    @property
    def usage_metadata(self) -> _GeminiUsageMetadata:
        return _GeminiUsageMetadata(self._prompt_tokens, self._completion_tokens)


class _GeminiCandidate:
    def __init__(self, content: str) -> None:
        self.content = _GeminiContent(content)


class _GeminiContent:
    def __init__(self, content: str) -> None:
        self.parts = [_GeminiPart(content)]


class _GeminiPart:
    def __init__(self, text: str) -> None:
        self.text = text


class _GeminiUsageMetadata:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens
        self.total_token_count = prompt_tokens + completion_tokens


def _convert_gemini(model: str, resp: Any) -> Any:
    """Convert to Gemini-compatible proxy object."""
    return _GeminiResponseProxy(
        content=resp.content if resp.content else "",
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
    )
