"""Convert any provider response to OpenAI ChatCompletion format."""

from __future__ import annotations

import json
import uuid
from typing import Any

from stateloom.core.response_helpers import _make_completion as _make_completion  # re-export


def to_openai_completion_dict(
    raw_response: Any,
    provider: str,
    model: str,
    request_id: str = "",
) -> dict[str, Any]:
    """Convert any provider response to OpenAI ChatCompletion dict.

    Handles: OpenAI ChatCompletion, Anthropic Message, Gemini response, dicts.
    """
    if not request_id:
        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]

    # None → empty completion (no response from provider)
    if raw_response is None:
        return _make_completion(
            content="",
            model=model,
            request_id=request_id,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    # Dict passthrough (e.g. kill switch response, cached dict)
    if isinstance(raw_response, dict):
        return _normalize_dict_response(raw_response, model, request_id)

    # Use adapter for provider-specific conversion
    from stateloom.intercept.provider_registry import get_adapter, get_all_adapters

    adapter = get_adapter(provider)
    if adapter is not None:
        try:
            return adapter.to_openai_dict(raw_response, model, request_id)
        except Exception:
            pass

    # Fallback: try all registered adapters (first one that extracts content wins)
    for name, fallback in get_all_adapters().items():
        if name == provider:
            continue
        try:
            fb_content = fallback.extract_content(raw_response)
            if fb_content:
                return fallback.to_openai_dict(raw_response, model, request_id)
        except Exception:
            pass

    # Structural fallback when adapter registry is empty (e.g. tests, standalone use)
    if not get_all_adapters():
        result = _structural_fallback(raw_response, model, request_id)
        if result is not None:
            return result

    # Last resort: wrap as text
    content = str(raw_response) if raw_response is not None else ""
    return _make_completion(
        content=content,
        model=model,
        request_id=request_id,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )


def to_openai_sse_event(chunk_dict: dict[str, Any]) -> str:
    """Format a chunk dict as an SSE data event."""
    return f"data: {json.dumps(chunk_dict)}\n\n"


def to_openai_done_event() -> str:
    """Format the SSE [DONE] sentinel."""
    return "data: [DONE]\n\n"


def _structural_fallback(raw_response: Any, model: str, request_id: str) -> dict[str, Any] | None:
    """Try builtin adapter classes based on structural markers when registry is empty.

    This handles standalone usage (tests, scripts) where Gate hasn't been
    initialized but callers still need response conversion.
    """
    try:
        # OpenAI: has .choices and .usage
        if hasattr(raw_response, "choices") and hasattr(raw_response, "usage"):
            from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter

            return OpenAIAdapter().to_openai_dict(raw_response, model, request_id)
        # Anthropic: has .content list and .stop_reason
        if hasattr(raw_response, "content") and hasattr(raw_response, "stop_reason"):
            from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter

            return AnthropicAdapter().to_openai_dict(raw_response, model, request_id)
        # Gemini: has .candidates or .text
        if hasattr(raw_response, "candidates") or hasattr(raw_response, "text"):
            from stateloom.intercept.adapters.gemini_adapter import GeminiAdapter

            return GeminiAdapter().to_openai_dict(raw_response, model, request_id)
    except Exception:
        pass
    return None


def _normalize_dict_response(data: dict[str, Any], model: str, request_id: str) -> dict[str, Any]:
    """Normalize a dict response to OpenAI ChatCompletion format."""
    # Already in OpenAI format
    if "choices" in data and "object" in data:
        data.setdefault("id", request_id)
        return data

    # Try to extract content from common dict shapes
    content = ""
    choices = data.get("choices", [])
    if choices and isinstance(choices, list):
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
    elif "content" in data:
        content = data["content"]
    elif "message" in data:
        content = data["message"]

    usage = data.get("usage", {})
    return _make_completion(
        content=content,
        model=model,
        request_id=request_id,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )
