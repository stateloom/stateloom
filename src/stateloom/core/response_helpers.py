"""Shared response format helpers used by core adapters and proxy."""

from __future__ import annotations

import time
from typing import Any


def _make_completion(
    *,
    content: str,
    model: str,
    request_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a minimal OpenAI ChatCompletion dict."""
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
