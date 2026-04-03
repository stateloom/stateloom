"""Shared utilities for E2E integration tests."""

from __future__ import annotations

import time
from typing import Any

from stateloom.core.event import ShadowDraftEvent
from stateloom.core.session import Session
from stateloom.core.types import Provider
from stateloom.gate import Gate
from stateloom.local.client import OllamaResponse

try:
    from openai.types.chat import ChatCompletion as _ChatCompletion
except ImportError:
    _ChatCompletion = None  # type: ignore[assignment,misc]


def make_openai_response(
    content: str = "Hello!",
    model: str = "gpt-3.5-turbo",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> Any:
    """Build a minimal OpenAI ChatCompletion object."""
    if _ChatCompletion is None:
        raise ImportError("openai is required for make_openai_response")
    return _ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


def make_ollama_response(
    content: str = "Hi from local!",
    model: str = "llama3.2",
    prompt_tokens: int = 8,
    completion_tokens: int = 4,
) -> OllamaResponse:
    """Build an OllamaResponse dataclass."""
    return OllamaResponse(
        model=model,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=50.0,
    )


def invoke_pipeline(
    gate: Gate,
    session: Session,
    request_kwargs: dict[str, Any],
    llm_call: Any,
    provider: str = Provider.OPENAI,
    model: str = "gpt-3.5-turbo",
    auto_route_eligible: bool = True,
) -> Any:
    """Call the pipeline the same way the interceptor does.

    Calls ``session.next_step()`` then ``gate.pipeline.execute_sync()``.
    """
    session.next_step()
    return gate.pipeline.execute_sync(
        provider=provider,
        method="chat.completions.create",
        model=model,
        request_kwargs=request_kwargs,
        session=session,
        config=gate.config,
        llm_call=llm_call,
        auto_route_eligible=auto_route_eligible,
    )


def wait_for_shadow_events(
    gate: Gate,
    timeout: float = 2.0,
) -> list[ShadowDraftEvent]:
    """Poll the store until at least one ShadowDraftEvent appears (or timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = gate.store.get_session_events("", event_type="shadow_draft")
        shadow_events = [e for e in events if isinstance(e, ShadowDraftEvent)]
        if shadow_events:
            return shadow_events
        time.sleep(0.05)
    return []
