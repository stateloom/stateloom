"""Additional helpers for production integration tests.

Re-exports e2e helpers and adds production-specific utilities.
"""

from __future__ import annotations

from typing import Any

from stateloom.core.session import Session
from stateloom.core.types import Provider
from stateloom.gate import Gate
from tests.test_e2e.helpers import (  # noqa: F401
    invoke_pipeline,
    make_ollama_response,
    make_openai_response,
    wait_for_shadow_events,
)


def make_anthropic_response(
    content: str = "Hello from Claude!",
    model: str = "claude-3-sonnet-20240229",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> Any:
    """Build a mock Anthropic Message-like object using SimpleNamespace."""
    import types

    usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    block = types.SimpleNamespace(type="text", text=content)
    return types.SimpleNamespace(
        id="msg-test",
        type="message",
        role="assistant",
        content=[block],
        model=model,
        usage=usage,
        stop_reason="end_turn",
    )


def make_failing_llm_call(
    fail_count: int,
    then_response: Any,
    error: Exception | None = None,
) -> Any:
    """Create a callable that fails N times then returns a response."""
    if error is None:
        error = RuntimeError("LLM provider error")
    call_count = 0

    def _call():
        nonlocal call_count
        call_count += 1
        if call_count <= fail_count:
            raise error
        return then_response

    return _call


def assert_event_exists(
    events: list[dict],
    event_type: str,
    **field_checks: Any,
) -> dict:
    """Assert that an event of the given type exists with matching fields.

    Returns the first matching event.
    """
    matching = [e for e in events if e["event_type"] == event_type]
    assert matching, (
        f"No '{event_type}' event found. Event types: {[e['event_type'] for e in events]}"
    )

    for evt in matching:
        if all(evt.get(k) == v for k, v in field_checks.items()):
            return evt

    if field_checks:
        assert False, (
            f"Found {len(matching)} '{event_type}' event(s) but none matched {field_checks}. "
            f"Events: {matching}"
        )
    return matching[0]


def run_pipeline_calls(
    gate: Gate,
    session: Session,
    request_kwargs: dict[str, Any],
    llm_call: Any,
    count: int,
    model: str = "gpt-3.5-turbo",
    provider: str = Provider.OPENAI,
) -> list[Any]:
    """Run N pipeline calls and return results."""
    results = []
    for _ in range(count):
        result = invoke_pipeline(
            gate,
            session,
            request_kwargs,
            llm_call,
            provider=provider,
            model=model,
        )
        results.append(result)
    return results
