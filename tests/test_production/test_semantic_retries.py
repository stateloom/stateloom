"""Production tests: Semantic Retries (Self-Healing).

retry_loop and durable_task with automatic retries on LLM output failures.
"""

from __future__ import annotations

import json

import pytest

from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomRetryError,
)
from stateloom.retry import RetryLoop
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_retry_loop_succeeds_after_failures(e2e_gate, api_client):
    """Fail twice, succeed third → returns result."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Success")

    attempt_count = 0

    with gate.session(session_id="prod-retry-ok-1") as session:
        for attempt in RetryLoop(retries=3):
            with attempt:
                attempt_count += 1
                if attempt_count < 3:
                    raise ValueError("Bad JSON output")
                result = invoke_pipeline(
                    gate,
                    session,
                    {"messages": [{"role": "user", "content": "Generate JSON"}]},
                    llm_call=lambda: response,
                )

    assert attempt_count == 3
    assert result is response


def test_retry_loop_exhausted(e2e_gate, api_client):
    """All attempts fail → last exception propagates.

    RetryLoop suppresses errors for intermediate attempts but lets the
    final attempt's exception propagate directly (it does NOT wrap in
    StateLoomRetryError — that wrapping is only done by durable_task).
    """
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    with gate.session(session_id="prod-retry-fail-1") as session:
        with pytest.raises(ValueError, match="Always fails"):
            for attempt in RetryLoop(retries=3):
                with attempt:
                    raise ValueError("Always fails")


def test_retry_loop_non_retryable_propagates(e2e_gate, api_client):
    """Budget error inside retry → propagates immediately."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    with gate.session(session_id="prod-retry-nonret-1") as session:
        with pytest.raises(StateLoomBudgetError):
            for attempt in RetryLoop(retries=5):
                with attempt:
                    raise StateLoomBudgetError(limit=0.0001, spent=0.01, session_id="test")


def test_retry_events_recorded(e2e_gate, api_client):
    """Failed attempts → SemanticRetryEvent per attempt."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Success")

    attempt_count = 0

    with gate.session(session_id="prod-retry-events-1") as session:
        for attempt in RetryLoop(retries=3):
            with attempt:
                attempt_count += 1
                if attempt_count < 2:
                    raise ValueError("Retry me")
                invoke_pipeline(
                    gate,
                    session,
                    {"messages": [{"role": "user", "content": "Q"}]},
                    llm_call=lambda: response,
                )

    events = client.get("/sessions/prod-retry-events-1/events").json()
    retry_events = [e for e in events["events"] if e["event_type"] == "semantic_retry"]
    assert len(retry_events) >= 1


def test_durable_task_decorator(e2e_gate, api_client):
    """@durable_task(retries=3) → function retries on failure."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    from stateloom.retry import durable_task

    call_count = 0

    @durable_task(retries=3)
    def generate_report():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Bad output")
        return {"status": "ok"}

    result = generate_report()
    assert result == {"status": "ok"}
    assert call_count == 2


def test_durable_task_validate_callback(e2e_gate, api_client):
    """Validation fails → triggers retry."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    from stateloom.retry import durable_task

    call_count = 0

    def validate_output(result):
        return result.get("valid") is True

    @durable_task(retries=3, validate=validate_output)
    def generate_validated():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return {"valid": False}
        return {"valid": True}

    result = generate_validated()
    assert result == {"valid": True}
    assert call_count == 2
