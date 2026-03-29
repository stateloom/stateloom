"""Production tests: Loop Detection.

Repeated identical requests trigger loop detection with configurable thresholds.
The loop detector blocks the request (skip_call + static response) instead of raising.
"""

from __future__ import annotations

import pytest

from tests.test_production.helpers import (
    assert_event_exists,
    invoke_pipeline,
    make_openai_response,
)


def test_loop_detected_after_threshold(e2e_gate, api_client):
    """With threshold=3, 3rd identical request is blocked (not raised)."""
    gate = e2e_gate(cache=False, loop_exact_threshold=3)
    client = api_client(gate)
    response = make_openai_response("Loop reply")
    request_kwargs = {"messages": [{"role": "user", "content": "Same thing"}]}

    with gate.session(session_id="prod-loop-1") as session:
        for _ in range(2):
            invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)

        # 3rd call is blocked — returns static response instead of raising
        result = invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)
        assert isinstance(result, dict)
        assert "loop detected" in result["choices"][0]["message"]["content"].lower()


def test_loop_different_messages_no_detection(e2e_gate, api_client):
    """Varied messages → no loop detected."""
    gate = e2e_gate(cache=False, loop_exact_threshold=3)
    client = api_client(gate)
    response = make_openai_response("Reply")

    with gate.session(session_id="prod-loop-varied-1") as session:
        for i in range(5):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": f"Unique message {i}"}]},
                llm_call=lambda: response,
            )

    # All 5 calls should have succeeded
    sess_resp = client.get("/sessions/prod-loop-varied-1").json()
    assert sess_resp["call_count"] == 5


def test_loop_event_persisted(e2e_gate, api_client):
    """Loop detected → request blocked, loop_detection event persisted."""
    gate = e2e_gate(cache=False, loop_exact_threshold=2)
    client = api_client(gate)
    response = make_openai_response("Loop")
    request_kwargs = {"messages": [{"role": "user", "content": "Repeat"}]}

    with gate.session(session_id="prod-loop-event-1") as session:
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)
        # 2nd call is blocked — no exception
        result = invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)
        assert isinstance(result, dict)

    # Both calls persisted — first as llm_call, second has loop_detection event
    events = client.get("/sessions/prod-loop-event-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    assert "llm_call" in event_types
    assert "loop_detection" in event_types


def test_loop_threshold_configurable(e2e_gate, api_client):
    """Custom threshold=5 — no block on first 4, blocked on 5th."""
    gate = e2e_gate(cache=False, loop_exact_threshold=5)
    client = api_client(gate)
    response = make_openai_response("OK")
    request_kwargs = {"messages": [{"role": "user", "content": "Again and again"}]}

    with gate.session(session_id="prod-loop-thresh-1") as session:
        for _ in range(4):
            invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)

        # 5th call is blocked
        result = invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)
        assert isinstance(result, dict)
        assert "loop detected" in result["choices"][0]["message"]["content"].lower()
