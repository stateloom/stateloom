"""Production tests: Session Timeouts & Cancellation.

Session timeout, idle timeout, heartbeat, and cancellation through pipeline
and dashboard API.
"""

from __future__ import annotations

import time

import pytest

from stateloom.core.errors import StateLoomCancellationError, StateLoomTimeoutError
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_session_timeout_expires(e2e_gate, api_client):
    """Session timeout=0.1s → sleep → StateLoomTimeoutError."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Timeout")

    with gate.session(session_id="prod-timeout-1", timeout=0.1) as session:
        time.sleep(0.2)
        with pytest.raises(StateLoomTimeoutError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: response,
            )


def test_idle_timeout_expires(e2e_gate, api_client):
    """idle_timeout=0.1s → no heartbeat → timeout."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Idle")

    with gate.session(session_id="prod-idle-1", idle_timeout=0.1) as session:
        time.sleep(0.2)
        with pytest.raises(StateLoomTimeoutError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: response,
            )


def test_heartbeat_extends_idle(e2e_gate, api_client):
    """Heartbeat updates → idle timeout doesn't trigger."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Alive")

    with gate.session(session_id="prod-heartbeat-1", idle_timeout=0.3) as session:
        time.sleep(0.1)
        session.heartbeat()
        time.sleep(0.1)
        session.heartbeat()
        # Should still work — heartbeat kept it alive
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        assert result is response


def test_cancel_session(e2e_gate, api_client):
    """cancel_session(id) → next call raises StateLoomCancellationError."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Never")

    with gate.session(session_id="prod-cancel-1") as session:
        # First call succeeds
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "First"}]},
            llm_call=lambda: response,
        )

        # Cancel
        gate.cancel_session("prod-cancel-1")

        # Next call should fail
        with pytest.raises(StateLoomCancellationError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Cancelled"}]},
                llm_call=lambda: response,
            )


def test_cancel_via_dashboard(e2e_gate, api_client):
    """Dashboard POST /sessions/{id}/cancel → session cancelled."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    with gate.session(session_id="prod-dash-cancel-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

        # Cancel via dashboard API
        cancel_resp = client.post("/sessions/prod-dash-cancel-1/cancel")
        assert cancel_resp.status_code == 200

        with pytest.raises(StateLoomCancellationError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Cancelled"}]},
                llm_call=lambda: response,
            )


def test_timeout_status_in_dashboard(e2e_gate, api_client):
    """Timed-out session → status=timed_out in dashboard."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Timeout")

    with gate.session(session_id="prod-timeout-status-1", timeout=0.1) as session:
        time.sleep(0.2)
        try:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: response,
            )
        except StateLoomTimeoutError:
            pass

    sess_resp = client.get("/sessions/prod-timeout-status-1").json()
    assert sess_resp["status"] == "timed_out"
