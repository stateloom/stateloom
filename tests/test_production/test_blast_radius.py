"""Production tests: Blast Radius Containment.

Auto-pause on consecutive failures, agent-level pause, unpause,
and dashboard API verification.
"""

from __future__ import annotations

import pytest

from stateloom.core.errors import (
    StateLoomBlastRadiusError,
    StateLoomKillSwitchError,
)
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def _failing_call():
    """A callable that always raises RuntimeError."""
    raise RuntimeError("Provider down")


def test_blast_radius_pauses_after_failures(e2e_gate, api_client):
    """Consecutive failures → StateLoomBlastRadiusError."""
    gate = e2e_gate(
        cache=False,
        blast_radius_enabled=True,
        blast_radius_consecutive_failures=5,
    )
    client = api_client(gate)

    got_blast_radius = False

    with gate.session(session_id="prod-br-fail-1") as session:
        for i in range(6):
            try:
                invoke_pipeline(
                    gate,
                    session,
                    {"messages": [{"role": "user", "content": f"Fail {i}"}]},
                    llm_call=_failing_call,
                )
            except StateLoomBlastRadiusError:
                got_blast_radius = True
                break
            except RuntimeError:
                pass

    assert got_blast_radius, "Expected StateLoomBlastRadiusError after consecutive failures"


def test_blast_radius_reset_on_success(e2e_gate, api_client):
    """4 failures then 1 success → counter resets, no pause."""
    gate = e2e_gate(
        cache=False,
        blast_radius_enabled=True,
        blast_radius_consecutive_failures=5,
    )
    client = api_client(gate)
    response = make_openai_response("Success")

    with gate.session(session_id="prod-br-reset-1") as session:
        for i in range(4):
            try:
                invoke_pipeline(
                    gate,
                    session,
                    {"messages": [{"role": "user", "content": f"Fail {i}"}]},
                    llm_call=_failing_call,
                )
            except RuntimeError:
                pass

        # Success resets the counter
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Recover"}]},
            llm_call=lambda: response,
        )
        assert result is response

        # Next calls should still work (counter reset)
        result2 = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "OK"}]},
            llm_call=lambda: response,
        )
        assert result2 is response


def test_blast_radius_agent_pause(e2e_gate, api_client):
    """Failures with agent_name across sessions → agent paused."""
    gate = e2e_gate(
        cache=False,
        blast_radius_enabled=True,
        blast_radius_consecutive_failures=3,
    )
    client = api_client(gate)

    # Use multiple sessions, each failing fewer than session threshold (3),
    # but the agent accumulates failures across sessions.
    for sid_idx in range(2):
        with gate.session(session_id=f"prod-br-agent-s{sid_idx}") as session:
            session.agent_name = "my-agent"
            session.metadata["agent_name"] = "my-agent"
            for i in range(2):
                try:
                    invoke_pipeline(
                        gate,
                        session,
                        {"messages": [{"role": "user", "content": f"Fail {i}"}]},
                        llm_call=_failing_call,
                    )
                except (RuntimeError, StateLoomBlastRadiusError):
                    pass

    # Agent should now be paused (accumulated 4 agent failures >= threshold 3)
    # New session with same agent should be blocked
    with gate.session(session_id="prod-br-agent-new") as session2:
        session2.metadata["agent_name"] = "my-agent"
        with pytest.raises(StateLoomBlastRadiusError):
            invoke_pipeline(
                gate,
                session2,
                {"messages": [{"role": "user", "content": "Blocked"}]},
                llm_call=lambda: make_openai_response("Never"),
            )


def test_blast_radius_unpause_session(e2e_gate, api_client):
    """Paused session → unpause_session() → resumes."""
    gate = e2e_gate(
        cache=False,
        blast_radius_enabled=True,
        blast_radius_consecutive_failures=3,
    )
    client = api_client(gate)
    response = make_openai_response("Resumed")

    with gate.session(session_id="prod-br-unpause-1") as session:
        for i in range(4):
            try:
                invoke_pipeline(
                    gate,
                    session,
                    {"messages": [{"role": "user", "content": f"Fail {i}"}]},
                    llm_call=_failing_call,
                )
            except (RuntimeError, StateLoomBlastRadiusError):
                pass

        # Should be paused now — verify
        with pytest.raises(StateLoomBlastRadiusError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Blocked"}]},
                llm_call=lambda: response,
            )

        # Unpause
        gate._blast_radius.unpause_session("prod-br-unpause-1")

        # Should work now
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Unpaused"}]},
            llm_call=lambda: response,
        )
        assert result is response


def test_blast_radius_unpause_agent(e2e_gate, api_client):
    """Paused agent → unpause_agent() → resumes."""
    gate = e2e_gate(
        cache=False,
        blast_radius_enabled=True,
        blast_radius_consecutive_failures=3,
    )
    client = api_client(gate)
    response = make_openai_response("Resumed")

    with gate.session(session_id="prod-br-unp-agent-1") as session:
        session.agent_name = "pausable-agent"
        session.metadata["agent_name"] = "pausable-agent"
        for i in range(4):
            try:
                invoke_pipeline(
                    gate,
                    session,
                    {"messages": [{"role": "user", "content": f"Fail {i}"}]},
                    llm_call=_failing_call,
                )
            except (RuntimeError, StateLoomBlastRadiusError):
                pass

    # Unpause agent
    gate._blast_radius.unpause_agent("agent:pausable-agent")

    # New session with same agent should now work
    with gate.session(session_id="prod-br-unp-agent-2") as session2:
        session2.agent_name = "pausable-agent"
        session2.metadata["agent_name"] = "pausable-agent"
        result = invoke_pipeline(
            gate,
            session2,
            {"messages": [{"role": "user", "content": "Unpaused"}]},
            llm_call=lambda: response,
        )
        assert result is response


def test_blast_radius_status_api(e2e_gate, api_client):
    """Dashboard /blast-radius shows paused sessions/agents."""
    gate = e2e_gate(
        cache=False,
        blast_radius_enabled=True,
        blast_radius_consecutive_failures=3,
    )
    client = api_client(gate)

    with gate.session(session_id="prod-br-status-1") as session:
        for i in range(4):
            try:
                invoke_pipeline(
                    gate,
                    session,
                    {"messages": [{"role": "user", "content": f"Fail {i}"}]},
                    llm_call=_failing_call,
                )
            except (RuntimeError, StateLoomBlastRadiusError):
                pass

    br_resp = client.get("/blast-radius").json()
    assert br_resp["enabled"] is True
    assert "prod-br-status-1" in br_resp["paused_sessions"]


def test_blast_radius_excludes_non_retryable(e2e_gate, api_client):
    """Kill switch errors don't count toward blast radius threshold."""
    gate = e2e_gate(
        cache=False,
        blast_radius_enabled=True,
        blast_radius_consecutive_failures=3,
        kill_switch_active=True,
    )
    client = api_client(gate)

    with gate.session(session_id="prod-br-exclude-1") as session:
        for i in range(5):
            try:
                invoke_pipeline(
                    gate,
                    session,
                    {"messages": [{"role": "user", "content": f"Call {i}"}]},
                    llm_call=lambda: make_openai_response("Never"),
                )
            except StateLoomKillSwitchError:
                pass

    # Session should NOT be paused (kill switch errors excluded)
    br_resp = client.get("/blast-radius").json()
    assert "prod-br-exclude-1" not in br_resp.get("paused_sessions", [])
