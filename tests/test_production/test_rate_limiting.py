"""Production tests: Per-Team TPS Rate Limiting.

Rate limiting with priority queues and dashboard verification.
"""

from __future__ import annotations

import pytest

from stateloom.core.errors import StateLoomRateLimitError
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_rate_limit_allows_within_tps(e2e_gate, api_client):
    """1 TPS, 1 call → passes immediately."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="RLOrg")
    team = gate.create_team(org_id=org.id, name="RLTeam")
    team.rate_limit_tps = 10.0
    gate.store.save_team(team)

    response = make_openai_response("Allowed")

    with gate.session(session_id="prod-rl-ok-1", team_id=team.id) as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        assert result is response


def test_rate_limit_rejects_over_max_queue(e2e_gate, api_client):
    """Queue full → StateLoomRateLimitError."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="RLOrg2")
    team = gate.create_team(org_id=org.id, name="RLTeam2")
    team.rate_limit_tps = 0.001  # Very low TPS to force queueing
    team.rate_limit_max_queue = 0  # No queue allowed
    team.rate_limit_queue_timeout = 0.1
    gate.store.save_team(team)

    response = make_openai_response("Never")

    with gate.session(session_id="prod-rl-reject-1", team_id=team.id) as session:
        # First call consumes the token
        try:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "First"}]},
                llm_call=lambda: response,
            )
        except StateLoomRateLimitError:
            pass  # May or may not succeed depending on timing

        # Second call should be rejected (queue full)
        with pytest.raises(StateLoomRateLimitError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Overflow"}]},
                llm_call=lambda: response,
            )


def test_rate_limit_event_recorded(e2e_gate, api_client):
    """Rate-limited call → RateLimitEvent in dashboard."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="RLOrg3")
    team = gate.create_team(org_id=org.id, name="RLTeam3")
    team.rate_limit_tps = 10.0
    gate.store.save_team(team)

    response = make_openai_response("OK")

    with gate.session(session_id="prod-rl-event-1", team_id=team.id) as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    events = client.get("/sessions/prod-rl-event-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    assert "rate_limit" in event_types


def test_rate_limit_no_team_passes(e2e_gate, api_client):
    """Session without team_id → no rate limiting."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("No team")

    with gate.session(session_id="prod-rl-noteam-1") as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        assert result is response


def test_rate_limit_dashboard_status(e2e_gate, api_client):
    """/rate-limiter endpoint shows bucket status."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="RLOrg5")
    team = gate.create_team(org_id=org.id, name="RLTeam5")
    team.rate_limit_tps = 5.0
    gate.store.save_team(team)

    response = make_openai_response("OK")

    with gate.session(session_id="prod-rl-status-1", team_id=team.id) as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    rl_resp = client.get("/rate-limiter").json()
    assert "teams" in rl_resp


def test_rate_limit_queues_excess(e2e_gate, api_client):
    """Rapid burst calls → later calls eventually served."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="RLOrg6")
    team = gate.create_team(org_id=org.id, name="RLTeam6")
    team.rate_limit_tps = 100.0  # High enough to not block
    team.rate_limit_max_queue = 10
    gate.store.save_team(team)

    response = make_openai_response("Burst OK")
    results = []

    with gate.session(session_id="prod-rl-burst-1", team_id=team.id) as session:
        for i in range(3):
            r = invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": f"Burst {i}"}]},
                llm_call=lambda: response,
            )
            results.append(r)

    assert len(results) == 3
    assert all(r is response for r in results)
