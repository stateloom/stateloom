"""Production tests: Team-Level and Org-Level Budget Enforcement.

End-to-end tests exercising the full stack: Gate → middleware pipeline →
budget enforcement → cost propagation → org/team accumulators.
"""

from __future__ import annotations

import pytest

from stateloom.core.errors import StateLoomBudgetError
from stateloom.core.types import BudgetAction
from tests.test_production.helpers import invoke_pipeline, make_openai_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# High token counts guarantee the tiny budgets are exceeded on the first call.
EXPENSIVE_RESPONSE = make_openai_response(
    "Expensive reply", prompt_tokens=1000, completion_tokens=500
)


def _setup_org_team(gate, *, org_budget=None, team_budget=None):
    """Create an org and team with optional budgets."""
    org = gate.create_organization(name="BudgetOrg", budget=org_budget)
    team = gate.create_team(org_id=org.id, name="BudgetTeam", budget=team_budget)
    return org, team


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_team_budget_hard_stop(e2e_gate):
    """Team with tiny budget — first call succeeds, second is blocked."""
    gate = e2e_gate(
        cache=False,
        budget=100.0,  # large per-session budget so BudgetEnforcer is in pipeline
        budget_on_middleware_failure="block",
    )
    org, team = _setup_org_team(gate, team_budget=0.0001)

    with gate.session(
        session_id="team-budget-hard-1", org_id=org.id, team_id=team.id, budget=100.0
    ) as s:
        # First call — succeeds and accumulates cost into team
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )

        # Second call — team budget exceeded → blocked
        with pytest.raises(StateLoomBudgetError) as exc_info:
            invoke_pipeline(
                gate,
                s,
                {"messages": [{"role": "user", "content": "Again"}]},
                llm_call=lambda: EXPENSIVE_RESPONSE,
            )

        err = exc_info.value
        assert err.limit == 0.0001
        assert err.spent > 0.0001


def test_org_budget_hard_stop(e2e_gate):
    """Org with tiny budget, team without budget — org budget blocks."""
    gate = e2e_gate(
        cache=False,
        budget=100.0,
        budget_on_middleware_failure="block",
    )
    org, team = _setup_org_team(gate, org_budget=0.0001, team_budget=None)

    # First session — succeeds
    with gate.session(
        session_id="org-budget-hard-1", org_id=org.id, team_id=team.id, budget=100.0
    ) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )

    # Second session — org budget exceeded
    with gate.session(
        session_id="org-budget-hard-2", org_id=org.id, team_id=team.id, budget=100.0
    ) as s:
        with pytest.raises(StateLoomBudgetError) as exc_info:
            invoke_pipeline(
                gate,
                s,
                {"messages": [{"role": "user", "content": "More"}]},
                llm_call=lambda: EXPENSIVE_RESPONSE,
            )

        err = exc_info.value
        assert err.limit == 0.0001
        assert err.spent > 0.0001


def test_team_budget_blocks_before_org(e2e_gate):
    """Team has tiny budget, org has large budget — team trips first."""
    gate = e2e_gate(
        cache=False,
        budget=100.0,
        budget_on_middleware_failure="block",
    )
    org, team = _setup_org_team(gate, org_budget=100.0, team_budget=0.0001)

    with gate.session(
        session_id="team-before-org-1", org_id=org.id, team_id=team.id, budget=100.0
    ) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )

        with pytest.raises(StateLoomBudgetError) as exc_info:
            invoke_pipeline(
                gate,
                s,
                {"messages": [{"role": "user", "content": "Again"}]},
                llm_call=lambda: EXPENSIVE_RESPONSE,
            )

        # Error should show the team budget limit, not the org limit
        err = exc_info.value
        assert err.limit == 0.0001


def test_session_budget_blocks_before_team(e2e_gate):
    """Session has tiny budget, team has large budget — session trips first."""
    gate = e2e_gate(
        cache=False,
        budget=0.0001,  # per-session budget
        budget_on_middleware_failure="block",
    )
    org, team = _setup_org_team(gate, team_budget=100.0)

    with gate.session(session_id="session-before-team-1", org_id=org.id, team_id=team.id) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )

        with pytest.raises(StateLoomBudgetError) as exc_info:
            invoke_pipeline(
                gate,
                s,
                {"messages": [{"role": "user", "content": "Again"}]},
                llm_call=lambda: EXPENSIVE_RESPONSE,
            )

        # Error should show per-session budget limit (checked first)
        err = exc_info.value
        assert err.limit == 0.0001


def test_team_budget_across_sessions(e2e_gate):
    """Two sessions in same team — second blocked by team-level total."""
    gate = e2e_gate(
        cache=False,
        budget=100.0,
        budget_on_middleware_failure="block",
    )
    org, team = _setup_org_team(gate, team_budget=0.0001)

    # First session — spends enough to exceed team budget
    with gate.session(session_id="team-cross-1", org_id=org.id, team_id=team.id, budget=100.0) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )

    # Second session (different session_id) — team budget already exceeded
    with gate.session(session_id="team-cross-2", org_id=org.id, team_id=team.id, budget=100.0) as s:
        with pytest.raises(StateLoomBudgetError) as exc_info:
            invoke_pipeline(
                gate,
                s,
                {"messages": [{"role": "user", "content": "Hello"}]},
                llm_call=lambda: EXPENSIVE_RESPONSE,
            )

        err = exc_info.value
        assert err.limit == 0.0001


def test_org_budget_across_teams(e2e_gate):
    """Two teams in one org — second team blocked by org-level total."""
    gate = e2e_gate(
        cache=False,
        budget=100.0,
        budget_on_middleware_failure="block",
    )
    org = gate.create_organization(name="CrossTeamOrg", budget=0.0001)
    team1 = gate.create_team(org_id=org.id, name="Team1")
    team2 = gate.create_team(org_id=org.id, name="Team2")

    # Team 1 session — spends enough to exceed org budget
    with gate.session(
        session_id="org-cross-t1", org_id=org.id, team_id=team1.id, budget=100.0
    ) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Hi from T1"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )

    # Team 2 session — org budget already exceeded
    with gate.session(
        session_id="org-cross-t2", org_id=org.id, team_id=team2.id, budget=100.0
    ) as s:
        with pytest.raises(StateLoomBudgetError) as exc_info:
            invoke_pipeline(
                gate,
                s,
                {"messages": [{"role": "user", "content": "Hi from T2"}]},
                llm_call=lambda: EXPENSIVE_RESPONSE,
            )

        err = exc_info.value
        assert err.limit == 0.0001


def test_team_budget_warn_mode(e2e_gate, api_client):
    """With budget_action=WARN, team budget exceeded → call goes through."""
    gate = e2e_gate(
        cache=False,
        budget=100.0,
        budget_action=BudgetAction.WARN,
    )
    client = api_client(gate)
    org, team = _setup_org_team(gate, team_budget=0.0001)

    with gate.session(session_id="team-warn-1", org_id=org.id, team_id=team.id, budget=100.0) as s:
        # First call
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )

        # Second call — team budget exceeded but warn mode allows it
        result = invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "More"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )
        assert result is EXPENSIVE_RESPONSE

    # Verify budget_enforcement event was recorded
    events = client.get("/sessions/team-warn-1/events").json()
    budget_events = [e for e in events["events"] if e["event_type"] == "budget_enforcement"]
    assert len(budget_events) >= 1
    assert budget_events[0]["action"] == "warn"


def test_budget_enforcement_event_persisted(e2e_gate, api_client):
    """When team budget blocks a call, the enforcement event is persisted."""
    gate = e2e_gate(
        cache=False,
        budget=100.0,
        budget_on_middleware_failure="block",
    )
    client = api_client(gate)
    org, team = _setup_org_team(gate, team_budget=0.0001)

    with gate.session(
        session_id="team-event-persist-1", org_id=org.id, team_id=team.id, budget=100.0
    ) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: EXPENSIVE_RESPONSE,
        )

        with pytest.raises(StateLoomBudgetError):
            invoke_pipeline(
                gate,
                s,
                {"messages": [{"role": "user", "content": "Again"}]},
                llm_call=lambda: EXPENSIVE_RESPONSE,
            )

    # The budget_enforcement event should be persisted via _save_events_directly
    events = client.get("/sessions/team-event-persist-1/events").json()
    budget_events = [e for e in events["events"] if e["event_type"] == "budget_enforcement"]
    assert len(budget_events) >= 1
    evt = budget_events[-1]
    assert evt["limit"] == 0.0001
    assert evt["spent"] > 0.0001
    assert evt["action"] == "hard_stop"
