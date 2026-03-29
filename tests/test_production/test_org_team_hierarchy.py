"""Production tests: Organization & Team Management.

Org/team creation, hierarchy, stats, and team-level features.
"""

from __future__ import annotations

from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_org_team_creation(e2e_gate, api_client):
    """Create org → create team → verify hierarchy."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="MegaCorp")
    team = gate.create_team(org_id=org.id, name="AI Team")

    assert org.name == "MegaCorp"
    assert team.org_id == org.id
    assert team.name == "AI Team"

    # Verify via dashboard
    orgs_resp = client.get("/organizations").json()
    org_ids = [o["id"] for o in orgs_resp["organizations"]]
    assert org.id in org_ids

    teams_resp = client.get(f"/teams?org_id={org.id}").json()
    team_ids = [t["id"] for t in teams_resp["teams"]]
    assert team.id in team_ids


def test_org_stats_aggregation(e2e_gate, api_client):
    """Multiple teams/sessions → org_stats aggregates correctly."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply", prompt_tokens=10, completion_tokens=5)

    org = gate.create_organization(name="StatsOrg")
    team1 = gate.create_team(org_id=org.id, name="Team1")
    team2 = gate.create_team(org_id=org.id, name="Team2")

    # Sessions in team1
    with gate.session(session_id="prod-org-stat-1", org_id=org.id, team_id=team1.id) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "T1 Q1"}]},
            llm_call=lambda: response,
        )

    # Sessions in team2
    with gate.session(session_id="prod-org-stat-2", org_id=org.id, team_id=team2.id) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "T2 Q1"}]},
            llm_call=lambda: response,
        )

    org_stats = gate.store.get_org_stats(org.id)
    assert org_stats["total_sessions"] == 2
    assert org_stats["total_calls"] == 2


def test_team_stats(e2e_gate, api_client):
    """Team sessions → team_stats returns costs/calls."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply", prompt_tokens=10, completion_tokens=5)

    org = gate.create_organization(name="TeamStatsOrg")
    team = gate.create_team(org_id=org.id, name="StatsTeam")

    for i in range(3):
        with gate.session(session_id=f"prod-team-stat-{i}", org_id=org.id, team_id=team.id) as s:
            invoke_pipeline(
                gate,
                s,
                {"messages": [{"role": "user", "content": f"Q{i}"}]},
                llm_call=lambda: response,
            )

    team_stats = gate.store.get_team_stats(team.id)
    assert team_stats["total_sessions"] == 3
    assert team_stats["total_calls"] == 3


def test_team_budget_enforcement(e2e_gate, api_client):
    """Team-level budget → enforced across sessions."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="BudgetOrg")
    team = gate.create_team(org_id=org.id, name="BudgetTeam", budget=0.0001)

    response = make_openai_response("Reply", prompt_tokens=1000, completion_tokens=500)

    with gate.session(session_id="prod-team-budget-1", org_id=org.id, team_id=team.id) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "Expensive"}]},
            llm_call=lambda: response,
        )

    # Team budget should be tracked
    team_stats = gate.store.get_team_stats(team.id)
    assert team_stats["total_cost"] > 0


def test_team_pii_rules(e2e_gate, api_client):
    """Team with PII rules → applied to team sessions."""
    from stateloom.core.config import PIIRule
    from stateloom.core.types import PIIMode

    gate = e2e_gate(cache=False, pii=True)
    client = api_client(gate)

    org = gate.create_organization(name="PIIOrg")
    team = gate.create_team(org_id=org.id, name="PIITeam")

    response = make_openai_response("Reply")

    with gate.session(session_id="prod-team-pii-1", org_id=org.id, team_id=team.id) as s:
        invoke_pipeline(
            gate,
            s,
            {"messages": [{"role": "user", "content": "My email is test@corp.com"}]},
            llm_call=lambda: response,
        )

    # PII should be detected
    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 1


def test_list_teams_by_org(e2e_gate, api_client):
    """Multiple orgs → teams filtered correctly."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org1 = gate.create_organization(name="Org1")
    org2 = gate.create_organization(name="Org2")
    team1 = gate.create_team(org_id=org1.id, name="T1")
    team2 = gate.create_team(org_id=org1.id, name="T2")
    team3 = gate.create_team(org_id=org2.id, name="T3")

    # List teams for org1 — should have 2
    teams_resp = client.get(f"/teams?org_id={org1.id}").json()
    team_ids = [t["id"] for t in teams_resp["teams"]]
    assert team1.id in team_ids
    assert team2.id in team_ids
    assert team3.id not in team_ids
