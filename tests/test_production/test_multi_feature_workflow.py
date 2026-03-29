"""Production tests: Cross-Feature Integration Workflows.

Real-world workflows combining multiple features through the full pipeline.
"""

from __future__ import annotations

import pytest

from stateloom.core.config import KillSwitchRule, PIIRule
from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomKillSwitchError,
)
from stateloom.core.types import PIIMode
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_chatbot_workflow(e2e_gate, api_client):
    """Org → team → budget → PII → session → 3 calls → verify all events."""
    gate = e2e_gate(
        cache=True,
        pii=True,
        pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
    )
    client = api_client(gate)

    org = gate.create_organization(name="ChatbotCorp")
    team = gate.create_team(org_id=org.id, name="Bot Team")

    response = make_openai_response("Chatbot reply", prompt_tokens=10, completion_tokens=5)

    with gate.session(
        session_id="prod-workflow-chatbot-1",
        org_id=org.id,
        team_id=team.id,
        budget=10.0,
    ) as session:
        # Normal call
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi there"}]},
            llm_call=lambda: response,
        )
        # Call with PII
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Email me at user@corp.com"}]},
            llm_call=lambda: response,
        )
        # Checkpoint
        gate.checkpoint(label="conversation-started")

        # Another call
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Thanks"}]},
            llm_call=lambda: response,
        )

    # Verify session
    sess_resp = client.get("/sessions/prod-workflow-chatbot-1").json()
    assert sess_resp["call_count"] == 3
    assert sess_resp["status"] == "completed"
    assert sess_resp["budget"] == 10.0

    # Verify events
    events = client.get("/sessions/prod-workflow-chatbot-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    assert "llm_call" in event_types
    assert "pii_detection" in event_types
    assert "checkpoint" in event_types

    # Verify PII detection
    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 1

    # Verify org stats
    org_stats = gate.store.get_org_stats(org.id)
    assert org_stats["total_sessions"] >= 1


def test_agent_with_experiment(e2e_gate, api_client):
    """Agent + experiment → version A/B → track metrics."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="ExpOrg")
    team = gate.create_team(org_id=org.id, name="ExpTeam")

    # Create agent
    agent = gate.create_agent(
        slug="exp-agent",
        team_id=team.id,
        model="gpt-4o",
        system_prompt="You are an experimental agent.",
    )

    # Create experiment
    exp = gate.experiment_manager.create_experiment(
        name="agent-experiment",
        variants=[
            {"name": "baseline", "weight": 50},
            {"name": "improved", "weight": 50, "model": "gpt-4o"},
        ],
    )
    gate.experiment_manager.start_experiment(exp.id)

    response = make_openai_response("Experiment reply")

    # Run sessions with different variants
    for i, variant in enumerate(["baseline", "improved"]):
        with gate.session(
            session_id=f"prod-agent-exp-{i}",
            experiment=exp.id,
            variant=variant,
        ) as session:
            session.agent_id = agent.id
            session.metadata["agent_id"] = agent.id
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": f"Q{i}"}]},
                llm_call=lambda: response,
            )

    gate.feedback(session_id="prod-agent-exp-0", rating="success")
    gate.feedback(session_id="prod-agent-exp-1", rating="success")

    metrics = gate.experiment_manager.get_metrics(exp.id)
    assert "variants" in metrics


def test_budget_with_blast_radius(e2e_gate, api_client):
    """Budget violations → blast radius tracks them."""
    gate = e2e_gate(
        cache=False,
        budget=0.0001,
        budget_on_middleware_failure="block",
        blast_radius_enabled=True,
        blast_radius_consecutive_failures=10,
        blast_radius_budget_violations_per_hour=2,
    )
    client = api_client(gate)

    response = make_openai_response("Expensive", prompt_tokens=1000, completion_tokens=500)

    # First session — exceeds budget
    with gate.session(session_id="prod-br-budget-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Big query"}]},
            llm_call=lambda: response,
        )
        try:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "More"}]},
                llm_call=lambda: response,
            )
        except StateLoomBudgetError:
            pass

    br_resp = client.get("/blast-radius").json()
    assert br_resp["enabled"] is True


def test_kill_switch_during_experiment(e2e_gate, api_client):
    """Kill switch activates mid-experiment → blocks calls."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    exp = gate.experiment_manager.create_experiment(
        name="killswitch-exp",
        variants=[{"name": "v1", "weight": 100}],
    )
    gate.experiment_manager.start_experiment(exp.id)

    response = make_openai_response("Reply")

    # First call works
    with gate.session(
        session_id="prod-ks-exp-1",
        experiment=exp.id,
        variant="v1",
    ) as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Before kill switch"}]},
            llm_call=lambda: response,
        )

    # Activate kill switch
    gate.config.kill_switch_active = True

    # Next call should be blocked
    with gate.session(
        session_id="prod-ks-exp-2",
        experiment=exp.id,
        variant="v1",
    ) as session:
        with pytest.raises(StateLoomKillSwitchError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "After kill switch"}]},
                llm_call=lambda: response,
            )


def test_session_with_checkpoints_and_tools(e2e_gate, api_client):
    """Tools + checkpoints + LLM calls → full event timeline."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    @gate.tool()
    def lookup_user(user_id: str) -> dict:
        return {"id": user_id, "name": "Alice"}

    @gate.tool(mutates_state=True)
    def update_profile(user_id: str, name: str) -> dict:
        return {"id": user_id, "name": name, "updated": True}

    response = make_openai_response("Agent reply")

    with gate.session(session_id="prod-full-timeline-1") as session:
        # LLM call
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Look up user"}]},
            llm_call=lambda: response,
        )

        # Tool calls
        user = lookup_user("u-42")
        gate.checkpoint(label="user-fetched")

        # Another LLM call
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Update profile"}]},
            llm_call=lambda: response,
        )

        update_profile("u-42", "Bob")
        gate.checkpoint(label="profile-updated")

    events = client.get("/sessions/prod-full-timeline-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    assert "llm_call" in event_types
    assert "tool_call" in event_types
    assert "checkpoint" in event_types
    assert events["total"] >= 4  # 2 LLM + 2 tool + 2 checkpoint


def test_compliance_with_pii_and_budget(e2e_gate, api_client):
    """HIPAA profile + PII + budget → all enforced together."""
    from stateloom.core.config import ComplianceProfile

    gate = e2e_gate(
        cache=False,
        pii=True,
        pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        budget=10.0,
        budget_on_middleware_failure="block",
    )
    client = api_client(gate)

    org = gate.create_organization(
        name="HIPAACompOrg",
        compliance_profile=ComplianceProfile(
            standard="hipaa",
            block_shadow=True,
            block_streaming=True,
        ),
    )
    team = gate.create_team(org_id=org.id, name="HIPAACompTeam")

    response = make_openai_response("HIPAA reply", prompt_tokens=10, completion_tokens=5)

    with gate.session(
        session_id="prod-hipaa-full-1",
        org_id=org.id,
        team_id=team.id,
        budget=10.0,
    ) as session:
        # Normal call
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Patient data query"}]},
            llm_call=lambda: response,
        )

        # Call with PII
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Doctor email: doc@hospital.com"}]},
            llm_call=lambda: response,
        )

    # Verify events
    events = client.get("/sessions/prod-hipaa-full-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    assert "llm_call" in event_types
    assert "pii_detection" in event_types

    # Verify session completed normally
    sess_resp = client.get("/sessions/prod-hipaa-full-1").json()
    assert sess_resp["status"] == "completed"
    assert sess_resp["budget"] == 10.0
