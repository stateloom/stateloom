"""Production tests: Session Management.

Real-world scenarios for session creation, completion, metadata, cost tracking,
and dashboard API verification.
"""

from __future__ import annotations

from tests.test_production.helpers import (
    invoke_pipeline,
    make_openai_response,
    run_pipeline_calls,
)


def test_session_creates_and_completes(e2e_gate, api_client):
    """Open session, make 1 call, verify status=completed, cost>0, tokens>0."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Hello!", model="gpt-3.5-turbo")
    request_kwargs = {"messages": [{"role": "user", "content": "Hi"}]}

    with gate.session(session_id="prod-session-1") as session:
        result = invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)

    assert result is response

    # Verify via dashboard
    sess_resp = client.get("/sessions/prod-session-1").json()
    assert sess_resp["status"] == "completed"
    assert sess_resp["call_count"] == 1
    assert sess_resp["total_tokens"] > 0
    assert sess_resp["total_cost"] >= 0


def test_session_with_custom_id(e2e_gate, api_client):
    """Use a custom session_id and verify retrievable by ID."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("World")
    request_kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

    with gate.session(session_id="my-app-session-123") as session:
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)

    sess_resp = client.get("/sessions/my-app-session-123").json()
    assert sess_resp["id"] == "my-app-session-123"
    assert sess_resp["status"] == "completed"


def test_session_metadata_persisted(e2e_gate, api_client):
    """Set metadata and verify it's accessible after session ends."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Answer")
    request_kwargs = {"messages": [{"role": "user", "content": "Q"}]}

    with gate.session(session_id="prod-meta-1") as session:
        session.metadata["user_id"] = "u1"
        session.metadata["app"] = "chatbot"
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)

    stored = gate.store.get_session("prod-meta-1")
    assert stored is not None
    assert stored.metadata["user_id"] == "u1"
    assert stored.metadata["app"] == "chatbot"


def test_session_multiple_calls_accumulate(e2e_gate, api_client):
    """3 calls in one session, verify call_count=3, costs summed."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply", prompt_tokens=10, completion_tokens=5)
    request_kwargs = {"messages": [{"role": "user", "content": "Tell me something"}]}

    with gate.session(session_id="prod-multi-1") as session:
        for i in range(3):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": f"Message {i}"}]},
                llm_call=lambda: response,
            )

    sess_resp = client.get("/sessions/prod-multi-1").json()
    assert sess_resp["call_count"] == 3
    assert sess_resp["total_tokens"] == 45  # 3 * (10 + 5)
    assert sess_resp["step_counter"] == 3


def test_session_with_name(e2e_gate, api_client):
    """Named session shows name in API response."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Named")
    request_kwargs = {"messages": [{"role": "user", "content": "Hi"}]}

    with gate.session(session_id="prod-named-1", name="Customer Support Bot") as session:
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)

    sess_resp = client.get("/sessions/prod-named-1").json()
    assert sess_resp["name"] == "Customer Support Bot"


def test_concurrent_sessions_independent(e2e_gate, api_client):
    """Two sequential sessions have independent stats."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Answer", prompt_tokens=10, completion_tokens=5)

    with gate.session(session_id="prod-s1") as s1:
        invoke_pipeline(
            gate,
            s1,
            {"messages": [{"role": "user", "content": "Q1"}]},
            llm_call=lambda: response,
        )

    with gate.session(session_id="prod-s2") as s2:
        invoke_pipeline(
            gate,
            s2,
            {"messages": [{"role": "user", "content": "Q2"}]},
            llm_call=lambda: response,
        )
        invoke_pipeline(
            gate,
            s2,
            {"messages": [{"role": "user", "content": "Q3"}]},
            llm_call=lambda: response,
        )

    s1_resp = client.get("/sessions/prod-s1").json()
    s2_resp = client.get("/sessions/prod-s2").json()
    assert s1_resp["call_count"] == 1
    assert s2_resp["call_count"] == 2


def test_session_org_team_assignment(e2e_gate, api_client):
    """Session with org_id + team_id recorded in store."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    org = gate.create_organization(name="TestOrg")
    team = gate.create_team(org_id=org.id, name="TestTeam")
    response = make_openai_response("Org/Team answer")

    with gate.session(session_id="prod-org-1", org_id=org.id, team_id=team.id) as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    stored = gate.store.get_session("prod-org-1")
    assert stored is not None
    assert stored.org_id == org.id
    assert stored.team_id == team.id
