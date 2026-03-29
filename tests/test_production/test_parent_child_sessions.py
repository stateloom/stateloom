"""Production tests: Parent-Child Sessions.

Hierarchical session relationships, inheritance, isolation, and dashboard API.
"""

from __future__ import annotations

from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_child_session_inherits_org_team(e2e_gate, api_client):
    """Parent has org_id/team_id → child inherits automatically."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="ParentOrg")
    team = gate.create_team(org_id=org.id, name="ParentTeam")
    response = make_openai_response("Child reply")

    with gate.session(session_id="prod-parent-1", org_id=org.id, team_id=team.id) as parent:
        invoke_pipeline(
            gate,
            parent,
            {"messages": [{"role": "user", "content": "Parent call"}]},
            llm_call=lambda: response,
        )

        with gate.session(session_id="prod-child-1", parent=parent.id) as child:
            invoke_pipeline(
                gate,
                child,
                {"messages": [{"role": "user", "content": "Child call"}]},
                llm_call=lambda: response,
            )

    child_sess = gate.store.get_session("prod-child-1")
    assert child_sess is not None
    assert child_sess.parent_session_id == "prod-parent-1"
    assert child_sess.org_id == org.id
    assert child_sess.team_id == team.id


def test_child_session_independent_budget(e2e_gate, api_client):
    """Parent and child have separate budgets."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply", prompt_tokens=10, completion_tokens=5)

    with gate.session(session_id="prod-parent-budget-1", budget=10.0) as parent:
        invoke_pipeline(
            gate,
            parent,
            {"messages": [{"role": "user", "content": "Parent"}]},
            llm_call=lambda: response,
        )

        with gate.session(session_id="prod-child-budget-1", parent=parent.id, budget=5.0) as child:
            invoke_pipeline(
                gate,
                child,
                {"messages": [{"role": "user", "content": "Child"}]},
                llm_call=lambda: response,
            )

    parent_resp = client.get("/sessions/prod-parent-budget-1").json()
    child_resp = client.get("/sessions/prod-child-budget-1").json()
    assert parent_resp["budget"] == 10.0
    assert child_resp["budget"] == 5.0


def test_child_session_listed_in_dashboard(e2e_gate, api_client):
    """/sessions/{parent_id}/children returns children."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    with gate.session(session_id="prod-parent-list-1") as parent:
        invoke_pipeline(
            gate,
            parent,
            {"messages": [{"role": "user", "content": "Parent"}]},
            llm_call=lambda: response,
        )

        for i in range(3):
            with gate.session(session_id=f"prod-child-list-{i}", parent=parent.id) as child:
                invoke_pipeline(
                    gate,
                    child,
                    {"messages": [{"role": "user", "content": f"Child {i}"}]},
                    llm_call=lambda: response,
                )

    children_resp = client.get("/sessions/prod-parent-list-1/children").json()
    assert children_resp["total"] == 3
    child_ids = {c["id"] for c in children_resp["children"]}
    assert "prod-child-list-0" in child_ids
    assert "prod-child-list-1" in child_ids
    assert "prod-child-list-2" in child_ids


def test_nested_sessions(e2e_gate, api_client):
    """Parent → child → grandchild hierarchy."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    with gate.session(session_id="prod-grandparent-1") as gp:
        invoke_pipeline(
            gate,
            gp,
            {"messages": [{"role": "user", "content": "GP"}]},
            llm_call=lambda: response,
        )

        with gate.session(session_id="prod-parent-nested-1", parent=gp.id) as p:
            invoke_pipeline(
                gate,
                p,
                {"messages": [{"role": "user", "content": "P"}]},
                llm_call=lambda: response,
            )

            with gate.session(session_id="prod-grandchild-1", parent=p.id) as gc:
                invoke_pipeline(
                    gate,
                    gc,
                    {"messages": [{"role": "user", "content": "GC"}]},
                    llm_call=lambda: response,
                )

    gc_sess = gate.store.get_session("prod-grandchild-1")
    assert gc_sess.parent_session_id == "prod-parent-nested-1"

    p_sess = gate.store.get_session("prod-parent-nested-1")
    assert p_sess.parent_session_id == "prod-grandparent-1"


def test_parent_child_independent_cost(e2e_gate, api_client):
    """Costs tracked separately per session."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply", prompt_tokens=10, completion_tokens=5)

    with gate.session(session_id="prod-parent-cost-1") as parent:
        invoke_pipeline(
            gate,
            parent,
            {"messages": [{"role": "user", "content": "Parent"}]},
            llm_call=lambda: response,
        )

        with gate.session(session_id="prod-child-cost-1", parent=parent.id) as child:
            invoke_pipeline(
                gate,
                child,
                {"messages": [{"role": "user", "content": "Child 1"}]},
                llm_call=lambda: response,
            )
            invoke_pipeline(
                gate,
                child,
                {"messages": [{"role": "user", "content": "Child 2"}]},
                llm_call=lambda: response,
            )

    parent_resp = client.get("/sessions/prod-parent-cost-1").json()
    child_resp = client.get("/sessions/prod-child-cost-1").json()
    assert parent_resp["call_count"] == 1
    assert child_resp["call_count"] == 2
    # Child cost should be approximately double
    assert child_resp["total_tokens"] == parent_resp["total_tokens"] * 2
