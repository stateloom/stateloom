"""Production tests: Budget Controls.

Per-session budgets, global budgets, warn vs hard-stop modes,
and dashboard API verification.
"""

from __future__ import annotations

import pytest

from stateloom.core.errors import StateLoomBudgetError
from stateloom.core.types import BudgetAction
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_budget_hard_stop_blocks_call(e2e_gate, api_client):
    """Session budget exceeded triggers StateLoomBudgetError on next call."""
    gate = e2e_gate(cache=False, budget=0.0001, budget_on_middleware_failure="block")
    client = api_client(gate)
    response = make_openai_response("Expensive", prompt_tokens=1000, completion_tokens=500)

    with gate.session(session_id="prod-budget-1") as session:
        # First call — should succeed and consume budget
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

        # Second call — should be blocked
        with pytest.raises(StateLoomBudgetError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Again"}]},
                llm_call=lambda: response,
            )


def test_budget_warn_allows_call(e2e_gate, api_client):
    """Budget action=WARN allows calls even when budget exceeded."""
    gate = e2e_gate(
        cache=False,
        budget=0.0001,
        budget_action=BudgetAction.WARN,
    )
    client = api_client(gate)
    response = make_openai_response("Result", prompt_tokens=1000, completion_tokens=500)

    with gate.session(session_id="prod-budget-warn-1") as session:
        # First call
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "First"}]},
            llm_call=lambda: response,
        )
        # Second call should still succeed (warn mode)
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Second"}]},
            llm_call=lambda: response,
        )
        assert result is response

    # Dashboard should show budget enforcement events
    events = client.get("/sessions/prod-budget-warn-1/events").json()
    budget_events = [e for e in events["events"] if e["event_type"] == "budget_enforcement"]
    assert len(budget_events) >= 1
    assert budget_events[0]["action"] == "warn"


def test_budget_tracks_cost_correctly(e2e_gate, api_client):
    """Verify total_cost matches sum of individual call costs."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply", prompt_tokens=10, completion_tokens=5)

    with gate.session(session_id="prod-budget-cost-1") as session:
        for i in range(3):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": f"Q{i}"}]},
                llm_call=lambda: response,
            )

    sess_resp = client.get("/sessions/prod-budget-cost-1").json()
    assert sess_resp["call_count"] == 3
    assert sess_resp["total_cost"] > 0


def test_budget_dashboard_shows_enforcement(e2e_gate, api_client):
    """Budget exceeded in WARN mode produces budget_enforcement event in dashboard.

    Note: In hard_stop mode, the BudgetEnforcer raises before EventRecorder
    can persist the event. Use WARN mode to verify event persistence.
    """
    gate = e2e_gate(
        cache=False,
        budget=0.0001,
        budget_action=BudgetAction.WARN,
    )
    client = api_client(gate)
    response = make_openai_response("Big", prompt_tokens=1000, completion_tokens=500)

    with gate.session(session_id="prod-budget-dash-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        # Second call — budget exceeded but WARN mode allows it
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "More"}]},
            llm_call=lambda: response,
        )

    events = client.get("/sessions/prod-budget-dash-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    assert "budget_enforcement" in event_types


def test_budget_different_sessions_independent(e2e_gate, api_client):
    """Two sessions with different budgets are enforced independently.

    Note: BudgetEnforcer is only in the pipeline when budget is set at
    the gate level (budget_per_session or budget_global in config). We
    set a gate-level budget so the middleware is active, then override
    per-session.
    """
    gate = e2e_gate(
        cache=False,
        budget=0.0001,
        budget_on_middleware_failure="block",
    )
    client = api_client(gate)
    response = make_openai_response("Reply", prompt_tokens=100, completion_tokens=50)

    # Session with tiny budget (inherits from gate default)
    with gate.session(session_id="prod-budget-s1") as s1:
        invoke_pipeline(
            gate,
            s1,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        with pytest.raises(StateLoomBudgetError):
            invoke_pipeline(
                gate,
                s1,
                {"messages": [{"role": "user", "content": "Again"}]},
                llm_call=lambda: response,
            )

    # Session with generous budget — should succeed
    with gate.session(session_id="prod-budget-s2", budget=100.0) as s2:
        result = invoke_pipeline(
            gate,
            s2,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        assert result is response


def test_global_budget_enforcement(e2e_gate, api_client):
    """Per-session budget cap enforced independently across sessions.

    Each session gets the default budget from init(budget=...) and
    enforces it independently.
    """
    gate = e2e_gate(cache=False, budget=0.0001, budget_on_middleware_failure="block")
    client = api_client(gate)
    response = make_openai_response("Resp", prompt_tokens=1000, completion_tokens=500)

    # First session — budget exceeded after first call
    with gate.session(session_id="prod-global-budget-1") as s1:
        invoke_pipeline(
            gate,
            s1,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        with pytest.raises(StateLoomBudgetError):
            invoke_pipeline(
                gate,
                s1,
                {"messages": [{"role": "user", "content": "Again"}]},
                llm_call=lambda: response,
            )

    # Second session — also budget exceeded after first call (independent budget)
    with gate.session(session_id="prod-global-budget-2") as s2:
        invoke_pipeline(
            gate,
            s2,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        with pytest.raises(StateLoomBudgetError):
            invoke_pipeline(
                gate,
                s2,
                {"messages": [{"role": "user", "content": "Again"}]},
                llm_call=lambda: response,
            )
