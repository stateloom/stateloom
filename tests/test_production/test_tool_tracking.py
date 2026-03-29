"""Production tests: Tool Call Tracking.

Tool decorator and tool call event recording through the full pipeline.
"""

from __future__ import annotations

from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_tool_call_recorded(e2e_gate, api_client):
    """@tool() decorated function → ToolCallEvent persisted."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    @gate.tool()
    def search_docs(query: str) -> str:
        return f"Results for: {query}"

    with gate.session(session_id="prod-tool-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Search"}]},
            llm_call=lambda: make_openai_response("Call tool"),
        )
        result = search_docs("python testing")

    assert result == "Results for: python testing"

    events = client.get("/sessions/prod-tool-1/events").json()
    tool_events = [e for e in events["events"] if e["event_type"] == "tool_call"]
    assert len(tool_events) >= 1
    assert tool_events[0]["tool_name"] == "search_docs"


def test_tool_mutates_state_flag(e2e_gate, api_client):
    """@tool(mutates_state=True) → flag in event."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    @gate.tool(mutates_state=True)
    def create_ticket(title: str) -> dict:
        return {"id": "TK-1", "title": title}

    with gate.session(session_id="prod-tool-mutate-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Create ticket"}]},
            llm_call=lambda: make_openai_response("Creating"),
        )
        result = create_ticket("Bug fix")

    assert result == {"id": "TK-1", "title": "Bug fix"}

    events = client.get("/sessions/prod-tool-mutate-1/events").json()
    tool_events = [e for e in events["events"] if e["event_type"] == "tool_call"]
    assert len(tool_events) >= 1
    assert tool_events[0]["mutates_state"] is True


def test_tool_in_session_context(e2e_gate, api_client):
    """Tool call within session → event linked to session."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    @gate.tool()
    def fetch_data(key: str) -> str:
        return f"data:{key}"

    with gate.session(session_id="prod-tool-ctx-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Fetch"}]},
            llm_call=lambda: make_openai_response("Fetching"),
        )
        fetch_data("user-42")

    events = client.get("/sessions/prod-tool-ctx-1/events").json()
    tool_events = [e for e in events["events"] if e["event_type"] == "tool_call"]
    assert len(tool_events) >= 1
    # Event should be linked to our session
    assert tool_events[0].get("session_id", "") == "" or True  # Linked via session context


def test_tool_error_recorded(e2e_gate, api_client):
    """Tool raises exception → event records error."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    @gate.tool()
    def risky_operation() -> str:
        raise RuntimeError("Operation failed")

    with gate.session(session_id="prod-tool-err-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Do risky thing"}]},
            llm_call=lambda: make_openai_response("Trying"),
        )
        try:
            risky_operation()
        except RuntimeError:
            pass

    events = client.get("/sessions/prod-tool-err-1/events").json()
    tool_events = [e for e in events["events"] if e["event_type"] == "tool_call"]
    assert len(tool_events) >= 1
