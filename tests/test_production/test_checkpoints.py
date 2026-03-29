"""Production tests: Named Checkpoints.

Milestone marking within sessions and dashboard visibility.
"""

from __future__ import annotations

from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_checkpoint_created(e2e_gate, api_client):
    """checkpoint('data-loaded') → CheckpointEvent in store."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    with gate.session(session_id="prod-ckpt-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Step 1"}]},
            llm_call=lambda: response,
        )
        gate.checkpoint(label="data-loaded")

    events = client.get("/sessions/prod-ckpt-1/events").json()
    ckpt_events = [e for e in events["events"] if e["event_type"] == "checkpoint"]
    assert len(ckpt_events) == 1
    assert ckpt_events[0]["label"] == "data-loaded"


def test_multiple_checkpoints(e2e_gate, api_client):
    """3 checkpoints → all appear in session events."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    with gate.session(session_id="prod-ckpt-multi-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Step 1"}]},
            llm_call=lambda: response,
        )
        gate.checkpoint(label="step-1-done")

        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Step 2"}]},
            llm_call=lambda: response,
        )
        gate.checkpoint(label="step-2-done")

        gate.checkpoint(label="final")

    events = client.get("/sessions/prod-ckpt-multi-1/events").json()
    ckpt_events = [e for e in events["events"] if e["event_type"] == "checkpoint"]
    labels = [e["label"] for e in ckpt_events]
    assert "step-1-done" in labels
    assert "step-2-done" in labels
    assert "final" in labels


def test_checkpoint_in_dashboard_timeline(e2e_gate, api_client):
    """Checkpoint events visible in /sessions/{id}/events."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    with gate.session(session_id="prod-ckpt-dash-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Before"}]},
            llm_call=lambda: response,
        )
        gate.checkpoint(label="midpoint")
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "After"}]},
            llm_call=lambda: response,
        )

    events = client.get("/sessions/prod-ckpt-dash-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    assert "checkpoint" in event_types
    assert "llm_call" in event_types


def test_checkpoint_with_description(e2e_gate, api_client):
    """Checkpoint with description → description persisted."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    with gate.session(session_id="prod-ckpt-desc-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        gate.checkpoint(label="plan-complete", description="All data fetched and plan generated")

    events = client.get("/sessions/prod-ckpt-desc-1/events").json()
    ckpt_events = [e for e in events["events"] if e["event_type"] == "checkpoint"]
    assert len(ckpt_events) == 1
    assert ckpt_events[0]["label"] == "plan-complete"
    assert ckpt_events[0]["description"] == "All data fetched and plan generated"
