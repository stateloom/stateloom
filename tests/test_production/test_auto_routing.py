"""Production tests: Smart Auto-Routing.

Complexity-based routing to local models with event recording.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from tests.test_production.helpers import (
    invoke_pipeline,
    make_ollama_response,
    make_openai_response,
)


def test_simple_query_routed_locally(e2e_gate, api_client):
    """Simple 'Hi' → routed to local model."""
    gate = e2e_gate(
        local_model="llama3.2",
        auto_route=True,
        shadow=False,
        cache=False,
    )
    client = api_client(gate)

    ollama_resp = make_ollama_response("Local hi", model="llama3.2")
    router = gate._auto_router

    with (
        patch.object(router._client, "chat", return_value=ollama_resp),
        patch.object(router._client, "is_available", return_value=True),
    ):
        router._ollama_available = True
        router._ollama_check_time = time.monotonic()

        with gate.session(session_id="prod-route-local-1") as session:
            result = invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: (_ for _ in ()).throw(AssertionError("Should route locally")),
                model="gpt-3.5-turbo",
            )

    stats = client.get("/stats").json()
    assert stats["local_calls"] == 1
    assert stats["cloud_calls"] == 0


def test_complex_query_stays_cloud(e2e_gate, api_client):
    """Complex multi-step reasoning → stays on cloud."""
    gate = e2e_gate(
        local_model="llama3.2",
        auto_route=True,
        shadow=False,
        cache=False,
        auto_route_complexity_threshold=0.3,
    )
    client = api_client(gate)

    cloud_response = make_openai_response("Complex answer")
    router = gate._auto_router

    with (
        patch.object(router._client, "is_available", return_value=True),
    ):
        router._ollama_available = True
        router._ollama_check_time = time.monotonic()

        complex_msg = (
            "Analyze the following multi-step problem: Given a distributed system with "
            "eventual consistency, explain the CAP theorem trade-offs when implementing "
            "a real-time collaborative editing system with conflict resolution using CRDTs. "
            "Include a comparison of operational transformation vs CRDT approaches, with "
            "examples of how Google Docs and Figma solve this problem differently."
        )

        with gate.session(session_id="prod-route-cloud-1") as session:
            result = invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": complex_msg}]},
                llm_call=lambda: cloud_response,
                model="gpt-4o",
            )

    assert result is cloud_response


def test_force_local_mode(e2e_gate, api_client):
    """force_local=True → all requests routed locally."""
    gate = e2e_gate(
        local_model="llama3.2",
        auto_route=True,
        shadow=False,
        cache=False,
        auto_route_force_local=True,
    )
    client = api_client(gate)

    ollama_resp = make_ollama_response("Forced local", model="llama3.2")
    router = gate._auto_router

    with (
        patch.object(router._client, "chat", return_value=ollama_resp),
        patch.object(router._client, "is_available", return_value=True),
    ):
        router._ollama_available = True
        router._ollama_check_time = time.monotonic()

        with gate.session(session_id="prod-force-local-1") as session:
            result = invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Any query"}]},
                llm_call=lambda: (_ for _ in ()).throw(AssertionError("Should route locally")),
                model="gpt-4o",
            )

    stats = client.get("/stats").json()
    assert stats["local_calls"] >= 1


def test_local_failure_falls_back_to_cloud(e2e_gate, api_client):
    """Local model fails → cloud fallback (no error to user)."""
    gate = e2e_gate(
        local_model="llama3.2",
        auto_route=True,
        shadow=False,
        cache=False,
    )
    client = api_client(gate)

    cloud_response = make_openai_response("Cloud fallback")
    router = gate._auto_router

    with (
        patch.object(
            router._client,
            "chat",
            side_effect=ConnectionError("Ollama crashed"),
        ),
        patch.object(router._client, "is_available", return_value=True),
    ):
        router._ollama_available = True
        router._ollama_check_time = time.monotonic()

        with gate.session(session_id="prod-route-fallback-1") as session:
            result = invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: cloud_response,
                model="gpt-3.5-turbo",
            )

    assert result is cloud_response


def test_routing_event_recorded(e2e_gate, api_client):
    """Local routing → LocalRoutingEvent with decision metadata."""
    gate = e2e_gate(
        local_model="llama3.2",
        auto_route=True,
        shadow=False,
        cache=False,
    )
    client = api_client(gate)

    ollama_resp = make_ollama_response("Routed", model="llama3.2")
    router = gate._auto_router

    with (
        patch.object(router._client, "chat", return_value=ollama_resp),
        patch.object(router._client, "is_available", return_value=True),
    ):
        router._ollama_available = True
        router._ollama_check_time = time.monotonic()

        with gate.session(session_id="prod-route-event-1") as session:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: make_openai_response("Cloud"),
                model="gpt-3.5-turbo",
            )

    events = client.get("/sessions/prod-route-event-1/events").json()
    routing_events = [e for e in events["events"] if e["event_type"] == "local_routing"]
    assert len(routing_events) >= 1
    assert routing_events[0]["routing_success"] is True


def test_auto_route_disabled_always_cloud(e2e_gate, api_client):
    """auto_route=False → all calls go to cloud."""
    gate = e2e_gate(
        local_model=None,
        auto_route=False,
        shadow=False,
        cache=False,
    )
    client = api_client(gate)

    response = make_openai_response("Cloud only")

    with gate.session(session_id="prod-no-route-1") as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    assert result is response
    stats = client.get("/stats").json()
    assert stats["cloud_calls"] == 1
    assert stats["local_calls"] == 0
