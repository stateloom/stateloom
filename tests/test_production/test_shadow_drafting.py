"""Production tests: Shadow Drafting.

Fire-and-forget local model comparison with similarity scoring.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from tests.test_production.helpers import (
    invoke_pipeline,
    make_ollama_response,
    make_openai_response,
    wait_for_shadow_events,
)


def test_shadow_runs_in_parallel(e2e_gate, api_client):
    """Cloud call + shadow call → cloud returns immediately, shadow event appears."""
    gate = e2e_gate(
        local_model="llama3.2",
        shadow=True,
        auto_route=False,
        cache=False,
    )
    client = api_client(gate)

    cloud_response = make_openai_response("Cloud answer")
    shadow_response = make_ollama_response("Shadow answer", model="llama3.2")

    shadow_mw = gate._shadow_middleware
    with patch.object(shadow_mw._client, "chat", return_value=shadow_response):
        with gate.session(session_id="prod-shadow-1") as session:
            result = invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: cloud_response,
            )

    assert result is cloud_response

    shadow_events = wait_for_shadow_events(gate, timeout=3.0)
    assert len(shadow_events) >= 1


def test_shadow_similarity_computed(e2e_gate, api_client):
    """Both responses available → similarity_score > 0."""
    gate = e2e_gate(
        local_model="llama3.2",
        shadow=True,
        auto_route=False,
        cache=False,
    )
    client = api_client(gate)

    cloud_response = make_openai_response("The answer is 42")
    shadow_response = make_ollama_response("The answer is 42", model="llama3.2")

    shadow_mw = gate._shadow_middleware
    with patch.object(shadow_mw._client, "chat", return_value=shadow_response):
        with gate.session(session_id="prod-shadow-sim-1") as session:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "What is the answer?"}]},
                llm_call=lambda: cloud_response,
            )

    shadow_events = wait_for_shadow_events(gate, timeout=3.0)
    non_cancelled = [e for e in shadow_events if e.shadow_status != "cancelled"]
    assert len(non_cancelled) >= 1
    assert non_cancelled[0].similarity_score is not None
    assert non_cancelled[0].similarity_score > 0


def test_shadow_does_not_affect_cloud(e2e_gate, api_client):
    """Shadow failure → cloud call still succeeds."""
    gate = e2e_gate(
        local_model="llama3.2",
        shadow=True,
        auto_route=False,
        cache=False,
    )
    client = api_client(gate)

    cloud_response = make_openai_response("Cloud success")

    shadow_mw = gate._shadow_middleware
    with patch.object(
        shadow_mw._client,
        "chat",
        side_effect=ConnectionError("Ollama down"),
    ):
        with gate.session(session_id="prod-shadow-fail-1") as session:
            result = invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: cloud_response,
            )

    assert result is cloud_response
    # Give shadow thread time to fail and record
    time.sleep(0.5)

    stats = client.get("/stats").json()
    assert stats["cloud_calls"] == 1


def test_shadow_dashboard_metrics(e2e_gate, api_client):
    """/shadow/metrics shows correct counts."""
    gate = e2e_gate(
        local_model="llama3.2",
        shadow=True,
        auto_route=False,
        cache=False,
    )
    client = api_client(gate)

    cloud_response = make_openai_response("Cloud")
    shadow_response = make_ollama_response("Shadow", model="llama3.2")

    shadow_mw = gate._shadow_middleware
    with patch.object(shadow_mw._client, "chat", return_value=shadow_response):
        with gate.session(session_id="prod-shadow-metrics-1") as session:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: cloud_response,
            )

    wait_for_shadow_events(gate, timeout=3.0)

    shadow_resp = client.get("/shadow/metrics").json()
    assert shadow_resp["total_calls"] >= 1
    assert shadow_resp["success_count"] >= 1


def test_shadow_disabled_no_events(e2e_gate, api_client):
    """shadow=False → no shadow events."""
    gate = e2e_gate(
        local_model=None,
        shadow=False,
        auto_route=False,
        cache=False,
    )
    client = api_client(gate)

    response = make_openai_response("No shadow")

    with gate.session(session_id="prod-no-shadow-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    time.sleep(0.3)

    shadow_resp = client.get("/shadow/metrics").json()
    assert shadow_resp["total_calls"] == 0
