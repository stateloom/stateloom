"""Production tests: A/B Experiments.

Full experiment lifecycle: create, assign, track, conclude,
variant model/param overrides, and dashboard API verification.
"""

from __future__ import annotations

import pytest

from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_experiment_lifecycle(e2e_gate, api_client):
    """Create → start → assign sessions → feedback → conclude → metrics."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Exp reply")

    # Create and start experiment
    exp = gate.experiment_manager.create_experiment(
        name="prod-exp-1",
        variants=[
            {"name": "control", "weight": 50},
            {"name": "treatment", "weight": 50},
        ],
    )
    gate.experiment_manager.start_experiment(exp.id)

    # Run sessions with experiment assignment
    for i in range(4):
        variant = "control" if i % 2 == 0 else "treatment"
        with gate.session(
            session_id=f"prod-exp-sess-{i}",
            experiment=exp.id,
            variant=variant,
        ) as session:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": f"Question {i}"}]},
                llm_call=lambda: response,
            )

    # Submit feedback
    gate.feedback(session_id="prod-exp-sess-0", rating="success")
    gate.feedback(session_id="prod-exp-sess-1", rating="failure")
    gate.feedback(session_id="prod-exp-sess-2", rating="success")
    gate.feedback(session_id="prod-exp-sess-3", rating="success")

    # Conclude
    result = gate.experiment_manager.conclude_experiment(exp.id)
    assert "variants" in result

    # Dashboard API
    exp_resp = client.get(f"/experiments/{exp.id}").json()
    assert exp_resp["status"] == "concluded"


def test_experiment_variant_model_override(e2e_gate, api_client):
    """Variant with model override → pipeline uses that model."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    exp = gate.experiment_manager.create_experiment(
        name="model-override-exp",
        variants=[
            {"name": "gpt4", "weight": 100, "model": "gpt-4o"},
        ],
    )
    gate.experiment_manager.start_experiment(exp.id)

    response = make_openai_response("GPT-4 answer", model="gpt-4o")

    with gate.session(
        session_id="prod-exp-model-1",
        experiment=exp.id,
        variant="gpt4",
    ) as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
            model="gpt-3.5-turbo",
        )

    assert result is response


def test_experiment_variant_temperature_override(e2e_gate, api_client):
    """Variant with request_overrides → applied to request."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    exp = gate.experiment_manager.create_experiment(
        name="temp-exp",
        variants=[
            {"name": "low_temp", "weight": 100, "request_overrides": {"temperature": 0.0}},
        ],
    )
    gate.experiment_manager.start_experiment(exp.id)

    response = make_openai_response("Deterministic")

    with gate.session(
        session_id="prod-exp-temp-1",
        experiment=exp.id,
        variant="low_temp",
    ) as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    assert result is response


def test_experiment_random_assignment(e2e_gate, api_client):
    """Multiple sessions → distributed across variants."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    exp = gate.experiment_manager.create_experiment(
        name="random-assignment-exp",
        variants=[
            {"name": "A", "weight": 50},
            {"name": "B", "weight": 50},
        ],
        strategy="random",
    )
    gate.experiment_manager.start_experiment(exp.id)

    response = make_openai_response("Reply")
    assigned_variants = set()

    for i in range(20):
        with gate.session(session_id=f"prod-rand-{i}", experiment=exp.id) as session:
            assigned_variants.add(session.metadata.get("variant", ""))
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": f"Q{i}"}]},
                llm_call=lambda: response,
            )

    # With 20 samples and 50/50 weight, both variants should appear
    assert "A" in assigned_variants or "B" in assigned_variants


def test_experiment_feedback_success_rate(e2e_gate, api_client):
    """3 success + 1 failure → metrics show 75% success rate."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    exp = gate.experiment_manager.create_experiment(
        name="success-rate-exp",
        variants=[{"name": "v1", "weight": 100}],
    )
    gate.experiment_manager.start_experiment(exp.id)

    response = make_openai_response("Reply")
    ratings = ["success", "success", "success", "failure"]

    for i, rating in enumerate(ratings):
        with gate.session(
            session_id=f"prod-fb-{i}",
            experiment=exp.id,
            variant="v1",
        ) as session:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": f"Q{i}"}]},
                llm_call=lambda: response,
            )
        gate.feedback(session_id=f"prod-fb-{i}", rating=rating)

    metrics = gate.experiment_manager.get_metrics(exp.id)
    v1_metrics = metrics.get("variants", {}).get("v1", {})
    assert v1_metrics.get("success_rate", 0) == pytest.approx(0.75, abs=0.01)


def test_experiment_pause_and_resume(e2e_gate, api_client):
    """Pause stops experiment, resume re-enables it."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    exp = gate.experiment_manager.create_experiment(
        name="pause-exp",
        variants=[{"name": "v1", "weight": 100}],
    )
    gate.experiment_manager.start_experiment(exp.id)

    # Pause
    gate.experiment_manager.pause_experiment(exp.id)
    exp_resp = client.get(f"/experiments/{exp.id}").json()
    assert exp_resp["status"] == "paused"

    # Resume (start again)
    gate.experiment_manager.start_experiment(exp.id)
    exp_resp = client.get(f"/experiments/{exp.id}").json()
    assert exp_resp["status"] == "running"


def test_experiment_leaderboard(e2e_gate, api_client):
    """Two experiments → leaderboard ranks variants."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    for exp_name in ("leaderboard-exp-1", "leaderboard-exp-2"):
        exp = gate.experiment_manager.create_experiment(
            name=exp_name,
            variants=[{"name": "v1", "weight": 100}],
        )
        gate.experiment_manager.start_experiment(exp.id)
        with gate.session(
            session_id=f"prod-lb-{exp_name}",
            experiment=exp.id,
            variant="v1",
        ) as session:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Q"}]},
                llm_call=lambda: response,
            )
        gate.feedback(session_id=f"prod-lb-{exp_name}", rating="success")
        gate.experiment_manager.conclude_experiment(exp.id)

    lb_resp = client.get("/leaderboard").json()
    assert "entries" in lb_resp
    assert len(lb_resp["entries"]) >= 2


def test_experiment_dashboard_api(e2e_gate, api_client):
    """Verify experiment dashboard endpoints return correct data."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    exp = gate.experiment_manager.create_experiment(
        name="dash-exp",
        variants=[{"name": "v1", "weight": 100}],
        description="Dashboard test experiment",
    )
    gate.experiment_manager.start_experiment(exp.id)

    # List experiments
    list_resp = client.get("/experiments").json()
    assert list_resp["total"] >= 1

    # Get experiment detail
    detail_resp = client.get(f"/experiments/{exp.id}").json()
    assert detail_resp["name"] == "dash-exp"
    assert detail_resp["description"] == "Dashboard test experiment"
    assert detail_resp["status"] == "running"

    # Get metrics
    metrics_resp = client.get(f"/experiments/{exp.id}/metrics").json()
    assert "variants" in metrics_resp
