"""Production tests: Emergency Kill Switch.

Global and rule-based kill switch with response modes and dashboard verification.
"""

from __future__ import annotations

import pytest

from stateloom.core.config import KillSwitchRule
from stateloom.core.errors import StateLoomKillSwitchError
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_global_kill_switch_blocks_all(e2e_gate, api_client):
    """Activate global kill switch → all calls blocked."""
    gate = e2e_gate(cache=False, kill_switch_active=True)
    client = api_client(gate)
    response = make_openai_response("Never")

    with gate.session(session_id="prod-ks-global-1") as session:
        with pytest.raises(StateLoomKillSwitchError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: response,
            )


def test_kill_switch_rule_model_glob(e2e_gate, api_client):
    """Rule for 'gpt-4*' → blocks gpt-4o, allows gpt-3.5-turbo."""
    gate = e2e_gate(
        cache=False,
        kill_switch_rules=[KillSwitchRule(model="gpt-4*", reason="Testing")],
    )
    client = api_client(gate)
    response_4 = make_openai_response("GPT-4", model="gpt-4o")
    response_35 = make_openai_response("GPT-3.5", model="gpt-3.5-turbo")

    with gate.session(session_id="prod-ks-model-1") as session:
        # gpt-4o should be blocked
        with pytest.raises(StateLoomKillSwitchError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: response_4,
                model="gpt-4o",
            )

        # gpt-3.5-turbo should work
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi again"}]},
            llm_call=lambda: response_35,
            model="gpt-3.5-turbo",
        )
        assert result is response_35


def test_kill_switch_rule_provider(e2e_gate, api_client):
    """Rule for provider='openai' → blocks OpenAI calls."""
    gate = e2e_gate(
        cache=False,
        kill_switch_rules=[KillSwitchRule(provider="openai", reason="Outage")],
    )
    client = api_client(gate)
    response = make_openai_response("Never")

    with gate.session(session_id="prod-ks-provider-1") as session:
        with pytest.raises(StateLoomKillSwitchError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: response,
                model="gpt-3.5-turbo",
            )


def test_kill_switch_response_mode(e2e_gate, api_client):
    """response_mode='response' → returns static dict instead of error."""
    gate = e2e_gate(
        cache=False,
        kill_switch_active=True,
        kill_switch_response_mode="response",
    )
    client = api_client(gate)
    response = make_openai_response("Never")

    with gate.session(session_id="prod-ks-response-1") as session:
        # Should NOT raise — returns a static response
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        assert result is not response  # Got the static response, not the LLM response


def test_kill_switch_event_persisted(e2e_gate, api_client):
    """Kill switch triggered → event in dashboard."""
    gate = e2e_gate(cache=False, kill_switch_active=True)
    client = api_client(gate)
    response = make_openai_response("Never")

    with gate.session(session_id="prod-ks-event-1") as session:
        try:
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: response,
            )
        except StateLoomKillSwitchError:
            pass

    events = client.get("/sessions/prod-ks-event-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    assert "kill_switch" in event_types


def test_kill_switch_add_remove_rules(e2e_gate, api_client):
    """Add rule → blocks → clear rules → allows."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Reply")

    # No rules — should work
    with gate.session(session_id="prod-ks-rules-1") as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )
        assert result is response

    # Add a rule
    gate.config.kill_switch_rules.append(KillSwitchRule(model="gpt-3.5-turbo", reason="Test block"))

    with gate.session(session_id="prod-ks-rules-2") as session:
        with pytest.raises(StateLoomKillSwitchError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Blocked"}]},
                llm_call=lambda: response,
            )

    # Clear rules
    gate.config.kill_switch_rules.clear()

    with gate.session(session_id="prod-ks-rules-3") as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Allowed again"}]},
            llm_call=lambda: response,
        )
        assert result is response


def test_kill_switch_environment_rule(e2e_gate, api_client):
    """Environment-specific rule matches when environment is set."""
    gate = e2e_gate(
        cache=False,
        kill_switch_rules=[KillSwitchRule(environment="production", reason="Prod block")],
        kill_switch_environment="production",
    )
    client = api_client(gate)
    response = make_openai_response("Never")

    with gate.session(session_id="prod-ks-env-1") as session:
        with pytest.raises(StateLoomKillSwitchError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_call=lambda: response,
            )
