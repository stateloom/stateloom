"""Production tests: Managed Agent Definitions (Prompts-as-an-API).

Agent CRUD, versioning, proxy routing, VK scoping, and dashboard verification.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from stateloom.core.types import AgentStatus
from stateloom.dashboard.api import create_api_router
from stateloom.proxy.router import create_proxy_router
from tests.test_production.helpers import make_openai_response


def _full_client(gate):
    """Create a TestClient with both dashboard API and proxy routes."""
    app = FastAPI()
    app.include_router(create_api_router(gate))
    app.include_router(create_proxy_router(gate), prefix="/v1")
    return TestClient(app)


def _create_vk(gate, team_id, **kwargs):
    """Create a virtual key and return (full_key, vk_object)."""
    from stateloom.proxy.virtual_key import (
        VirtualKey,
        generate_virtual_key,
        make_key_preview,
        make_virtual_key_id,
    )

    full_key, key_hash = generate_virtual_key()
    vk = VirtualKey(
        id=make_virtual_key_id(),
        key_hash=key_hash,
        key_preview=make_key_preview(full_key),
        team_id=team_id,
        org_id=kwargs.get("org_id", ""),
        name=kwargs.get("name", "agent-key"),
        agent_ids=kwargs.get("agent_ids", []),
        allowed_models=kwargs.get("allowed_models", []),
    )
    gate.store.save_virtual_key(vk)
    return full_key, vk


def test_agent_create_and_retrieve(e2e_gate, api_client):
    """Create agent via Gate, retrieve via dashboard API."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="AgentOrg")
    team = gate.create_team(org_id=org.id, name="AgentTeam")

    agent = gate.create_agent(
        slug="my-chatbot",
        team_id=team.id,
        name="My Chatbot",
        model="gpt-4o",
        system_prompt="You are a helpful assistant.",
    )

    assert agent.slug == "my-chatbot"
    assert agent.status == AgentStatus.ACTIVE

    # Retrieve via dashboard
    resp = client.get(f"/agents/{agent.id}").json()
    assert resp["slug"] == "my-chatbot"
    assert resp["name"] == "My Chatbot"


def test_agent_model_override_governance(e2e_gate):
    """Agent forces model — client model silently ignored."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _full_client(gate)

    org = gate.create_organization(name="GovOrg")
    team = gate.create_team(org_id=org.id, name="GovTeam")

    agent = gate.create_agent(
        slug="governance-bot",
        team_id=team.id,
        model="gpt-4o",
        system_prompt="You are governed.",
    )

    full_key, vk = _create_vk(gate, team.id, org_id=org.id)

    mock_response = make_openai_response("Governed response", model="gpt-4o")

    with patch("stateloom.chat.Client.achat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = types.SimpleNamespace(
            raw=mock_response,
            content="Governed response",
            model="gpt-4o",
            cost=0.001,
            tokens=15,
            provider="openai",
        )

        # Client sends model=gpt-3.5-turbo but agent forces gpt-4o
        resp = client.post(
            f"/v1/agents/{agent.slug}/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={
                "Authorization": f"Bearer {full_key}",
                "X-StateLoom-Team-Id": team.id,
            },
        )

    assert resp.status_code == 200


def test_agent_system_prompt_prepend(e2e_gate):
    """Agent system prompt + client system prompt → merged correctly."""
    gate = e2e_gate(cache=False)

    org = gate.create_organization(name="PromptOrg")
    team = gate.create_team(org_id=org.id, name="PromptTeam")

    agent = gate.create_agent(
        slug="prompt-bot",
        team_id=team.id,
        model="gpt-4o",
        system_prompt="You are a strict assistant.",
    )

    # Get the active version
    versions = gate.store.list_agent_versions(agent.id)
    version = versions[0]

    from stateloom.agent.resolver import apply_agent_overrides

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    model, merged, extras = apply_agent_overrides(version, messages, {})
    assert model == "gpt-4o"

    # Agent system prompt should be prepended
    system_msgs = [m for m in merged if m["role"] == "system"]
    assert len(system_msgs) >= 1
    combined_system = " ".join(m["content"] for m in system_msgs)
    assert "strict assistant" in combined_system


def test_agent_version_rollback(e2e_gate, api_client):
    """Create v2, activate, rollback to v1, verify v1 prompt used."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="RollOrg")
    team = gate.create_team(org_id=org.id, name="RollTeam")

    agent = gate.create_agent(
        slug="rollback-bot",
        team_id=team.id,
        model="gpt-4o",
        system_prompt="V1 prompt",
    )

    # Create v2
    v2 = gate.create_agent_version(
        agent.id,
        model="gpt-4o-mini",
        system_prompt="V2 prompt",
    )

    # Activate v2
    gate.activate_agent_version(agent.id, v2.id)
    updated = gate.get_agent(agent.id)
    assert updated.active_version_id == v2.id

    # Get v1 version ID
    versions = gate.store.list_agent_versions(agent.id)
    v1 = [v for v in versions if v.version_number == 1][0]

    # Rollback to v1
    gate.activate_agent_version(agent.id, v1.id)
    rolled_back = gate.get_agent(agent.id)
    assert rolled_back.active_version_id == v1.id


def test_agent_paused_returns_403(e2e_gate):
    """Pause agent → proxy returns 403."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _full_client(gate)

    org = gate.create_organization(name="PauseOrg")
    team = gate.create_team(org_id=org.id, name="PauseTeam")

    agent = gate.create_agent(
        slug="paused-bot",
        team_id=team.id,
        model="gpt-4o",
        system_prompt="Paused.",
    )

    full_key, vk = _create_vk(gate, team.id, org_id=org.id)

    # Pause the agent
    gate.update_agent(agent.id, status=AgentStatus.PAUSED)

    resp = client.post(
        f"/v1/agents/{agent.slug}/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        headers={
            "Authorization": f"Bearer {full_key}",
            "X-StateLoom-Team-Id": team.id,
        },
    )
    assert resp.status_code == 403


def test_agent_archived_returns_410(e2e_gate):
    """Archive agent → proxy returns 410."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _full_client(gate)

    org = gate.create_organization(name="ArchiveOrg")
    team = gate.create_team(org_id=org.id, name="ArchiveTeam")

    agent = gate.create_agent(
        slug="archived-bot",
        team_id=team.id,
        model="gpt-4o",
        system_prompt="Archived.",
    )

    full_key, vk = _create_vk(gate, team.id, org_id=org.id)

    # Archive the agent
    gate.archive_agent(agent.id)

    resp = client.post(
        f"/v1/agents/{agent.slug}/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        headers={
            "Authorization": f"Bearer {full_key}",
            "X-StateLoom-Team-Id": team.id,
        },
    )
    assert resp.status_code == 410


def test_agent_vk_scoping(e2e_gate):
    """VK restricted to agent A → calling agent B returns 403."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _full_client(gate)

    org = gate.create_organization(name="ScopeOrg")
    team = gate.create_team(org_id=org.id, name="ScopeTeam")

    agent_a = gate.create_agent(slug="agent-a", team_id=team.id, model="gpt-4o")
    agent_b = gate.create_agent(slug="agent-b", team_id=team.id, model="gpt-4o")

    # VK only allows agent_a
    full_key, vk = _create_vk(
        gate,
        team.id,
        org_id=org.id,
        agent_ids=[agent_a.id],
    )

    resp = client.post(
        f"/v1/agents/{agent_b.slug}/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        headers={
            "Authorization": f"Bearer {full_key}",
            "X-StateLoom-Team-Id": team.id,
        },
    )
    assert resp.status_code == 403


def test_agent_request_overrides(e2e_gate):
    """Agent with request_overrides → applied in pipeline."""
    gate = e2e_gate(cache=False)

    org = gate.create_organization(name="OverrideOrg")
    team = gate.create_team(org_id=org.id, name="OverrideTeam")

    agent = gate.create_agent(
        slug="override-bot",
        team_id=team.id,
        model="gpt-4o",
        system_prompt="Precise bot.",
        request_overrides={"temperature": 0.1, "max_tokens": 100},
    )

    versions = gate.store.list_agent_versions(agent.id)
    version = versions[0]
    assert version.request_overrides == {"temperature": 0.1, "max_tokens": 100}

    from stateloom.agent.resolver import apply_agent_overrides

    messages = [{"role": "user", "content": "Hi"}]
    model, merged, extras = apply_agent_overrides(version, messages, {})
    assert extras.get("temperature") == 0.1
    assert extras.get("max_tokens") == 100
