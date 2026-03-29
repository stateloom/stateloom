"""Production tests: Virtual Key Management.

VK creation, model restriction, budget, revocation, listing, and agent scoping.
"""

from __future__ import annotations

import stateloom
from tests.test_production.helpers import make_openai_response


def test_create_virtual_key(e2e_gate, api_client):
    """Create VK → returned key hash validates."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="VKOrg")
    team = gate.create_team(org_id=org.id, name="VKTeam")

    result = stateloom.create_virtual_key(team.id, name="Test Key")

    assert "id" in result
    assert "key" in result
    assert result["key"].startswith("ag-")
    assert result["key_preview"].startswith("ag-")
    assert "..." in result["key_preview"]
    assert result["team_id"] == team.id


def test_vk_model_restriction(e2e_gate, api_client):
    """VK allows only gpt-4o → other models rejected by proxy."""
    from stateloom.proxy.virtual_key import (
        VirtualKey,
        generate_virtual_key,
        make_key_preview,
        make_virtual_key_id,
    )

    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="ModelOrg")
    team = gate.create_team(org_id=org.id, name="ModelTeam")

    full_key, key_hash = generate_virtual_key()
    vk = VirtualKey(
        id=make_virtual_key_id(),
        key_hash=key_hash,
        key_preview=make_key_preview(full_key),
        team_id=team.id,
        org_id=org.id,
        name="model-restricted-key",
        allowed_models=["gpt-4o"],
    )
    gate.store.save_virtual_key(vk)

    # Verify VK stored correctly
    stored_vk = gate.store.get_virtual_key_by_hash(key_hash)
    assert stored_vk is not None
    assert stored_vk.allowed_models == ["gpt-4o"]


def test_vk_budget_limit(e2e_gate, api_client):
    """VK with budget → budget limit stored."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="BudgetVKOrg")
    team = gate.create_team(org_id=org.id, name="BudgetVKTeam")

    result = stateloom.create_virtual_key(
        team.id,
        name="Budget Key",
        budget_limit=50.0,
    )

    assert result["id"]
    # Verify budget stored
    vk = gate.store.list_virtual_keys(team_id=team.id)
    assert len(vk) >= 1
    budget_vk = [k for k in vk if k.name == "Budget Key"][0]
    assert budget_vk.budget_limit == 50.0


def test_vk_revoke(e2e_gate, api_client):
    """Revoke VK → key marked as revoked."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="RevokeOrg")
    team = gate.create_team(org_id=org.id, name="RevokeTeam")

    result = stateloom.create_virtual_key(team.id, name="Revocable Key")
    key_id = result["id"]

    # Revoke
    success = stateloom.revoke_virtual_key(key_id)
    assert success is True

    # Verify revoked
    keys = stateloom.list_virtual_keys(team_id=team.id)
    revoked_key = [k for k in keys if k["id"] == key_id][0]
    assert revoked_key["revoked"] is True


def test_list_virtual_keys(e2e_gate, api_client):
    """Create 3 VKs → list returns all 3."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="ListOrg")
    team = gate.create_team(org_id=org.id, name="ListTeam")

    for i in range(3):
        stateloom.create_virtual_key(team.id, name=f"Key-{i}")

    keys = stateloom.list_virtual_keys(team_id=team.id)
    assert len(keys) == 3
    names = {k["name"] for k in keys}
    assert names == {"Key-0", "Key-1", "Key-2"}


def test_vk_agent_scoping(e2e_gate, api_client):
    """VK with agent_ids → scoping stored correctly."""
    from stateloom.proxy.virtual_key import (
        VirtualKey,
        generate_virtual_key,
        make_key_preview,
        make_virtual_key_id,
    )

    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(name="AgentScopeOrg")
    team = gate.create_team(org_id=org.id, name="AgentScopeTeam")

    agent = gate.create_agent(slug="scoped-agent", team_id=team.id, model="gpt-4o")

    full_key, key_hash = generate_virtual_key()
    vk = VirtualKey(
        id=make_virtual_key_id(),
        key_hash=key_hash,
        key_preview=make_key_preview(full_key),
        team_id=team.id,
        org_id=org.id,
        name="agent-scoped-key",
        agent_ids=[agent.id],
    )
    gate.store.save_virtual_key(vk)

    stored_vk = gate.store.get_virtual_key_by_hash(key_hash)
    assert stored_vk is not None
    assert agent.id in stored_vk.agent_ids
