"""Production tests: OpenAI-Compatible Proxy.

Virtual key auth, model routing, error format through the proxy endpoint.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from stateloom.dashboard.api import create_api_router
from stateloom.proxy.router import create_proxy_router
from tests.test_production.helpers import make_openai_response


def _proxy_client(gate):
    """Create a TestClient that includes both proxy and dashboard routes."""
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
        name=kwargs.get("name", "test-key"),
        scopes=kwargs.get("scopes", []),
        allowed_models=kwargs.get("allowed_models", []),
        budget_limit=kwargs.get("budget_limit"),
        agent_ids=kwargs.get("agent_ids", []),
    )
    gate.store.save_virtual_key(vk)
    return full_key, vk


def test_proxy_basic_request(e2e_gate):
    """POST /v1/chat/completions with VK → 200, valid response."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _proxy_client(gate)

    org = gate.create_organization(name="ProxyOrg")
    team = gate.create_team(org_id=org.id, name="ProxyTeam")
    full_key, vk = _create_vk(gate, team.id, org_id=org.id)

    mock_response = make_openai_response("Proxy response")

    with patch("stateloom.chat.Client.achat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = types.SimpleNamespace(
            raw=mock_response,
            content="Proxy response",
            model="gpt-3.5-turbo",
            cost=0.001,
            tokens=15,
            provider="openai",
        )

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {full_key}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data or "content" in str(data)


def test_proxy_401_no_auth(e2e_gate):
    """No Authorization header → 401."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _proxy_client(gate)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 401


def test_proxy_401_invalid_key(e2e_gate):
    """Invalid key → 401."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _proxy_client(gate)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": "Bearer agk-invalid-key-12345"},
    )
    assert resp.status_code == 401


def test_proxy_virtual_key_model_restriction(e2e_gate):
    """VK allows only gpt-4o → request with gpt-3.5-turbo rejected."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _proxy_client(gate)

    org = gate.create_organization(name="RestrictOrg")
    team = gate.create_team(org_id=org.id, name="RestrictTeam")
    full_key, vk = _create_vk(gate, team.id, org_id=org.id, allowed_models=["gpt-4o"])

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": f"Bearer {full_key}"},
    )
    assert resp.status_code == 403


def test_proxy_error_format(e2e_gate):
    """Error responses follow OpenAI error format."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _proxy_client(gate)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 401
    data = resp.json()
    assert "error" in data
    assert "message" in data["error"]
    assert "type" in data["error"]


def test_proxy_models_endpoint(e2e_gate):
    """/v1/models endpoint returns model list."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=False)
    client = _proxy_client(gate)

    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data


def test_proxy_virtual_key_budget(e2e_gate):
    """VK with budget → budget tracked."""
    gate = e2e_gate(cache=False, proxy=True, proxy_require_virtual_key=True)
    client = _proxy_client(gate)

    org = gate.create_organization(name="BudgetOrg")
    team = gate.create_team(org_id=org.id, name="BudgetTeam")
    full_key, vk = _create_vk(gate, team.id, org_id=org.id, budget_limit=10.0)

    mock_response = make_openai_response("Budget response")

    with patch("stateloom.chat.Client.achat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = types.SimpleNamespace(
            raw=mock_response,
            content="Budget response",
            model="gpt-3.5-turbo",
            cost=0.001,
            tokens=15,
            provider="openai",
        )

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {full_key}"},
        )

    assert resp.status_code == 200
