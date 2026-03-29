"""Tests for the agent proxy endpoint (/v1/agents/{ref}/chat/completions)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.agent.models import Agent, AgentVersion
from stateloom.core.config import StateLoomConfig
from stateloom.core.types import AgentStatus
from stateloom.proxy.router import create_proxy_router
from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore


@dataclass
class MockPricing:
    _prices: dict = field(
        default_factory=lambda: {
            "gpt-4o": MagicMock(input_per_token=0.000005, output_per_token=0.000015),
        }
    )


def _make_gate(*, require_virtual_key: bool = True) -> MagicMock:
    """Create a mock Gate."""
    gate = MagicMock()
    gate.store = MemoryStore()
    gate.config = StateLoomConfig(
        proxy_enabled=True,
        proxy_require_virtual_key=require_virtual_key,
        console_output=False,
        dashboard=False,
    )
    gate.pricing = MockPricing()
    gate._metrics_collector = None
    return gate


def _make_app(gate: MagicMock) -> FastAPI:
    app = FastAPI()
    router = create_proxy_router(gate)
    app.include_router(router, prefix="/v1")
    return app


def _setup_vk(gate: MagicMock, agent_ids: list[str] | None = None) -> str:
    """Create a virtual key and return the full key string."""
    full_key, key_hash = generate_virtual_key()
    vk = VirtualKey(
        id=make_virtual_key_id(),
        key_hash=key_hash,
        key_preview=make_key_preview(full_key),
        team_id="team-1",
        org_id="org-1",
        name="test-key",
        agent_ids=agent_ids or [],
    )
    gate.store.save_virtual_key(vk)
    return full_key


def _setup_agent(
    gate: MagicMock,
    slug: str = "legal-bot",
    status: AgentStatus = AgentStatus.ACTIVE,
    model: str = "gpt-4o",
    system_prompt: str = "You are a legal assistant.",
    request_overrides: dict | None = None,
    budget_per_session: float | None = None,
) -> tuple[Agent, AgentVersion]:
    """Create and persist an agent with an active version."""
    agent = Agent(
        slug=slug,
        team_id="team-1",
        org_id="org-1",
        name="Legal Bot",
        status=status,
    )
    version = AgentVersion(
        agent_id=agent.id,
        version_number=1,
        model=model,
        system_prompt=system_prompt,
        request_overrides=request_overrides or {},
        budget_per_session=budget_per_session,
    )
    agent.active_version_id = version.id
    gate.store.save_agent(agent)
    gate.store.save_agent_version(version)
    return agent, version


def _make_mock_raw(content: str = "Hello", model: str = "gpt-4o") -> SimpleNamespace:
    """Create a mock raw response that serializes cleanly (no model_dump)."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        model=model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="chatcmpl-test",
    )


def _make_mock_client(raw_response: SimpleNamespace) -> MagicMock:
    """Create a mock Client instance with proper async context manager."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.raw = raw_response
    mock_instance.achat = AsyncMock(return_value=mock_response)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    mock_instance._session = MagicMock()
    mock_instance._session.metadata = {}
    return mock_instance


class TestAgentProxyAuth:
    """Authentication tests for the agent proxy endpoint."""

    def test_401_without_auth(self):
        gate = _make_gate()
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/legal-bot/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 401

    def test_400_no_messages(self):
        gate = _make_gate()
        _setup_agent(gate)
        full_key = _setup_vk(gate)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/legal-bot/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"messages": []},
        )
        assert resp.status_code == 400


class TestAgentProxyResolution:
    """Agent resolution tests."""

    def test_404_unknown_agent(self):
        gate = _make_gate()
        full_key = _setup_vk(gate)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/nonexistent/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 404

    def test_403_paused_agent(self):
        gate = _make_gate()
        _setup_agent(gate, status=AgentStatus.PAUSED)
        full_key = _setup_vk(gate)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/legal-bot/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 403
        assert "paused" in resp.json()["error"]["message"]

    def test_410_archived_agent(self):
        gate = _make_gate()
        _setup_agent(gate, status=AgentStatus.ARCHIVED)
        full_key = _setup_vk(gate)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/legal-bot/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 410

    def test_403_vk_agent_ids_scoping(self):
        gate = _make_gate()
        agent, _ = _setup_agent(gate)
        # VK restricted to a different agent
        full_key = _setup_vk(gate, agent_ids=["other-agent-id"])
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/legal-bot/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 403
        assert "not accessible" in resp.json()["error"]["message"]


class TestAgentProxyOverrides:
    """Override application tests via mocked Client."""

    @patch("stateloom.proxy.router.Client")
    def test_model_override_applied(self, MockClient):
        """Agent's model should override any client-sent model."""
        gate = _make_gate()
        _setup_agent(gate, model="gpt-4o")
        full_key = _setup_vk(gate)

        mock_instance = _make_mock_client(_make_mock_raw("Hello", "gpt-4o"))
        MockClient.return_value = mock_instance

        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/legal-bot/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={
                "model": "gpt-3.5-turbo",  # should be ignored
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert resp.status_code == 200
        # Verify the model passed to achat was the agent's model
        call_kwargs = mock_instance.achat.call_args
        assert call_kwargs.kwargs.get("model") or call_kwargs.args[0] == "gpt-4o"

    @patch("stateloom.proxy.router.Client")
    def test_system_prompt_applied(self, MockClient):
        """Agent's system prompt should be prepended."""
        gate = _make_gate()
        _setup_agent(gate, system_prompt="Be concise.")
        full_key = _setup_vk(gate)

        mock_instance = _make_mock_client(_make_mock_raw("Hi", "gpt-4o"))
        MockClient.return_value = mock_instance

        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/legal-bot/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200

        # Verify system prompt was added
        call_kwargs = mock_instance.achat.call_args
        messages = call_kwargs.kwargs.get("messages", [])
        assert messages[0]["role"] == "system"
        assert "Be concise." in messages[0]["content"]

    @patch("stateloom.proxy.router.Client")
    def test_session_metadata_injected(self, MockClient):
        """Agent metadata should be set on the session."""
        gate = _make_gate()
        agent, version = _setup_agent(gate)
        full_key = _setup_vk(gate)

        mock_instance = _make_mock_client(_make_mock_raw("Hi", "gpt-4o"))
        session_metadata = {}
        mock_instance._session.metadata = session_metadata
        MockClient.return_value = mock_instance

        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/agents/legal-bot/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200

        # Verify agent metadata was injected
        assert session_metadata["agent_id"] == agent.id
        assert session_metadata["agent_slug"] == "legal-bot"
        assert session_metadata["agent_version_id"] == version.id
        assert session_metadata["agent_version_number"] == 1
