"""Tests for dashboard agent CRUD endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.agent.models import Agent, AgentVersion
from stateloom.core.types import AgentStatus
from stateloom.dashboard.api import create_api_router
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def mock_gate():
    """Create a mock Gate backed by a real MemoryStore."""
    gate = MagicMock()
    gate.store = MemoryStore()
    gate.config = MagicMock()
    gate.config.async_jobs_enabled = False

    # Wire agent methods to use the real store
    def _create_agent(
        slug,
        team_id,
        *,
        name="",
        description="",
        model="",
        system_prompt="",
        request_overrides=None,
        budget_per_session=None,
        metadata=None,
        created_by="",
        org_id="",
        compliance_profile_json="",
    ):
        agent = Agent(
            slug=slug,
            team_id=team_id,
            org_id=org_id,
            name=name,
            description=description,
            status=AgentStatus.ACTIVE,
            metadata=metadata or {},
        )
        version = AgentVersion(
            agent_id=agent.id,
            version_number=1,
            model=model,
            system_prompt=system_prompt,
            request_overrides=request_overrides or {},
            budget_per_session=budget_per_session,
            created_by=created_by,
            metadata={},
        )
        agent.active_version_id = version.id
        gate.store.save_agent(agent)
        gate.store.save_agent_version(version)
        gate._agents_cache = getattr(gate, "_agents_cache_real", {})
        gate._agents_cache[agent.id] = agent
        return agent

    def _get_agent(agent_id):
        return gate.store.get_agent(agent_id)

    def _get_agent_by_slug(slug, team_id):
        return gate.store.get_agent_by_slug(slug, team_id)

    def _list_agents(team_id=None, org_id=None, status=None):
        return gate.store.list_agents(team_id=team_id, org_id=org_id, status=status)

    def _update_agent(agent_id, *, name=None, description=None, status=None, metadata=None):
        from datetime import datetime, timezone

        agent = gate.store.get_agent(agent_id)
        if agent is None:
            raise Exception(f"Agent not found: {agent_id}")
        if name is not None:
            agent.name = name
        if description is not None:
            agent.description = description
        if status is not None:
            agent.status = AgentStatus(status)
        if metadata is not None:
            agent.metadata = metadata
        agent.updated_at = datetime.now(timezone.utc)
        gate.store.save_agent(agent)
        return agent

    def _create_agent_version(
        agent_id,
        *,
        model="",
        system_prompt="",
        request_overrides=None,
        budget_per_session=None,
        metadata=None,
        created_by="",
        compliance_profile_json="",
    ):
        agent = gate.store.get_agent(agent_id)
        if agent is None:
            raise Exception(f"Agent not found: {agent_id}")
        version_number = gate.store.get_next_version_number(agent_id)
        version = AgentVersion(
            agent_id=agent_id,
            version_number=version_number,
            model=model,
            system_prompt=system_prompt,
            request_overrides=request_overrides or {},
            budget_per_session=budget_per_session,
            metadata=metadata or {},
            created_by=created_by,
        )
        gate.store.save_agent_version(version)
        return version

    def _activate_agent_version(agent_id, version_id):
        from datetime import datetime, timezone

        agent = gate.store.get_agent(agent_id)
        if agent is None:
            raise Exception(f"Agent not found: {agent_id}")
        version = gate.store.get_agent_version(version_id)
        if version is None or version.agent_id != agent_id:
            raise Exception(f"Version not found: {version_id}")
        agent.active_version_id = version_id
        agent.updated_at = datetime.now(timezone.utc)
        gate.store.save_agent(agent)
        return agent

    def _archive_agent(agent_id):
        from datetime import datetime, timezone

        agent = gate.store.get_agent(agent_id)
        if agent is None:
            raise Exception(f"Agent not found: {agent_id}")
        agent.status = AgentStatus.ARCHIVED
        agent.updated_at = datetime.now(timezone.utc)
        gate.store.save_agent(agent)
        return agent

    gate.create_agent = _create_agent
    gate.get_agent = _get_agent
    gate.get_agent_by_slug = _get_agent_by_slug
    gate.list_agents = _list_agents
    gate.update_agent = _update_agent
    gate.create_agent_version = _create_agent_version
    gate.activate_agent_version = _activate_agent_version
    gate.archive_agent = _archive_agent

    return gate


@pytest.fixture
def client(mock_gate):
    app = FastAPI()
    router = create_api_router(mock_gate)
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


class TestCreateAgent:
    """POST /api/agents"""

    def test_create_agent_with_initial_version(self, client):
        resp = client.post(
            "/api/v1/agents",
            json={
                "slug": "legal-bot",
                "team_id": "team-1",
                "name": "Legal Bot",
                "model": "gpt-4o",
                "system_prompt": "You are a legal assistant.",
                "request_overrides": {"temperature": 0.2},
                "budget_per_session": 5.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"]["slug"] == "legal-bot"
        assert data["agent"]["team_id"] == "team-1"
        assert data["agent"]["name"] == "Legal Bot"
        assert data["agent"]["status"] == "active"
        assert data["agent"]["active_version_id"] is not None
        assert data["version"] is not None
        assert data["version"]["version_number"] == 1
        assert data["version"]["model"] == "gpt-4o"
        assert data["version"]["system_prompt"] == "You are a legal assistant."
        assert data["version"]["request_overrides"] == {"temperature": 0.2}
        assert data["version"]["budget_per_session"] == 5.0

    def test_create_agent_minimal(self, client):
        resp = client.post(
            "/api/v1/agents",
            json={
                "slug": "min-bot",
                "team_id": "team-1",
                "model": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"]["slug"] == "min-bot"
        assert data["version"]["model"] == "gpt-4o"

    def test_create_agent_invalid_slug(self, client):
        resp = client.post(
            "/api/v1/agents",
            json={
                "slug": "AB",  # too short and uppercase
                "team_id": "team-1",
                "model": "gpt-4o",
            },
        )
        assert resp.status_code == 422  # Pydantic validation


class TestListAgents:
    """GET /api/agents"""

    def test_list_all(self, client, mock_gate):
        mock_gate.create_agent("bot-a", "team-1", model="gpt-4o")
        mock_gate.create_agent("bot-b", "team-1", model="gpt-4o")
        mock_gate.create_agent("bot-c", "team-2", model="gpt-4o")

        resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_list_by_team(self, client, mock_gate):
        mock_gate.create_agent("bot-a", "team-1", model="gpt-4o")
        mock_gate.create_agent("bot-b", "team-1", model="gpt-4o")
        mock_gate.create_agent("bot-c", "team-2", model="gpt-4o")

        resp = client.get("/api/v1/agents", params={"team_id": "team-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_list_by_status(self, client, mock_gate):
        mock_gate.create_agent("bot-a", "team-1", model="gpt-4o")
        agent_b = mock_gate.create_agent("bot-b", "team-1", model="gpt-4o")
        mock_gate.archive_agent(agent_b.id)

        resp = client.get("/api/v1/agents", params={"status": "active"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["agents"][0]["slug"] == "bot-a"


class TestGetAgent:
    """GET /api/agents/{agent_ref}"""

    def test_get_by_id(self, client, mock_gate):
        agent = mock_gate.create_agent(
            "legal-bot", "team-1", model="gpt-4o", system_prompt="Be helpful."
        )
        resp = client.get(f"/api/v1/agents/{agent.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "legal-bot"
        assert data["active_version"] is not None
        assert data["active_version"]["model"] == "gpt-4o"

    def test_get_by_slug(self, client, mock_gate):
        mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o")
        resp = client.get("/api/v1/agents/legal-bot", params={"team_id": "team-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "legal-bot"

    def test_get_slug_without_team_id(self, client, mock_gate):
        """Slug lookup without team_id returns 404."""
        mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o")
        resp = client.get("/api/v1/agents/legal-bot")
        assert resp.status_code == 404

    def test_get_not_found(self, client):
        resp = client.get("/api/v1/agents/agt-nonexistent")
        assert resp.status_code == 404


class TestUpdateAgent:
    """PATCH /api/agents/{agent_ref}"""

    def test_update_name(self, client, mock_gate):
        agent = mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o", name="Old Name")
        resp = client.patch(
            f"/api/v1/agents/{agent.id}",
            json={"name": "New Name"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "New Name"

    def test_update_status(self, client, mock_gate):
        agent = mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o")
        resp = client.patch(
            f"/api/v1/agents/{agent.id}",
            json={"status": "paused"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_update_not_found(self, client):
        resp = client.patch(
            "/api/v1/agents/agt-nonexistent",
            json={"name": "x"},
        )
        assert resp.status_code == 404


class TestDeleteAgent:
    """DELETE /api/agents/{agent_ref}"""

    def test_delete_archives(self, client, mock_gate):
        agent = mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o")
        resp = client.delete(f"/api/v1/agents/{agent.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "archived"

    def test_delete_not_found(self, client):
        resp = client.delete("/api/v1/agents/agt-nonexistent")
        assert resp.status_code == 404


class TestCreateAgentVersion:
    """POST /api/agents/{agent_ref}/versions"""

    def test_create_new_version(self, client, mock_gate):
        agent = mock_gate.create_agent(
            "legal-bot", "team-1", model="gpt-4o", system_prompt="v1 prompt"
        )
        resp = client.post(
            f"/api/v1/agents/{agent.id}/versions",
            json={
                "model": "gpt-4o-mini",
                "system_prompt": "v2 prompt",
                "request_overrides": {"temperature": 0.5},
                "created_by": "alice",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["version_number"] == 2
        assert data["model"] == "gpt-4o-mini"
        assert data["system_prompt"] == "v2 prompt"
        assert data["created_by"] == "alice"

    def test_create_version_not_found(self, client):
        resp = client.post(
            "/api/v1/agents/agt-nonexistent/versions",
            json={"model": "gpt-4o"},
        )
        assert resp.status_code == 404


class TestListAgentVersions:
    """GET /api/agents/{agent_ref}/versions"""

    def test_list_versions(self, client, mock_gate):
        agent = mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o")
        mock_gate.create_agent_version(agent.id, model="gpt-4o-mini", system_prompt="v2")
        mock_gate.create_agent_version(agent.id, model="gpt-4o", system_prompt="v3")

        resp = client.get(f"/api/v1/agents/{agent.id}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3  # v1 + v2 + v3
        # Newest first
        assert data["versions"][0]["version_number"] == 3
        assert data["versions"][2]["version_number"] == 1

    def test_list_versions_not_found(self, client):
        resp = client.get("/api/v1/agents/agt-nonexistent/versions")
        assert resp.status_code == 404


class TestActivateAgentVersion:
    """PUT /api/agents/{agent_ref}/versions/{version_id}/activate"""

    def test_activate_rollback(self, client, mock_gate):
        agent = mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o", system_prompt="v1")
        v1_id = agent.active_version_id

        v2 = mock_gate.create_agent_version(agent.id, model="gpt-4o-mini", system_prompt="v2")
        mock_gate.activate_agent_version(agent.id, v2.id)

        # Rollback to v1
        resp = client.put(f"/api/v1/agents/{agent.id}/versions/{v1_id}/activate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_version_id"] == v1_id

    def test_activate_invalid_version(self, client, mock_gate):
        agent = mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o")
        resp = client.put(f"/api/v1/agents/{agent.id}/versions/agv-nonexistent/activate")
        assert resp.status_code == 404


class TestAgentSessions:
    """GET /api/agents/{agent_ref}/sessions"""

    def test_sessions_empty(self, client, mock_gate):
        agent = mock_gate.create_agent("legal-bot", "team-1", model="gpt-4o")
        resp = client.get(f"/api/v1/agents/{agent.id}/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["sessions"] == []

    def test_sessions_not_found(self, client):
        resp = client.get("/api/v1/agents/agt-nonexistent/sessions")
        assert resp.status_code == 404
