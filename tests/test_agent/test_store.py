"""Tests for agent persistence in both SQLite and Memory stores."""

from __future__ import annotations

import pytest

from stateloom.agent.models import Agent, AgentVersion
from stateloom.core.types import AgentStatus
from stateloom.store.memory_store import MemoryStore
from stateloom.store.sqlite_store import SQLiteStore


@pytest.fixture
def memory_store():
    return MemoryStore()


@pytest.fixture
def sqlite_store(tmp_path):
    return SQLiteStore(str(tmp_path / "test.db"))


def _make_agent(slug="test-bot", team_id="team-1", org_id="org-1") -> Agent:
    return Agent(
        slug=slug,
        team_id=team_id,
        org_id=org_id,
        name="Test Bot",
        description="A test bot",
        status=AgentStatus.ACTIVE,
    )


def _make_version(agent_id: str, version_number: int = 1) -> AgentVersion:
    return AgentVersion(
        agent_id=agent_id,
        version_number=version_number,
        model="gpt-4o",
        system_prompt="You are helpful.",
        request_overrides={"temperature": 0.5},
        budget_per_session=10.0,
        created_by="tester",
    )


class TestMemoryStoreAgent:
    """Agent CRUD on MemoryStore."""

    def test_save_and_get(self, memory_store):
        agent = _make_agent()
        memory_store.save_agent(agent)
        result = memory_store.get_agent(agent.id)
        assert result is not None
        assert result.slug == "test-bot"
        assert result.team_id == "team-1"

    def test_get_by_slug(self, memory_store):
        agent = _make_agent()
        memory_store.save_agent(agent)
        result = memory_store.get_agent_by_slug("test-bot", "team-1")
        assert result is not None
        assert result.id == agent.id

    def test_get_by_slug_wrong_team(self, memory_store):
        agent = _make_agent()
        memory_store.save_agent(agent)
        result = memory_store.get_agent_by_slug("test-bot", "team-other")
        assert result is None

    def test_list_agents(self, memory_store):
        a1 = _make_agent("bot-a", "team-1")
        a2 = _make_agent("bot-b", "team-1")
        a3 = _make_agent("bot-c", "team-2")
        memory_store.save_agent(a1)
        memory_store.save_agent(a2)
        memory_store.save_agent(a3)

        all_agents = memory_store.list_agents()
        assert len(all_agents) == 3

        team1 = memory_store.list_agents(team_id="team-1")
        assert len(team1) == 2

    def test_list_agents_by_status(self, memory_store):
        a1 = _make_agent("bot-a", "team-1")
        a2 = _make_agent("bot-b", "team-1")
        a2.status = AgentStatus.ARCHIVED
        memory_store.save_agent(a1)
        memory_store.save_agent(a2)

        active = memory_store.list_agents(status="active")
        assert len(active) == 1
        assert active[0].slug == "bot-a"

    def test_version_crud(self, memory_store):
        agent = _make_agent()
        memory_store.save_agent(agent)

        v1 = _make_version(agent.id, 1)
        v2 = _make_version(agent.id, 2)
        memory_store.save_agent_version(v1)
        memory_store.save_agent_version(v2)

        result = memory_store.get_agent_version(v1.id)
        assert result is not None
        assert result.version_number == 1

        versions = memory_store.list_agent_versions(agent.id)
        assert len(versions) == 2
        assert versions[0].version_number == 2  # newest first

    def test_next_version_number(self, memory_store):
        agent = _make_agent()
        memory_store.save_agent(agent)

        assert memory_store.get_next_version_number(agent.id) == 1

        v1 = _make_version(agent.id, 1)
        memory_store.save_agent_version(v1)
        assert memory_store.get_next_version_number(agent.id) == 2


class TestSQLiteStoreAgent:
    """Agent CRUD on SQLiteStore."""

    def test_save_and_get(self, sqlite_store):
        agent = _make_agent()
        sqlite_store.save_agent(agent)
        result = sqlite_store.get_agent(agent.id)
        assert result is not None
        assert result.slug == "test-bot"
        assert result.team_id == "team-1"
        assert result.org_id == "org-1"
        assert result.status == AgentStatus.ACTIVE

    def test_get_by_slug(self, sqlite_store):
        agent = _make_agent()
        sqlite_store.save_agent(agent)
        result = sqlite_store.get_agent_by_slug("test-bot", "team-1")
        assert result is not None
        assert result.id == agent.id

    def test_get_by_slug_wrong_team(self, sqlite_store):
        agent = _make_agent()
        sqlite_store.save_agent(agent)
        result = sqlite_store.get_agent_by_slug("test-bot", "team-other")
        assert result is None

    def test_slug_uniqueness_per_team(self, sqlite_store):
        a1 = _make_agent("bot", "team-1")
        sqlite_store.save_agent(a1)

        # Same slug, different team should work
        a2 = _make_agent("bot", "team-2")
        sqlite_store.save_agent(a2)
        assert sqlite_store.get_agent_by_slug("bot", "team-2") is not None

    def test_list_agents_filtered(self, sqlite_store):
        a1 = _make_agent("bot-a", "team-1")
        a2 = _make_agent("bot-b", "team-1")
        a3 = _make_agent("bot-c", "team-2")
        sqlite_store.save_agent(a1)
        sqlite_store.save_agent(a2)
        sqlite_store.save_agent(a3)

        all_agents = sqlite_store.list_agents()
        assert len(all_agents) == 3

        team1 = sqlite_store.list_agents(team_id="team-1")
        assert len(team1) == 2

    def test_version_roundtrip(self, sqlite_store):
        agent = _make_agent()
        sqlite_store.save_agent(agent)

        v = _make_version(agent.id, 1)
        sqlite_store.save_agent_version(v)

        result = sqlite_store.get_agent_version(v.id)
        assert result is not None
        assert result.model == "gpt-4o"
        assert result.system_prompt == "You are helpful."
        assert result.request_overrides == {"temperature": 0.5}
        assert result.budget_per_session == 10.0
        assert result.created_by == "tester"

    def test_list_versions_newest_first(self, sqlite_store):
        agent = _make_agent()
        sqlite_store.save_agent(agent)

        for i in range(1, 4):
            v = _make_version(agent.id, i)
            sqlite_store.save_agent_version(v)

        versions = sqlite_store.list_agent_versions(agent.id)
        assert len(versions) == 3
        assert versions[0].version_number == 3

    def test_next_version_number(self, sqlite_store):
        agent = _make_agent()
        sqlite_store.save_agent(agent)

        assert sqlite_store.get_next_version_number(agent.id) == 1

        v1 = _make_version(agent.id, 1)
        sqlite_store.save_agent_version(v1)
        assert sqlite_store.get_next_version_number(agent.id) == 2

        v2 = _make_version(agent.id, 2)
        sqlite_store.save_agent_version(v2)
        assert sqlite_store.get_next_version_number(agent.id) == 3

    def test_agent_not_found(self, sqlite_store):
        assert sqlite_store.get_agent("nonexistent") is None

    def test_version_not_found(self, sqlite_store):
        assert sqlite_store.get_agent_version("nonexistent") is None

    def test_metadata_roundtrip(self, sqlite_store):
        agent = _make_agent()
        agent.metadata = {"key": "value", "nested": {"a": 1}}
        sqlite_store.save_agent(agent)

        result = sqlite_store.get_agent(agent.id)
        assert result.metadata == {"key": "value", "nested": {"a": 1}}
