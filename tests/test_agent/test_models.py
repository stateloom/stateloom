"""Tests for agent data models and slug validation."""

from __future__ import annotations

from stateloom.agent.models import (
    Agent,
    AgentVersion,
    _make_agent_id,
    _make_agent_version_id,
    validate_slug,
)
from stateloom.core.types import AgentStatus


class TestSlugValidation:
    """Slug validation tests."""

    def test_valid_slugs(self):
        assert validate_slug("legal-bot") is True
        assert validate_slug("my-agent-v2") is True
        assert validate_slug("abc") is True
        assert validate_slug("a" * 64) is True
        assert validate_slug("a1b2c3") is True
        assert validate_slug("agent123") is True

    def test_invalid_slugs(self):
        assert validate_slug("") is False
        assert validate_slug("ab") is False  # too short
        assert validate_slug("-leading") is False
        assert validate_slug("trailing-") is False
        assert validate_slug("UPPERCASE") is False
        assert validate_slug("has spaces") is False
        assert validate_slug("special!chars") is False
        assert validate_slug("a" * 65) is False  # too long

    def test_hyphens_in_middle(self):
        assert validate_slug("a-b") is True
        assert validate_slug("my-long-agent-name") is True

    def test_numbers_only(self):
        assert validate_slug("123") is True
        assert validate_slug("1-2-3") is True


class TestAgentStatus:
    """AgentStatus enum tests."""

    def test_string_serialization(self):
        assert AgentStatus.ACTIVE == "active"
        assert AgentStatus.PAUSED == "paused"
        assert AgentStatus.ARCHIVED == "archived"

    def test_from_string(self):
        assert AgentStatus("active") == AgentStatus.ACTIVE
        assert AgentStatus("paused") == AgentStatus.PAUSED
        assert AgentStatus("archived") == AgentStatus.ARCHIVED


class TestAgentDataclass:
    """Agent dataclass tests."""

    def test_defaults(self):
        agent = Agent()
        assert agent.id.startswith("agt-")
        assert agent.slug == ""
        assert agent.status == AgentStatus.ACTIVE
        assert agent.active_version_id is None
        assert agent.metadata == {}

    def test_custom_fields(self):
        agent = Agent(
            slug="test-bot",
            team_id="team-1",
            org_id="org-1",
            name="Test Bot",
            description="A test bot",
            status=AgentStatus.PAUSED,
        )
        assert agent.slug == "test-bot"
        assert agent.team_id == "team-1"
        assert agent.status == AgentStatus.PAUSED


class TestAgentVersionDataclass:
    """AgentVersion dataclass tests."""

    def test_defaults(self):
        version = AgentVersion()
        assert version.id.startswith("agv-")
        assert version.version_number == 1
        assert version.model == ""
        assert version.system_prompt == ""
        assert version.request_overrides == {}
        assert version.budget_per_session is None

    def test_custom_fields(self):
        version = AgentVersion(
            agent_id="agt-test",
            version_number=3,
            model="gpt-4o",
            system_prompt="You are a legal assistant.",
            request_overrides={"temperature": 0.1},
            budget_per_session=5.0,
        )
        assert version.agent_id == "agt-test"
        assert version.version_number == 3
        assert version.model == "gpt-4o"
        assert version.budget_per_session == 5.0


class TestIdGeneration:
    """ID generation tests."""

    def test_agent_id_format(self):
        aid = _make_agent_id()
        assert aid.startswith("agt-")
        assert len(aid) == 16  # "agt-" + 12 hex chars

    def test_version_id_format(self):
        vid = _make_agent_version_id()
        assert vid.startswith("agv-")
        assert len(vid) == 16  # "agv-" + 12 hex chars

    def test_ids_unique(self):
        ids = {_make_agent_id() for _ in range(100)}
        assert len(ids) == 100
