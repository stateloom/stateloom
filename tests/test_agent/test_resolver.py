"""Tests for agent resolution and override application."""

from __future__ import annotations

import pytest

from stateloom.agent.models import Agent, AgentVersion
from stateloom.agent.resolver import (
    AgentResolutionError,
    apply_agent_overrides,
    resolve_agent,
)
from stateloom.core.types import AgentStatus
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


def _setup_agent(store, slug="test-bot", team_id="team-1", status=AgentStatus.ACTIVE):
    """Create and persist an agent with an active version."""
    agent = Agent(
        slug=slug,
        team_id=team_id,
        org_id="org-1",
        name="Test Bot",
        status=status,
    )
    version = AgentVersion(
        agent_id=agent.id,
        version_number=1,
        model="gpt-4o",
        system_prompt="You are a helpful assistant.",
        request_overrides={"temperature": 0.2},
        budget_per_session=5.0,
    )
    agent.active_version_id = version.id
    store.save_agent(agent)
    store.save_agent_version(version)
    return agent, version


class TestResolveAgent:
    """resolve_agent() tests."""

    def test_resolve_by_slug(self, store):
        agent, version = _setup_agent(store)
        result_agent, result_version = resolve_agent(store, "test-bot", "team-1")
        assert result_agent.id == agent.id
        assert result_version.id == version.id

    def test_resolve_by_id(self, store):
        agent, version = _setup_agent(store)
        result_agent, result_version = resolve_agent(store, agent.id, "team-1")
        assert result_agent.id == agent.id
        assert result_version.id == version.id

    def test_not_found(self, store):
        with pytest.raises(AgentResolutionError) as exc_info:
            resolve_agent(store, "nonexistent", "team-1")
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.message

    def test_paused_agent(self, store):
        _setup_agent(store, status=AgentStatus.PAUSED)
        with pytest.raises(AgentResolutionError) as exc_info:
            resolve_agent(store, "test-bot", "team-1")
        assert exc_info.value.status_code == 403
        assert "paused" in exc_info.value.message

    def test_archived_agent(self, store):
        _setup_agent(store, status=AgentStatus.ARCHIVED)
        with pytest.raises(AgentResolutionError) as exc_info:
            resolve_agent(store, "test-bot", "team-1")
        assert exc_info.value.status_code == 410
        assert "archived" in exc_info.value.message

    def test_no_active_version(self, store):
        agent = Agent(
            slug="no-version",
            team_id="team-1",
            status=AgentStatus.ACTIVE,
        )
        store.save_agent(agent)
        with pytest.raises(AgentResolutionError) as exc_info:
            resolve_agent(store, "no-version", "team-1")
        assert exc_info.value.status_code == 404
        assert "no active version" in exc_info.value.message

    def test_vk_agent_ids_scoping_allowed(self, store):
        agent, version = _setup_agent(store)
        result_agent, _ = resolve_agent(store, "test-bot", "team-1", vk_agent_ids=[agent.id])
        assert result_agent.id == agent.id

    def test_vk_agent_ids_scoping_denied(self, store):
        _setup_agent(store)
        with pytest.raises(AgentResolutionError) as exc_info:
            resolve_agent(store, "test-bot", "team-1", vk_agent_ids=["other-id"])
        assert exc_info.value.status_code == 403
        assert "not accessible" in exc_info.value.message

    def test_vk_agent_ids_empty_allows_all(self, store):
        agent, version = _setup_agent(store)
        # Empty list = no restriction
        result_agent, _ = resolve_agent(store, "test-bot", "team-1", vk_agent_ids=[])
        assert result_agent.id == agent.id


class TestApplyAgentOverrides:
    """apply_agent_overrides() tests."""

    def test_model_override(self):
        version = AgentVersion(model="gpt-4o")
        model, messages, kwargs = apply_agent_overrides(
            version,
            [{"role": "user", "content": "Hello"}],
            {"model": "gpt-3.5-turbo"},  # client model should be ignored
        )
        assert model == "gpt-4o"

    def test_system_prompt_no_existing(self):
        version = AgentVersion(
            model="gpt-4o",
            system_prompt="You are a legal assistant.",
        )
        _, messages, _ = apply_agent_overrides(
            version,
            [{"role": "user", "content": "Hello"}],
            {},
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a legal assistant."
        assert messages[1]["role"] == "user"

    def test_system_prompt_prepend_to_existing(self):
        version = AgentVersion(
            model="gpt-4o",
            system_prompt="Agent guardrails here.",
        )
        _, messages, _ = apply_agent_overrides(
            version,
            [
                {"role": "system", "content": "User context."},
                {"role": "user", "content": "Hello"},
            ],
            {},
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Agent guardrails here.\n\nUser context."
        assert messages[1]["role"] == "user"

    def test_empty_agent_prompt_no_modification(self):
        version = AgentVersion(model="gpt-4o", system_prompt="")
        original = [{"role": "user", "content": "Hello"}]
        _, messages, _ = apply_agent_overrides(version, original, {})
        assert len(messages) == 1

    def test_request_overrides_applied(self):
        version = AgentVersion(
            model="gpt-4o",
            request_overrides={"temperature": 0.1, "max_tokens": 100},
        )
        _, _, kwargs = apply_agent_overrides(
            version,
            [{"role": "user", "content": "Hello"}],
            {"temperature": 0.9},  # should be overridden by agent
        )
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_tokens"] == 100

    def test_body_kwargs_extracted(self):
        version = AgentVersion(model="gpt-4o")
        _, _, kwargs = apply_agent_overrides(
            version,
            [{"role": "user", "content": "Hello"}],
            {"temperature": 0.5, "top_p": 0.9, "unknown_field": "ignored"},
        )
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert "unknown_field" not in kwargs

    def test_original_messages_not_mutated(self):
        version = AgentVersion(
            model="gpt-4o",
            system_prompt="Agent prompt.",
        )
        original = [{"role": "user", "content": "Hello"}]
        _, messages, _ = apply_agent_overrides(version, original, {})
        # Original should not be mutated
        assert len(original) == 1
        assert len(messages) == 2
