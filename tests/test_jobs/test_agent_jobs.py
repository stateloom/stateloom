"""Tests for agent parameter support in async jobs."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from stateloom.agent.models import Agent, AgentVersion
from stateloom.core.job import Job
from stateloom.core.types import AgentStatus, JobStatus
from stateloom.jobs.processor import JobProcessor
from stateloom.jobs.queue import InProcessJobQueue
from stateloom.store.memory_store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = "You are a sentiment classifier. Reply POSITIVE, NEGATIVE, or NEUTRAL."


def _setup_agent(store, slug="sentiment-bot", model="gpt-4o"):
    """Create and persist an agent with an active version."""
    agent = Agent(
        slug=slug,
        team_id="team-1",
        org_id="org-1",
        name="Sentiment Bot",
        status=AgentStatus.ACTIVE,
    )
    version = AgentVersion(
        agent_id=agent.id,
        version_number=1,
        model=model,
        system_prompt=AGENT_SYSTEM_PROMPT,
        request_overrides={"temperature": 0.1},
        budget_per_session=5.0,
    )
    agent.active_version_id = version.id
    store.save_agent(agent)
    store.save_agent_version(version)
    return agent, version


def _make_mock_gate(store=None):
    """Create a minimal mock Gate backed by a real MemoryStore."""
    gate = MagicMock()
    gate.store = store or MemoryStore()
    gate.config = MagicMock()
    gate.config.async_jobs_enabled = True
    gate.config.async_jobs_default_ttl = 3600
    gate.config.async_jobs_webhook_timeout = 30.0
    gate.config.async_jobs_webhook_retries = 3
    gate.config.async_jobs_webhook_secret = ""
    # Mock session manager
    session = MagicMock()
    session.id = "test-session"
    session.agent_id = ""
    session.agent_slug = ""
    session.agent_version_id = ""
    session.agent_version_number = 0
    session.agent_name = ""
    gate.session_manager.create.return_value = session
    # Mock pipeline
    gate.pipeline.execute_sync.return_value = {"choices": [{"message": {"content": "POSITIVE"}}]}
    return gate


def _make_pending_job(**kwargs) -> Job:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        status=JobStatus.PENDING,
    )
    defaults.update(kwargs)
    return Job(**defaults)


# ---------------------------------------------------------------------------
# Gate.submit_job() with agent
# ---------------------------------------------------------------------------


class TestGateSubmitJobWithAgent:
    def test_agent_resolves_model_and_messages(self):
        """submit_job(agent=...) resolves agent model and prepends system prompt."""
        from stateloom.gate import Gate

        store = MemoryStore()
        agent, version = _setup_agent(store, model="claude-haiku-4-5-20251001")

        gate = MagicMock()
        gate.store = store
        gate.config = MagicMock()
        gate.config.async_jobs_enabled = True
        gate.config.async_jobs_default_ttl = 3600

        job = Gate.submit_job(
            gate,
            agent="sentiment-bot",
            messages=[{"role": "user", "content": "Great product!"}],
        )

        assert job.model == "claude-haiku-4-5-20251001"
        # System prompt should be prepended by apply_agent_overrides
        system_msgs = [m for m in job.messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        assert AGENT_SYSTEM_PROMPT in system_msgs[0]["content"]

    def test_agent_metadata_stored_on_job(self):
        """submit_job(agent=...) stores agent metadata for the processor."""
        from stateloom.gate import Gate

        store = MemoryStore()
        agent, version = _setup_agent(store)

        gate = MagicMock()
        gate.store = store
        gate.config = MagicMock()
        gate.config.async_jobs_enabled = True
        gate.config.async_jobs_default_ttl = 3600

        job = Gate.submit_job(
            gate,
            agent="sentiment-bot",
            messages=[{"role": "user", "content": "test"}],
        )

        assert job.metadata["agent_id"] == agent.id
        assert job.metadata["agent_slug"] == "sentiment-bot"
        assert job.metadata["agent_version_id"] == version.id
        assert job.metadata["agent_version_number"] == 1

    def test_no_agent_preserves_default_behavior(self):
        """Without agent param, submit_job works as before."""
        from stateloom.gate import Gate

        gate = MagicMock()
        gate.store = MagicMock()
        gate.config = MagicMock()
        gate.config.async_jobs_enabled = True
        gate.config.async_jobs_default_ttl = 3600

        job = Gate.submit_job(
            gate,
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert job.model == "gpt-4o"
        assert "agent_id" not in job.metadata

    def test_agent_request_overrides_merged(self):
        """Agent request_overrides are merged into job request_kwargs."""
        from stateloom.gate import Gate

        store = MemoryStore()
        agent, version = _setup_agent(store)

        gate = MagicMock()
        gate.store = store
        gate.config = MagicMock()
        gate.config.async_jobs_enabled = True
        gate.config.async_jobs_default_ttl = 3600

        job = Gate.submit_job(
            gate,
            agent="sentiment-bot",
            messages=[{"role": "user", "content": "test"}],
            request_kwargs={"top_p": 0.9},
        )

        # Both user's top_p and agent's temperature should be present
        assert job.request_kwargs.get("top_p") == 0.9
        assert job.request_kwargs.get("temperature") == 0.1

    def test_agent_not_found_raises(self):
        """submit_job(agent=...) raises when agent doesn't exist."""
        from stateloom.core.errors import StateLoomError
        from stateloom.gate import Gate

        store = MemoryStore()

        gate = MagicMock()
        gate.store = store
        gate.config = MagicMock()
        gate.config.async_jobs_enabled = True
        gate.config.async_jobs_default_ttl = 3600

        with pytest.raises(StateLoomError, match="not found"):
            Gate.submit_job(
                gate,
                agent="nonexistent",
                messages=[{"role": "user", "content": "test"}],
            )

    def test_user_metadata_preserved_alongside_agent(self):
        """User-provided metadata is not overwritten by agent metadata."""
        from stateloom.gate import Gate

        store = MemoryStore()
        _setup_agent(store)

        gate = MagicMock()
        gate.store = store
        gate.config = MagicMock()
        gate.config.async_jobs_enabled = True
        gate.config.async_jobs_default_ttl = 3600

        job = Gate.submit_job(
            gate,
            agent="sentiment-bot",
            messages=[{"role": "user", "content": "test"}],
            metadata={"custom_field": "my-value"},
        )

        assert job.metadata["custom_field"] == "my-value"
        assert job.metadata["agent_id"] != ""


# ---------------------------------------------------------------------------
# JobProcessor — agent session fields from metadata
# ---------------------------------------------------------------------------


class TestProcessorAgentSessionFields:
    def test_processor_sets_agent_session_fields(self):
        """When job metadata has agent_id, processor sets session fields."""
        store = MemoryStore()
        gate = _make_mock_gate(store=store)

        # Track what gets set on the session
        session = gate.session_manager.create.return_value

        job = _make_pending_job(
            metadata={
                "agent_id": "agt-123",
                "agent_slug": "sentiment-bot",
                "agent_version_id": "agv-456",
                "agent_version_number": 2,
            }
        )
        store.save_job(job)

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=1)
        try:
            processor.start()
            time.sleep(2.0)
        finally:
            processor.shutdown()

        assert session.agent_id == "agt-123"
        assert session.agent_slug == "sentiment-bot"
        assert session.agent_version_id == "agv-456"
        assert session.agent_version_number == 2
        assert session.agent_name == "sentiment-bot"

    def test_processor_no_agent_fields_without_metadata(self):
        """When job has no agent metadata, session fields stay default."""
        store = MemoryStore()
        gate = _make_mock_gate(store=store)
        session = gate.session_manager.create.return_value

        job = _make_pending_job(metadata={})
        store.save_job(job)

        processor = JobProcessor(gate, queue=InProcessJobQueue(gate.store), max_workers=1)
        try:
            processor.start()
            time.sleep(2.0)
        finally:
            processor.shutdown()

        assert session.agent_id == ""
