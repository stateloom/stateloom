"""Tests for experiment ↔ agent integration."""

from __future__ import annotations

import asyncio

import pytest

from stateloom.agent.models import AgentVersion
from stateloom.core.config import StateLoomConfig
from stateloom.core.session import Session
from stateloom.core.types import ExperimentStatus
from stateloom.experiment.assigner import ExperimentAssigner
from stateloom.experiment.models import Experiment, VariantConfig
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.experiment import ExperimentMiddleware
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def agent_version(store):
    """A saved agent version for testing."""
    version = AgentVersion(
        id="agv-test123",
        agent_id="agt-test456",
        version_number=1,
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful support agent.",
        request_overrides={"temperature": 0.3, "max_tokens": 1024},
    )
    store.save_agent_version(version)
    return version


def _make_ctx(
    store=None,
    provider="openai",
    model="gpt-4o",
    metadata=None,
    request_kwargs=None,
):
    session = Session(id="test-session")
    session.metadata = metadata or {}
    return MiddlewareContext(
        session=session,
        config=StateLoomConfig(dashboard=False, console_output=False),
        provider=provider,
        model=model,
        request_kwargs=request_kwargs or {"model": model, "messages": []},
    )


async def _passthrough(ctx):
    return ctx.response


# --- VariantConfig agent_version_id ---


class TestVariantConfigAgentVersionId:
    def test_default_none(self):
        v = VariantConfig(name="control")
        assert v.agent_version_id is None

    def test_set_agent_version_id(self):
        v = VariantConfig(name="agent-variant", agent_version_id="agv-abc123")
        assert v.agent_version_id == "agv-abc123"

    def test_serializes_to_dict(self):
        v = VariantConfig(name="test", agent_version_id="agv-xyz")
        d = v.to_dict()
        assert d["agent_version_id"] == "agv-xyz"

    def test_deserializes_from_dict(self):
        d = {"name": "test", "agent_version_id": "agv-xyz"}
        v = VariantConfig.from_dict(d)
        assert v.agent_version_id == "agv-xyz"

    def test_roundtrip(self):
        original = VariantConfig(
            name="test",
            model="gpt-4o",
            agent_version_id="agv-abc",
            request_overrides={"temperature": 0.5},
        )
        restored = VariantConfig.from_dict(original.to_dict())
        assert restored.agent_version_id == original.agent_version_id
        assert restored.model == original.model


# --- Experiment agent_id ---


class TestExperimentAgentId:
    def test_default_none(self):
        exp = Experiment(name="test")
        assert exp.agent_id is None

    def test_set_agent_id(self):
        exp = Experiment(name="test", agent_id="agt-abc123")
        assert exp.agent_id == "agt-abc123"

    def test_serializes(self):
        exp = Experiment(name="test", agent_id="agt-abc")
        d = exp.model_dump(mode="json")
        assert d["agent_id"] == "agt-abc"


# --- ExperimentMiddleware agent version resolution ---


class TestMiddlewareAgentResolution:
    def test_agent_model_as_base(self, store, agent_version):
        """Agent version model is used when variant doesn't specify one."""
        middleware = ExperimentMiddleware(store=store)
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "agent_version_id": "agv-test123",
                }
            }
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.model == "claude-sonnet-4-20250514"
        assert ctx.request_kwargs["model"] == "claude-sonnet-4-20250514"

    def test_variant_model_overrides_agent(self, store, agent_version):
        """Variant's explicit model takes precedence over agent version model."""
        middleware = ExperimentMiddleware(store=store)
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "model": "gpt-4o-mini",
                    "agent_version_id": "agv-test123",
                }
            }
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.model == "gpt-4o-mini"

    def test_agent_request_overrides_as_base(self, store, agent_version):
        """Agent version request_overrides are applied as base layer."""
        middleware = ExperimentMiddleware(store=store)
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "agent_version_id": "agv-test123",
                }
            }
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.request_kwargs["temperature"] == 0.3
        assert ctx.request_kwargs["max_tokens"] == 1024

    def test_variant_request_overrides_win(self, store, agent_version):
        """Variant's request_overrides take precedence over agent's."""
        middleware = ExperimentMiddleware(store=store)
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "agent_version_id": "agv-test123",
                    "request_overrides": {"temperature": 0.9},
                }
            }
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        # Variant temperature wins
        assert ctx.request_kwargs["temperature"] == 0.9
        # Agent max_tokens still applied (not overridden)
        assert ctx.request_kwargs["max_tokens"] == 1024

    def test_agent_system_prompt_applied(self, store, agent_version):
        """Agent version system prompt is applied when variant doesn't override."""
        middleware = ExperimentMiddleware(store=store)
        ctx = _make_ctx(
            provider="openai",
            request_kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "agent_version_id": "agv-test123",
                }
            },
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.request_kwargs["messages"][0]["role"] == "system"
        assert ctx.request_kwargs["messages"][0]["content"] == "You are a helpful support agent."

    def test_variant_system_prompt_overrides_agent(self, store, agent_version):
        """Variant's system_prompt in request_overrides overrides agent's."""
        middleware = ExperimentMiddleware(store=store)
        ctx = _make_ctx(
            provider="openai",
            request_kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "agent_version_id": "agv-test123",
                    "request_overrides": {"system_prompt": "Custom prompt"},
                }
            },
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.request_kwargs["messages"][0]["content"] == "Custom prompt"

    def test_no_store_graceful(self):
        """Middleware without store skips agent resolution gracefully."""
        middleware = ExperimentMiddleware(store=None)
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "agent_version_id": "agv-test123",
                }
            }
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        # Model unchanged — no agent resolution
        assert ctx.model == "gpt-4o"

    def test_missing_agent_version_graceful(self, store):
        """Missing agent version is handled gracefully."""
        middleware = ExperimentMiddleware(store=store)
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "agent_version_id": "agv-nonexistent",
                }
            }
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.model == "gpt-4o"


# --- Snapshot-based resolution (preferred path) ---


class TestMiddlewareSnapshotResolution:
    def test_uses_snapshot_over_live(self, store, agent_version):
        """Middleware prefers _resolved_agent_overrides snapshot over live lookup."""
        middleware = ExperimentMiddleware(store=store)
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "agent-variant",
                    "agent_version_id": "agv-test123",
                    "_resolved_agent_overrides": {
                        "model": "snapshot-model",
                        "system_prompt": "",
                        "request_overrides": {"temperature": 0.1},
                    },
                }
            }
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        # Should use snapshot model, not live agent version model
        assert ctx.model == "snapshot-model"
        assert ctx.request_kwargs["temperature"] == 0.1


# --- Assignment snapshot ---


class TestAssignmentSnapshot:
    def test_agent_version_snapshotted_at_assignment(self, store, agent_version):
        """When variant has agent_version_id, assignment snapshots the version."""
        experiment = Experiment(
            name="agent-test",
            status=ExperimentStatus.RUNNING,
            variants=[
                VariantConfig(name="agent-v1", agent_version_id="agv-test123"),
            ],
        )
        store.save_experiment(experiment)
        assigner = ExperimentAssigner(store)
        assigner.register(experiment)

        assignment = assigner.assign(
            session_id="s1",
            experiment_id=experiment.id,
        )
        assert assignment is not None
        config = assignment.variant_config
        assert "_resolved_agent_overrides" in config
        assert config["_resolved_agent_overrides"]["model"] == "claude-sonnet-4-20250514"
        assert config["_resolved_agent_overrides"]["system_prompt"] == "You are a helpful support agent."
        assert config["_resolved_agent_overrides"]["request_overrides"]["temperature"] == 0.3

        assert "_agent_meta" in config
        assert config["_agent_meta"]["agent_id"] == "agt-test456"
        assert config["_agent_meta"]["agent_version_id"] == "agv-test123"

    def test_assignment_without_agent_version(self, store):
        """Standard variants without agent_version_id produce normal snapshots."""
        experiment = Experiment(
            name="standard",
            status=ExperimentStatus.RUNNING,
            variants=[VariantConfig(name="control", model="gpt-4o")],
        )
        store.save_experiment(experiment)
        assigner = ExperimentAssigner(store)
        assigner.register(experiment)

        assignment = assigner.assign(session_id="s2", experiment_id=experiment.id)
        assert assignment is not None
        assert "_resolved_agent_overrides" not in assignment.variant_config

    def test_missing_agent_version_graceful(self, store):
        """Missing agent version at assignment time doesn't break assignment."""
        experiment = Experiment(
            name="missing-agent",
            status=ExperimentStatus.RUNNING,
            variants=[
                VariantConfig(name="broken", agent_version_id="agv-nonexistent"),
            ],
        )
        store.save_experiment(experiment)
        assigner = ExperimentAssigner(store)
        assigner.register(experiment)

        assignment = assigner.assign(session_id="s3", experiment_id=experiment.id)
        assert assignment is not None
        assert "_resolved_agent_overrides" not in assignment.variant_config
