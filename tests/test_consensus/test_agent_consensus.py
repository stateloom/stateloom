"""Tests for agent parameter support in consensus."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateloom.consensus.models import (
    ConsensusConfig,
    ConsensusResult,
    DebateRound,
    DebaterResponse,
)
from stateloom.consensus.orchestrator import ConsensusOrchestrator
from stateloom.consensus.prompts import DEFAULT_CONFIDENCE_INSTRUCTION, VOTE_SYSTEM_PROMPT
from stateloom.consensus.strategies.debate import DebateStrategy
from stateloom.consensus.strategies.self_consistency import SelfConsistencyStrategy
from stateloom.consensus.strategies.vote import VoteStrategy
from stateloom.core.session import Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = "You are a medical AI advisor specializing in radiology."


def _make_mock_gate():
    gate = MagicMock()
    gate.store = MagicMock()
    gate.config = MagicMock()
    return gate


def _make_parent_session(session_id="parent-1"):
    session = Session(id=session_id)
    session.total_cost = 0.0
    session.total_prompt_tokens = 0
    session.total_completion_tokens = 0
    return session


def _mock_chat_response(text="Answer text [Confidence: 0.85]", cost=0.01, tokens=50):
    return types.SimpleNamespace(
        content=text,
        cost=cost,
        tokens=tokens,
        model="gpt-4o",
        provider="openai",
        raw=None,
    )


def _mock_client_session(cost=0.01, prompt_tokens=30, completion_tokens=20):
    session = Session(id="child-1")
    session.total_cost = cost
    session.total_prompt_tokens = prompt_tokens
    session.total_completion_tokens = completion_tokens
    return session


def _make_mock_result(session_id="parent-1"):
    return ConsensusResult(
        answer="The answer is 42",
        confidence=0.9,
        cost=0.05,
        session_id=session_id,
        strategy="vote",
        models=["gpt-4o", "claude"],
        rounds=[
            DebateRound(
                round_number=1,
                responses=[
                    DebaterResponse(model="gpt-4o", content="42", confidence=0.9),
                    DebaterResponse(model="claude", content="42", confidence=0.85),
                ],
                agreement_score=0.95,
                cost=0.05,
            )
        ],
        total_rounds=1,
    )


@pytest.fixture
def mock_client_class():
    """Patch Client at stateloom.chat.Client (lazy import target)."""
    with patch("stateloom.chat.Client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(return_value=_mock_chat_response())
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = mock_instance
        yield mock_client


# ---------------------------------------------------------------------------
# ConsensusConfig agent fields
# ---------------------------------------------------------------------------


class TestConsensusConfigAgentFields:
    def test_agent_fields_default_empty(self):
        config = ConsensusConfig(prompt="test", models=["gpt-4o"])
        assert config.agent is None
        assert config.agent_system_prompt == ""
        assert config.agent_id == ""
        assert config.agent_slug == ""
        assert config.agent_version_id == ""
        assert config.agent_version_number == 0

    def test_agent_fields_populated(self):
        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            agent="medical-advisor",
            agent_system_prompt=AGENT_SYSTEM_PROMPT,
            agent_id="agt-123",
            agent_slug="medical-advisor",
            agent_version_id="agv-456",
            agent_version_number=3,
        )
        assert config.agent == "medical-advisor"
        assert config.agent_system_prompt == AGENT_SYSTEM_PROMPT
        assert config.agent_id == "agt-123"
        assert config.agent_slug == "medical-advisor"
        assert config.agent_version_id == "agv-456"
        assert config.agent_version_number == 3


# ---------------------------------------------------------------------------
# Orchestrator — agent session fields
# ---------------------------------------------------------------------------


class TestOrchestratorAgentSessionFields:
    @pytest.mark.asyncio
    async def test_agent_fields_set_on_parent_session(self):
        gate = MagicMock()
        gate.store = MagicMock()
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        mock_strategy = AsyncMock()
        mock_strategy.execute = AsyncMock(return_value=_make_mock_result())

        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            agent="medical-advisor",
            agent_id="agt-123",
            agent_slug="medical-advisor",
            agent_version_id="agv-456",
            agent_version_number=3,
        )

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            await orch.run(config)

        assert parent_session.agent_id == "agt-123"
        assert parent_session.agent_slug == "medical-advisor"
        assert parent_session.agent_version_id == "agv-456"
        assert parent_session.agent_version_number == 3
        assert parent_session.agent_name == "medical-advisor"

    @pytest.mark.asyncio
    async def test_no_agent_fields_when_not_set(self):
        gate = MagicMock()
        gate.store = MagicMock()
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        mock_strategy = AsyncMock()
        mock_strategy.execute = AsyncMock(return_value=_make_mock_result())

        config = ConsensusConfig(prompt="test", models=["gpt-4o"])

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            await orch.run(config)

        # Should NOT have set agent fields (they stay at Session defaults)
        assert parent_session.agent_id == ""


# ---------------------------------------------------------------------------
# Vote strategy — agent system prompt
# ---------------------------------------------------------------------------


class TestVoteStrategyAgentPrompt:
    @pytest.mark.asyncio
    async def test_uses_agent_system_prompt(self, mock_client_class):
        """When agent_system_prompt is set, it replaces VOTE_SYSTEM_PROMPT."""
        captured_messages = []

        async def capture_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="Should we use AI for radiology?",
            models=["gpt-4o"],
            strategy="vote",
            agent_system_prompt=AGENT_SYSTEM_PROMPT,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        await strategy.execute(config, gate, parent)

        # System prompt should be agent prompt + confidence instruction
        assert len(captured_messages) >= 1
        system_msg = captured_messages[0][0]
        assert system_msg["role"] == "system"
        assert AGENT_SYSTEM_PROMPT in system_msg["content"]
        assert "Confidence" in system_msg["content"]
        assert VOTE_SYSTEM_PROMPT not in system_msg["content"]

    @pytest.mark.asyncio
    async def test_uses_default_prompt_without_agent(self, mock_client_class):
        """When agent_system_prompt is empty, uses VOTE_SYSTEM_PROMPT."""
        captured_messages = []

        async def capture_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="What is 2+2?",
            models=["gpt-4o"],
            strategy="vote",
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        await strategy.execute(config, gate, parent)

        system_msg = captured_messages[0][0]
        assert system_msg["content"] == VOTE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Debate strategy — agent system prompt
# ---------------------------------------------------------------------------


class TestDebateStrategyAgentPrompt:
    @pytest.mark.asyncio
    async def test_uses_agent_system_prompt_round1(self, mock_client_class):
        """Agent system prompt replaces VOTE_SYSTEM_PROMPT in round 1."""
        captured_messages = []

        async def capture_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="Should we use AI for radiology?",
            models=["gpt-4o", "claude"],
            rounds=1,
            strategy="debate",
            agent_system_prompt=AGENT_SYSTEM_PROMPT,
            early_stop_enabled=False,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        with patch(
            "stateloom.consensus.strategies.debate.judge_synthesis",
            new_callable=AsyncMock,
            return_value=("Answer", 0.8),
        ):
            strategy = DebateStrategy()
            await strategy.execute(config, gate, parent)

        # Round 1 system prompt should contain agent prompt
        system_msg = captured_messages[0][0]
        assert system_msg["role"] == "system"
        assert AGENT_SYSTEM_PROMPT in system_msg["content"]
        assert VOTE_SYSTEM_PROMPT not in system_msg["content"]

    @pytest.mark.asyncio
    async def test_uses_default_prompt_without_agent(self, mock_client_class):
        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            rounds=1,
            strategy="debate",
            early_stop_enabled=False,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        captured_messages = []

        async def capture_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        with patch(
            "stateloom.consensus.strategies.debate.judge_synthesis",
            new_callable=AsyncMock,
            return_value=("Answer", 0.8),
        ):
            strategy = DebateStrategy()
            await strategy.execute(config, gate, parent)

        system_msg = captured_messages[0][0]
        assert system_msg["content"] == VOTE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Self-consistency strategy — agent system prompt
# ---------------------------------------------------------------------------


class TestSelfConsistencyStrategyAgentPrompt:
    @pytest.mark.asyncio
    async def test_uses_agent_system_prompt(self, mock_client_class):
        captured_messages = []

        async def capture_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="Is this scan normal?",
            models=["gpt-4o"],
            strategy="self_consistency",
            samples=3,
            temperature=0.7,
            agent_system_prompt=AGENT_SYSTEM_PROMPT,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = SelfConsistencyStrategy()
        await strategy.execute(config, gate, parent)

        assert len(captured_messages) == 3
        for msgs in captured_messages:
            system_msg = msgs[0]
            assert system_msg["role"] == "system"
            assert AGENT_SYSTEM_PROMPT in system_msg["content"]
            assert VOTE_SYSTEM_PROMPT not in system_msg["content"]

    @pytest.mark.asyncio
    async def test_uses_default_prompt_without_agent(self, mock_client_class):
        captured_messages = []

        async def capture_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            strategy="self_consistency",
            samples=2,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = SelfConsistencyStrategy()
        await strategy.execute(config, gate, parent)

        system_msg = captured_messages[0][0]
        assert system_msg["content"] == VOTE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Gate.consensus() — agent resolution
# ---------------------------------------------------------------------------


class TestGateConsensusAgentResolution:
    @pytest.mark.asyncio
    async def test_agent_resolved_and_config_populated(self):
        """Gate.consensus(agent=...) resolves the agent and populates config fields."""
        from stateloom.agent.models import Agent, AgentVersion
        from stateloom.core.feature_registry import FeatureRegistry
        from stateloom.core.types import AgentStatus
        from stateloom.gate import Gate

        agent_obj = Agent(
            id="agt-test-1",
            slug="medical-advisor",
            team_id="team-1",
            org_id="org-1",
            name="Medical Advisor",
            status=AgentStatus.ACTIVE,
            active_version_id="agv-v1",
        )
        version = AgentVersion(
            id="agv-v1",
            agent_id="agt-test-1",
            version_number=2,
            model="gpt-4o",
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

        gate = MagicMock()
        gate.store = MagicMock()
        gate.store.get_agent.return_value = None
        gate.store.list_agents.return_value = [agent_obj]
        gate.store.get_agent_version.return_value = version

        registry = FeatureRegistry()
        registry.define("consensus_advanced", tier="enterprise")
        gate._feature_registry = registry
        gate._consensus_orchestrator = None

        defaults = MagicMock()
        defaults.default_models = ["gpt-4o"]
        defaults.default_strategy = "debate"
        defaults.default_rounds = 2
        defaults.default_budget = None
        defaults.greedy = False
        defaults.greedy_agreement_threshold = 0.7
        defaults.early_stop_enabled = True
        defaults.early_stop_threshold = 0.9
        gate.config.consensus_defaults = defaults

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=_make_mock_result())

        with patch(
            "stateloom.consensus.orchestrator.ConsensusOrchestrator",
            return_value=mock_orch,
        ):
            result = await Gate.consensus(gate, agent="medical-advisor", prompt="test")

        # Verify config was populated with agent fields
        call_config = mock_orch.run.call_args[0][0]
        assert call_config.agent == "medical-advisor"
        assert call_config.agent_system_prompt == AGENT_SYSTEM_PROMPT
        assert call_config.agent_id == "agt-test-1"
        assert call_config.agent_slug == "medical-advisor"
        assert call_config.agent_version_id == "agv-v1"
        assert call_config.agent_version_number == 2

    @pytest.mark.asyncio
    async def test_agent_model_used_as_default(self):
        """When models not provided, agent model is used."""
        from stateloom.agent.models import Agent, AgentVersion
        from stateloom.core.feature_registry import FeatureRegistry
        from stateloom.core.types import AgentStatus
        from stateloom.gate import Gate

        agent_obj = Agent(
            id="agt-test-1",
            slug="bot",
            team_id="team-1",
            org_id="org-1",
            status=AgentStatus.ACTIVE,
            active_version_id="agv-v1",
        )
        version = AgentVersion(
            id="agv-v1",
            agent_id="agt-test-1",
            version_number=1,
            model="claude-haiku-4-5-20251001",
            system_prompt="You are a bot.",
        )

        gate = MagicMock()
        gate.store = MagicMock()
        gate.store.get_agent.return_value = None
        gate.store.list_agents.return_value = [agent_obj]
        gate.store.get_agent_version.return_value = version

        registry = FeatureRegistry()
        registry.define("consensus_advanced", tier="enterprise")
        gate._feature_registry = registry
        gate._consensus_orchestrator = None

        defaults = MagicMock()
        defaults.default_models = ["gpt-4o"]
        defaults.default_strategy = "vote"
        defaults.default_rounds = 2
        defaults.default_budget = None
        defaults.greedy = False
        defaults.greedy_agreement_threshold = 0.7
        defaults.early_stop_enabled = True
        defaults.early_stop_threshold = 0.9
        gate.config.consensus_defaults = defaults

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=_make_mock_result())

        with patch(
            "stateloom.consensus.orchestrator.ConsensusOrchestrator",
            return_value=mock_orch,
        ):
            await Gate.consensus(gate, agent="bot", prompt="test")

        call_config = mock_orch.run.call_args[0][0]
        assert call_config.models == ["claude-haiku-4-5-20251001"]

    @pytest.mark.asyncio
    async def test_explicit_models_override_agent_model(self):
        """When models are explicitly provided, agent model is NOT used."""
        from stateloom.agent.models import Agent, AgentVersion
        from stateloom.core.feature_registry import FeatureRegistry
        from stateloom.core.types import AgentStatus
        from stateloom.gate import Gate

        agent_obj = Agent(
            id="agt-test-1",
            slug="bot",
            team_id="team-1",
            org_id="org-1",
            status=AgentStatus.ACTIVE,
            active_version_id="agv-v1",
        )
        version = AgentVersion(
            id="agv-v1",
            agent_id="agt-test-1",
            version_number=1,
            model="claude-haiku-4-5-20251001",
            system_prompt="You are a bot.",
        )

        gate = MagicMock()
        gate.store = MagicMock()
        gate.store.get_agent.return_value = None
        gate.store.list_agents.return_value = [agent_obj]
        gate.store.get_agent_version.return_value = version

        registry = FeatureRegistry()
        registry.define("consensus_advanced", tier="enterprise")
        gate._feature_registry = registry
        gate._consensus_orchestrator = None

        defaults = MagicMock()
        defaults.default_models = ["gpt-4o"]
        defaults.default_strategy = "vote"
        defaults.default_rounds = 2
        defaults.default_budget = None
        defaults.greedy = False
        defaults.greedy_agreement_threshold = 0.7
        defaults.early_stop_enabled = True
        defaults.early_stop_threshold = 0.9
        gate.config.consensus_defaults = defaults

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=_make_mock_result())

        with patch(
            "stateloom.consensus.orchestrator.ConsensusOrchestrator",
            return_value=mock_orch,
        ):
            await Gate.consensus(
                gate,
                agent="bot",
                models=["gpt-4o", "gemini-2.0-flash"],
                prompt="test",
            )

        call_config = mock_orch.run.call_args[0][0]
        assert call_config.models == ["gpt-4o", "gemini-2.0-flash"]

    @pytest.mark.asyncio
    async def test_no_agent_preserves_default_behavior(self):
        """Without agent param, behavior is unchanged."""
        from stateloom.core.feature_registry import FeatureRegistry
        from stateloom.gate import Gate

        gate = MagicMock()
        gate.store = MagicMock()

        registry = FeatureRegistry()
        registry.define("consensus_advanced", tier="enterprise")
        gate._feature_registry = registry
        gate._consensus_orchestrator = None

        defaults = MagicMock()
        defaults.default_models = ["gpt-4o"]
        defaults.default_strategy = "vote"
        defaults.default_rounds = 2
        defaults.default_budget = None
        defaults.greedy = False
        defaults.greedy_agreement_threshold = 0.7
        defaults.early_stop_enabled = True
        defaults.early_stop_threshold = 0.9
        gate.config.consensus_defaults = defaults

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=_make_mock_result())

        with patch(
            "stateloom.consensus.orchestrator.ConsensusOrchestrator",
            return_value=mock_orch,
        ):
            await Gate.consensus(gate, prompt="test")

        call_config = mock_orch.run.call_args[0][0]
        assert call_config.agent is None
        assert call_config.agent_system_prompt == ""
        assert call_config.agent_id == ""
        assert call_config.models == ["gpt-4o"]
