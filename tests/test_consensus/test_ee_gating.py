"""Tests for consensus EE gating — core vs enterprise feature split."""

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
from stateloom.core.errors import StateLoomFeatureError
from stateloom.core.feature_registry import FeatureRegistry
from stateloom.core.session import Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gate(*, ee: bool = False):
    """Build a mock Gate with a real FeatureRegistry.

    When ``ee=False``, ``consensus_advanced`` is defined but NOT provided
    (simulates no license + no dev mode).  When ``ee=True``, the feature
    is registered (define + provide).
    """
    gate = MagicMock()
    gate.store = MagicMock()
    gate.store.save_event = MagicMock()

    registry = FeatureRegistry()
    registry.define(
        "consensus_advanced",
        tier="enterprise",
        description="Advanced consensus features",
    )
    if ee:
        registry.provide("consensus_advanced")

    gate._feature_registry = registry
    gate._consensus_orchestrator = None

    # Config defaults
    gate.config = MagicMock()
    defaults = MagicMock()
    defaults.default_models = ["gpt-4o", "claude-sonnet-4-20250514"]
    defaults.default_strategy = "debate"
    defaults.default_rounds = 2
    defaults.default_budget = None
    defaults.greedy = False
    defaults.greedy_agreement_threshold = 0.7
    defaults.early_stop_enabled = True
    defaults.early_stop_threshold = 0.9
    gate.config.consensus_defaults = defaults

    return gate


def _make_parent_session(session_id="test-parent"):
    session = Session(id=session_id)
    session.total_cost = 0.0
    return session


def _make_mock_result(session_id="test-parent", agg_method="confidence_weighted"):
    return ConsensusResult(
        answer="42",
        confidence=0.9,
        cost=0.01,
        session_id=session_id,
        strategy="vote",
        models=["gpt-4o", "claude-sonnet-4-20250514"],
        rounds=[
            DebateRound(
                round_number=1,
                responses=[
                    DebaterResponse(
                        model="gpt-4o",
                        content="42",
                        confidence=0.9,
                    ),
                    DebaterResponse(
                        model="claude-sonnet-4-20250514",
                        content="42",
                        confidence=0.85,
                    ),
                ],
                agreement_score=0.95,
                cost=0.01,
            )
        ],
        total_rounds=1,
        aggregation_method=agg_method,
    )


def _mock_chat_response(text="Answer [Confidence: 0.85]"):
    return types.SimpleNamespace(
        content=text,
        cost=0.01,
        tokens=50,
        model="gpt-4o",
        provider="openai",
        raw=None,
    )


def _mock_client_session():
    s = Session(id="child-1")
    s.total_cost = 0.01
    s.total_prompt_tokens = 30
    s.total_completion_tokens = 20
    return s


@pytest.fixture
def mock_client():
    """Patch Client at stateloom.chat.Client (lazy import target)."""
    with patch("stateloom.chat.Client") as mock_cls:
        instance = AsyncMock()
        instance.achat = AsyncMock(return_value=_mock_chat_response())
        instance.session = _mock_client_session()
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = instance
        yield mock_cls


# ---------------------------------------------------------------------------
# Gate-level gating tests (use Gate.consensus directly)
# ---------------------------------------------------------------------------


class TestGateConsensusGating:
    """Tests that Gate.consensus() enforces EE gates before dispatching."""

    @pytest.mark.asyncio
    async def test_models_over_3_blocked_without_ee(self):
        gate = _make_gate(ee=False)
        from stateloom.gate import Gate

        with pytest.raises(StateLoomFeatureError) as exc_info:
            await Gate.consensus(gate, models=["a", "b", "c", "d"], prompt="test")
        assert "4 models" in str(exc_info.value)
        assert exc_info.value.feature == "consensus_advanced"

    @pytest.mark.asyncio
    async def test_models_3_or_fewer_allowed_without_ee(self):
        """3 models should pass the gate check (dispatch to orchestrator)."""
        gate = _make_gate(ee=False)

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=_make_mock_result())
        gate._consensus_orchestrator = mock_orch

        from stateloom.gate import Gate

        result = await Gate.consensus(gate, models=["a", "b", "c"], prompt="test")
        assert result.answer == "42"
        call_config = mock_orch.run.call_args[0][0]
        assert call_config.ee_consensus is False

    @pytest.mark.asyncio
    async def test_greedy_blocked_without_ee(self):
        gate = _make_gate(ee=False)
        from stateloom.gate import Gate

        with pytest.raises(StateLoomFeatureError) as exc_info:
            await Gate.consensus(
                gate,
                models=["a", "b"],
                prompt="test",
                greedy=True,
            )
        assert "Greedy" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_judge_synthesis_aggregation_blocked_without_ee(self):
        gate = _make_gate(ee=False)
        from stateloom.gate import Gate

        with pytest.raises(StateLoomFeatureError) as exc_info:
            await Gate.consensus(
                gate,
                models=["a", "b"],
                prompt="test",
                aggregation="judge_synthesis",
            )
        assert "Judge synthesis" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_judge_model_blocked_without_ee(self):
        gate = _make_gate(ee=False)
        from stateloom.gate import Gate

        with pytest.raises(StateLoomFeatureError) as exc_info:
            await Gate.consensus(
                gate,
                models=["a", "b"],
                prompt="test",
                judge_model="gpt-4o",
            )
        assert "Custom judge model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_all_features_allowed_with_ee(self):
        """With EE available, 10 models + greedy + judge_model all pass."""
        gate = _make_gate(ee=True)

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=_make_mock_result())
        gate._consensus_orchestrator = mock_orch

        from stateloom.gate import Gate

        result = await Gate.consensus(
            gate,
            models=[f"model-{i}" for i in range(10)],
            prompt="test",
            greedy=True,
            judge_model="gpt-4o",
            aggregation="judge_synthesis",
        )
        assert result.answer == "42"
        call_config = mock_orch.run.call_args[0][0]
        assert call_config.ee_consensus is True

    @pytest.mark.asyncio
    async def test_error_message_includes_guidance(self):
        gate = _make_gate(ee=False)
        from stateloom.gate import Gate

        with pytest.raises(StateLoomFeatureError) as exc_info:
            await Gate.consensus(
                gate,
                models=["a", "b", "c", "d"],
                prompt="test",
            )
        msg = str(exc_info.value)
        assert "STATELOOM_LICENSE_KEY" in msg
        assert "STATELOOM_ENV=development" in msg


# ---------------------------------------------------------------------------
# Orchestrator durable flag tests
# ---------------------------------------------------------------------------


class TestOrchestratorDurableFlag:
    @pytest.mark.asyncio
    async def test_durable_disabled_without_ee(self):
        """Orchestrator passes durable=False when ee_consensus=False."""
        gate = _make_gate(ee=False)
        parent_session = _make_parent_session()

        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            strategy="vote",
            ee_consensus=False,
        )

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        mock_strategy = AsyncMock()
        mock_strategy.execute = AsyncMock(return_value=_make_mock_result())

        from stateloom.consensus.orchestrator import ConsensusOrchestrator

        with patch(
            "stateloom.consensus.strategies.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            await orch.run(config)

        gate.async_session.assert_called_once()
        _, kwargs = gate.async_session.call_args
        assert kwargs["durable"] is False

    @pytest.mark.asyncio
    async def test_durable_enabled_with_ee(self):
        """Orchestrator passes durable=True when ee_consensus=True."""
        gate = _make_gate(ee=True)
        parent_session = _make_parent_session()

        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            strategy="vote",
            ee_consensus=True,
        )

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        mock_strategy = AsyncMock()
        mock_strategy.execute = AsyncMock(return_value=_make_mock_result())

        from stateloom.consensus.orchestrator import ConsensusOrchestrator

        with patch(
            "stateloom.consensus.strategies.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            await orch.run(config)

        gate.async_session.assert_called_once()
        _, kwargs = gate.async_session.call_args
        assert kwargs["durable"] is True


# ---------------------------------------------------------------------------
# Debate strategy judge synthesis gating
# ---------------------------------------------------------------------------


class TestDebateJudgeGating:
    @pytest.mark.asyncio
    async def test_debate_skips_judge_without_ee(self, mock_client):
        """Debate uses confidence_weighted when ee_consensus=False."""
        gate = _make_gate(ee=False)
        parent_session = _make_parent_session()

        config = ConsensusConfig(
            prompt="What is 2+2?",
            models=["gpt-4o", "claude-sonnet-4-20250514"],
            rounds=1,
            strategy="debate",
            ee_consensus=False,
        )

        from stateloom.consensus.strategies.debate import DebateStrategy

        result = await DebateStrategy().execute(config, gate, parent_session)
        assert result.aggregation_method == "confidence_weighted"

    @pytest.mark.asyncio
    async def test_debate_aggregation_method_accurate_without_ee(self, mock_client):
        """Result.aggregation_method reflects actual method, not hardcoded."""
        gate = _make_gate(ee=False)
        parent_session = _make_parent_session()

        config = ConsensusConfig(
            prompt="test",
            models=["m1"],
            rounds=1,
            strategy="debate",
            ee_consensus=False,
        )

        from stateloom.consensus.strategies.debate import DebateStrategy

        result = await DebateStrategy().execute(config, gate, parent_session)
        assert result.aggregation_method == "confidence_weighted"

    @pytest.mark.asyncio
    async def test_debate_uses_judge_with_ee(self, mock_client):
        """Debate calls judge_synthesis when ee_consensus=True."""
        gate = _make_gate(ee=True)
        parent_session = _make_parent_session()

        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            rounds=1,
            strategy="debate",
            ee_consensus=True,
        )

        mock_judge = AsyncMock(return_value=("synthesized answer", 0.95))
        with patch(
            "stateloom.consensus.strategies.debate.judge_synthesis",
            mock_judge,
        ):
            from stateloom.consensus.strategies.debate import DebateStrategy

            result = await DebateStrategy().execute(config, gate, parent_session)

        assert result.aggregation_method == "judge_synthesis"
        mock_judge.assert_called_once()


# ---------------------------------------------------------------------------
# Strategy durable propagation
# ---------------------------------------------------------------------------


class TestStrategyDurablePropagation:
    @pytest.mark.asyncio
    async def test_vote_works_without_ee(self, mock_client):
        """Vote strategy works with 2 models and ee_consensus=False."""
        gate = _make_gate(ee=False)
        parent_session = _make_parent_session()

        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o", "claude-sonnet-4-20250514"],
            strategy="vote",
            ee_consensus=False,
        )

        from stateloom.consensus.strategies.vote import VoteStrategy

        result = await VoteStrategy().execute(config, gate, parent_session)
        assert result.answer != ""

        # Verify durable=False was passed to Client
        for call in mock_client.call_args_list:
            assert call.kwargs.get("durable") is False

    @pytest.mark.asyncio
    async def test_self_consistency_works_without_ee(self, mock_client):
        """Self-consistency works with ee_consensus=False."""
        gate = _make_gate(ee=False)
        parent_session = _make_parent_session()

        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            strategy="self_consistency",
            samples=3,
            ee_consensus=False,
        )

        from stateloom.consensus.strategies.self_consistency import (
            SelfConsistencyStrategy,
        )

        result = await SelfConsistencyStrategy().execute(
            config,
            gate,
            parent_session,
        )
        assert result.answer != ""

        # Verify durable=False was passed to Client
        for call in mock_client.call_args_list:
            assert call.kwargs.get("durable") is False


# ---------------------------------------------------------------------------
# Dev mode bypass
# ---------------------------------------------------------------------------


class TestDevModeBypass:
    @pytest.mark.asyncio
    async def test_dev_mode_bypasses_gating(self):
        """With EE available (dev mode), all advanced features pass."""
        gate = _make_gate(ee=True)

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=_make_mock_result())
        gate._consensus_orchestrator = mock_orch

        from stateloom.gate import Gate

        result = await Gate.consensus(
            gate,
            models=["a", "b", "c", "d", "e"],  # >3 models
            prompt="test",
            greedy=True,
            judge_model="gpt-4o",
            aggregation="judge_synthesis",
        )
        assert result.answer == "42"
        call_config = mock_orch.run.call_args[0][0]
        assert call_config.ee_consensus is True
