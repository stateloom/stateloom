"""Tests for ConsensusOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateloom.consensus.models import (
    ConsensusConfig,
    ConsensusResult,
    DebateRound,
    DebaterResponse,
)
from stateloom.consensus.orchestrator import ConsensusOrchestrator
from stateloom.core.session import Session


def _make_mock_gate():
    gate = MagicMock()
    gate.store = MagicMock()
    gate.config = MagicMock()
    gate.config.consensus_defaults = MagicMock()
    gate.config.consensus_defaults.default_models = ["gpt-4o"]
    gate.config.consensus_defaults.default_strategy = "debate"
    gate.config.consensus_defaults.default_rounds = 2
    gate.config.consensus_defaults.default_budget = None
    gate.config.consensus_defaults.greedy = False
    gate.config.consensus_defaults.greedy_agreement_threshold = 0.7
    gate.config.consensus_defaults.early_stop_enabled = True
    gate.config.consensus_defaults.early_stop_threshold = 0.9
    return gate


def _make_parent_session(session_id="orch-parent-1"):
    session = Session(id=session_id)
    session.total_cost = 0.0
    return session


def _make_mock_result(session_id="orch-parent-1"):
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
def mock_strategy():
    strategy = AsyncMock()
    strategy.execute = AsyncMock(return_value=_make_mock_result())
    return strategy


class TestConsensusOrchestrator:
    @pytest.mark.asyncio
    async def test_basic_orchestration(self, mock_strategy):
        gate = _make_mock_gate()
        parent_session = _make_parent_session()

        # Mock async_session context manager
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(
                prompt="test",
                models=["gpt-4o", "claude"],
                strategy="vote",
            )
            result = await orch.run(config)

        assert result.answer == "The answer is 42"
        assert result.session_id == "orch-parent-1"
        mock_strategy.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_parent_session_created_durable(self, mock_strategy):
        gate = _make_mock_gate()
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(prompt="test", models=["gpt-4o"])
            await orch.run(config)

        # Verify async_session was called with durable=True
        gate.async_session.assert_called_once()
        call_kwargs = gate.async_session.call_args[1]
        assert call_kwargs["durable"] is True

    @pytest.mark.asyncio
    async def test_events_recorded(self, mock_strategy):
        gate = _make_mock_gate()
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(
                prompt="test",
                models=["gpt-4o", "claude"],
                strategy="vote",
            )
            await orch.run(config)

        # Should have saved events: 1 DebateRoundEvent + 1 ConsensusEvent
        assert gate.store.save_event.call_count == 2
        saved_events = [call.args[0] for call in gate.store.save_event.call_args_list]
        event_types = [e.event_type.value for e in saved_events]
        assert "debate_round" in event_types
        assert "consensus" in event_types

    @pytest.mark.asyncio
    async def test_consensus_event_fields(self, mock_strategy):
        gate = _make_mock_gate()
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(
                prompt="test",
                models=["gpt-4o", "claude"],
                strategy="vote",
            )
            await orch.run(config)

        # Find the ConsensusEvent
        consensus_event = None
        for call in gate.store.save_event.call_args_list:
            evt = call.args[0]
            if evt.event_type.value == "consensus":
                consensus_event = evt
                break

        assert consensus_event is not None
        assert consensus_event.strategy == "vote"
        assert consensus_event.models == ["gpt-4o", "claude"]
        assert consensus_event.confidence == 0.9

    @pytest.mark.asyncio
    async def test_budget_exhaustion_returns_partial(self):
        from stateloom.core.errors import StateLoomBudgetError

        gate = _make_mock_gate()
        parent_session = _make_parent_session()
        parent_session.total_cost = 0.5

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        mock_strategy = AsyncMock()
        mock_strategy.execute = AsyncMock(
            side_effect=StateLoomBudgetError(limit=1.0, spent=1.5, session_id="s1")
        )

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(
                prompt="test",
                models=["gpt-4o"],
                budget=1.0,
            )
            result = await orch.run(config)

        assert "Budget exhausted" in result.answer

    @pytest.mark.asyncio
    async def test_non_budget_error_propagates(self):
        gate = _make_mock_gate()
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        mock_strategy = AsyncMock()
        mock_strategy.execute = AsyncMock(side_effect=RuntimeError("unexpected"))

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(prompt="test", models=["gpt-4o"])
            with pytest.raises(RuntimeError, match="unexpected"):
                await orch.run(config)

    @pytest.mark.asyncio
    async def test_session_id_in_result(self, mock_strategy):
        gate = _make_mock_gate()
        parent_session = _make_parent_session("my-debate-1")

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        mock_strategy.execute.return_value = _make_mock_result("my-debate-1")

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(
                prompt="test",
                models=["gpt-4o"],
                session_id="my-debate-1",
            )
            result = await orch.run(config)

        assert result.session_id == "my-debate-1"

    @pytest.mark.asyncio
    async def test_greedy_passed_to_config(self, mock_strategy):
        gate = _make_mock_gate()
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(
                prompt="test",
                models=["gpt-4o"],
                greedy=True,
            )
            await orch.run(config)

        # Strategy received the config with greedy=True
        call_args = mock_strategy.execute.call_args
        passed_config = call_args.args[0] if call_args.args else call_args[0][0]
        assert passed_config.greedy is True

    @pytest.mark.asyncio
    async def test_event_save_failure_does_not_crash(self, mock_strategy):
        """Event persistence is fail-open."""
        gate = _make_mock_gate()
        gate.store.save_event.side_effect = RuntimeError("DB down")
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(prompt="test", models=["gpt-4o"])
            result = await orch.run(config)

        # Should still return result despite event save failures
        assert result.answer == "The answer is 42"

    @pytest.mark.asyncio
    async def test_multiple_rounds_recorded(self):
        gate = _make_mock_gate()
        parent_session = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent_session)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        # Strategy returns 3 rounds
        multi_round_result = ConsensusResult(
            answer="Final answer",
            confidence=0.85,
            cost=0.15,
            strategy="debate",
            models=["gpt-4o", "claude"],
            rounds=[
                DebateRound(
                    round_number=i,
                    responses=[DebaterResponse(model="gpt-4o", content=f"R{i}")],
                    cost=0.05,
                )
                for i in range(1, 4)
            ],
            total_rounds=3,
        )

        mock_strategy = AsyncMock()
        mock_strategy.execute = AsyncMock(return_value=multi_round_result)

        with patch(
            "stateloom.consensus.orchestrator.get_strategy",
            return_value=mock_strategy,
        ):
            orch = ConsensusOrchestrator(gate)
            config = ConsensusConfig(
                prompt="test",
                models=["gpt-4o", "claude"],
                rounds=3,
            )
            await orch.run(config)

        # 3 DebateRoundEvents + 1 ConsensusEvent = 4 total
        assert gate.store.save_event.call_count == 4
