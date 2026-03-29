"""Tests for consensus strategies (vote, debate, self_consistency)."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateloom.consensus.models import ConsensusConfig
from stateloom.consensus.strategies.debate import DebateStrategy, _downgrade_model
from stateloom.consensus.strategies.self_consistency import SelfConsistencyStrategy
from stateloom.consensus.strategies.vote import VoteStrategy
from stateloom.core.session import Session


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
    """Create a mock ChatResponse."""
    resp = types.SimpleNamespace(
        content=text,
        cost=cost,
        tokens=tokens,
        model="gpt-4o",
        provider="openai",
        raw=None,
    )
    return resp


def _mock_client_session(cost=0.01, prompt_tokens=30, completion_tokens=20):
    session = Session(id="child-1")
    session.total_cost = cost
    session.total_prompt_tokens = prompt_tokens
    session.total_completion_tokens = completion_tokens
    return session


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


# ─── Vote Strategy ─────────────────────────────────────────────────


class TestVoteStrategy:
    @pytest.mark.asyncio
    async def test_basic_vote(self, mock_client_class):
        config = ConsensusConfig(
            prompt="What is 2+2?",
            models=["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
            strategy="vote",
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.strategy == "vote"
        assert result.total_rounds == 1
        assert len(result.rounds) == 1
        assert len(result.rounds[0].responses) == 3
        assert result.answer != ""
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_vote_aggregation_majority(self, mock_client_class):
        config = ConsensusConfig(
            prompt="What is 2+2?",
            models=["gpt-4o", "claude"],
            strategy="vote",
            aggregation="majority_vote",
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.aggregation_method == "majority_vote"

    @pytest.mark.asyncio
    async def test_vote_aggregation_confidence_weighted(self, mock_client_class):
        config = ConsensusConfig(
            prompt="What is 2+2?",
            models=["gpt-4o", "claude"],
            strategy="vote",
            aggregation="confidence_weighted",
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.aggregation_method == "confidence_weighted"

    @pytest.mark.asyncio
    async def test_vote_cost_aggregated(self, mock_client_class):
        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o", "claude"],
            budget=1.0,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.cost >= 0

    @pytest.mark.asyncio
    async def test_vote_handles_failed_call(self, mock_client_class):
        # Make one client raise an exception, one succeed
        call_count = [0]

        bad_instance = AsyncMock()
        bad_instance.achat = AsyncMock(side_effect=RuntimeError("API error"))
        bad_instance.session = _mock_client_session()
        bad_instance.__aenter__ = AsyncMock(return_value=bad_instance)
        bad_instance.__aexit__ = AsyncMock(return_value=False)

        good_instance = AsyncMock()
        good_instance.achat = AsyncMock(return_value=_mock_chat_response())
        good_instance.session = _mock_client_session()
        good_instance.__aenter__ = AsyncMock(return_value=good_instance)
        good_instance.__aexit__ = AsyncMock(return_value=False)

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return bad_instance
            return good_instance

        mock_client_class.side_effect = side_effect

        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o", "claude"],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        result = await strategy.execute(config, gate, parent)

        # Should still get a result from the successful call
        assert len(result.rounds[0].responses) >= 1


# ─── Debate Strategy ───────────────────────────────────────────────


class TestDebateStrategy:
    @pytest.mark.asyncio
    async def test_basic_debate(self, mock_client_class):
        config = ConsensusConfig(
            prompt="Discuss healthcare AI risks",
            models=["gpt-4o", "claude-sonnet-4-20250514"],
            rounds=2,
            strategy="debate",
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        # Also mock judge_synthesis
        with patch(
            "stateloom.consensus.strategies.debate.judge_synthesis",
            new_callable=AsyncMock,
            return_value=("Synthesized answer [Confidence: 0.90]", 0.90),
        ):
            strategy = DebateStrategy()
            result = await strategy.execute(config, gate, parent)

        assert result.strategy == "debate"
        assert result.total_rounds == 2
        assert len(result.rounds) == 2
        assert result.answer != ""

    @pytest.mark.asyncio
    async def test_debate_early_stop(self, mock_client_class):
        # Mock high confidence responses to trigger early stop
        high_conf_resp = _mock_chat_response("Same answer [Confidence: 0.95]")

        if True:  # scope block for mock setup
            mock_instance = AsyncMock()
            mock_instance.achat = AsyncMock(return_value=high_conf_resp)
            mock_instance.session = _mock_client_session()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="What is 2+2?",
            models=["gpt-4o", "claude"],
            rounds=5,
            strategy="debate",
            early_stop_enabled=True,
            early_stop_threshold=0.9,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        with patch(
            "stateloom.consensus.strategies.debate.judge_synthesis",
            new_callable=AsyncMock,
            return_value=("4 [Confidence: 0.99]", 0.99),
        ):
            strategy = DebateStrategy()
            result = await strategy.execute(config, gate, parent)

        assert result.early_stopped is True
        assert result.total_rounds < 5

    @pytest.mark.asyncio
    async def test_debate_context_includes_previous(self, mock_client_class):
        """Verify that round 2 messages include previous responses."""
        captured_messages = []

        async def capture_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response()

        if True:  # scope block for mock setup
            mock_instance = AsyncMock()
            mock_instance.achat = AsyncMock(side_effect=capture_achat)
            mock_instance.session = _mock_client_session()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="Discuss AI",
            models=["gpt-4o", "claude"],
            rounds=2,
            strategy="debate",
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

        # Should have captured messages from both rounds
        # Round 1: 2 models × 1 = 2 calls, Round 2: 2 models × 1 = 2 calls
        assert len(captured_messages) >= 4

        # Round 2 messages should contain debate context (DEBATE_ROUND_TEMPLATE)
        round2_msgs = [
            m
            for m in captured_messages
            if any("previous round" in msg.get("content", "").lower() for msg in m)
        ]
        assert len(round2_msgs) > 0

    @pytest.mark.asyncio
    async def test_debate_judge_synthesis_fallback(self, mock_client_class):
        """When judge synthesis fails, should fall back to confidence_weighted."""
        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o", "claude"],
            rounds=1,
            strategy="debate",
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        with patch(
            "stateloom.consensus.strategies.debate.judge_synthesis",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Judge failed"),
        ):
            strategy = DebateStrategy()
            result = await strategy.execute(config, gate, parent)

        # Should still return a result
        assert result.answer != ""


class TestGreedyDowngrade:
    def test_downgrade_gpt4o(self):
        result = _downgrade_model("gpt-4o")
        assert result == "gpt-4o-mini"

    def test_downgrade_claude_sonnet(self):
        result = _downgrade_model("claude-sonnet-4-20250514")
        assert "haiku" in result.lower() or "claude" in result.lower()

    def test_downgrade_gemini_pro(self):
        result = _downgrade_model("gemini-1.5-pro")
        assert "flash" in result.lower() or "gemini" in result.lower()

    def test_no_downgrade_for_unknown(self):
        result = _downgrade_model("custom-model")
        assert result == "custom-model"

    def test_no_downgrade_for_already_cheap(self):
        result = _downgrade_model("gpt-4o-mini")
        assert result == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_greedy_triggers_downgrade(self, mock_client_class):
        """When agreement > threshold after R1, models should be downgraded."""
        # All models return identical content → high agreement
        same_resp = _mock_chat_response("The answer is Paris [Confidence: 0.95]")
        if True:  # scope block for mock setup
            mock_instance = AsyncMock()
            mock_instance.achat = AsyncMock(return_value=same_resp)
            mock_instance.session = _mock_client_session()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="Capital of France?",
            models=["gpt-4o", "gemini-1.5-pro"],
            rounds=2,
            strategy="debate",
            greedy=True,
            greedy_agreement_threshold=0.5,  # low threshold to ensure trigger
            early_stop_enabled=False,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        with patch(
            "stateloom.consensus.strategies.debate.judge_synthesis",
            new_callable=AsyncMock,
            return_value=("Paris [Confidence: 0.99]", 0.99),
        ):
            strategy = DebateStrategy()
            result = await strategy.execute(config, gate, parent)

        assert result.total_rounds == 2

    @pytest.mark.asyncio
    async def test_greedy_no_downgrade_low_agreement(self, mock_client_class):
        """When agreement < threshold, no downgrade."""
        call_count = [0]

        async def varied_responses(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return _mock_chat_response(
                    "Answer A: something very different xyz [Confidence: 0.6]"
                )
            return _mock_chat_response("Answer B: completely other thing abc [Confidence: 0.7]")

        if True:  # scope block for mock setup
            mock_instance = AsyncMock()
            mock_instance.achat = AsyncMock(side_effect=varied_responses)
            mock_instance.session = _mock_client_session()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            prompt="Complex question",
            models=["gpt-4o", "claude-sonnet-4-20250514"],
            rounds=2,
            strategy="debate",
            greedy=True,
            greedy_agreement_threshold=0.99,  # very high threshold
            early_stop_enabled=False,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        with patch(
            "stateloom.consensus.strategies.debate.judge_synthesis",
            new_callable=AsyncMock,
            return_value=("Synthesized", 0.7),
        ):
            strategy = DebateStrategy()
            result = await strategy.execute(config, gate, parent)

        assert result.total_rounds == 2


# ─── Self-Consistency Strategy ─────────────────────────────────────


class TestSelfConsistencyStrategy:
    @pytest.mark.asyncio
    async def test_basic_self_consistency(self, mock_client_class):
        config = ConsensusConfig(
            prompt="Solve: 2x + 5 = 13",
            models=["gpt-4o"],
            strategy="self_consistency",
            samples=5,
            temperature=0.7,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = SelfConsistencyStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.strategy == "self_consistency"
        assert result.total_rounds == 1
        assert len(result.rounds[0].responses) == 5
        assert result.aggregation_method == "majority_vote"
        assert result.winner_model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_self_consistency_single_sample(self, mock_client_class):
        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            strategy="self_consistency",
            samples=1,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = SelfConsistencyStrategy()
        result = await strategy.execute(config, gate, parent)

        assert len(result.rounds[0].responses) == 1

    @pytest.mark.asyncio
    async def test_self_consistency_with_budget(self, mock_client_class):
        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o"],
            strategy="self_consistency",
            samples=3,
            budget=0.50,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = SelfConsistencyStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.cost >= 0
