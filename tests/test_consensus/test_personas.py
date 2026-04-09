"""Tests for the personas feature in the consensus API."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateloom.consensus.models import ConsensusConfig, ConsensusResult, Persona
from stateloom.consensus.strategies.debate import (
    DebateStrategy,
    _build_debate_messages,
    _sanitize_session_id,
)
from stateloom.consensus.strategies.self_consistency import SelfConsistencyStrategy
from stateloom.consensus.strategies.vote import VoteStrategy
from stateloom.core.errors import StateLoomFeatureError
from stateloom.core.event import ConsensusEvent, DebateRoundEvent
from stateloom.core.feature_registry import FeatureRegistry
from stateloom.core.session import Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_gate():
    gate = MagicMock()
    gate.store = MagicMock()
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


def _make_gate_for_validation(*, ee: bool = False):
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


@pytest.fixture
def mock_client_class():
    with patch("stateloom.chat.Client") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(return_value=_mock_chat_response())
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = mock_instance
        yield mock_client


# ---------------------------------------------------------------------------
# Session ID sanitization
# ---------------------------------------------------------------------------


class TestSanitizeSessionId:
    def test_basic(self):
        assert _sanitize_session_id("Author") == "author"

    def test_spaces(self):
        assert _sanitize_session_id("Tech Reviewer") == "tech-reviewer"

    def test_special_chars(self):
        assert _sanitize_session_id("V.C. Partner!") == "v-c-partner"

    def test_truncation(self):
        long_name = "a" * 60
        assert len(_sanitize_session_id(long_name)) == 40

    def test_strips_leading_trailing_hyphens(self):
        assert _sanitize_session_id("  --hello--  ") == "hello"


# ---------------------------------------------------------------------------
# Persona validation (Gate level)
# ---------------------------------------------------------------------------


class TestPersonaValidation:
    @pytest.mark.asyncio
    async def test_empty_personas_raises(self):
        gate = _make_gate_for_validation(ee=True)
        from stateloom.gate import Gate

        with pytest.raises(ValueError, match="must not be empty"):
            await Gate.consensus(gate, personas=[], prompt="test")

    @pytest.mark.asyncio
    async def test_missing_name_raises(self):
        gate = _make_gate_for_validation(ee=True)
        from stateloom.gate import Gate

        with pytest.raises(ValueError, match="non-empty 'name'"):
            await Gate.consensus(
                gate,
                personas=[{"name": "", "model": "gpt-4o"}],
                prompt="test",
            )

    @pytest.mark.asyncio
    async def test_missing_model_raises(self):
        gate = _make_gate_for_validation(ee=True)
        from stateloom.gate import Gate

        with pytest.raises(ValueError, match="non-empty 'model'"):
            await Gate.consensus(
                gate,
                personas=[{"name": "Author", "model": ""}],
                prompt="test",
            )

    @pytest.mark.asyncio
    async def test_duplicate_names_raises(self):
        gate = _make_gate_for_validation(ee=True)
        from stateloom.gate import Gate

        with pytest.raises(ValueError, match="Duplicate persona name"):
            await Gate.consensus(
                gate,
                personas=[
                    {"name": "Author", "model": "gpt-4o"},
                    {"name": "Author", "model": "claude-sonnet-4-20250514"},
                ],
                prompt="test",
            )

    @pytest.mark.asyncio
    async def test_sees_self_raises(self):
        gate = _make_gate_for_validation(ee=True)
        from stateloom.gate import Gate

        with pytest.raises(ValueError, match="cannot list itself"):
            await Gate.consensus(
                gate,
                personas=[
                    {"name": "Author", "model": "gpt-4o", "sees": ["Author"]},
                    {"name": "Reviewer", "model": "claude-sonnet-4-20250514"},
                ],
                prompt="test",
            )

    @pytest.mark.asyncio
    async def test_sees_unknown_persona_raises(self):
        gate = _make_gate_for_validation(ee=True)
        from stateloom.gate import Gate

        with pytest.raises(ValueError, match="unknown persona 'Ghost'"):
            await Gate.consensus(
                gate,
                personas=[
                    {"name": "Author", "model": "gpt-4o", "sees": ["Ghost"]},
                    {"name": "Reviewer", "model": "claude-sonnet-4-20250514"},
                ],
                prompt="test",
            )


# ---------------------------------------------------------------------------
# EE gating counts personas as debaters
# ---------------------------------------------------------------------------


class TestPersonaEEGating:
    @pytest.mark.asyncio
    async def test_4_personas_blocked_without_ee(self):
        gate = _make_gate_for_validation(ee=False)
        from stateloom.gate import Gate

        with pytest.raises(StateLoomFeatureError) as exc_info:
            await Gate.consensus(
                gate,
                personas=[{"name": f"P{i}", "model": "gpt-4o"} for i in range(4)],
                prompt="test",
            )
        assert "4 debaters" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_3_personas_allowed_without_ee(self):
        gate = _make_gate_for_validation(ee=False)
        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=ConsensusResult(answer="ok", confidence=0.9))
        gate._consensus_orchestrator = mock_orch

        from stateloom.gate import Gate

        result = await Gate.consensus(
            gate,
            personas=[
                {"name": "A", "model": "gpt-4o"},
                {"name": "B", "model": "gpt-4o"},
                {"name": "C", "model": "claude-sonnet-4-20250514"},
            ],
            prompt="test",
        )
        assert result.answer == "ok"

    @pytest.mark.asyncio
    async def test_4_personas_same_model_blocked(self):
        """4 personas even with only 1 unique model requires EE."""
        gate = _make_gate_for_validation(ee=False)
        from stateloom.gate import Gate

        with pytest.raises(StateLoomFeatureError):
            await Gate.consensus(
                gate,
                personas=[{"name": f"P{i}", "model": "gpt-4o"} for i in range(4)],
                prompt="test",
            )


# ---------------------------------------------------------------------------
# Debate strategy with personas
# ---------------------------------------------------------------------------


class TestDebatePersonas:
    @pytest.mark.asyncio
    async def test_round1_each_persona_gets_own_prompt(self, mock_client_class):
        """Round 1: each persona receives their own system_prompt and prompt."""
        captured = []

        async def capture_achat(**kwargs):
            captured.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            models=["gpt-4o", "gemini-2.5-flash"],
            rounds=1,
            strategy="debate",
            early_stop_enabled=False,
            ee_consensus=False,
            personas=[
                Persona(
                    name="Author",
                    model="gpt-4o",
                    system_prompt="You are the author.",
                    prompt="Present this.",
                ),
                Persona(
                    name="Reviewer",
                    model="gemini-2.5-flash",
                    system_prompt="You are a reviewer.",
                    prompt="Critique this.",
                ),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = DebateStrategy()
        result = await strategy.execute(config, gate, parent)

        # Two calls, each with their own system_prompt + prompt
        assert len(captured) == 2
        # Author messages
        author_sys = captured[0][0]["content"]
        assert "author" in author_sys.lower()
        author_user = captured[0][1]["content"]
        assert "Present this" in author_user
        # Reviewer messages
        reviewer_sys = captured[1][0]["content"]
        assert "reviewer" in reviewer_sys.lower()
        reviewer_user = captured[1][1]["content"]
        assert "Critique this" in reviewer_user

    @pytest.mark.asyncio
    async def test_round2_sees_filtering(self, mock_client_class):
        """Round 2+: personas only see responses from personas in their `sees` list."""
        captured = []

        async def capture_achat(**kwargs):
            captured.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            models=["gpt-4o", "gemini-2.5-flash", "gemini-2.5-flash"],
            rounds=2,
            strategy="debate",
            early_stop_enabled=False,
            ee_consensus=False,
            personas=[
                Persona(
                    name="Author",
                    model="gpt-4o",
                    system_prompt="Author sys",
                    prompt="Write.",
                    sees=["Reviewer", "VC"],
                ),
                Persona(
                    name="Reviewer",
                    model="gemini-2.5-flash",
                    system_prompt="Reviewer sys",
                    prompt="Review.",
                    sees=["Author"],
                ),
                Persona(
                    name="VC",
                    model="gemini-2.5-flash",
                    system_prompt="VC sys",
                    prompt="Evaluate.",
                    sees=["Author"],
                ),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = DebateStrategy()
        result = await strategy.execute(config, gate, parent)

        # Round 1: 3 calls, Round 2: 3 calls = 6 total
        assert len(captured) == 6

        # Round 2 messages (indices 3, 4, 5)
        # Author (idx 3) should see Reviewer and VC
        r2_author_msgs = captured[3]
        r2_author_text = " ".join(m.get("content", "") for m in r2_author_msgs)
        assert "Reviewer" in r2_author_text
        assert "VC" in r2_author_text

        # Reviewer (idx 4) should see only Author
        r2_reviewer_msgs = captured[4]
        r2_reviewer_text = " ".join(m.get("content", "") for m in r2_reviewer_msgs)
        assert "Author" in r2_reviewer_text
        # Should NOT see VC's response
        assert "**VC**" not in r2_reviewer_text

        # VC (idx 5) should see only Author
        r2_vc_msgs = captured[5]
        r2_vc_text = " ".join(m.get("content", "") for m in r2_vc_msgs)
        assert "Author" in r2_vc_text
        assert "**Reviewer**" not in r2_vc_text

    @pytest.mark.asyncio
    async def test_sees_none_sees_all(self, mock_client_class):
        """When sees=None, a persona sees all other personas."""
        captured = []

        async def capture_achat(**kwargs):
            captured.append(kwargs.get("messages", []))
            return _mock_chat_response()

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=capture_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_instance

        config = ConsensusConfig(
            models=["gpt-4o", "gemini-2.5-flash"],
            rounds=2,
            strategy="debate",
            early_stop_enabled=False,
            ee_consensus=False,
            personas=[
                Persona(name="A", model="gpt-4o", system_prompt="sys A", prompt="Go A."),
                Persona(name="B", model="gemini-2.5-flash", system_prompt="sys B", prompt="Go B."),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = DebateStrategy()
        await strategy.execute(config, gate, parent)

        # Round 2: A (idx 2) and B (idx 3)
        r2_a = " ".join(m.get("content", "") for m in captured[2])
        r2_b = " ".join(m.get("content", "") for m in captured[3])
        assert "**B**" in r2_a  # A sees B
        assert "**A**" in r2_b  # B sees A

    @pytest.mark.asyncio
    async def test_same_model_distinct_session_ids(self, mock_client_class):
        """Same model with different personas gets distinct session IDs."""
        config = ConsensusConfig(
            models=["gemini-2.5-flash"],
            rounds=1,
            strategy="debate",
            ee_consensus=False,
            personas=[
                Persona(
                    name="Reviewer", model="gemini-2.5-flash", system_prompt="Review.", prompt="Go."
                ),
                Persona(name="VC", model="gemini-2.5-flash", system_prompt="VC.", prompt="Go."),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = DebateStrategy()
        result = await strategy.execute(config, gate, parent)

        session_ids = [r.session_id for r in result.rounds[0].responses]
        assert len(set(session_ids)) == 2  # distinct
        assert "reviewer" in session_ids[0]
        assert "vc" in session_ids[1]

    @pytest.mark.asyncio
    async def test_result_carries_persona_metadata(self, mock_client_class):
        """ConsensusResult carries personas list and winner_persona."""
        config = ConsensusConfig(
            models=["gpt-4o", "gemini-2.5-flash"],
            rounds=1,
            strategy="debate",
            ee_consensus=False,
            personas=[
                Persona(name="Author", model="gpt-4o", system_prompt="Auth", prompt="Write."),
                Persona(
                    name="Reviewer", model="gemini-2.5-flash", system_prompt="Rev", prompt="Review."
                ),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = DebateStrategy()
        result = await strategy.execute(config, gate, parent)

        assert len(result.personas) == 2
        assert result.personas[0]["name"] == "Author"
        assert result.personas[1]["name"] == "Reviewer"
        assert result.winner_persona in ("Author", "Reviewer")

    @pytest.mark.asyncio
    async def test_debater_response_has_persona_name(self, mock_client_class):
        """DebaterResponse.persona_name is set for each response."""
        config = ConsensusConfig(
            models=["gpt-4o"],
            rounds=1,
            strategy="debate",
            ee_consensus=False,
            personas=[
                Persona(name="Alpha", model="gpt-4o", system_prompt="sys", prompt="go"),
                Persona(name="Beta", model="gpt-4o", system_prompt="sys", prompt="go"),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = DebateStrategy()
        result = await strategy.execute(config, gate, parent)

        names = [r.persona_name for r in result.rounds[0].responses]
        assert "Alpha" in names
        assert "Beta" in names


# ---------------------------------------------------------------------------
# Vote strategy with personas
# ---------------------------------------------------------------------------


class TestVotePersonas:
    @pytest.mark.asyncio
    async def test_vote_with_personas(self, mock_client_class):
        """Vote strategy works with personas."""
        config = ConsensusConfig(
            models=["gpt-4o", "gemini-2.5-flash"],
            strategy="vote",
            ee_consensus=False,
            personas=[
                Persona(
                    name="Optimist",
                    model="gpt-4o",
                    system_prompt="Be optimistic.",
                    prompt="Rate this.",
                ),
                Persona(
                    name="Pessimist",
                    model="gemini-2.5-flash",
                    system_prompt="Be pessimistic.",
                    prompt="Rate this.",
                ),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.strategy == "vote"
        assert len(result.rounds[0].responses) == 2
        assert result.personas[0]["name"] == "Optimist"
        assert result.winner_persona in ("Optimist", "Pessimist")


# ---------------------------------------------------------------------------
# Self-consistency with personas
# ---------------------------------------------------------------------------


class TestSelfConsistencyPersonas:
    @pytest.mark.asyncio
    async def test_multiple_personas_raises(self, mock_client_class):
        config = ConsensusConfig(
            models=["gpt-4o"],
            strategy="self_consistency",
            samples=3,
            personas=[
                Persona(name="A", model="gpt-4o"),
                Persona(name="B", model="gpt-4o"),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = SelfConsistencyStrategy()
        with pytest.raises(ValueError, match="exactly 1 persona"):
            await strategy.execute(config, gate, parent)

    @pytest.mark.asyncio
    async def test_single_persona_works(self, mock_client_class):
        config = ConsensusConfig(
            models=["gpt-4o"],
            strategy="self_consistency",
            samples=3,
            ee_consensus=False,
            personas=[
                Persona(name="Expert", model="gpt-4o", system_prompt="Expert sys", prompt="Solve."),
            ],
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = SelfConsistencyStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.strategy == "self_consistency"
        assert result.personas == [{"name": "Expert", "model": "gpt-4o"}]
        assert result.winner_persona == "Expert"


# ---------------------------------------------------------------------------
# _build_debate_messages with persona mode
# ---------------------------------------------------------------------------


class TestBuildDebateMessagesPersonas:
    def test_use_personas_skips_self(self):
        from stateloom.consensus.models import DebaterResponse

        responses = [
            DebaterResponse(
                model="gpt-4o", content="Answer A", confidence=0.8, persona_name="Alpha"
            ),
            DebaterResponse(
                model="gpt-4o", content="Answer B", confidence=0.7, persona_name="Beta"
            ),
        ]
        base = [{"role": "user", "content": "Question?"}]
        msgs = _build_debate_messages(base, responses, "Alpha", 2, use_personas=True)
        combined = " ".join(m.get("content", "") for m in msgs)
        assert "**Beta**" in combined
        assert "**Alpha**" not in combined

    def test_use_personas_sees_filter(self):
        from stateloom.consensus.models import DebaterResponse

        responses = [
            DebaterResponse(model="m1", content="A", confidence=0.8, persona_name="Author"),
            DebaterResponse(model="m2", content="B", confidence=0.7, persona_name="Reviewer"),
            DebaterResponse(model="m3", content="C", confidence=0.6, persona_name="VC"),
        ]
        base = [{"role": "user", "content": "Q?"}]
        msgs = _build_debate_messages(
            base,
            responses,
            "Reviewer",
            2,
            use_personas=True,
            sees=["Author"],
        )
        combined = " ".join(m.get("content", "") for m in msgs)
        assert "**Author**" in combined
        assert "**VC**" not in combined
        assert "**Reviewer**" not in combined

    def test_use_personas_sees_none_sees_all(self):
        from stateloom.consensus.models import DebaterResponse

        responses = [
            DebaterResponse(model="m1", content="A", confidence=0.8, persona_name="Author"),
            DebaterResponse(model="m2", content="B", confidence=0.7, persona_name="Reviewer"),
        ]
        base = [{"role": "user", "content": "Q?"}]
        msgs = _build_debate_messages(
            base,
            responses,
            "Author",
            2,
            use_personas=True,
            sees=None,
        )
        combined = " ".join(m.get("content", "") for m in msgs)
        assert "**Reviewer**" in combined

    def test_non_persona_mode_unchanged(self):
        """Without use_personas, behavior is identical to original."""
        from stateloom.consensus.models import DebaterResponse

        responses = [
            DebaterResponse(model="gpt-4o", content="A", confidence=0.8),
            DebaterResponse(model="claude", content="B", confidence=0.7),
        ]
        base = [{"role": "user", "content": "Q?"}]
        msgs = _build_debate_messages(base, responses, "gpt-4o", 2)
        combined = " ".join(m.get("content", "") for m in msgs)
        assert "**claude**" in combined
        assert "**gpt-4o**" not in combined


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_no_personas_debate_unchanged(self, mock_client_class):
        """Without personas, debate works exactly as before."""
        config = ConsensusConfig(
            prompt="What is 2+2?",
            models=["gpt-4o", "claude-sonnet-4-20250514"],
            rounds=1,
            strategy="debate",
            ee_consensus=False,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = DebateStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.strategy == "debate"
        assert result.personas == []
        assert result.winner_persona == ""
        assert len(result.rounds[0].responses) == 2
        for r in result.rounds[0].responses:
            assert r.persona_name == ""

    @pytest.mark.asyncio
    async def test_no_personas_vote_unchanged(self, mock_client_class):
        config = ConsensusConfig(
            prompt="test",
            models=["gpt-4o", "claude-sonnet-4-20250514"],
            strategy="vote",
            ee_consensus=False,
        )
        gate = _make_mock_gate()
        parent = _make_parent_session()

        strategy = VoteStrategy()
        result = await strategy.execute(config, gate, parent)

        assert result.personas == []
        assert result.winner_persona == ""


# ---------------------------------------------------------------------------
# Aggregation uses persona names in judge transcript
# ---------------------------------------------------------------------------


class TestJudgeTranscriptPersonaNames:
    @pytest.mark.asyncio
    async def test_judge_uses_persona_names(self):
        """judge_synthesis uses persona_name as label when set."""
        from stateloom.consensus.models import DebaterResponse

        responses = [
            DebaterResponse(
                model="gpt-4o",
                content="Answer A [Confidence: 0.9]",
                confidence=0.9,
                persona_name="Author",
                round_number=1,
            ),
            DebaterResponse(
                model="gemini-2.5-flash",
                content="Answer B [Confidence: 0.8]",
                confidence=0.8,
                persona_name="Reviewer",
                round_number=1,
            ),
        ]

        captured_messages = []

        async def mock_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response("Synthesized [Confidence: 0.95]")

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=mock_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.aclose = AsyncMock()

        with patch("stateloom.chat.Client", return_value=mock_instance):
            from stateloom.consensus.aggregation import judge_synthesis

            parent = _make_parent_session()
            answer, conf = await judge_synthesis(responses, "gpt-4o", None, parent)

        # The judge prompt should use persona names, not model names
        judge_prompt = captured_messages[0][0]["content"]
        assert "**Author**" in judge_prompt
        assert "**Reviewer**" in judge_prompt

    @pytest.mark.asyncio
    async def test_judge_falls_back_to_model(self):
        """When persona_name is empty, judge uses model name."""
        from stateloom.consensus.models import DebaterResponse

        responses = [
            DebaterResponse(
                model="gpt-4o", content="Answer [Confidence: 0.9]", confidence=0.9, round_number=1
            ),
        ]

        captured_messages = []

        async def mock_achat(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _mock_chat_response("Synthesized [Confidence: 0.95]")

        mock_instance = AsyncMock()
        mock_instance.achat = AsyncMock(side_effect=mock_achat)
        mock_instance.session = _mock_client_session()
        mock_instance.aclose = AsyncMock()

        with patch("stateloom.chat.Client", return_value=mock_instance):
            from stateloom.consensus.aggregation import judge_synthesis

            parent = _make_parent_session()
            await judge_synthesis(responses, "gpt-4o", None, parent)

        judge_prompt = captured_messages[0][0]["content"]
        assert "**gpt-4o**" in judge_prompt


# ---------------------------------------------------------------------------
# Event recording with personas
# ---------------------------------------------------------------------------


class TestEventPersonaData:
    @pytest.mark.asyncio
    async def test_orchestrator_records_persona_events(self, mock_client_class):
        """Orchestrator records persona data in events."""
        gate = _make_mock_gate()
        parent = _make_parent_session()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=parent)
        cm.__aexit__ = AsyncMock(return_value=False)
        gate.async_session = MagicMock(return_value=cm)

        config = ConsensusConfig(
            models=["gpt-4o", "gemini-2.5-flash"],
            rounds=1,
            strategy="debate",
            ee_consensus=False,
            personas=[
                Persona(name="Author", model="gpt-4o", system_prompt="Auth", prompt="Write."),
                Persona(
                    name="Reviewer", model="gemini-2.5-flash", system_prompt="Rev", prompt="Review."
                ),
            ],
        )

        from stateloom.consensus.orchestrator import ConsensusOrchestrator

        orch = ConsensusOrchestrator(gate)
        result = await orch.run(config)

        # Check that save_event was called with persona data
        saved_events = [call[0][0] for call in gate.store.save_event.call_args_list]

        # Find the ConsensusEvent
        consensus_events = [e for e in saved_events if isinstance(e, ConsensusEvent)]
        assert len(consensus_events) == 1
        ce = consensus_events[0]
        assert len(ce.personas) == 2
        assert ce.personas[0]["name"] == "Author"

        # Find DebateRoundEvents
        round_events = [e for e in saved_events if isinstance(e, DebateRoundEvent)]
        assert len(round_events) >= 1
        re0 = round_events[0]
        assert len(re0.persona_names) == 2
        assert "Author" in re0.persona_names
        assert "Reviewer" in re0.persona_names
