"""Tests for DebateRoundEvent and ConsensusEvent."""

from __future__ import annotations

from pydantic import TypeAdapter

from stateloom.core.event import (
    AnyEvent,
    ConsensusEvent,
    DebateRoundEvent,
)
from stateloom.core.types import EventType


class TestDebateRoundEvent:
    def test_construction(self):
        evt = DebateRoundEvent(
            session_id="s1",
            round_number=2,
            strategy="debate",
            models=["gpt-4o", "claude"],
            responses_summary=[
                {"model": "gpt-4o", "confidence": 0.9, "cost": 0.01},
            ],
            agreement_score=0.85,
            consensus_reached=False,
            round_cost=0.02,
            round_duration_ms=1500.0,
        )
        assert evt.event_type == EventType.DEBATE_ROUND
        assert evt.round_number == 2
        assert len(evt.models) == 2
        assert evt.agreement_score == 0.85

    def test_serialization_round_trip(self):
        evt = DebateRoundEvent(
            session_id="s1",
            round_number=1,
            strategy="vote",
            models=["gpt-4o"],
        )
        data = evt.model_dump(mode="json")
        evt2 = DebateRoundEvent.model_validate(data)
        assert evt2.round_number == 1
        assert evt2.strategy == "vote"

    def test_any_event_discriminator(self):
        """DebateRoundEvent resolves via AnyEvent discriminated union."""
        evt = DebateRoundEvent(session_id="s1", round_number=1)
        data = evt.model_dump(mode="json")
        ta = TypeAdapter(AnyEvent)
        resolved = ta.validate_python(data)
        assert isinstance(resolved, DebateRoundEvent)
        assert resolved.round_number == 1


class TestConsensusEvent:
    def test_construction(self):
        evt = ConsensusEvent(
            session_id="s1",
            strategy="debate",
            models=["gpt-4o", "claude", "gemini-2.0-flash"],
            total_rounds=3,
            final_answer_preview="The answer is 42...",
            confidence=0.92,
            total_cost=0.15,
            total_duration_ms=8000.0,
            early_stopped=True,
            aggregation_method="judge_synthesis",
            winner_model="claude",
        )
        assert evt.event_type == EventType.CONSENSUS
        assert evt.strategy == "debate"
        assert len(evt.models) == 3
        assert evt.early_stopped is True

    def test_serialization_round_trip(self):
        evt = ConsensusEvent(
            session_id="s1",
            strategy="vote",
            models=["gpt-4o"],
            confidence=0.88,
        )
        data = evt.model_dump(mode="json")
        evt2 = ConsensusEvent.model_validate(data)
        assert evt2.confidence == 0.88

    def test_any_event_discriminator(self):
        """ConsensusEvent resolves via AnyEvent discriminated union."""
        evt = ConsensusEvent(session_id="s1", strategy="debate")
        data = evt.model_dump(mode="json")
        ta = TypeAdapter(AnyEvent)
        resolved = ta.validate_python(data)
        assert isinstance(resolved, ConsensusEvent)
        assert resolved.strategy == "debate"
