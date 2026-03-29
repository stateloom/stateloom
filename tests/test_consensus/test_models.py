"""Tests for consensus Pydantic models."""

from __future__ import annotations

import pytest

from stateloom.consensus.models import (
    ConsensusConfig,
    ConsensusResult,
    DebateRound,
    DebaterResponse,
)


class TestDebaterResponse:
    def test_default_construction(self):
        r = DebaterResponse()
        assert r.model == ""
        assert r.content == ""
        assert r.confidence == 0.5
        assert r.cost == 0.0
        assert r.latency_ms == 0.0
        assert r.tokens == 0
        assert r.session_id == ""
        assert r.round_number == 0

    def test_full_construction(self):
        r = DebaterResponse(
            model="gpt-4o",
            content="The answer is 42",
            confidence=0.95,
            cost=0.01,
            latency_ms=500.0,
            tokens=100,
            session_id="s1",
            round_number=2,
        )
        assert r.model == "gpt-4o"
        assert r.confidence == 0.95
        assert r.round_number == 2

    def test_confidence_bounds(self):
        r = DebaterResponse(confidence=0.0)
        assert r.confidence == 0.0
        r = DebaterResponse(confidence=1.0)
        assert r.confidence == 1.0

    def test_confidence_out_of_bounds(self):
        with pytest.raises(Exception):
            DebaterResponse(confidence=-0.1)
        with pytest.raises(Exception):
            DebaterResponse(confidence=1.1)

    def test_round_trip_serialization(self):
        r = DebaterResponse(model="claude", content="yes", confidence=0.8)
        data = r.model_dump(mode="json")
        r2 = DebaterResponse.model_validate(data)
        assert r2.model == "claude"
        assert r2.confidence == 0.8


class TestDebateRound:
    def test_default_construction(self):
        rnd = DebateRound()
        assert rnd.round_number == 0
        assert rnd.responses == []
        assert rnd.consensus_reached is False
        assert rnd.agreement_score == 0.0

    def test_with_responses(self):
        resps = [
            DebaterResponse(model="gpt-4o", content="A", confidence=0.9),
            DebaterResponse(model="claude", content="B", confidence=0.8),
        ]
        rnd = DebateRound(round_number=1, responses=resps, agreement_score=0.5)
        assert len(rnd.responses) == 2
        assert rnd.agreement_score == 0.5

    def test_round_trip_serialization(self):
        rnd = DebateRound(
            round_number=1,
            responses=[DebaterResponse(model="gpt-4o", content="hello")],
            cost=0.02,
        )
        data = rnd.model_dump(mode="json")
        rnd2 = DebateRound.model_validate(data)
        assert rnd2.round_number == 1
        assert len(rnd2.responses) == 1


class TestConsensusResult:
    def test_default_construction(self):
        r = ConsensusResult()
        assert r.answer == ""
        assert r.confidence == 0.0
        assert r.cost == 0.0
        assert r.rounds == []
        assert r.early_stopped is False
        assert r.human_verdict is None

    def test_full_construction(self):
        r = ConsensusResult(
            answer="The answer is 42",
            confidence=0.92,
            cost=0.15,
            session_id="s1",
            strategy="debate",
            models=["gpt-4o", "claude"],
            total_rounds=2,
            early_stopped=True,
            aggregation_method="judge_synthesis",
            winner_model="claude",
            duration_ms=5000.0,
        )
        assert r.strategy == "debate"
        assert r.total_rounds == 2
        assert r.early_stopped is True
        assert r.winner_model == "claude"

    def test_round_trip_serialization(self):
        r = ConsensusResult(
            answer="Paris",
            confidence=0.99,
            strategy="vote",
            models=["gpt-4o"],
        )
        data = r.model_dump(mode="json")
        r2 = ConsensusResult.model_validate(data)
        assert r2.answer == "Paris"
        assert r2.confidence == 0.99

    def test_confidence_bounds(self):
        r = ConsensusResult(confidence=0.0)
        assert r.confidence == 0.0
        r = ConsensusResult(confidence=1.0)
        assert r.confidence == 1.0


class TestConsensusConfig:
    def test_defaults(self):
        c = ConsensusConfig()
        assert c.prompt == ""
        assert c.rounds == 2
        assert c.strategy == "debate"
        assert c.greedy is False
        assert c.samples == 5
        assert c.temperature == 0.7

    def test_custom(self):
        c = ConsensusConfig(
            prompt="test",
            models=["gpt-4o", "claude"],
            rounds=3,
            strategy="vote",
            budget=1.0,
        )
        assert c.prompt == "test"
        assert len(c.models) == 2
        assert c.rounds == 3
        assert c.budget == 1.0
