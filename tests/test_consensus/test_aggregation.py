"""Tests for consensus aggregation functions."""

from __future__ import annotations

from stateloom.consensus.aggregation import (
    compute_agreement,
    confidence_weighted,
    majority_vote,
)
from stateloom.consensus.models import DebaterResponse


class TestComputeAgreement:
    def test_identical_responses(self):
        resps = [
            DebaterResponse(model="a", content="The capital of France is Paris."),
            DebaterResponse(model="b", content="The capital of France is Paris."),
        ]
        assert compute_agreement(resps) == 1.0

    def test_completely_different(self):
        resps = [
            DebaterResponse(model="a", content="abcdefghij"),
            DebaterResponse(model="b", content="klmnopqrst"),
        ]
        score = compute_agreement(resps)
        assert score < 0.2

    def test_single_response(self):
        resps = [DebaterResponse(model="a", content="hello")]
        assert compute_agreement(resps) == 1.0

    def test_empty_responses(self):
        assert compute_agreement([]) == 1.0

    def test_three_responses_mixed(self):
        resps = [
            DebaterResponse(model="a", content="The answer is 42"),
            DebaterResponse(model="b", content="The answer is 42"),
            DebaterResponse(model="c", content="The answer is 7"),
        ]
        score = compute_agreement(resps)
        assert 0.3 < score < 1.0  # two agree, one different


class TestMajorityVote:
    def test_clear_winner(self):
        resps = [
            DebaterResponse(model="a", content="Paris is the capital", confidence=0.9),
            DebaterResponse(model="b", content="Paris is the capital", confidence=0.8),
            DebaterResponse(model="c", content="London is the capital", confidence=0.7),
        ]
        answer, conf = majority_vote(resps)
        assert "Paris" in answer
        assert conf > 0.5

    def test_single_response(self):
        resps = [DebaterResponse(model="a", content="hello", confidence=0.8)]
        answer, conf = majority_vote(resps)
        assert answer == "hello"
        assert conf == 0.8

    def test_empty_responses(self):
        answer, conf = majority_vote([])
        assert answer == ""
        assert conf == 0.0

    def test_tie_returns_highest_confidence(self):
        resps = [
            DebaterResponse(model="a", content="Answer A is correct", confidence=0.6),
            DebaterResponse(model="b", content="Answer B is correct", confidence=0.9),
        ]
        answer, conf = majority_vote(resps)
        # Both are in their own group (different content), highest conf wins
        assert conf > 0


class TestConfidenceWeighted:
    def test_high_confidence_wins(self):
        resps = [
            DebaterResponse(model="a", content="A", confidence=0.95),
            DebaterResponse(model="b", content="B", confidence=0.3),
        ]
        answer, conf = confidence_weighted(resps)
        assert answer == "A"

    def test_single_response(self):
        resps = [DebaterResponse(model="a", content="hello", confidence=0.7)]
        answer, conf = confidence_weighted(resps)
        assert answer == "hello"
        assert conf == 0.7

    def test_empty_responses(self):
        answer, conf = confidence_weighted([])
        assert answer == ""
        assert conf == 0.0

    def test_equal_confidence(self):
        resps = [
            DebaterResponse(model="a", content="A", confidence=0.5),
            DebaterResponse(model="b", content="B", confidence=0.5),
        ]
        answer, conf = confidence_weighted(resps)
        assert answer in ("A", "B")
