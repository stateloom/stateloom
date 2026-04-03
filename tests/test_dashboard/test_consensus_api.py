"""Tests for consensus dashboard API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.core.event import ConsensusEvent, DebateRoundEvent
from stateloom.dashboard.api import create_api_router
from stateloom.store.memory_store import MemoryStore


def _make_gate():
    """Create a mock Gate backed by a real MemoryStore."""
    gate = MagicMock()
    gate.store = MemoryStore()
    gate.config = MagicMock()
    gate.config.async_jobs_enabled = False
    return gate


def _seed_consensus_data(store, session_id="sess-consensus-1", strategy="debate"):
    """Seed store with a consensus event and debate round events."""
    from stateloom.core.session import Session

    session = Session(id=session_id, name=f"consensus-{strategy}")
    session.total_cost = 0.05
    store.save_session(session)

    # Debate round event
    round_event = DebateRoundEvent(
        session_id=session_id,
        round_number=1,
        strategy=strategy,
        models=["gpt-4o", "claude-sonnet-4-20250514"],
        responses_summary=[
            {
                "model": "gpt-4o",
                "confidence": 0.85,
                "cost": 0.01,
                "content_preview": "The answer is 42.",
            },
            {
                "model": "claude-sonnet-4-20250514",
                "confidence": 0.90,
                "cost": 0.012,
                "content_preview": "I believe the answer is 42.",
            },
        ],
        agreement_score=0.82,
        consensus_reached=True,
        round_cost=0.022,
        round_duration_ms=1500.0,
    )
    store.save_event(round_event)

    # Consensus event
    consensus_event = ConsensusEvent(
        session_id=session_id,
        strategy=strategy,
        models=["gpt-4o", "claude-sonnet-4-20250514"],
        total_rounds=1,
        final_answer_preview="The answer is 42.",
        confidence=0.88,
        total_cost=0.05,
        total_duration_ms=2000.0,
        early_stopped=False,
        aggregation_method="judge_synthesis",
        winner_model="claude-sonnet-4-20250514",
    )
    store.save_event(consensus_event)

    return session


@pytest.fixture
def gate():
    return _make_gate()


@pytest.fixture
def client(gate):
    app = FastAPI()
    router = create_api_router(gate)
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


class TestListConsensusRuns:
    def test_empty_list(self, client):
        resp = client.get("/api/v1/consensus-runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"] == []
        assert data["total"] == 0

    def test_list_with_data(self, client, gate):
        _seed_consensus_data(gate.store)
        resp = client.get("/api/v1/consensus-runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        run = data["runs"][0]
        assert run["session_id"] == "sess-consensus-1"
        assert run["strategy"] == "debate"
        assert run["total_rounds"] == 1
        assert run["confidence"] == 0.88
        assert run["total_cost"] == 0.05
        assert run["winner_model"] == "claude-sonnet-4-20250514"

    def test_list_filter_by_strategy(self, client, gate):
        _seed_consensus_data(gate.store, session_id="s1", strategy="debate")
        _seed_consensus_data(gate.store, session_id="s2", strategy="vote")
        resp = client.get("/api/v1/consensus-runs?strategy=vote")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["runs"][0]["strategy"] == "vote"

    def test_list_multiple_runs(self, client, gate):
        _seed_consensus_data(gate.store, session_id="s1", strategy="debate")
        _seed_consensus_data(gate.store, session_id="s2", strategy="vote")
        _seed_consensus_data(gate.store, session_id="s3", strategy="self_consistency")
        resp = client.get("/api/v1/consensus-runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_list_has_timestamp(self, client, gate):
        _seed_consensus_data(gate.store)
        resp = client.get("/api/v1/consensus-runs")
        run = resp.json()["runs"][0]
        assert "timestamp" in run
        assert "T" in run["timestamp"]  # ISO format


class TestGetConsensusRun:
    def test_detail_success(self, client, gate):
        _seed_consensus_data(gate.store)
        resp = client.get("/api/v1/consensus-runs/sess-consensus-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-consensus-1"
        assert data["consensus"]["strategy"] == "debate"
        assert data["consensus"]["confidence"] == 0.88
        assert data["consensus"]["winner_model"] == "claude-sonnet-4-20250514"
        assert data["consensus"]["aggregation_method"] == "judge_synthesis"

    def test_detail_includes_rounds(self, client, gate):
        _seed_consensus_data(gate.store)
        resp = client.get("/api/v1/consensus-runs/sess-consensus-1")
        data = resp.json()
        assert len(data["rounds"]) == 1
        rd = data["rounds"][0]
        assert rd["round_number"] == 1
        assert rd["agreement_score"] == 0.82
        assert rd["consensus_reached"] is True
        assert len(rd["responses_summary"]) == 2
        assert rd["responses_summary"][0]["model"] == "gpt-4o"

    def test_detail_includes_session(self, client, gate):
        _seed_consensus_data(gate.store)
        resp = client.get("/api/v1/consensus-runs/sess-consensus-1")
        data = resp.json()
        assert "session" in data
        assert data["session"]["id"] == "sess-consensus-1"

    def test_detail_includes_children(self, client, gate):
        from stateloom.core.session import Session

        _seed_consensus_data(gate.store)
        # Add a child session
        child = Session(id="child-debater-1", name="debate-gpt-4o-r1")
        child.parent_session_id = "sess-consensus-1"
        child.total_cost = 0.01
        gate.store.save_session(child)

        resp = client.get("/api/v1/consensus-runs/sess-consensus-1")
        data = resp.json()
        assert len(data["children"]) == 1
        assert data["children"][0]["id"] == "child-debater-1"
        assert data["children"][0]["name"] == "debate-gpt-4o-r1"

    def test_detail_404_no_session(self, client):
        resp = client.get("/api/v1/consensus-runs/nonexistent")
        assert resp.status_code == 404

    def test_detail_404_no_consensus_event(self, client, gate):
        from stateloom.core.session import Session

        # Session exists but has no consensus event
        session = Session(id="plain-session")
        gate.store.save_session(session)
        resp = client.get("/api/v1/consensus-runs/plain-session")
        assert resp.status_code == 404
        assert "No consensus data" in resp.json()["detail"]

    def test_detail_multi_round(self, client, gate):
        from stateloom.core.session import Session

        session = Session(id="multi-round", name="consensus-debate")
        gate.store.save_session(session)

        for rn in range(1, 4):
            gate.store.save_event(
                DebateRoundEvent(
                    session_id="multi-round",
                    round_number=rn,
                    strategy="debate",
                    models=["gpt-4o", "claude-sonnet-4-20250514"],
                    responses_summary=[],
                    agreement_score=0.5 + rn * 0.1,
                    consensus_reached=rn == 3,
                    round_cost=0.01,
                    round_duration_ms=500.0,
                )
            )

        gate.store.save_event(
            ConsensusEvent(
                session_id="multi-round",
                strategy="debate",
                models=["gpt-4o", "claude-sonnet-4-20250514"],
                total_rounds=3,
                final_answer_preview="Final answer.",
                confidence=0.92,
                total_cost=0.03,
                total_duration_ms=3000.0,
                early_stopped=False,
                aggregation_method="judge_synthesis",
                winner_model="gpt-4o",
            )
        )

        resp = client.get("/api/v1/consensus-runs/multi-round")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["rounds"]) == 3
        assert data["rounds"][0]["round_number"] == 1
        assert data["rounds"][2]["round_number"] == 3
        assert data["consensus"]["total_rounds"] == 3
