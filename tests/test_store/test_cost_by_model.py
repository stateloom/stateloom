"""Tests for per-model cost breakdown persistence in SQLite store."""

import pytest

from stateloom.core.session import Session
from stateloom.store.sqlite_store import SQLiteStore


@pytest.fixture()
def store(tmp_path):
    return SQLiteStore(str(tmp_path / "test.db"))


def test_cost_by_model_roundtrip(store):
    session = Session(id="cbm-1")
    session.add_cost(0.001, prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini")
    session.add_cost(0.05, prompt_tokens=200, completion_tokens=100, model="gpt-4o")
    store.save_session(session)

    loaded = store.get_session("cbm-1")
    assert loaded is not None
    assert loaded.cost_by_model["gpt-4o-mini"] == 0.001
    assert loaded.cost_by_model["gpt-4o"] == 0.05
    assert loaded.tokens_by_model["gpt-4o-mini"]["prompt_tokens"] == 100
    assert loaded.tokens_by_model["gpt-4o"]["total_tokens"] == 300


def test_empty_cost_by_model_roundtrip(store):
    session = Session(id="cbm-2")
    session.add_cost(0.01, prompt_tokens=100, completion_tokens=50)
    store.save_session(session)

    loaded = store.get_session("cbm-2")
    assert loaded is not None
    assert loaded.cost_by_model == {}
    assert loaded.tokens_by_model == {}


def test_multi_model_roundtrip_with_tokens(store):
    session = Session(id="cbm-3")
    session.add_cost(0.001, prompt_tokens=50, completion_tokens=25, model="gpt-4o-mini")
    session.add_cost(0.002, prompt_tokens=80, completion_tokens=40, model="gpt-4o-mini")
    session.add_cost(
        0.10, prompt_tokens=500, completion_tokens=250, model="claude-sonnet-4-20250514"
    )
    store.save_session(session)

    loaded = store.get_session("cbm-3")
    assert loaded is not None
    assert len(loaded.cost_by_model) == 2
    assert loaded.cost_by_model["gpt-4o-mini"] == 0.003
    assert loaded.cost_by_model["claude-sonnet-4-20250514"] == 0.10
    assert loaded.tokens_by_model["gpt-4o-mini"] == {
        "prompt_tokens": 130,
        "completion_tokens": 65,
        "total_tokens": 195,
    }
    assert loaded.tokens_by_model["claude-sonnet-4-20250514"] == {
        "prompt_tokens": 500,
        "completion_tokens": 250,
        "total_tokens": 750,
    }
