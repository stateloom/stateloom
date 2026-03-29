"""Tests for session management."""

from stateloom.core.session import Session, SessionManager
from stateloom.core.types import SessionStatus


def test_session_defaults():
    session = Session()
    assert session.id is not None
    assert len(session.id) == 12
    assert session.status == SessionStatus.ACTIVE
    assert session.total_cost == 0.0
    assert session.step_counter == 0


def test_session_next_step():
    session = Session()
    assert session.next_step() == 1
    assert session.next_step() == 2
    assert session.next_step() == 3
    assert session.step_counter == 3


def test_session_add_cost():
    session = Session()
    session.add_cost(0.01, prompt_tokens=100, completion_tokens=50)
    assert session.total_cost == 0.01
    assert session.total_tokens == 150
    assert session.total_prompt_tokens == 100
    assert session.total_completion_tokens == 50
    assert session.call_count == 1

    session.add_cost(0.02, prompt_tokens=200, completion_tokens=100)
    assert session.total_cost == 0.03
    assert session.total_tokens == 450
    assert session.call_count == 2


def test_session_cache_hit():
    session = Session()
    session.add_cache_hit(0.05)
    assert session.cache_hits == 1
    assert session.cache_savings == 0.05


def test_session_end():
    session = Session()
    assert session.ended_at is None
    session.end()
    assert session.ended_at is not None
    assert session.status == SessionStatus.COMPLETED


def test_session_manager_create():
    manager = SessionManager()
    session = manager.create(session_id="test-123", name="Test")
    assert session.id == "test-123"
    assert session.name == "Test"


def test_session_manager_get():
    manager = SessionManager()
    manager.create(session_id="s1")
    assert manager.get("s1") is not None
    assert manager.get("s2") is None


def test_session_manager_get_or_create():
    manager = SessionManager()
    s1 = manager.get_or_create("s1")
    s2 = manager.get_or_create("s1")
    assert s1 is s2  # Same session


def test_session_manager_list_active():
    manager = SessionManager()
    s1 = manager.create(session_id="s1")
    s2 = manager.create(session_id="s2")
    s1.end()
    active = manager.list_active()
    assert len(active) == 1
    assert active[0].id == "s2"


def test_session_manager_default_budget():
    manager = SessionManager()
    manager.set_default_budget(10.0)
    session = manager.create()
    assert session.budget == 10.0

    # Override budget
    session2 = manager.create(budget=5.0)
    assert session2.budget == 5.0


def test_session_cost_by_model_default():
    session = Session()
    assert session.cost_by_model == {}
    assert session.tokens_by_model == {}


def test_session_add_cost_with_model():
    session = Session()
    session.add_cost(0.01, prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini")
    assert session.cost_by_model == {"gpt-4o-mini": 0.01}
    assert session.tokens_by_model == {
        "gpt-4o-mini": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    }


def test_session_add_cost_multiple_models():
    session = Session()
    session.add_cost(0.001, prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini")
    session.add_cost(0.05, prompt_tokens=200, completion_tokens=100, model="gpt-4o")
    session.add_cost(0.002, prompt_tokens=150, completion_tokens=75, model="gpt-4o-mini")

    assert session.cost_by_model["gpt-4o-mini"] == 0.003
    assert session.cost_by_model["gpt-4o"] == 0.05
    assert session.tokens_by_model["gpt-4o-mini"] == {
        "prompt_tokens": 250,
        "completion_tokens": 125,
        "total_tokens": 375,
    }
    assert session.tokens_by_model["gpt-4o"] == {
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "total_tokens": 300,
    }
    # Aggregates still work
    assert abs(session.total_cost - 0.053) < 1e-9
    assert session.total_tokens == 675
    assert session.call_count == 3


def test_session_add_cost_without_model_no_breakdown():
    session = Session()
    session.add_cost(0.01, prompt_tokens=100, completion_tokens=50)
    assert session.cost_by_model == {}
    assert session.tokens_by_model == {}
    assert session.total_cost == 0.01


def test_session_add_cost_empty_model_no_breakdown():
    session = Session()
    session.add_cost(0.01, prompt_tokens=100, completion_tokens=50, model="")
    assert session.cost_by_model == {}
    assert session.tokens_by_model == {}
