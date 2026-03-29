"""Tests for in-memory store."""

from stateloom.core.event import LLMCallEvent
from stateloom.core.session import Session
from stateloom.core.types import SessionStatus
from stateloom.store.memory_store import MemoryStore


def test_save_and_get_session():
    store = MemoryStore()
    session = Session(id="s1", name="Test")
    store.save_session(session)
    retrieved = store.get_session("s1")
    assert retrieved is not None
    assert retrieved.id == "s1"
    assert retrieved.name == "Test"


def test_get_nonexistent_session():
    store = MemoryStore()
    assert store.get_session("nonexistent") is None


def test_list_sessions():
    store = MemoryStore()
    store.save_session(Session(id="s1"))
    store.save_session(Session(id="s2"))
    store.save_session(Session(id="s3"))
    sessions = store.list_sessions()
    assert len(sessions) == 3


def test_list_sessions_with_status():
    store = MemoryStore()
    s1 = Session(id="s1")
    s2 = Session(id="s2")
    s2.end()
    store.save_session(s1)
    store.save_session(s2)
    active = store.list_sessions(status="active")
    assert len(active) == 1
    assert active[0].id == "s1"


def test_save_and_get_events():
    store = MemoryStore()
    event = LLMCallEvent(session_id="s1", step=1, provider="openai", model="gpt-4o")
    store.save_event(event)
    events = store.get_session_events("s1")
    assert len(events) == 1
    assert events[0].model == "gpt-4o"


def test_get_events_filtered_by_type():
    store = MemoryStore()
    store.save_event(LLMCallEvent(session_id="s1", step=1))
    store.save_event(LLMCallEvent(session_id="s1", step=2))
    events = store.get_session_events("s1", event_type="llm_call")
    assert len(events) == 2


def test_cross_session_events():
    """get_session_events("") should return events across all sessions."""
    store = MemoryStore()
    store.save_event(LLMCallEvent(session_id="s1", step=1, provider="openai", model="gpt-4o"))
    store.save_event(LLMCallEvent(session_id="s2", step=1, provider="openai", model="gpt-4o"))
    store.save_event(LLMCallEvent(session_id="s3", step=1, provider="anthropic", model="claude"))
    events = store.get_session_events("")
    assert len(events) == 3


def test_cross_session_events_with_type_filter():
    """get_session_events("", event_type=...) filters by type across sessions."""
    from stateloom.core.event import PIIDetectionEvent

    store = MemoryStore()
    store.save_event(LLMCallEvent(session_id="s1", step=1))
    store.save_event(
        PIIDetectionEvent(
            session_id="s2",
            step=1,
            pii_type="email",
            mode="audit",
            pii_field="f",
            action_taken="logged",
        )
    )
    events = store.get_session_events("", event_type="llm_call")
    assert len(events) == 1
    assert events[0].session_id == "s1"


def test_global_stats():
    store = MemoryStore()
    s = Session(id="s1")
    s.total_cost = 0.05
    s.total_tokens = 1000
    s.call_count = 5
    s.cache_hits = 2
    s.cache_savings = 0.01
    store.save_session(s)
    stats = store.get_global_stats()
    assert stats["total_sessions"] == 1
    assert stats["total_cost"] == 0.05
    assert stats["total_calls"] == 5
