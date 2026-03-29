"""Tests for SQLite store."""

import os
import tempfile

import pytest

from stateloom.core.event import KillSwitchEvent, LLMCallEvent, PIIDetectionEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType
from stateloom.store.sqlite_store import SQLiteStore


@pytest.fixture
def sqlite_store():
    """Create a temporary SQLite store."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(path=path)
    yield store
    store.close()
    os.unlink(path)


def test_save_and_get_session(sqlite_store):
    session = Session(id="s1", name="Test")
    session.total_cost = 0.05
    session.total_tokens = 500
    sqlite_store.save_session(session)

    retrieved = sqlite_store.get_session("s1")
    assert retrieved is not None
    assert retrieved.id == "s1"
    assert retrieved.name == "Test"
    assert retrieved.total_cost == 0.05
    assert retrieved.total_tokens == 500


def test_list_sessions(sqlite_store):
    sqlite_store.save_session(Session(id="s1"))
    sqlite_store.save_session(Session(id="s2"))
    sessions = sqlite_store.list_sessions()
    assert len(sessions) == 2


def test_save_and_get_llm_event(sqlite_store):
    event = LLMCallEvent(
        session_id="s1",
        step=1,
        provider="openai",
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.00075,
        latency_ms=1234.5,
    )
    sqlite_store.save_event(event)

    events = sqlite_store.get_session_events("s1")
    assert len(events) == 1
    e = events[0]
    assert e.provider == "openai"
    assert e.model == "gpt-4o"
    assert e.prompt_tokens == 100
    assert e.cost == 0.00075


def test_save_and_get_pii_event(sqlite_store):
    event = PIIDetectionEvent(
        session_id="s1",
        step=1,
        pii_type="email",
        mode="audit",
        pii_field="messages[0].content",
        action_taken="logged",
    )
    sqlite_store.save_event(event)

    events = sqlite_store.get_session_events("s1", event_type="pii_detection")
    assert len(events) == 1
    assert events[0].pii_type == "email"


def test_cross_session_events(sqlite_store):
    """get_session_events("") should return events across all sessions."""
    sqlite_store.save_event(
        LLMCallEvent(session_id="s1", step=1, provider="openai", model="gpt-4o")
    )
    sqlite_store.save_event(
        LLMCallEvent(session_id="s2", step=1, provider="openai", model="gpt-4o")
    )
    sqlite_store.save_event(
        LLMCallEvent(session_id="s3", step=1, provider="anthropic", model="claude")
    )
    events = sqlite_store.get_session_events("")
    assert len(events) == 3


def test_cross_session_events_with_type_filter(sqlite_store):
    """get_session_events("", event_type=...) filters by type across sessions."""
    sqlite_store.save_event(LLMCallEvent(session_id="s1", step=1))
    sqlite_store.save_event(
        PIIDetectionEvent(
            session_id="s2",
            step=1,
            pii_type="email",
            mode="audit",
            pii_field="f",
            action_taken="logged",
        )
    )
    events = sqlite_store.get_session_events("", event_type="llm_call")
    assert len(events) == 1
    assert events[0].session_id == "s1"


def test_global_stats(sqlite_store):
    s = Session(id="s1")
    s.total_cost = 0.10
    s.call_count = 10
    sqlite_store.save_session(s)

    stats = sqlite_store.get_global_stats()
    assert stats["total_cost"] == 0.10
    assert stats["total_calls"] == 10


def test_save_session_with_events_atomic(sqlite_store):
    """Atomic save persists session + multiple events in one transaction."""
    session = Session(id="s-atomic", name="Atomic Test")
    session.total_cost = 0.10
    session.call_count = 2

    events = [
        LLMCallEvent(
            session_id="s-atomic",
            step=1,
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.05,
            latency_ms=500.0,
        ),
        PIIDetectionEvent(
            session_id="s-atomic",
            step=2,
            pii_type="email",
            mode="audit",
            pii_field="messages[0].content",
            action_taken="logged",
        ),
    ]

    sqlite_store.save_session_with_events(session, events)

    # Verify session
    retrieved = sqlite_store.get_session("s-atomic")
    assert retrieved is not None
    assert retrieved.total_cost == 0.10
    assert retrieved.call_count == 2

    # Verify events
    stored_events = sqlite_store.get_session_events("s-atomic")
    assert len(stored_events) == 2
    assert stored_events[0].provider == "openai"
    assert stored_events[1].pii_type == "email"


def test_cleanup(sqlite_store):
    from datetime import datetime, timedelta, timezone

    event = LLMCallEvent(session_id="s1", step=1)
    # Manually set old timestamp
    event.timestamp = datetime.now(timezone.utc) - timedelta(days=60)
    sqlite_store.save_event(event)

    deleted = sqlite_store.cleanup(retention_days=30)
    assert deleted == 1


def test_kill_switch_event_roundtrip(sqlite_store):
    """KillSwitchEvent with webhook fields survives save/load cycle."""
    session = Session(id="ks-session")
    sqlite_store.save_session(session)

    event = KillSwitchEvent(
        session_id="ks-session",
        step=1,
        reason="cost_overrun",
        message="GPT-4 suspended due to cost overrun",
        matched_rule={"model": "gpt-4", "reason": "cost_overrun"},
        blocked_model="gpt-4",
        blocked_provider="openai",
        webhook_fired=True,
        webhook_url="https://hooks.example.com/ks",
    )
    sqlite_store.save_event(event)

    events = sqlite_store.get_session_events("ks-session")
    assert len(events) == 1
    loaded = events[0]
    assert isinstance(loaded, KillSwitchEvent)
    assert loaded.event_type == EventType.KILL_SWITCH
    assert loaded.reason == "cost_overrun"
    assert loaded.message == "GPT-4 suspended due to cost overrun"
    assert loaded.matched_rule == {"model": "gpt-4", "reason": "cost_overrun"}
    assert loaded.blocked_model == "gpt-4"
    assert loaded.blocked_provider == "openai"
    assert loaded.webhook_fired is True
    assert loaded.webhook_url == "https://hooks.example.com/ks"


def test_kill_switch_event_roundtrip_no_webhook(sqlite_store):
    """KillSwitchEvent without webhook has defaults after round-trip."""
    session = Session(id="ks-session-2")
    sqlite_store.save_session(session)

    event = KillSwitchEvent(
        session_id="ks-session-2",
        step=1,
        reason="kill_switch_active",
        message="Service temporarily unavailable.",
        blocked_model="gpt-4",
        blocked_provider="openai",
    )
    sqlite_store.save_event(event)

    events = sqlite_store.get_session_events("ks-session-2")
    assert len(events) == 1
    loaded = events[0]
    assert isinstance(loaded, KillSwitchEvent)
    assert loaded.webhook_fired is False
    assert loaded.webhook_url == ""
