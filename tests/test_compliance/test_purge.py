"""Tests for the compliance purge engine."""

from __future__ import annotations

import pytest

from stateloom.compliance.purge import PurgeEngine, PurgeResult
from stateloom.core.event import LLMCallEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType
from stateloom.store.memory_store import MemoryStore


class TestPurgeEngineMemoryStore:
    def _setup_store(self) -> MemoryStore:
        store = MemoryStore()
        # Create sessions with user metadata
        s1 = Session(id="sess-1", metadata={"user_email": "alice@example.com"})
        s2 = Session(id="sess-2", metadata={"user_email": "bob@example.com"})
        s3 = Session(id="sess-3", metadata={"user_email": "alice@example.com"})
        store.save_session(s1)
        store.save_session(s2)
        store.save_session(s3)

        # Create events
        store.save_event(LLMCallEvent(session_id="sess-1", step=1, model="gpt-4"))
        store.save_event(LLMCallEvent(session_id="sess-1", step=2, model="gpt-4"))
        store.save_event(LLMCallEvent(session_id="sess-2", step=1, model="gpt-4"))
        store.save_event(LLMCallEvent(session_id="sess-3", step=1, model="gpt-4"))
        return store

    def test_purge_matching_user(self):
        store = self._setup_store()
        engine = PurgeEngine(store)
        result = engine.purge("alice@example.com")
        assert result.sessions_deleted == 2
        assert result.events_deleted == 3  # 2 from sess-1 + 1 from sess-3
        assert result.audit_event_id != ""

    def test_purge_non_matching(self):
        store = self._setup_store()
        engine = PurgeEngine(store)
        result = engine.purge("nobody@example.com")
        assert result.sessions_deleted == 0
        assert result.events_deleted == 0

    def test_purge_creates_audit_event(self):
        store = self._setup_store()
        engine = PurgeEngine(store)
        result = engine.purge("alice@example.com", standard="gdpr")
        # Check audit event was saved
        events = store.get_session_events("", event_type="compliance_audit")
        assert len(events) == 1
        audit = events[0]
        assert audit.compliance_standard == "gdpr"
        assert audit.action == "data_purged"
        assert "alice@example.com" in audit.justification

    def test_purge_bob_only(self):
        store = self._setup_store()
        engine = PurgeEngine(store)
        result = engine.purge("bob@example.com")
        assert result.sessions_deleted == 1
        assert result.events_deleted == 1

    def test_purge_with_cache(self):
        store = self._setup_store()

        class FakeCacheStore:
            def purge_by_content(self, identifier):
                return 5

        engine = PurgeEngine(store, cache_store=FakeCacheStore())
        result = engine.purge("alice@example.com")
        assert result.cache_entries_deleted == 5

    def test_purge_no_cache_store(self):
        store = self._setup_store()
        engine = PurgeEngine(store, cache_store=None)
        result = engine.purge("alice@example.com")
        assert result.cache_entries_deleted == 0

    def test_purge_result_dataclass(self):
        r = PurgeResult(user_identifier="test")
        assert r.sessions_deleted == 0
        assert r.events_deleted == 0
        assert r.cache_entries_deleted == 0
        assert r.audit_event_id == ""


class TestPurgeSession:
    def test_purge_single_session(self):
        store = MemoryStore()
        s = Session(id="sess-1")
        store.save_session(s)
        store.save_event(LLMCallEvent(session_id="sess-1", step=1))
        store.save_event(LLMCallEvent(session_id="sess-1", step=2))

        deleted = store.purge_session("sess-1")
        assert deleted == 2
        assert store.get_session("sess-1") is None

    def test_purge_nonexistent_session(self):
        store = MemoryStore()
        deleted = store.purge_session("nonexistent")
        assert deleted == 0

    def test_purge_preserves_other_sessions(self):
        store = MemoryStore()
        store.save_session(Session(id="sess-1"))
        store.save_session(Session(id="sess-2"))
        store.save_event(LLMCallEvent(session_id="sess-1", step=1))
        store.save_event(LLMCallEvent(session_id="sess-2", step=1))

        store.purge_session("sess-1")
        assert store.get_session("sess-2") is not None
        events = store.get_session_events("sess-2")
        assert len(events) == 1
