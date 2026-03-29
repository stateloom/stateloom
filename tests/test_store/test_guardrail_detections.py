"""Tests for guardrail_detections persistence in stores."""

from __future__ import annotations

import pytest

from stateloom.core.session import Session
from stateloom.store.memory_store import MemoryStore
from stateloom.store.sqlite_store import SQLiteStore


class TestGuardrailDetectionsMemoryStore:
    def test_guardrail_detections_persisted(self):
        store = MemoryStore()
        session = Session(id="test-gr-1")
        session.add_guardrail_detection()
        session.add_guardrail_detection()
        store.save_session(session)

        loaded = store.get_session("test-gr-1")
        assert loaded is not None
        assert loaded.guardrail_detections == 2


class TestGuardrailDetectionsSQLiteStore:
    def test_guardrail_detections_persisted(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)

        session = Session(id="test-gr-sqlite-1")
        session.add_guardrail_detection()
        session.add_guardrail_detection()
        session.add_guardrail_detection()
        store.save_session(session)

        loaded = store.get_session("test-gr-sqlite-1")
        assert loaded is not None
        assert loaded.guardrail_detections == 3

    def test_guardrail_detections_in_global_stats(self, tmp_path):
        db_path = str(tmp_path / "test_stats.db")
        store = SQLiteStore(db_path)

        s1 = Session(id="test-gr-stats-1")
        s1.add_guardrail_detection()
        store.save_session(s1)

        s2 = Session(id="test-gr-stats-2")
        s2.add_guardrail_detection()
        s2.add_guardrail_detection()
        store.save_session(s2)

        stats = store.get_global_stats()
        assert stats["total_guardrail_detections"] == 3
