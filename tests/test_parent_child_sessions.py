"""Tests for parent-child session hierarchy."""

from __future__ import annotations

import pytest

from stateloom.core.session import Session, SessionManager
from stateloom.store.memory_store import MemoryStore


class TestSessionParentField:
    """Session dataclass parent_session_id field."""

    def test_default_is_none(self):
        session = Session(id="child")
        assert session.parent_session_id is None

    def test_set_parent_id(self):
        session = Session(id="child", parent_session_id="parent-123")
        assert session.parent_session_id == "parent-123"


class TestSessionManagerCreate:
    """SessionManager.create() accepts parent_session_id."""

    def test_create_with_parent(self):
        mgr = SessionManager()
        session = mgr.create(session_id="child-1", parent_session_id="parent-1")
        assert session.parent_session_id == "parent-1"
        assert session.id == "child-1"

    def test_create_without_parent(self):
        mgr = SessionManager()
        session = mgr.create(session_id="standalone")
        assert session.parent_session_id is None


class TestMemoryStoreChildSessions:
    """MemoryStore.list_child_sessions() filtering."""

    def test_list_child_sessions_empty(self):
        store = MemoryStore()
        result = store.list_child_sessions("parent-1")
        assert result == []

    def test_list_child_sessions_returns_children(self):
        store = MemoryStore()

        parent = Session(id="parent-1")
        child1 = Session(id="child-1", parent_session_id="parent-1")
        child2 = Session(id="child-2", parent_session_id="parent-1")
        unrelated = Session(id="other", parent_session_id="different-parent")

        for s in [parent, child1, child2, unrelated]:
            store.save_session(s)

        children = store.list_child_sessions("parent-1")
        ids = [s.id for s in children]
        assert set(ids) == {"child-1", "child-2"}

    def test_list_child_sessions_respects_limit(self):
        store = MemoryStore()
        parent = Session(id="parent-1")
        store.save_session(parent)

        for i in range(10):
            store.save_session(Session(id=f"child-{i}", parent_session_id="parent-1"))

        children = store.list_child_sessions("parent-1", limit=3)
        assert len(children) == 3

    def test_parent_not_in_children(self):
        store = MemoryStore()
        parent = Session(id="parent-1")
        child = Session(id="child-1", parent_session_id="parent-1")
        store.save_session(parent)
        store.save_session(child)

        children = store.list_child_sessions("parent-1")
        ids = [s.id for s in children]
        assert "parent-1" not in ids
        assert "child-1" in ids
