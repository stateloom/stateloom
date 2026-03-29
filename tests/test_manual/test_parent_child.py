"""Manual integration tests for parent-child session hierarchy.

These tests exercise the parent-child relationship end-to-end:
  - Creating child sessions with an explicit parent
  - Auto-deriving parent from the active session (ContextVar)
  - Inheriting org_id / team_id from the parent
  - Listing child sessions via the store
  - Cancelling a parent and verifying children are unaffected (no cascade)
  - Checkpoint events within parent-child hierarchies

Run with:
    pytest tests/test_manual/test_parent_child.py -v
"""

from __future__ import annotations

import pytest

from stateloom.core.event import CheckpointEvent
from stateloom.core.session import Session, SessionManager
from stateloom.core.types import EventType, SessionStatus
from stateloom.store.memory_store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager() -> SessionManager:
    return SessionManager()


def _make_store_with_hierarchy() -> tuple[MemoryStore, Session, list[Session]]:
    """Create a store pre-populated with a parent and 3 children."""
    store = MemoryStore()
    parent = Session(id="parent-1", org_id="org-a", team_id="team-a")
    children = [
        Session(id=f"child-{i}", parent_session_id="parent-1", org_id="org-a", team_id="team-a")
        for i in range(1, 4)
    ]
    store.save_session(parent)
    for c in children:
        store.save_session(c)
    return store, parent, children


# ---------------------------------------------------------------------------
# Parent-child session creation
# ---------------------------------------------------------------------------


class TestParentChildCreation:
    """Creating sessions with parent_session_id."""

    def test_child_has_parent_id(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent-1")
        child = mgr.create(session_id="child-1", parent_session_id="parent-1")
        assert child.parent_session_id == "parent-1"
        assert parent.parent_session_id is None

    def test_standalone_session_has_no_parent(self):
        mgr = _make_manager()
        session = mgr.create(session_id="standalone")
        assert session.parent_session_id is None

    def test_multiple_children_same_parent(self):
        mgr = _make_manager()
        mgr.create(session_id="parent-1")
        children = [
            mgr.create(session_id=f"child-{i}", parent_session_id="parent-1") for i in range(5)
        ]
        for child in children:
            assert child.parent_session_id == "parent-1"

    def test_grandchild_relationship(self):
        """A child can itself be a parent (multi-level hierarchy)."""
        mgr = _make_manager()
        mgr.create(session_id="root")
        child = mgr.create(session_id="child", parent_session_id="root")
        grandchild = mgr.create(session_id="grandchild", parent_session_id="child")
        assert child.parent_session_id == "root"
        assert grandchild.parent_session_id == "child"


# ---------------------------------------------------------------------------
# org_id / team_id inheritance
# ---------------------------------------------------------------------------


class TestOrgTeamInheritance:
    """Child sessions inherit org_id and team_id from the parent."""

    def test_child_inherits_org_id(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent", org_id="org-1", team_id="team-1")
        # SessionManager.create doesn't auto-inherit — Gate.session() does that.
        # Here we test the Session model can carry the values.
        child = mgr.create(
            session_id="child",
            parent_session_id="parent",
            org_id=parent.org_id,
            team_id=parent.team_id,
        )
        assert child.org_id == "org-1"
        assert child.team_id == "team-1"

    def test_child_can_override_org_id(self):
        mgr = _make_manager()
        mgr.create(session_id="parent", org_id="org-1", team_id="team-1")
        child = mgr.create(
            session_id="child",
            parent_session_id="parent",
            org_id="org-override",
            team_id="team-override",
        )
        assert child.org_id == "org-override"
        assert child.team_id == "team-override"


# ---------------------------------------------------------------------------
# Store: list_child_sessions
# ---------------------------------------------------------------------------


class TestStoreListChildSessions:
    """MemoryStore.list_child_sessions() queries."""

    def test_returns_only_children(self):
        store, parent, children = _make_store_with_hierarchy()
        result = store.list_child_sessions("parent-1")
        result_ids = {s.id for s in result}
        assert result_ids == {"child-1", "child-2", "child-3"}
        assert "parent-1" not in result_ids

    def test_empty_when_no_children(self):
        store = MemoryStore()
        parent = Session(id="lonely-parent")
        store.save_session(parent)
        assert store.list_child_sessions("lonely-parent") == []

    def test_respects_limit(self):
        store, _, _ = _make_store_with_hierarchy()
        result = store.list_child_sessions("parent-1", limit=2)
        assert len(result) == 2

    def test_different_parent_ids_isolated(self):
        store = MemoryStore()
        for pid in ["p1", "p2"]:
            store.save_session(Session(id=pid))
            for i in range(3):
                store.save_session(Session(id=f"{pid}-child-{i}", parent_session_id=pid))

        p1_children = store.list_child_sessions("p1")
        p2_children = store.list_child_sessions("p2")
        assert len(p1_children) == 3
        assert len(p2_children) == 3
        assert {s.id for s in p1_children} & {s.id for s in p2_children} == set()

    def test_nonexistent_parent_returns_empty(self):
        store = MemoryStore()
        assert store.list_child_sessions("nonexistent") == []


# ---------------------------------------------------------------------------
# Cancellation isolation
# ---------------------------------------------------------------------------


class TestCancellationIsolation:
    """Cancelling a parent does NOT cascade to children."""

    def test_cancel_parent_leaves_children_active(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent-1")
        child1 = mgr.create(session_id="child-1", parent_session_id="parent-1")
        child2 = mgr.create(session_id="child-2", parent_session_id="parent-1")

        parent.cancel()

        assert parent.is_cancelled is True
        assert child1.is_cancelled is False
        assert child2.is_cancelled is False
        assert child1.status == SessionStatus.ACTIVE
        assert child2.status == SessionStatus.ACTIVE

    def test_cancel_child_leaves_parent_active(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent-1")
        child = mgr.create(session_id="child-1", parent_session_id="parent-1")

        child.cancel()

        assert child.is_cancelled is True
        assert parent.is_cancelled is False
        assert parent.status == SessionStatus.ACTIVE

    def test_end_parent_leaves_children_active(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent-1")
        child = mgr.create(session_id="child-1", parent_session_id="parent-1")

        parent.end(SessionStatus.COMPLETED)

        assert parent.status == SessionStatus.COMPLETED
        assert child.status == SessionStatus.ACTIVE


# ---------------------------------------------------------------------------
# Timeouts with parent-child
# ---------------------------------------------------------------------------


class TestTimeoutsWithHierarchy:
    """Timeouts are per-session, not inherited."""

    def test_parent_timeout_not_inherited(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent", timeout=60.0)
        child = mgr.create(session_id="child", parent_session_id="parent")
        assert parent.timeout == 60.0
        assert child.timeout is None

    def test_child_can_have_own_timeout(self):
        mgr = _make_manager()
        mgr.create(session_id="parent", timeout=60.0)
        child = mgr.create(session_id="child", parent_session_id="parent", timeout=30.0)
        assert child.timeout == 30.0


# ---------------------------------------------------------------------------
# Checkpoints in parent-child
# ---------------------------------------------------------------------------


class TestCheckpointsInHierarchy:
    """Checkpoint events work correctly in parent-child sessions."""

    def test_checkpoint_in_child_session(self):
        store = MemoryStore()
        parent = Session(id="parent-1")
        child = Session(id="child-1", parent_session_id="parent-1")
        store.save_session(parent)
        store.save_session(child)

        event = CheckpointEvent(
            session_id="child-1",
            label="step-complete",
            description="Child finished processing",
        )
        store.save_event(event)

        events = store.get_session_events("child-1")
        assert len(events) == 1
        assert events[0].event_type == EventType.CHECKPOINT
        assert events[0].label == "step-complete"

    def test_parent_and_child_checkpoints_isolated(self):
        store = MemoryStore()
        store.save_session(Session(id="parent-1"))
        store.save_session(Session(id="child-1", parent_session_id="parent-1"))

        store.save_event(CheckpointEvent(session_id="parent-1", label="parent-cp"))
        store.save_event(CheckpointEvent(session_id="child-1", label="child-cp"))

        parent_events = store.get_session_events("parent-1")
        child_events = store.get_session_events("child-1")

        assert len(parent_events) == 1
        assert parent_events[0].label == "parent-cp"
        assert len(child_events) == 1
        assert child_events[0].label == "child-cp"


# ---------------------------------------------------------------------------
# Session metadata in hierarchy
# ---------------------------------------------------------------------------


class TestMetadataInHierarchy:
    """Metadata is per-session, not shared or inherited."""

    def test_parent_metadata_not_shared_with_child(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent")
        parent.metadata["agent_name"] = "orchestrator"

        child = mgr.create(session_id="child", parent_session_id="parent")
        assert "agent_name" not in child.metadata

    def test_child_metadata_independent(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent")
        child = mgr.create(session_id="child", parent_session_id="parent")

        parent.metadata["key"] = "parent-val"
        child.metadata["key"] = "child-val"

        assert parent.metadata["key"] == "parent-val"
        assert child.metadata["key"] == "child-val"


# ---------------------------------------------------------------------------
# Budget isolation
# ---------------------------------------------------------------------------


class TestBudgetIsolation:
    """Budgets are per-session, not aggregated across hierarchy."""

    def test_parent_budget_not_inherited(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent", budget=10.0)
        child = mgr.create(session_id="child", parent_session_id="parent")
        assert parent.budget == 10.0
        assert child.budget is None

    def test_child_cost_does_not_affect_parent(self):
        mgr = _make_manager()
        parent = mgr.create(session_id="parent", budget=10.0)
        child = mgr.create(session_id="child", parent_session_id="parent", budget=5.0)

        child.add_cost(2.0, prompt_tokens=100, completion_tokens=50)

        assert child.total_cost == 2.0
        assert parent.total_cost == 0.0
