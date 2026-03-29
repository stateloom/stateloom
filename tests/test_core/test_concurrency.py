"""Tests for stateloom.patch_threading() — ContextVar propagation to child threads."""

import threading

import stateloom
from stateloom.concurrency import patch_threading
from stateloom.core.context import get_current_session, set_current_session
from stateloom.core.session import Session
from stateloom.intercept.unpatch import unpatch_all


def _init_gate():
    return stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )


def test_patch_threading_propagates_session():
    """Child thread inherits parent session after patch_threading()."""
    _init_gate()
    patch_threading()

    parent_session = Session(id="parent-session", name="Parent")
    set_current_session(parent_session)

    child_session_ids = []

    def child_target():
        session = get_current_session()
        if session:
            child_session_ids.append(session.id)

    t = threading.Thread(target=child_target)
    t.start()
    t.join()

    assert len(child_session_ids) == 1
    assert child_session_ids[0] == "parent-session"

    # Cleanup
    set_current_session(None)
    unpatch_all()


def test_without_patch_threading_no_propagation():
    """Without patch_threading, child thread does NOT see parent session."""
    _init_gate()

    parent_session = Session(id="parent-session", name="Parent")
    set_current_session(parent_session)

    child_session_ids = []

    def child_target():
        session = get_current_session()
        child_session_ids.append(session.id if session else None)

    t = threading.Thread(target=child_target)
    t.start()
    t.join()

    assert len(child_session_ids) == 1
    assert child_session_ids[0] is None  # ContextVar doesn't propagate

    set_current_session(None)


def test_patch_threading_cleanup():
    """unpatch_all() restores original threading.Thread."""
    original_init = threading.Thread.__init__
    original_run = threading.Thread.run

    patch_threading()

    # After patching, they should be different
    assert threading.Thread.__init__ is not original_init
    assert threading.Thread.run is not original_run

    unpatch_all()

    # After unpatching, they should be restored
    assert threading.Thread.__init__ is original_init
    assert threading.Thread.run is original_run


def test_patch_threading_via_public_api():
    """stateloom.patch_threading() works through the public API."""
    _init_gate()

    original_init = threading.Thread.__init__

    stateloom.patch_threading()

    assert threading.Thread.__init__ is not original_init

    unpatch_all()

    assert threading.Thread.__init__ is original_init
