"""Tests for ContextVar-based session tracking."""

from stateloom.core.context import (
    get_current_session,
    get_current_session_id,
    set_current_session,
    set_current_session_id,
)
from stateloom.core.session import Session


def test_get_set_session():
    assert get_current_session() is None

    session = Session(id="ctx-test")
    set_current_session(session)
    assert get_current_session() is session
    assert get_current_session_id() == "ctx-test"

    set_current_session(None)
    assert get_current_session() is None


def test_set_session_id_directly():
    set_current_session_id("manual-id")
    assert get_current_session_id() == "manual-id"
    set_current_session_id(None)
