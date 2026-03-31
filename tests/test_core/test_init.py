"""Tests for stateloom.init() and public API."""

import stateloom
from stateloom.gate import Gate


def test_init_returns_gate():
    gate = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )
    assert isinstance(gate, Gate)


def test_init_reinitializes():
    """Calling init() when already initialized shuts down the old gate and creates a fresh one."""
    gate1 = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )
    gate2 = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )
    assert gate1 is not gate2  # Fresh gate on re-init
    assert isinstance(gate2, Gate)


def test_get_gate_after_init():
    stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )
    gate = stateloom.get_gate()
    assert isinstance(gate, Gate)


def test_get_gate_before_init():
    import pytest

    with pytest.raises(stateloom.StateLoomError):
        stateloom.get_gate()


def test_shutdown():
    stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )
    stateloom.shutdown()
    import pytest

    with pytest.raises(stateloom.StateLoomError):
        stateloom.get_gate()


def test_auto_route_enabled_when_explicitly_set():
    """auto_route=True enables auto-routing (opt-in only)."""
    gate = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model="llama3.2",
        auto_route=True,
    )
    assert gate.config.auto_route_enabled is True


def test_auto_route_not_auto_enabled_by_local_model():
    """Setting local_model alone does NOT auto-enable auto_route."""
    gate = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model="llama3.2",
    )
    assert gate.config.auto_route_enabled is False


def test_auto_route_disabled_explicitly():
    """auto_route=False should disable even when local_model is set."""
    gate = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model="llama3.2",
        auto_route=False,
    )
    assert gate.config.auto_route_enabled is False


def test_auto_route_not_enabled_without_local_model():
    """With local_model=None, auto_route should not auto-enable."""
    gate = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )
    assert gate.config.auto_route_enabled is False


def test_auto_route_explicit_true():
    """auto_route=True should still work explicitly."""
    gate = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        auto_route=True,
    )
    assert gate.config.auto_route_enabled is True


def test_session_context_manager():
    gate = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        local_model=None,
    )
    with stateloom.session("test-session", budget=5.0) as s:
        assert s.id == "test-session"
        assert s.budget == 5.0
    # Session should be ended after context manager exits
    assert s.status.value == "completed"
