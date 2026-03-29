"""Tests for the AuditHookManager."""

from __future__ import annotations

import pytest

from stateloom.security.audit_hook import _MONITORED_EVENTS, AuditHookManager, _extract_detail


def test_configure():
    mgr = AuditHookManager()
    mgr.configure(
        enabled=True,
        mode="enforce",
        deny_events=["subprocess.Popen", "os.system"],
        allow_paths=["/tmp/**"],
    )
    assert mgr._enabled is True
    assert mgr._mode == "enforce"
    assert mgr._deny_events == {"subprocess.Popen", "os.system"}
    assert mgr._allow_paths == ["/tmp/**"]


def test_configure_disable():
    mgr = AuditHookManager()
    mgr.configure(enabled=True, mode="audit")
    assert mgr._enabled is True
    mgr.configure(enabled=False)
    assert mgr._enabled is False


def test_get_status_default():
    mgr = AuditHookManager()
    status = mgr.get_status()
    assert status["installed"] is False
    assert status["enabled"] is False
    assert status["mode"] == "audit"
    assert status["deny_events"] == []
    assert status["event_count"] == 0
    assert status["blocked_count"] == 0


def test_install_only_once():
    mgr = AuditHookManager()
    # Note: sys.addaudithook() is irreversible and global, so we test
    # that the second call returns False (idempotent guard).
    result1 = mgr.install()
    assert result1 is True
    result2 = mgr.install()
    assert result2 is False
    # Disable so it doesn't interfere with other tests
    mgr.configure(enabled=False)


def test_hook_disabled_noop():
    mgr = AuditHookManager()
    mgr.configure(enabled=False)
    # Directly call _hook — should be a no-op
    mgr._hook("open", ("/some/file", "r", 0))
    assert mgr._event_count == 0


def test_hook_enabled_logs():
    mgr = AuditHookManager()
    mgr.configure(enabled=True, mode="audit", deny_events=["open"])
    mgr._hook("open", ("/some/file", "r", 0))
    assert mgr._event_count == 1
    assert mgr._blocked_count == 0
    status = mgr.get_status()
    assert len(status["recent_events"]) == 1
    assert status["recent_events"][0]["action"] == "logged"
    mgr.configure(enabled=False)


def test_hook_enforce_blocks():
    mgr = AuditHookManager()
    mgr.configure(enabled=True, mode="enforce", deny_events=["subprocess.Popen"])
    with pytest.raises(RuntimeError, match="security policy blocked"):
        mgr._hook("subprocess.Popen", (["ls", "-la"],))
    assert mgr._blocked_count == 1
    mgr.configure(enabled=False)


def test_hook_allow_paths():
    mgr = AuditHookManager()
    mgr.configure(
        enabled=True,
        mode="enforce",
        deny_events=["open"],
        allow_paths=["/usr/lib/*"],
    )
    # Should NOT block — path matches allow list
    mgr._hook("open", ("/usr/lib/libfoo.so", "r", 0))
    assert mgr._blocked_count == 0

    # Should block — path does NOT match allow list
    with pytest.raises(RuntimeError):
        mgr._hook("open", ("/etc/passwd", "r", 0))
    assert mgr._blocked_count == 1
    mgr.configure(enabled=False)


def test_extract_detail():
    assert _extract_detail("open", ("/foo/bar",)) == "/foo/bar"
    assert _extract_detail("socket.connect", (None, ("127.0.0.1", 8080))) == "127.0.0.1:8080"
    assert _extract_detail("subprocess.Popen", (["ls", "-la"],)) == "['ls', '-la']"
    assert _extract_detail("os.system", ("echo hello",)) == "echo hello"
    assert _extract_detail("import", ("os",)) == "os"
    assert _extract_detail("ctypes.dlopen", ("libfoo.so",)) == "libfoo.so"
    # Unknown/empty
    assert _extract_detail("unknown", ()) == ""


def test_monitored_events_coverage():
    """Key dangerous events should be in the monitored set."""
    for event in [
        "open",
        "socket.connect",
        "subprocess.Popen",
        "os.system",
        "exec",
        "import",
    ]:
        assert event in _MONITORED_EVENTS
