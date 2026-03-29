"""Tests for the sticky session manager."""

from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.proxy.sticky_session import StickySessionManager, resolve_session_id


def _mock_request(ip: str = "127.0.0.1", user_agent: str = "test-agent", forwarded: str = ""):
    """Create a mock Starlette Request."""
    req = MagicMock()
    headers: dict[str, str] = {"user-agent": user_agent}
    if forwarded:
        headers["x-forwarded-for"] = forwarded

    req.headers = headers
    req.client = MagicMock()
    req.client.host = ip
    return req


class TestStickySessionManager:
    def test_same_fingerprint_same_session(self):
        """Same IP + UA → same session ID."""
        mgr = StickySessionManager()
        req1 = _mock_request(ip="10.0.0.1", user_agent="claude-cli/1.0")
        req2 = _mock_request(ip="10.0.0.1", user_agent="claude-cli/1.0")

        sid1 = mgr.get_session_id(req1)
        sid2 = mgr.get_session_id(req2)

        assert sid1 == sid2
        assert sid1.startswith("sticky-")

    def test_different_fingerprint_different_session(self):
        """Different IP → different session ID."""
        mgr = StickySessionManager()
        req1 = _mock_request(ip="10.0.0.1", user_agent="claude-cli/1.0")
        req2 = _mock_request(ip="10.0.0.2", user_agent="claude-cli/1.0")

        sid1 = mgr.get_session_id(req1)
        sid2 = mgr.get_session_id(req2)

        assert sid1 != sid2

    def test_different_user_agent(self):
        """Different UAs from same IP → different sessions."""
        mgr = StickySessionManager()
        req1 = _mock_request(ip="10.0.0.1", user_agent="claude-cli/1.0")
        req2 = _mock_request(ip="10.0.0.1", user_agent="gemini-cli/2.0")

        sid1 = mgr.get_session_id(req1)
        sid2 = mgr.get_session_id(req2)

        assert sid1 != sid2

    def test_stale_mapping_expires(self):
        """After idle timeout, new session ID generated."""
        mgr = StickySessionManager(idle_timeout=0.1)
        # Force prune interval to 0 so pruning happens every time
        mgr._prune_interval = 0.0

        req = _mock_request(ip="10.0.0.1", user_agent="test")
        sid1 = mgr.get_session_id(req)

        time.sleep(0.15)

        sid2 = mgr.get_session_id(req)
        assert sid1 != sid2

    def test_access_refreshes_timestamp(self):
        """Access within window resets idle timer."""
        mgr = StickySessionManager(idle_timeout=0.3)
        req = _mock_request(ip="10.0.0.1", user_agent="test")

        sid1 = mgr.get_session_id(req)
        time.sleep(0.15)
        # Access again — should refresh the timer
        sid2 = mgr.get_session_id(req)
        assert sid1 == sid2

        time.sleep(0.15)
        # Still within window because we refreshed 0.15s ago
        sid3 = mgr.get_session_id(req)
        assert sid1 == sid3

    def test_thread_safety(self):
        """10 concurrent threads don't corrupt state."""
        mgr = StickySessionManager()
        results: dict[str, list[str]] = {}
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                req = _mock_request(ip=f"10.0.0.{thread_id}", user_agent="test")
                sids = []
                for _ in range(20):
                    sids.append(mgr.get_session_id(req))
                results[f"t{thread_id}"] = sids
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # Each thread should get the same session ID for all 20 calls
        for key, sids in results.items():
            assert len(set(sids)) == 1, f"{key}: got {len(set(sids))} unique session IDs"

    def test_x_forwarded_for(self):
        """Uses forwarded IP when present."""
        mgr = StickySessionManager()
        req1 = _mock_request(ip="127.0.0.1", user_agent="test", forwarded="203.0.113.5")
        req2 = _mock_request(ip="127.0.0.1", user_agent="test", forwarded="203.0.113.5")
        req3 = _mock_request(ip="127.0.0.1", user_agent="test", forwarded="198.51.100.1")

        sid1 = mgr.get_session_id(req1)
        sid2 = mgr.get_session_id(req2)
        sid3 = mgr.get_session_id(req3)

        # Same forwarded IP → same session
        assert sid1 == sid2
        # Different forwarded IP → different session
        assert sid1 != sid3

    def test_x_forwarded_for_multiple_entries(self):
        """Uses first entry from X-Forwarded-For."""
        mgr = StickySessionManager()
        req = _mock_request(
            ip="127.0.0.1",
            user_agent="test",
            forwarded="203.0.113.5, 10.0.0.1, 10.0.0.2",
        )
        # Should use 203.0.113.5 as the IP
        sid1 = mgr.get_session_id(req)

        req2 = _mock_request(ip="127.0.0.1", user_agent="test", forwarded="203.0.113.5")
        sid2 = mgr.get_session_id(req2)
        assert sid1 == sid2


class TestResolveSessionId:
    def test_explicit_header_priority(self):
        """X-StateLoom-Session-Id header overrides sticky."""
        mgr = StickySessionManager()
        req = _mock_request()

        sid = resolve_session_id("my-explicit-session", req, mgr)
        assert sid == "my-explicit-session"

    def test_sticky_used_when_no_explicit(self):
        """Uses sticky session when no explicit header."""
        mgr = StickySessionManager()
        req = _mock_request(ip="10.0.0.1", user_agent="test")

        sid = resolve_session_id("", req, mgr)
        assert sid.startswith("sticky-")

    def test_resolve_fallback(self):
        """When sticky=None, falls back to random UUID."""
        req = _mock_request()
        sid = resolve_session_id("", req, None)
        assert sid.startswith("proxy-")

    def test_resolve_fallback_no_sticky_no_header(self):
        """When both sticky and header absent, generates proxy- prefixed ID."""
        req = _mock_request()
        sid1 = resolve_session_id("", req, None)
        sid2 = resolve_session_id("", req, None)
        # Each call generates a unique ID
        assert sid1 != sid2
        assert sid1.startswith("proxy-")
