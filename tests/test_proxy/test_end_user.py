"""Tests for end-user attribution via X-StateLoom-End-User header."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.proxy.auth import sanitize_end_user
from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore


def _make_gate(store=None):
    gate = MagicMock()
    gate.store = store or MemoryStore()
    gate.config = MagicMock()
    gate.config.proxy_require_virtual_key = True
    gate.config.proxy_upstream_openai = "https://api.openai.com"
    gate.config.proxy_upstream_anthropic = "https://api.anthropic.com"
    gate.config.proxy_upstream_gemini = "https://generativelanguage.googleapis.com"
    gate.config.proxy_timeout = 600.0
    gate.config.provider_api_key_openai = ""
    gate.config.provider_api_key_anthropic = ""
    gate.config.provider_api_key_google = ""
    gate.config.kill_switch_active = False
    gate.config.kill_switch_rules = []
    gate.config.blast_radius_enabled = False
    gate.pricing = MagicMock()
    gate.pricing.calculate_cost = MagicMock(return_value=0.001)
    gate._metrics_collector = None
    return gate


def _make_vk(**overrides) -> tuple[str, VirtualKey]:
    full_key, key_hash = generate_virtual_key()
    defaults = {
        "id": make_virtual_key_id(),
        "key_hash": key_hash,
        "key_preview": make_key_preview(full_key),
        "team_id": "team-1",
        "org_id": "org-1",
        "name": "test-key",
    }
    defaults.update(overrides)
    return full_key, VirtualKey(**defaults)


# ── sanitize_end_user unit tests ──────────────────────────────────────


class TestSanitizeEndUser:
    """sanitize_end_user() — header value sanitization."""

    def test_normal_value(self):
        assert sanitize_end_user("user@example.com") == "user@example.com"

    def test_strips_non_printable(self):
        assert sanitize_end_user("user\x00name") == "username"
        assert sanitize_end_user("user\x01\x02\x03name") == "username"

    def test_strips_tabs_and_newlines(self):
        assert sanitize_end_user("user\tname") == "username"
        assert sanitize_end_user("user\nname") == "username"
        assert sanitize_end_user("user\rname") == "username"

    def test_truncates_to_256_chars(self):
        long_value = "a" * 300
        result = sanitize_end_user(long_value)
        assert len(result) == 256

    def test_empty_string(self):
        assert sanitize_end_user("") == ""

    def test_all_non_printable(self):
        assert sanitize_end_user("\x00\x01\x02") == ""

    def test_preserves_spaces_and_punctuation(self):
        assert sanitize_end_user("John Doe (admin)") == "John Doe (admin)"

    def test_unicode_high_bytes_stripped(self):
        # Characters above 0x7E are stripped
        assert sanitize_end_user("user\x80\x90name") == "username"

    def test_sanitize_then_truncate(self):
        # Non-printable chars stripped first, then truncate
        value = "\x00" * 100 + "a" * 300
        result = sanitize_end_user(value)
        assert len(result) == 256
        assert result == "a" * 256


# ── End-user header stripping in passthrough ──────────────────────────


class TestEndUserHeaderStripping:
    """X-StateLoom-End-User is stripped before upstream forwarding."""

    def test_header_in_stateloom_headers(self):
        from stateloom.proxy.passthrough import _STATELOOM_HEADERS

        assert "x-stateloom-end-user" in _STATELOOM_HEADERS

    def test_filter_headers_strips_end_user(self):
        from stateloom.proxy.passthrough import filter_headers

        headers = {
            "content-type": "application/json",
            "x-stateloom-end-user": "test-user",
            "authorization": "Bearer sk-test",
        }
        filtered = filter_headers(headers)
        assert "x-stateloom-end-user" not in filtered
        assert "content-type" in filtered
        assert "authorization" in filtered


# ── End-user on session via dashboard API ─────────────────────────────


class TestEndUserInDashboardAPI:
    """Dashboard API exposes end_user field on sessions."""

    def _create_app(self):
        from stateloom.dashboard.api import create_api_router

        store = MemoryStore()
        gate = _make_gate(store)
        router = create_api_router(gate)
        app = FastAPI()
        app.include_router(router, prefix="/api")
        return app, gate, store

    def test_session_detail_includes_end_user(self):
        from stateloom.core.session import Session

        app, gate, store = self._create_app()
        session = Session(id="test-123", name="test")
        session.end_user = "user@example.com"
        store.save_session(session)

        client = TestClient(app)
        resp = client.get("/api/sessions/test-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["end_user"] == "user@example.com"

    def test_session_list_includes_end_user(self):
        from stateloom.core.session import Session

        app, gate, store = self._create_app()
        session = Session(id="test-456", name="test")
        session.end_user = "another-user"
        store.save_session(session)

        client = TestClient(app)
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) >= 1
        found = [s for s in data["sessions"] if s["id"] == "test-456"]
        assert len(found) == 1
        assert found[0]["end_user"] == "another-user"

    def test_session_list_filter_by_end_user(self):
        from stateloom.core.session import Session

        app, gate, store = self._create_app()
        s1 = Session(id="s-1", name="test1")
        s1.end_user = "alice"
        store.save_session(s1)
        s2 = Session(id="s-2", name="test2")
        s2.end_user = "bob"
        store.save_session(s2)
        s3 = Session(id="s-3", name="test3")
        s3.end_user = "alice"
        store.save_session(s3)

        client = TestClient(app)
        resp = client.get("/api/sessions", params={"end_user": "alice"})
        assert resp.status_code == 200
        data = resp.json()
        ids = [s["id"] for s in data["sessions"]]
        assert "s-1" in ids
        assert "s-3" in ids
        assert "s-2" not in ids

    def test_session_list_filter_by_end_user_no_match(self):
        from stateloom.core.session import Session

        app, gate, store = self._create_app()
        s = Session(id="s-10", name="test")
        s.end_user = "alice"
        store.save_session(s)

        client = TestClient(app)
        resp = client.get("/api/sessions", params={"end_user": "nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 0
