"""Tests for proxy auth helper functions."""

from __future__ import annotations

from stateloom.proxy.auth import strip_bearer


class TestStripBearer:
    """Test strip_bearer() extracts tokens from Bearer headers."""

    def test_valid_bearer(self):
        assert strip_bearer("Bearer sk-abc123") == "sk-abc123"

    def test_extra_whitespace(self):
        assert strip_bearer("Bearer   sk-abc123") == "sk-abc123"

    def test_no_prefix(self):
        assert strip_bearer("sk-abc123") == ""

    def test_empty_string(self):
        assert strip_bearer("") == ""

    def test_bearer_only(self):
        assert strip_bearer("Bearer ") == ""

    def test_lowercase_bearer(self):
        assert strip_bearer("bearer sk-abc123") == "sk-abc123"

    def test_mixed_case_bearer(self):
        assert strip_bearer("BEARER sk-abc123") == "sk-abc123"
