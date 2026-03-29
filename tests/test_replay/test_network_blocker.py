"""Tests for the strict replay network blocker."""

import pytest

from stateloom.core.errors import StateLoomSideEffectError
from stateloom.replay.network_blocker import NetworkBlocker


class TestNetworkBlocker:
    def setup_method(self):
        self.blocker = NetworkBlocker(session_id="test-session")

    def teardown_method(self):
        self.blocker.deactivate()

    def test_activate_deactivate(self):
        self.blocker.activate({"api.openai.com"})
        assert self.blocker._active is True
        self.blocker.deactivate()
        assert self.blocker._active is False

    def test_allowed_host_passes(self):
        self.blocker.activate({"api.openai.com"})
        # Should not raise for allowed host
        self.blocker._check_host("api.openai.com")

    def test_blocked_host_raises(self):
        self.blocker.activate({"api.openai.com"})
        with pytest.raises(StateLoomSideEffectError) as exc_info:
            self.blocker._check_host("evil.example.com")
        assert "evil.example.com" in str(exc_info.value)
        assert "test-session" in str(exc_info.value)

    def test_blocked_host_with_port(self):
        self.blocker.activate({"api.openai.com"})
        with pytest.raises(StateLoomSideEffectError):
            self.blocker._check_host("evil.example.com:443")

    def test_no_allowed_hosts_blocks_all(self):
        self.blocker.activate(set())
        with pytest.raises(StateLoomSideEffectError):
            self.blocker._check_host("api.openai.com")

    def test_inactive_blocker_allows_all(self):
        # Not activated, should not block
        self.blocker._check_host("anything.example.com")

    def test_deactivated_blocker_allows_all(self):
        self.blocker.activate(set())
        self.blocker.deactivate()
        # After deactivation, should not block
        self.blocker._check_host("anything.example.com")

    def test_extract_host_from_url(self):
        assert (
            self.blocker._extract_host_from_url("https://api.openai.com/v1/chat")
            == "api.openai.com"
        )
        assert self.blocker._extract_host_from_url("http://localhost:8080/test") == "localhost"

    def test_patches_httpx(self):
        """Test that httpx is patched when activated."""
        import httpx

        original_send = httpx.Client.send
        self.blocker.activate(set())
        # The send method should be patched
        assert httpx.Client.send is not original_send
        self.blocker.deactivate()
        # After deactivation, should be restored
        assert httpx.Client.send is original_send

    def test_error_includes_fix_options(self):
        self.blocker.activate(set())
        with pytest.raises(StateLoomSideEffectError) as exc_info:
            self.blocker._check_host("example.com")
        error_msg = str(exc_info.value)
        assert "@stateloom.tool()" in error_msg
        assert "allow_hosts" in error_msg

    def test_step_counter_increments(self):
        self.blocker.activate(set())
        for i in range(3):
            with pytest.raises(StateLoomSideEffectError) as exc_info:
                self.blocker._check_host(f"host{i}.example.com")
            assert exc_info.value.step == i + 1
