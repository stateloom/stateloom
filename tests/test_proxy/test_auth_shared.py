"""Tests for shared proxy auth utilities (AuthResult, _StubKey, authenticate_request, enforce_vk_policies)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stateloom.proxy.auth import (
    AuthResult,
    _StubKey,
    authenticate_request,
    enforce_vk_policies,
    format_policy_error,
    resolve_vk_rate_limit_id,
)
from stateloom.proxy.virtual_key import VirtualKey


class TestStubKey:
    """Tests for _StubKey duck-type compatibility with VirtualKey."""

    def test_default_fields(self):
        stub = _StubKey()
        assert stub.id == ""
        assert stub.org_id == ""
        assert stub.team_id == ""
        assert stub.name == "anonymous"
        assert stub.scopes == []
        assert stub.allowed_models == []
        assert stub.budget_limit is None
        assert stub.budget_spent == 0.0
        assert stub.rate_limit_tps is None
        assert stub.billing_mode == ""
        assert stub.agent_ids == []
        assert stub.revoked is False

    def test_no_shared_mutable_state(self):
        """Each _StubKey gets its own lists (no shared default_factory bug)."""
        a = _StubKey()
        b = _StubKey()
        a.scopes.append("chat")
        assert b.scopes == []

    def test_duck_type_compat_with_vk_fields(self):
        """_StubKey has all fields that proxy code accesses on VirtualKey."""
        vk_fields = {
            "id",
            "org_id",
            "team_id",
            "name",
            "scopes",
            "allowed_models",
            "budget_limit",
            "budget_spent",
            "rate_limit_tps",
            "billing_mode",
            "agent_ids",
            "revoked",
        }
        stub = _StubKey()
        for field in vk_fields:
            assert hasattr(stub, field), f"_StubKey missing field: {field}"


class TestAuthResult:
    def test_defaults(self):
        result = AuthResult()
        assert result.vk is None
        assert result.byok_key == ""
        assert result.raw_token == ""
        assert result.error_hint == ""

    def test_with_vk(self):
        vk = MagicMock()
        result = AuthResult(vk=vk, byok_key="key", raw_token="tok")
        assert result.vk is vk
        assert result.byok_key == "key"
        assert result.raw_token == "tok"

    def test_error_hint_field(self):
        result = AuthResult(vk=None, error_hint="Some hint")
        assert result.error_hint == "Some hint"


class TestAuthenticateRequest:
    def _make_proxy_auth(self, vk=None):
        pa = MagicMock()
        pa.authenticate.return_value = vk
        return pa

    def _make_config(self, require_vk=False):
        import types

        config = MagicMock()
        config.proxy = types.SimpleNamespace(require_virtual_key=require_vk)
        return config

    def test_ag_token_valid_vk(self):
        vk = MagicMock(spec=VirtualKey)
        pa = self._make_proxy_auth(vk)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "ag-abc123", config)
        assert result.vk is vk
        assert result.byok_key == ""
        assert result.raw_token == ""
        assert result.error_hint == ""

    def test_ag_token_invalid_vk(self):
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "ag-invalid", config)
        assert result.vk is None
        assert "not found or revoked" in result.error_hint

    def test_byok_managed_mode(self):
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "sk-openai-key", config)
        assert isinstance(result.vk, _StubKey)
        assert result.byok_key == "sk-openai-key"
        assert result.raw_token == "sk-openai-key"

    def test_no_auth_mode_passthrough(self):
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=False)
        result = authenticate_request(pa, "session-token", config)
        assert isinstance(result.vk, _StubKey)
        assert result.byok_key == ""
        assert result.raw_token == "session-token"

    def test_no_token_no_auth_required(self):
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=False)
        result = authenticate_request(pa, "", config)
        assert isinstance(result.vk, _StubKey)
        assert result.byok_key == ""
        assert result.raw_token == ""

    def test_no_token_auth_required(self):
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "", config)
        assert result.vk is None
        assert "No authorization token provided" in result.error_hint

    def test_has_byok_false_skips_byok(self):
        """When has_byok=False, non-VK tokens in managed mode still forward raw."""
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=False)
        result = authenticate_request(pa, "oauth-token", config, has_byok=False)
        assert isinstance(result.vk, _StubKey)
        assert result.byok_key == ""
        assert result.raw_token == "oauth-token"

    def test_has_byok_false_require_vk_non_ag(self):
        """With has_byok=False and require_vk=True, non-ag-* tokens are
        forwarded as-is (e.g. Code Assist OAuth passthrough)."""
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "oauth-token", config, has_byok=False)
        assert result.vk is not None
        assert result.raw_token == "oauth-token"


class TestEnforceVkPolicies:
    @pytest.mark.asyncio
    async def test_all_pass_stub_key(self):
        """_StubKey with defaults passes all checks."""
        stub = _StubKey()
        pa = MagicMock()
        rl = MagicMock()
        result = await enforce_vk_policies(stub, "gpt-4o", "chat", pa, rl)
        assert result is None

    @pytest.mark.asyncio
    async def test_model_not_allowed(self):
        stub = _StubKey(allowed_models=["claude-*"])
        pa = MagicMock()
        pa.check_model_access.return_value = False
        rl = MagicMock()
        result = await enforce_vk_policies(stub, "gpt-4o", "chat", pa, rl)
        assert result is not None
        assert result.startswith("model_not_allowed")

    @pytest.mark.asyncio
    async def test_budget_exceeded(self):
        stub = _StubKey(budget_limit=10.0, budget_spent=15.0)
        pa = MagicMock()
        pa.check_budget.return_value = False
        rl = MagicMock()
        result = await enforce_vk_policies(stub, "gpt-4o", "chat", pa, rl)
        assert result == "key_budget_exceeded"

    @pytest.mark.asyncio
    async def test_scope_denied(self):
        stub = _StubKey(scopes=["agents"])
        pa = MagicMock()
        pa.check_model_access.return_value = True
        pa.check_scope.return_value = False
        rl = MagicMock()
        result = await enforce_vk_policies(stub, "gpt-4o", "chat", pa, rl)
        assert result is not None
        assert result.startswith("scope_denied")


class TestResolveVkRateLimitId:
    def test_with_rate_limit(self):
        stub = _StubKey(id="vk-abc", rate_limit_tps=10.0)
        assert resolve_vk_rate_limit_id(stub) == "vk-abc"

    def test_without_rate_limit(self):
        stub = _StubKey(id="vk-abc")
        assert resolve_vk_rate_limit_id(stub) is None

    def test_no_id(self):
        stub = _StubKey(rate_limit_tps=10.0)
        assert resolve_vk_rate_limit_id(stub) is None


class TestFormatPolicyError:
    def test_model_not_allowed(self):
        status, code, msg = format_policy_error("model_not_allowed:gpt-4o", "gpt-4o", "chat")
        assert status == 403
        assert code == "model_not_allowed"
        assert "gpt-4o" in msg
        assert "allowed_models" in msg

    def test_key_budget_exceeded(self):
        status, code, msg = format_policy_error("key_budget_exceeded", "gpt-4o", "chat")
        assert status == 403
        assert code == "key_budget_exceeded"
        assert "budget" in msg.lower()

    def test_key_rate_limit_exceeded(self):
        status, code, msg = format_policy_error("key_rate_limit_exceeded", "gpt-4o", "chat")
        assert status == 429
        assert code == "key_rate_limit_exceeded"
        assert "rate limit" in msg.lower()

    def test_scope_denied_with_scope(self):
        status, code, msg = format_policy_error("scope_denied:agents", "gpt-4o", "chat")
        assert status == 403
        assert code == "scope_denied"
        assert "'agents'" in msg

    def test_scope_denied_default_scope(self):
        status, code, msg = format_policy_error("scope_denied", "gpt-4o", "messages")
        assert status == 403
        assert code == "scope_denied"
        assert "'messages'" in msg

    def test_unknown_policy_error(self):
        status, code, msg = format_policy_error("unknown_error", "gpt-4o", "chat")
        assert status == 403
        assert code == "policy_error"
        assert msg == "unknown_error"


class TestAuthErrorHint:
    """Tests that error_hint is set correctly on various auth failure paths."""

    def _make_proxy_auth(self, vk=None):
        pa = MagicMock()
        pa.authenticate.return_value = vk
        return pa

    def _make_config(self, require_vk=False):
        import types

        config = MagicMock()
        config.proxy = types.SimpleNamespace(require_virtual_key=require_vk)
        return config

    def test_success_no_hint(self):
        """Successful auth has empty error_hint."""
        vk = MagicMock(spec=VirtualKey)
        pa = self._make_proxy_auth(vk)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "ag-valid", config)
        assert result.vk is vk
        assert result.error_hint == ""

    def test_invalid_vk_hint(self):
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "ag-bad", config)
        assert result.vk is None
        assert "not found or revoked" in result.error_hint

    def test_no_token_hint(self):
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "", config)
        assert result.vk is None
        assert "No authorization token" in result.error_hint

    def test_byok_mode_no_hint(self):
        """BYOK mode succeeds — no error_hint."""
        pa = self._make_proxy_auth(None)
        config = self._make_config(require_vk=True)
        result = authenticate_request(pa, "sk-openai-key", config)
        assert isinstance(result.vk, _StubKey)
        assert result.error_hint == ""
