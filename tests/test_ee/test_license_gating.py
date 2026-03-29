"""Tests for EE license gating behavior."""

from __future__ import annotations

import os

import pytest

from stateloom.core.errors import StateLoomError
from stateloom.ee import (
    _validate_license,
    is_dev_mode,
    is_licensed,
    license_info,
    require_ee,
    reset_license,
)
from stateloom.ee.license import (
    LicensePayload,
    generate_license_key,
    validate_license_key,
)
from stateloom.ee.setup import register_ee

# --- Test helpers ---

_test_private_key: bytes | None = None
_test_public_key_b64: str | None = None


def _get_test_keys() -> tuple[bytes, str]:
    """Generate a test Ed25519 keypair (cached per module)."""
    global _test_private_key, _test_public_key_b64
    if _test_private_key is not None:
        assert _test_public_key_b64 is not None
        return _test_private_key, _test_public_key_b64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    _test_private_key = private_key.private_bytes_raw()
    pub_bytes = private_key.public_key().public_bytes_raw()
    import base64

    _test_public_key_b64 = base64.urlsafe_b64encode(pub_bytes).decode()
    return _test_private_key, _test_public_key_b64


def _make_valid_key() -> tuple[str, str]:
    """Create a valid license key and return (key, public_key_b64)."""
    priv, pub = _get_test_keys()
    payload = LicensePayload(
        org="Test Org",
        org_id="org-test-123",
        features=["proxy", "dashboard", "auth", "agents"],
        max_seats=10,
        issued_at="2025-01-01T00:00:00Z",
        expires_at="2099-01-01T00:00:00Z",
    )
    key = generate_license_key(payload, priv)
    return key, pub


@pytest.fixture(autouse=True)
def _reset_license_state():
    """Reset license state before and after each test."""
    reset_license()
    yield
    reset_license()


class TestDevMode:
    def test_dev_mode_bypass(self):
        """In dev mode, require_ee() should not raise even without a license."""
        old_env = os.environ.get("STATELOOM_ENV", "")
        try:
            os.environ["STATELOOM_ENV"] = "development"
            assert is_dev_mode()
            require_ee("proxy")  # Should not raise
        finally:
            if old_env:
                os.environ["STATELOOM_ENV"] = old_env
            else:
                os.environ.pop("STATELOOM_ENV", None)

    def test_test_mode_bypass(self):
        """STATELOOM_ENV=test also bypasses license checks."""
        old_env = os.environ.get("STATELOOM_ENV", "")
        try:
            os.environ["STATELOOM_ENV"] = "test"
            assert is_dev_mode()
            require_ee("dashboard")  # Should not raise
        finally:
            if old_env:
                os.environ["STATELOOM_ENV"] = old_env
            else:
                os.environ.pop("STATELOOM_ENV", None)

    def test_production_requires_license(self):
        """In production mode, require_ee() raises without a license."""
        old_env = os.environ.get("STATELOOM_ENV", "")
        try:
            os.environ["STATELOOM_ENV"] = "production"
            assert not is_dev_mode()
            assert not is_licensed()
            with pytest.raises(StateLoomError, match="requires a valid license"):
                require_ee("proxy")
        finally:
            if old_env:
                os.environ["STATELOOM_ENV"] = old_env
            else:
                os.environ.pop("STATELOOM_ENV", None)


class TestLicenseGating:
    def test_no_license_means_not_licensed(self):
        """Without a license key, is_licensed() returns False."""
        _validate_license("")
        assert not is_licensed()
        assert license_info() is None

    def test_invalid_key_means_not_licensed(self):
        """An invalid license key doesn't grant access."""
        _validate_license("ag-lic-v1.invalid.key")
        assert not is_licensed()
        assert license_info() is None

    def test_valid_key_grants_access(self):
        """A valid license key grants access."""
        key, pub = _make_valid_key()

        import stateloom.ee.license as lic_mod

        old_pub = lic_mod._PUBLIC_KEY_B64
        try:
            lic_mod._PUBLIC_KEY_B64 = pub
            _validate_license(key)
            assert is_licensed()
            info = license_info()
            assert info is not None
            assert info["org"] == "Test Org"
            assert "proxy" in info["features"]
        finally:
            lic_mod._PUBLIC_KEY_B64 = old_pub

    def test_valid_license_allows_require_ee(self):
        """With a valid license, require_ee() does not raise."""
        key, pub = _make_valid_key()

        import stateloom.ee.license as lic_mod

        old_pub = lic_mod._PUBLIC_KEY_B64
        old_env = os.environ.get("STATELOOM_ENV", "")
        try:
            lic_mod._PUBLIC_KEY_B64 = pub
            os.environ["STATELOOM_ENV"] = "production"
            _validate_license(key)
            require_ee("proxy")  # Should not raise
        finally:
            lic_mod._PUBLIC_KEY_B64 = old_pub
            if old_env:
                os.environ["STATELOOM_ENV"] = old_env
            else:
                os.environ.pop("STATELOOM_ENV", None)

    def test_expired_key_not_licensed(self):
        """An expired license key is not valid."""
        priv, pub = _get_test_keys()
        payload = LicensePayload(
            org="Expired Org",
            org_id="org-expired",
            features=["proxy"],
            max_seats=5,
            issued_at="2020-01-01T00:00:00Z",
            expires_at="2020-06-01T00:00:00Z",
        )
        key = generate_license_key(payload, priv)

        import stateloom.ee.license as lic_mod

        old_pub = lic_mod._PUBLIC_KEY_B64
        try:
            lic_mod._PUBLIC_KEY_B64 = pub
            _validate_license(key)
            assert not is_licensed()
        finally:
            lic_mod._PUBLIC_KEY_B64 = old_pub


class TestRegisterEE:
    def test_register_ee_skips_without_license(self):
        """register_ee() is a no-op without a valid license or dev mode."""
        old_env = os.environ.get("STATELOOM_ENV", "")
        try:
            os.environ["STATELOOM_ENV"] = "production"
            reset_license()
            # Should not raise, just skip silently
            from unittest.mock import MagicMock

            mock_gate = MagicMock()
            register_ee(mock_gate)
            # No callbacks should be set
        finally:
            if old_env:
                os.environ["STATELOOM_ENV"] = old_env
            else:
                os.environ.pop("STATELOOM_ENV", None)

    def test_register_ee_runs_in_dev_mode(self):
        """register_ee() executes in dev mode even without a license."""
        old_env = os.environ.get("STATELOOM_ENV", "")
        try:
            os.environ["STATELOOM_ENV"] = "development"
            reset_license()
            assert not is_licensed()
            assert is_dev_mode()
            # Should not raise — dev mode bypasses license check
            from unittest.mock import MagicMock

            mock_gate = MagicMock()
            # Restricted dev mode checks config fields — set valid defaults
            mock_gate.config.dashboard_host = "127.0.0.1"
            mock_gate.config.store_backend = "sqlite"
            register_ee(mock_gate)
        finally:
            if old_env:
                os.environ["STATELOOM_ENV"] = old_env
            else:
                os.environ.pop("STATELOOM_ENV", None)
