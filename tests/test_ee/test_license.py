"""Tests for EE license key generation, validation, and gating."""

from __future__ import annotations

import base64
import json
import time

import pytest

from stateloom.ee.license import (
    _LICENSE_PREFIX,
    LicensePayload,
    generate_license_key,
    validate_license_key,
)

# Generate a test keypair once for the module
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
    _test_public_key_b64 = base64.urlsafe_b64encode(pub_bytes).decode()
    return _test_private_key, _test_public_key_b64


def _make_payload(**overrides) -> LicensePayload:
    """Create a test license payload with defaults."""
    defaults = {
        "org": "Test Org",
        "org_id": "org-test-123",
        "features": ["proxy", "dashboard", "auth", "agents"],
        "max_seats": 10,
        "issued_at": "2025-01-01T00:00:00Z",
        "expires_at": "2099-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return LicensePayload(**defaults)


class TestLicensePayload:
    def test_not_expired(self):
        payload = _make_payload(expires_at="2099-12-31T23:59:59Z")
        assert not payload.is_expired

    def test_expired(self):
        payload = _make_payload(expires_at="2020-01-01T00:00:00Z")
        assert payload.is_expired

    def test_no_expiry(self):
        payload = _make_payload(expires_at="")
        assert not payload.is_expired

    def test_has_feature(self):
        payload = _make_payload(features=["proxy", "dashboard"])
        assert payload.has_feature("proxy")
        assert not payload.has_feature("compliance")

    def test_wildcard_feature(self):
        payload = _make_payload(features=["*"])
        assert payload.has_feature("proxy")
        assert payload.has_feature("anything")

    def test_to_dict(self):
        payload = _make_payload()
        d = payload.to_dict()
        assert d["org"] == "Test Org"
        assert d["org_id"] == "org-test-123"
        assert isinstance(d["features"], list)
        assert "is_expired" in d


class TestGenerateAndValidate:
    def test_roundtrip(self):
        priv, pub = _get_test_keys()
        payload = _make_payload()
        key = generate_license_key(payload, priv)

        assert key.startswith(_LICENSE_PREFIX)
        result = validate_license_key(key, public_key_b64=pub)
        assert result is not None
        assert result.org == "Test Org"
        assert result.org_id == "org-test-123"
        assert "proxy" in result.features

    def test_key_format(self):
        priv, pub = _get_test_keys()
        payload = _make_payload()
        key = generate_license_key(payload, priv)

        assert key.startswith("ag-lic-v1.")
        body = key[len("ag-lic-v1.") :]
        parts = body.split(".")
        assert len(parts) == 2  # payload.signature

    def test_all_features_preserved(self):
        priv, pub = _get_test_keys()
        features = ["proxy", "dashboard", "auth", "agents", "compliance", "jobs", "observability"]
        payload = _make_payload(features=features)
        key = generate_license_key(payload, priv)

        result = validate_license_key(key, public_key_b64=pub)
        assert result is not None
        assert result.features == features

    def test_max_seats_preserved(self):
        priv, pub = _get_test_keys()
        payload = _make_payload(max_seats=50)
        key = generate_license_key(payload, priv)

        result = validate_license_key(key, public_key_b64=pub)
        assert result is not None
        assert result.max_seats == 50


class TestValidationFailures:
    def test_empty_key(self):
        assert validate_license_key("") is None

    def test_wrong_prefix(self):
        assert validate_license_key("not-a-key") is None

    def test_missing_signature(self):
        assert validate_license_key("ag-lic-v1.onlypayload") is None

    def test_invalid_payload_encoding(self):
        assert validate_license_key("ag-lic-v1.!!!invalid!!!.sig") is None

    def test_tampered_payload(self):
        priv, pub = _get_test_keys()
        payload = _make_payload()
        key = generate_license_key(payload, priv)

        # Tamper with the payload
        parts = key[len(_LICENSE_PREFIX) :].split(".")
        payload_bytes = base64.urlsafe_b64decode(parts[0] + "==")
        payload_dict = json.loads(payload_bytes)
        payload_dict["max_seats"] = 9999
        tampered = (
            base64.urlsafe_b64encode(
                json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode()
            )
            .decode()
            .rstrip("=")
        )
        tampered_key = f"{_LICENSE_PREFIX}{tampered}.{parts[1]}"

        result = validate_license_key(tampered_key, public_key_b64=pub)
        assert result is None

    def test_tampered_signature(self):
        priv, pub = _get_test_keys()
        payload = _make_payload()
        key = generate_license_key(payload, priv)

        # Replace signature with garbage
        parts = key[len(_LICENSE_PREFIX) :].split(".")
        bad_sig = base64.urlsafe_b64encode(b"x" * 64).decode().rstrip("=")
        tampered_key = f"{_LICENSE_PREFIX}{parts[0]}.{bad_sig}"

        result = validate_license_key(tampered_key, public_key_b64=pub)
        assert result is None

    def test_wrong_public_key(self):
        """Key signed with one keypair cannot be validated with a different public key."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        priv, _ = _get_test_keys()
        payload = _make_payload()
        key = generate_license_key(payload, priv)

        # Generate a different keypair
        other_priv = Ed25519PrivateKey.generate()
        other_pub_b64 = base64.urlsafe_b64encode(
            other_priv.public_key().public_bytes_raw()
        ).decode()

        result = validate_license_key(key, public_key_b64=other_pub_b64)
        assert result is None

    def test_expired_key(self):
        priv, pub = _get_test_keys()
        payload = _make_payload(expires_at="2020-01-01T00:00:00Z")
        key = generate_license_key(payload, priv)

        result = validate_license_key(key, public_key_b64=pub)
        assert result is None  # Expired

    def test_placeholder_key_returns_none_with_warning(self, caplog):
        """When the placeholder public key is still in place, validation
        returns None with a WARNING-level log message instead of silently
        failing via invalid base64 decoding."""
        import logging

        priv, _ = _get_test_keys()
        payload = _make_payload()
        key = generate_license_key(payload, priv)

        # Call without public_key_b64 override — uses the placeholder
        with caplog.at_level(logging.WARNING, logger="stateloom.ee.license"):
            result = validate_license_key(key)

        assert result is None
        assert any("public key has not been configured" in r.message for r in caplog.records)


class TestEEModule:
    def test_is_licensed_false_by_default(self):
        import stateloom.ee as ee

        ee.reset_license()
        assert not ee.is_licensed()

    def test_license_info_none_by_default(self):
        import stateloom.ee as ee

        ee.reset_license()
        assert ee.license_info() is None

    def test_require_ee_raises_without_license(self):
        import stateloom.ee as ee

        ee.reset_license()

        import os

        old_env = os.environ.get("STATELOOM_ENV", "")
        try:
            os.environ["STATELOOM_ENV"] = "production"
            from stateloom.core.errors import StateLoomError

            with pytest.raises(StateLoomError, match="requires a valid license"):
                ee.require_ee("proxy")
        finally:
            if old_env:
                os.environ["STATELOOM_ENV"] = old_env
            else:
                os.environ.pop("STATELOOM_ENV", None)

    def test_require_ee_passes_in_dev_mode(self):
        import stateloom.ee as ee

        ee.reset_license()

        import os

        old_env = os.environ.get("STATELOOM_ENV", "")
        try:
            os.environ["STATELOOM_ENV"] = "development"
            ee.require_ee("proxy")  # Should not raise
        finally:
            if old_env:
                os.environ["STATELOOM_ENV"] = old_env
            else:
                os.environ.pop("STATELOOM_ENV", None)

    def test_validate_and_check(self):
        import stateloom.ee as ee

        ee.reset_license()

        priv, pub = _get_test_keys()
        payload = _make_payload()
        key = generate_license_key(payload, priv)

        # Monkey-patch the module's public key for this test
        import stateloom.ee.license as lic_mod

        old_pub = lic_mod._PUBLIC_KEY_B64
        try:
            lic_mod._PUBLIC_KEY_B64 = pub
            ee._validate_license(key)
            assert ee.is_licensed()
            info = ee.license_info()
            assert info is not None
            assert info["org"] == "Test Org"
        finally:
            lic_mod._PUBLIC_KEY_B64 = old_pub
            ee.reset_license()

    def test_validate_invalid_key(self):
        import stateloom.ee as ee

        ee.reset_license()

        ee._validate_license("ag-lic-v1.bad.key")
        assert not ee.is_licensed()
        ee.reset_license()

    def test_reset_license(self):
        import stateloom.ee as ee

        ee.reset_license()
        assert not ee.is_licensed()
        assert ee.license_info() is None
