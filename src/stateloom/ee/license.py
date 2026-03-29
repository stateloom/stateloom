"""Enterprise license key generation and validation.

License key format:
    ag-lic-v1.<base64(json_payload)>.<base64(Ed25519_signature)>

The public key is embedded here for offline verification.
The private key is kept offline and never shipped.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("stateloom.ee.license")

# Ed25519 public key for license verification (32 bytes, base64-encoded).
# Replace with your actual public key before shipping.
_PUBLIC_KEY_PLACEHOLDER = "REPLACE_WITH_REAL_PUBLIC_KEY_BASE64_BEFORE_RELEASE"
_PUBLIC_KEY_B64 = _PUBLIC_KEY_PLACEHOLDER

_LICENSE_PREFIX = "ag-lic-v1."


@dataclass
class LicensePayload:
    """Decoded license payload."""

    org: str = ""
    org_id: str = ""
    features: list[str] = field(default_factory=list)
    max_seats: int = 0
    issued_at: str = ""
    expires_at: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if the license has expired."""
        if not self.expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) > expiry
        except (ValueError, TypeError):
            return True

    def has_feature(self, feature: str) -> bool:
        """Check if a specific feature is licensed."""
        return feature in self.features or "*" in self.features

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "org": self.org,
            "org_id": self.org_id,
            "features": self.features,
            "max_seats": self.max_seats,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired,
        }


def generate_license_key(payload: LicensePayload, private_key_bytes: bytes) -> str:
    """Generate a signed license key.

    Args:
        payload: The license payload to sign.
        private_key_bytes: Ed25519 private key (64 bytes).

    Returns:
        License key string in the format ``ag-lic-v1.<payload>.<signature>``.

    This function is used by the license issuer (internal tooling),
    NOT shipped in production builds.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        raise ImportError(
            "cryptography package required for license key generation. "
            "Install with: pip install cryptography"
        )

    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes[:32])
    payload_json = json.dumps(payload.to_dict(), sort_keys=True, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
    signature = private_key.sign(payload_json.encode())
    signature_b64 = base64.urlsafe_b64encode(signature).decode()
    return f"{_LICENSE_PREFIX}{payload_b64}.{signature_b64}"


def validate_license_key(
    key: str,
    public_key_b64: str | None = None,
) -> LicensePayload | None:
    """Validate a license key and return the payload if valid.

    Args:
        key: The license key string.
        public_key_b64: Override public key (for testing). Uses the
            embedded key if not provided.

    Returns:
        ``LicensePayload`` if valid, ``None`` if invalid.
    """
    if not key or not key.startswith(_LICENSE_PREFIX):
        return None

    # Strip prefix
    body = key[len(_LICENSE_PREFIX) :]
    parts = body.split(".")
    if len(parts) != 2:
        return None

    payload_b64, signature_b64 = parts

    # Decode payload
    try:
        payload_json = base64.urlsafe_b64decode(payload_b64 + "==").decode()
        payload_dict = json.loads(payload_json)
    except Exception:
        logger.debug("License key: invalid payload encoding")
        return None

    # Decode signature
    try:
        signature = base64.urlsafe_b64decode(signature_b64 + "==")
    except Exception:
        logger.debug("License key: invalid signature encoding")
        return None

    # Verify signature
    pub_key_str = public_key_b64 or _PUBLIC_KEY_B64
    if pub_key_str == _PUBLIC_KEY_PLACEHOLDER:
        logger.warning(
            "License validation skipped: public key has not been configured. "
            "Replace _PUBLIC_KEY_B64 in ee/license.py before release."
        )
        return None
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        pub_bytes = base64.urlsafe_b64decode(pub_key_str + "==")
        public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        public_key.verify(signature, payload_json.encode())
    except ImportError:
        # cryptography not installed — cannot verify, reject
        logger.warning("License validation requires the cryptography package")
        return None
    except Exception:
        logger.debug("License key: signature verification failed")
        return None

    # Parse payload
    try:
        payload = LicensePayload(
            org=payload_dict.get("org", ""),
            org_id=payload_dict.get("org_id", ""),
            features=payload_dict.get("features", []),
            max_seats=payload_dict.get("max_seats", 0),
            issued_at=payload_dict.get("issued_at", ""),
            expires_at=payload_dict.get("expires_at", ""),
        )
    except Exception:
        logger.debug("License key: invalid payload structure")
        return None

    # Check expiry
    if payload.is_expired:
        logger.info("License key expired at %s", payload.expires_at)
        return None

    return payload
