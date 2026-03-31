"""StateLoom Enterprise Edition.

This package contains enterprise features that require a valid license key.
The code ships in the same ``pip install stateloom`` package but is gated
at runtime by an offline Ed25519 license key.

License: Proprietary source-available (see ee/LICENSE).
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

logger = logging.getLogger("stateloom.ee")

# Cached license state (set on first validation)
_license_payload: Any = None  # LicensePayload | None
_license_checked: bool = False


def _validate_license(key: str) -> Any:
    """Validate a license key and cache the result.

    Returns LicensePayload or None.
    """
    global _license_payload, _license_checked

    if _license_checked:
        return _license_payload

    _license_checked = True

    if not key:
        _license_payload = None
        return None

    from stateloom.ee.license import validate_license_key

    _license_payload = validate_license_key(key)
    if _license_payload is not None:
        logger.info("Enterprise license valid for org: %s", _license_payload.org)
    else:
        logger.warning("Enterprise license key is invalid or expired")

    return _license_payload


def is_licensed() -> bool:
    """Check if a valid enterprise license is active."""
    return _license_payload is not None


def is_dev_mode() -> bool:
    """Check if running in development mode (license check bypassed)."""
    return os.environ.get("STATELOOM_ENV", "").lower() in ("development", "dev", "test")


def require_ee(feature: str = "") -> None:
    """Raise if enterprise features are not licensed.

    Args:
        feature: Optional feature name for a targeted error message.

    Raises:
        StateLoomError: If no valid license is present and not in dev mode.
    """
    if is_licensed() or is_dev_mode():
        return

    from stateloom.core.errors import StateLoomError

    feature_msg = f" ({feature})" if feature else ""
    raise StateLoomError(
        f"Enterprise feature{feature_msg} requires a valid license key.",
        details=(
            "Set STATELOOM_LICENSE_KEY or pass license_key= to init(). "
            "Contact sales@stateloom.io for a license."
        ),
    )


def license_has_feature(feature: str) -> bool:
    """Check if the current license authorizes a specific feature."""
    if _license_payload is None:
        return False
    return cast(bool, _license_payload.has_feature(feature))


def license_info() -> dict[str, Any] | None:
    """Return license details, or None if not licensed."""
    if _license_payload is None:
        return None
    return cast(dict[str, Any], _license_payload.to_dict())


def is_restricted_dev_mode() -> bool:
    """True when dev mode is active WITHOUT a valid license.

    In this state, guardrails are enforced to prevent production abuse
    of unlicensed dev mode (scale caps, network lock, etc.).
    """
    return is_dev_mode() and not is_licensed()


DEV_MODE_LIMITS: dict[str, int | float] = {
    "max_teams": 3,
    "max_users": 5,
    "max_virtual_keys": 5,
    "max_tps": 5.0,
}


def get_dev_mode_limits() -> dict[str, int | float]:
    """Return scale cap limits for restricted dev mode."""
    return dict(DEV_MODE_LIMITS)


def reset_license() -> None:
    """Reset cached license state (for testing)."""
    global _license_payload, _license_checked
    _license_payload = None
    _license_checked = False
