"""JWT token creation and verification for StateLoom auth."""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from typing import Any, cast

from pydantic import BaseModel

logger = logging.getLogger("stateloom.auth.jwt")

try:
    import jwt as pyjwt

    _JWT_AVAILABLE = True
except ImportError:
    pyjwt = None  # type: ignore[assignment]  # pyjwt optional dep stub
    _JWT_AVAILABLE = False


class TokenPayload(BaseModel):
    """Decoded JWT access token payload."""

    sub: str  # user_id
    email: str = ""
    org_id: str = ""
    org_role: str | None = None
    exp: int = 0
    iat: int = 0
    jti: str = ""


def _get_jwt_secret(store: Any, config: Any) -> str:
    """Resolve the JWT signing secret.

    Priority:
    1. STATELOOM_JWT_SECRET env var (via config)
    2. Persisted secret in store
    3. Auto-generate and persist
    """
    import os

    env_secret = os.environ.get("STATELOOM_JWT_SECRET", "")
    if env_secret:
        return env_secret

    stored = store.get_secret("jwt_secret_key")
    if stored:
        return cast(str, stored)

    new_secret = secrets.token_urlsafe(64)
    store.save_secret("jwt_secret_key", new_secret)
    return new_secret


def create_access_token(
    user: Any,
    secret: str,
    algorithm: str = "HS256",
    ttl: int = 900,
) -> str:
    """Create a signed JWT access token for a user.

    Args:
        user: User object with id, email, org_id, org_role attributes.
        secret: Signing secret.
        algorithm: JWT algorithm (default HS256).
        ttl: Token time-to-live in seconds (default 15 min).

    Returns:
        Encoded JWT string.
    """
    if not _JWT_AVAILABLE:
        raise ImportError(
            "pyjwt is required for JWT auth. Install with: pip install stateloom[auth]"
        )

    now = int(time.time())
    org_role = getattr(user, "org_role", None)
    payload = {
        "sub": user.id,
        "email": user.email,
        "org_id": getattr(user, "org_id", ""),
        "org_role": org_role.value if org_role and hasattr(org_role, "value") else org_role,
        "iat": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex[:16],
    }
    return pyjwt.encode(payload, secret, algorithm=algorithm)


def create_refresh_token(
    user: Any,
    secret: str,
    algorithm: str = "HS256",
    ttl: int = 604800,
) -> tuple[str, str]:
    """Create a refresh token.

    Returns:
        (encoded_token, token_hash) — the hash is stored server-side.
    """
    if not _JWT_AVAILABLE:
        raise ImportError(
            "pyjwt is required for JWT auth. Install with: pip install stateloom[auth]"
        )

    import hashlib

    now = int(time.time())
    jti = uuid.uuid4().hex
    payload = {
        "sub": user.id,
        "type": "refresh",
        "iat": now,
        "exp": now + ttl,
        "jti": jti,
    }
    token = pyjwt.encode(payload, secret, algorithm=algorithm)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return token, token_hash


def decode_access_token(
    token: str,
    secret: str,
    algorithm: str = "HS256",
) -> TokenPayload | None:
    """Decode and verify a JWT access token.

    Returns:
        TokenPayload if valid, None if invalid/expired.
    """
    if not _JWT_AVAILABLE:
        return None

    try:
        payload = pyjwt.decode(token, secret, algorithms=[algorithm])
        return TokenPayload(
            sub=payload.get("sub", ""),
            email=payload.get("email", ""),
            org_id=payload.get("org_id", ""),
            org_role=payload.get("org_role"),
            exp=payload.get("exp", 0),
            iat=payload.get("iat", 0),
            jti=payload.get("jti", ""),
        )
    except Exception:
        return None


def decode_refresh_token(
    token: str,
    secret: str,
    algorithm: str = "HS256",
) -> dict[str, Any] | None:
    """Decode a refresh token. Returns dict payload or None."""
    if not _JWT_AVAILABLE:
        return None

    try:
        payload = pyjwt.decode(token, secret, algorithms=[algorithm])
        if payload.get("type") != "refresh":
            return None
        return cast(dict[str, Any], payload)
    except Exception:
        return None
