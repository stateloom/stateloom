"""Password hashing with argon2 (preferred) and PBKDF2 (stdlib fallback)."""

from __future__ import annotations

import hashlib
import os
import secrets

try:
    import argon2

    _ARGON2_AVAILABLE = True
except ImportError:
    _ARGON2_AVAILABLE = False


def hash_password(password: str) -> str:
    """Hash a password using argon2id (preferred) or PBKDF2-SHA256 fallback.

    Returns a string prefixed with the algorithm identifier.
    """
    if _ARGON2_AVAILABLE:
        ph = argon2.PasswordHasher()
        return ph.hash(password)
    return _pbkdf2_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a stored hash. Auto-detects algorithm."""
    if not password_hash:
        return False

    if password_hash.startswith("$argon2"):
        if not _ARGON2_AVAILABLE:
            return False
        try:
            ph = argon2.PasswordHasher()
            return ph.verify(password_hash, password)
        except Exception:
            return False

    if password_hash.startswith("$pbkdf2$"):
        return _pbkdf2_verify(password, password_hash)

    return False


def _pbkdf2_hash(password: str) -> str:
    """Hash password using PBKDF2-SHA256 (stdlib)."""
    salt = os.urandom(16)
    iterations = 600_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"$pbkdf2${iterations}${salt.hex()}${dk.hex()}"


def _pbkdf2_verify(password: str, stored: str) -> bool:
    """Verify password against PBKDF2 hash."""
    try:
        parts = stored.split("$")
        # Format: $pbkdf2${iterations}${salt_hex}${dk_hex}
        if len(parts) != 5 or parts[1] != "pbkdf2":
            return False
        iterations = int(parts[2])
        salt = bytes.fromhex(parts[3])
        expected_dk = bytes.fromhex(parts[4])
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return secrets.compare_digest(dk, expected_dk)
    except (ValueError, IndexError):
        return False
