"""Tests for password hashing and verification."""

from stateloom.auth.password import (
    _ARGON2_AVAILABLE,
    _pbkdf2_hash,
    _pbkdf2_verify,
    hash_password,
    verify_password,
)


def test_pbkdf2_hash_format():
    h = _pbkdf2_hash("password123")
    assert h.startswith("$pbkdf2$")
    parts = h.split("$")
    assert len(parts) == 5
    assert parts[1] == "pbkdf2"
    assert int(parts[2]) == 600_000


def test_pbkdf2_roundtrip():
    h = _pbkdf2_hash("my-secret")
    assert _pbkdf2_verify("my-secret", h) is True
    assert _pbkdf2_verify("wrong-secret", h) is False


def test_pbkdf2_different_passwords_different_hashes():
    h1 = _pbkdf2_hash("password1")
    h2 = _pbkdf2_hash("password2")
    assert h1 != h2


def test_pbkdf2_verify_invalid_format():
    assert _pbkdf2_verify("pass", "$pbkdf2$bad") is False
    assert _pbkdf2_verify("pass", "") is False
    assert _pbkdf2_verify("pass", "$other$123$aa$bb") is False


def test_hash_password_returns_prefixed():
    h = hash_password("test-password")
    assert h.startswith("$argon2") or h.startswith("$pbkdf2$")


def test_verify_password_roundtrip():
    h = hash_password("secure-pass-123")
    assert verify_password("secure-pass-123", h) is True
    assert verify_password("wrong-pass", h) is False


def test_verify_password_empty_hash():
    assert verify_password("anything", "") is False


def test_verify_password_unknown_algorithm():
    assert verify_password("pass", "$unknown$stuff") is False
