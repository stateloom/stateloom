"""Tests for JWT token creation and verification."""

import time

import pytest

pytest.importorskip("jwt")

from stateloom.auth.jwt import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
)
from stateloom.auth.models import User
from stateloom.core.types import Role

SECRET = "test-secret-key-for-jwt-tests"


@pytest.fixture
def user():
    return User(
        id="usr-testjwt001",
        email="jwt@test.com",
        org_id="org-1",
        org_role=Role.ORG_ADMIN,
    )


class TestAccessToken:
    def test_create_and_decode(self, user):
        token = create_access_token(user, SECRET, ttl=60)
        assert isinstance(token, str)
        payload = decode_access_token(token, SECRET)
        assert payload is not None
        assert payload.sub == user.id
        assert payload.email == user.email
        assert payload.org_id == user.org_id
        assert payload.org_role == "org_admin"

    def test_expired_token_returns_none(self, user):
        token = create_access_token(user, SECRET, ttl=0)
        time.sleep(1)
        payload = decode_access_token(token, SECRET)
        assert payload is None

    def test_wrong_secret_returns_none(self, user):
        token = create_access_token(user, SECRET, ttl=60)
        payload = decode_access_token(token, "wrong-secret")
        assert payload is None

    def test_invalid_token_returns_none(self):
        payload = decode_access_token("not-a-jwt", SECRET)
        assert payload is None

    def test_token_has_jti(self, user):
        token = create_access_token(user, SECRET, ttl=60)
        payload = decode_access_token(token, SECRET)
        assert payload.jti != ""

    def test_different_tokens_have_different_jti(self, user):
        t1 = create_access_token(user, SECRET, ttl=60)
        t2 = create_access_token(user, SECRET, ttl=60)
        p1 = decode_access_token(t1, SECRET)
        p2 = decode_access_token(t2, SECRET)
        assert p1.jti != p2.jti

    def test_user_without_org_role(self):
        user = User(email="norole@test.com")
        token = create_access_token(user, SECRET, ttl=60)
        payload = decode_access_token(token, SECRET)
        assert payload.org_role is None


class TestRefreshToken:
    def test_create_returns_token_and_hash(self, user):
        token, token_hash = create_refresh_token(user, SECRET, ttl=60)
        assert isinstance(token, str)
        assert isinstance(token_hash, str)
        assert len(token_hash) == 64  # SHA256 hex

    def test_decode_refresh_token(self, user):
        token, _ = create_refresh_token(user, SECRET, ttl=60)
        payload = decode_refresh_token(token, SECRET)
        assert payload is not None
        assert payload["sub"] == user.id
        assert payload["type"] == "refresh"

    def test_access_token_not_decodable_as_refresh(self, user):
        token = create_access_token(user, SECRET, ttl=60)
        payload = decode_refresh_token(token, SECRET)
        assert payload is None  # type != "refresh"

    def test_expired_refresh_returns_none(self, user):
        token, _ = create_refresh_token(user, SECRET, ttl=0)
        time.sleep(1)
        payload = decode_refresh_token(token, SECRET)
        assert payload is None

    def test_wrong_secret_returns_none(self, user):
        token, _ = create_refresh_token(user, SECRET, ttl=60)
        payload = decode_refresh_token(token, "wrong")
        assert payload is None


class TestTokenPayload:
    def test_payload_model(self):
        p = TokenPayload(sub="usr-1", email="a@b.com", org_id="org-1")
        assert p.sub == "usr-1"
        assert p.exp == 0
        assert p.jti == ""
