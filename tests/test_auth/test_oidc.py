"""Tests for OIDC client logic."""

import hashlib
import secrets

import pytest

from stateloom.auth.oidc import OIDCClient, OIDCError


class TestBuildAuthorizationUrl:
    def test_returns_url_state_nonce_verifier(self):
        oidc = OIDCClient(
            issuer_url="https://accounts.google.com",
            client_id="my-client",
        )
        url, state, nonce, verifier = oidc.build_authorization_url(
            redirect_uri="http://localhost/callback",
        )
        assert "https://accounts.google.com/authorize" in url
        assert "client_id=my-client" in url
        assert "code_challenge_method=S256" in url
        assert "state=" in url
        assert len(state) > 10
        assert len(nonce) > 10
        assert len(verifier) > 10

    def test_custom_state_and_nonce(self):
        oidc = OIDCClient(
            issuer_url="https://example.com",
            client_id="c1",
        )
        url, state, nonce, _ = oidc.build_authorization_url(
            redirect_uri="http://localhost/cb",
            state="my-state",
            nonce="my-nonce",
        )
        assert state == "my-state"
        assert nonce == "my-nonce"
        assert "state=my-state" in url

    def test_pkce_challenge_is_sha256(self):
        oidc = OIDCClient(
            issuer_url="https://example.com",
            client_id="c1",
        )
        url, _, _, verifier = oidc.build_authorization_url(
            redirect_uri="http://localhost/cb",
        )
        # Verify the code_challenge is base64url(sha256(verifier))
        import base64

        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert f"code_challenge={expected}" in url


class TestExtractGroups:
    def test_list_groups(self):
        claims = {"groups": ["admin", "dev"]}
        assert OIDCClient.extract_groups(claims, "groups") == ["admin", "dev"]

    def test_string_group(self):
        claims = {"role": "admin"}
        assert OIDCClient.extract_groups(claims, "role") == ["admin"]

    def test_empty_claim(self):
        claims = {}
        assert OIDCClient.extract_groups(claims, "groups") == []

    def test_no_group_claim(self):
        claims = {"groups": ["admin"]}
        assert OIDCClient.extract_groups(claims, "") == []


class TestValidateIdToken:
    def test_hs256_token(self):
        pytest.importorskip("jwt")
        import jwt

        oidc = OIDCClient(
            issuer_url="https://example.com",
            client_id="c1",
            client_secret="secret123",
        )
        import time

        token = jwt.encode(
            {
                "sub": "user123",
                "iss": "https://example.com",
                "aud": "c1",
                "exp": int(time.time()) + 60,
                "nonce": "n1",
            },
            "secret123",
            algorithm="HS256",
        )
        claims = oidc.validate_id_token(token, nonce="n1")
        assert claims["sub"] == "user123"

    def test_nonce_mismatch(self):
        pytest.importorskip("jwt")
        import time

        import jwt

        oidc = OIDCClient(
            issuer_url="https://example.com",
            client_id="c1",
            client_secret="secret123",
        )
        token = jwt.encode(
            {
                "sub": "u1",
                "iss": "https://example.com",
                "aud": "c1",
                "exp": int(time.time()) + 60,
                "nonce": "correct",
            },
            "secret123",
            algorithm="HS256",
        )
        with pytest.raises(OIDCError, match="Nonce"):
            oidc.validate_id_token(token, nonce="wrong")

    def test_issuer_mismatch(self):
        pytest.importorskip("jwt")
        import time

        import jwt

        oidc = OIDCClient(
            issuer_url="https://example.com",
            client_id="c1",
            client_secret="secret123",
        )
        token = jwt.encode(
            {
                "sub": "u1",
                "iss": "https://evil.com",
                "aud": "c1",
                "exp": int(time.time()) + 60,
            },
            "secret123",
            algorithm="HS256",
        )
        with pytest.raises(OIDCError, match="Issuer"):
            oidc.validate_id_token(token)

    def test_expired_token(self):
        pytest.importorskip("jwt")
        import time

        import jwt

        oidc = OIDCClient(
            issuer_url="https://example.com",
            client_id="c1",
            client_secret="secret123",
        )
        token = jwt.encode(
            {
                "sub": "u1",
                "iss": "https://example.com",
                "aud": "c1",
                "exp": int(time.time()) - 10,
            },
            "secret123",
            algorithm="HS256",
        )
        with pytest.raises(OIDCError, match="expired"):
            oidc.validate_id_token(token)


class TestOIDCProviderModel:
    def test_default_fields(self):
        from stateloom.auth.oidc_models import OIDCProvider

        p = OIDCProvider(issuer_url="https://accounts.google.com", client_id="c1")
        assert p.id.startswith("oidc-")
        assert p.scopes == "openid email profile"
        assert p.is_active is True
        assert p.group_role_mapping == {}

    def test_all_fields(self):
        from stateloom.auth.oidc_models import OIDCProvider

        p = OIDCProvider(
            name="Google",
            issuer_url="https://accounts.google.com",
            client_id="c1",
            client_secret_encrypted="encrypted",
            scopes="openid email",
            group_claim="groups",
            group_role_mapping={"admin": {"team_id": "team-1", "role": "team_admin"}},
        )
        assert p.name == "Google"
        assert p.group_claim == "groups"
        assert "admin" in p.group_role_mapping


class TestOIDCProviderStore:
    @pytest.fixture
    def store(self):
        from stateloom.store.memory_store import MemoryStore

        return MemoryStore()

    def test_save_and_get(self, store):
        from stateloom.auth.oidc_models import OIDCProvider

        p = OIDCProvider(
            issuer_url="https://accounts.google.com",
            client_id="c1",
            name="Google",
        )
        store.save_oidc_provider(p)
        fetched = store.get_oidc_provider(p.id)
        assert fetched is not None
        assert fetched.issuer_url == "https://accounts.google.com"
        assert fetched.name == "Google"

    def test_get_by_issuer(self, store):
        from stateloom.auth.oidc_models import OIDCProvider

        p = OIDCProvider(
            issuer_url="https://okta.example.com",
            client_id="c2",
        )
        store.save_oidc_provider(p)
        fetched = store.get_oidc_provider_by_issuer("https://okta.example.com")
        assert fetched is not None
        assert fetched.id == p.id

    def test_list_providers(self, store):
        from stateloom.auth.oidc_models import OIDCProvider

        p1 = OIDCProvider(issuer_url="https://a.com", client_id="c1")
        p2 = OIDCProvider(issuer_url="https://b.com", client_id="c2")
        store.save_oidc_provider(p1)
        store.save_oidc_provider(p2)
        providers = store.list_oidc_providers()
        assert len(providers) == 2

    def test_delete_provider(self, store):
        from stateloom.auth.oidc_models import OIDCProvider

        p = OIDCProvider(issuer_url="https://c.com", client_id="c3")
        store.save_oidc_provider(p)
        assert store.delete_oidc_provider(p.id) is True
        assert store.get_oidc_provider(p.id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete_oidc_provider("oidc-nonexistent") is False

    def test_update_provider(self, store):
        from stateloom.auth.oidc_models import OIDCProvider

        p = OIDCProvider(
            issuer_url="https://d.com",
            client_id="c4",
            name="Original",
        )
        store.save_oidc_provider(p)
        p.name = "Updated"
        store.save_oidc_provider(p)
        fetched = store.get_oidc_provider(p.id)
        assert fetched.name == "Updated"
