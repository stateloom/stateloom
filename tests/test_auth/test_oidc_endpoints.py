"""Tests for OIDC API endpoints (providers list, authorize, callback, link)."""

import time

import pytest

pytest.importorskip("jwt")

import jwt
from fastapi.testclient import TestClient

from stateloom.auth.endpoints import create_auth_router
from stateloom.auth.models import User
from stateloom.auth.oidc_models import OIDCProvider
from stateloom.auth.password import hash_password
from stateloom.core.types import Role
from stateloom.store.memory_store import MemoryStore


class FakeConfig:
    auth_enabled = True
    auth_jwt_algorithm = "HS256"
    auth_jwt_access_ttl = 900
    auth_jwt_refresh_ttl = 604800


class FakeGate:
    def __init__(self):
        self.store = MemoryStore()
        self.config = FakeConfig()
        from stateloom.core.feature_registry import FeatureRegistry

        self._feature_registry = FeatureRegistry()
        self._feature_registry.define("oidc", tier="enterprise")
        self._feature_registry.provide("oidc")


@pytest.fixture
def gate():
    return FakeGate()


@pytest.fixture
def app(gate):
    from fastapi import FastAPI

    app = FastAPI()
    router = create_auth_router(gate)
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def provider(gate):
    """Create and save a test OIDC provider."""
    p = OIDCProvider(
        name="TestIDP",
        issuer_url="https://idp.example.com",
        client_id="test-client",
        client_secret_encrypted="test-secret",
    )
    gate.store.save_oidc_provider(p)
    return p


class TestOIDCProvidersList:
    def test_list_providers_empty(self, client):
        resp = client.get("/api/v1/auth/oidc/providers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_providers_with_data(self, client, provider):
        resp = client.get("/api/v1/auth/oidc/providers")
        assert resp.status_code == 200
        providers = resp.json()
        assert len(providers) == 1
        assert providers[0]["name"] == "TestIDP"
        assert providers[0]["issuer_url"] == "https://idp.example.com"
        # Should not expose client_secret
        assert "client_secret" not in providers[0]

    def test_inactive_providers_hidden(self, client, gate):
        p = OIDCProvider(
            name="Inactive",
            issuer_url="https://inactive.example.com",
            client_id="c1",
            is_active=False,
        )
        gate.store.save_oidc_provider(p)
        resp = client.get("/api/v1/auth/oidc/providers")
        assert resp.status_code == 200
        assert len(resp.json()) == 0


class TestOIDCAuthorize:
    def test_authorize_returns_url(self, client, provider):
        resp = client.get(f"/api/v1/auth/oidc/authorize/{provider.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "authorization_url" in data
        assert "state" in data
        assert "idp.example.com" in data["authorization_url"]

    def test_authorize_nonexistent_provider(self, client):
        resp = client.get("/api/v1/auth/oidc/authorize/oidc-nonexistent")
        assert resp.status_code == 404


class TestOIDCLink:
    def test_link_success(self, client, gate, provider):
        """TOFU: link existing user to OIDC identity with password proof."""
        user = User(
            email="existing@test.com",
            password_hash=hash_password("mypassword"),
            email_verified=True,
            org_role=Role.ORG_ADMIN,
        )
        gate.store.save_user(user)

        resp = client.post(
            "/api/v1/auth/oidc/link",
            json={
                "email": "existing@test.com",
                "password": "mypassword",
                "oidc_provider_id": provider.id,
                "oidc_subject": "sub-123",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["linked"] is True
        assert "access_token" in data

        # Verify OIDC identity is saved
        updated = gate.store.get_user(user.id)
        assert updated.oidc_provider_id == provider.id
        assert updated.oidc_subject == "sub-123"

    def test_link_wrong_password(self, client, gate, provider):
        user = User(
            email="linkfail@test.com",
            password_hash=hash_password("correct"),
            email_verified=True,
        )
        gate.store.save_user(user)

        resp = client.post(
            "/api/v1/auth/oidc/link",
            json={
                "email": "linkfail@test.com",
                "password": "wrong",
                "oidc_provider_id": provider.id,
                "oidc_subject": "sub-456",
            },
        )
        assert resp.status_code == 401

    def test_link_missing_fields(self, client):
        resp = client.post(
            "/api/v1/auth/oidc/link",
            json={
                "email": "test@test.com",
            },
        )
        assert resp.status_code == 422

    def test_link_unknown_user(self, client, provider):
        resp = client.post(
            "/api/v1/auth/oidc/link",
            json={
                "email": "nobody@test.com",
                "password": "any",
                "oidc_provider_id": provider.id,
                "oidc_subject": "sub-789",
            },
        )
        assert resp.status_code == 401


class TestOIDCCallback:
    def test_callback_missing_code(self, client):
        resp = client.post(
            "/api/v1/auth/oidc/callback",
            json={
                "state": "some-state",
            },
        )
        assert resp.status_code == 422

    def test_callback_invalid_state(self, client):
        resp = client.post(
            "/api/v1/auth/oidc/callback",
            json={
                "code": "some-code",
                "state": "invalid-state",
            },
        )
        assert resp.status_code == 400
        assert "Invalid or expired state" in resp.json()["detail"]
