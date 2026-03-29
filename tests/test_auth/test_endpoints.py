"""Tests for auth API endpoints (bootstrap, login, refresh, logout, me)."""

import pytest

pytest.importorskip("jwt")

from fastapi.testclient import TestClient

from stateloom.auth.endpoints import create_auth_router
from stateloom.auth.models import User
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


class TestBootstrap:
    def test_bootstrap_success(self, client):
        resp = client.post(
            "/api/v1/auth/bootstrap",
            json={
                "email": "admin@test.com",
                "password": "secure123456",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["email"] == "admin@test.com"

    def test_bootstrap_twice_fails(self, client, gate):
        client.post(
            "/api/v1/auth/bootstrap",
            json={
                "email": "admin@test.com",
                "password": "secure123456",
            },
        )
        resp = client.post(
            "/api/v1/auth/bootstrap",
            json={
                "email": "admin2@test.com",
                "password": "secure123456",
            },
        )
        assert resp.status_code == 409

    def test_bootstrap_missing_fields(self, client):
        resp = client.post("/api/v1/auth/bootstrap", json={})
        assert resp.status_code == 422

    def test_bootstrap_short_password(self, client):
        resp = client.post(
            "/api/v1/auth/bootstrap",
            json={
                "email": "admin@test.com",
                "password": "short",
            },
        )
        assert resp.status_code == 422


class TestLogin:
    def test_login_success(self, client, gate):
        # Create user first
        user = User(
            email="user@test.com",
            password_hash=hash_password("mypassword"),
            org_id="org-1",
            org_role=Role.ORG_ADMIN,
        )
        gate.store.save_user(user)

        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": "user@test.com",
                "password": "mypassword",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["email"] == "user@test.com"

    def test_login_wrong_password(self, client, gate):
        user = User(
            email="user2@test.com",
            password_hash=hash_password("correct"),
        )
        gate.store.save_user(user)

        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": "user2@test.com",
                "password": "wrong",
            },
        )
        assert resp.status_code == 401

    def test_login_unknown_user(self, client):
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": "nobody@test.com",
                "password": "whatever",
            },
        )
        assert resp.status_code == 401

    def test_login_inactive_user(self, client, gate):
        user = User(
            email="inactive@test.com",
            password_hash=hash_password("pass1234"),
            is_active=False,
        )
        gate.store.save_user(user)

        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": "inactive@test.com",
                "password": "pass1234",
            },
        )
        assert resp.status_code == 401


class TestRefresh:
    def test_refresh_success(self, client, gate):
        user = User(
            email="refresh@test.com",
            password_hash=hash_password("pass1234"),
            org_role=Role.ORG_ADMIN,
        )
        gate.store.save_user(user)

        login_resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": "refresh@test.com",
                "password": "pass1234",
            },
        )
        refresh_token = login_resp.json()["refresh_token"]

        resp = client.post(
            "/api/v1/auth/refresh",
            json={
                "refresh_token": refresh_token,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_invalid_token(self, client):
        resp = client.post(
            "/api/v1/auth/refresh",
            json={
                "refresh_token": "invalid-token",
            },
        )
        assert resp.status_code == 401

    def test_refresh_reuse_revoked(self, client, gate):
        user = User(
            email="reuse@test.com",
            password_hash=hash_password("pass1234"),
        )
        gate.store.save_user(user)

        login_resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": "reuse@test.com",
                "password": "pass1234",
            },
        )
        refresh_token = login_resp.json()["refresh_token"]

        # First refresh should succeed
        resp1 = client.post(
            "/api/v1/auth/refresh",
            json={
                "refresh_token": refresh_token,
            },
        )
        assert resp1.status_code == 200

        # Second refresh with same token should fail (revoked)
        resp2 = client.post(
            "/api/v1/auth/refresh",
            json={
                "refresh_token": refresh_token,
            },
        )
        assert resp2.status_code == 401


class TestLogout:
    def test_logout(self, client, gate):
        user = User(
            email="logout@test.com",
            password_hash=hash_password("pass1234"),
        )
        gate.store.save_user(user)

        login_resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": "logout@test.com",
                "password": "pass1234",
            },
        )
        refresh_token = login_resp.json()["refresh_token"]

        resp = client.post(
            "/api/v1/auth/logout",
            json={
                "refresh_token": refresh_token,
            },
        )
        assert resp.status_code == 200

        # Refresh with revoked token should fail
        resp2 = client.post(
            "/api/v1/auth/refresh",
            json={
                "refresh_token": refresh_token,
            },
        )
        assert resp2.status_code == 401
