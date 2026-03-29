"""Tests for device authorization grant (CLI login flow)."""

import pytest

pytest.importorskip("jwt")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
    app = FastAPI()
    router = create_auth_router(gate)
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def admin_user(gate):
    user = User(
        email="admin@test.com",
        password_hash=hash_password("adminpass"),
        org_role=Role.ORG_ADMIN,
        is_active=True,
    )
    gate.store.save_user(user)
    return user


class TestDeviceAuthorize:
    def test_returns_codes(self, client):
        resp = client.post("/api/v1/auth/device/authorize")
        assert resp.status_code == 200
        data = resp.json()
        assert "device_code" in data
        assert "user_code" in data
        assert "verification_uri" in data
        assert "expires_in" in data
        assert data["interval"] == 5

    def test_unique_codes_per_request(self, client):
        resp1 = client.post("/api/v1/auth/device/authorize")
        resp2 = client.post("/api/v1/auth/device/authorize")
        assert resp1.json()["device_code"] != resp2.json()["device_code"]
        assert resp1.json()["user_code"] != resp2.json()["user_code"]


class TestDeviceToken:
    def test_pending_before_approval(self, client):
        auth_resp = client.post("/api/v1/auth/device/authorize")
        device_code = auth_resp.json()["device_code"]

        resp = client.post(
            "/api/v1/auth/device/token",
            json={
                "device_code": device_code,
            },
        )
        assert resp.status_code == 428
        assert resp.json()["error"] == "authorization_pending"

    def test_invalid_device_code(self, client):
        resp = client.post(
            "/api/v1/auth/device/token",
            json={
                "device_code": "nonexistent",
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "expired_token"


class TestDeviceVerifyAndToken:
    def test_full_device_flow(self, gate, admin_user):
        """Complete device flow with single router: authorize → verify → token."""
        # Use single app/router so device state is shared
        app = FastAPI()
        router = create_auth_router(gate)

        @app.middleware("http")
        async def inject_user_for_verify(request: Request, call_next):
            if "/device/verify" in request.url.path:
                request.state.user = admin_user
            return await call_next(request)

        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        # Step 1: Device requests authorization
        auth_resp = client.post("/api/v1/auth/device/authorize")
        data = auth_resp.json()
        device_code = data["device_code"]
        user_code = data["user_code"]

        # Step 2: Poll should return "authorization_pending"
        poll_resp = client.post(
            "/api/v1/auth/device/token",
            json={
                "device_code": device_code,
            },
        )
        assert poll_resp.status_code == 428

        # Step 3: User verifies
        verify_resp = client.post(
            "/api/v1/auth/device/verify",
            json={
                "user_code": user_code,
            },
        )
        assert verify_resp.status_code == 200
        assert verify_resp.json()["status"] == "approved"

        # Step 4: Poll should now return tokens
        token_resp = client.post(
            "/api/v1/auth/device/token",
            json={
                "device_code": device_code,
            },
        )
        assert token_resp.status_code == 200
        assert "access_token" in token_resp.json()

    def test_verify_unauthenticated(self, client):
        """Verify without auth returns 401."""
        resp = client.post(
            "/api/v1/auth/device/verify",
            json={
                "user_code": "ABCD1234",
            },
        )
        assert resp.status_code == 401

    def test_verify_invalid_code(self, client, gate, admin_user):
        """Verify with invalid user code returns 404."""
        # Create app with injected user
        app = FastAPI()
        router = create_auth_router(gate)

        @app.middleware("http")
        async def inject_user(request: Request, call_next):
            request.state.user = admin_user
            return await call_next(request)

        app.include_router(router, prefix="/api/v1")
        test_client = TestClient(app)

        resp = test_client.post(
            "/api/v1/auth/device/verify",
            json={
                "user_code": "INVALID",
            },
        )
        assert resp.status_code == 404


class TestDeviceFlowEndToEnd:
    def test_complete_flow_single_router(self, gate, admin_user):
        """Full flow with single router to test state sharing."""
        app = FastAPI()
        router = create_auth_router(gate)

        # Middleware to inject user for /device/verify only
        @app.middleware("http")
        async def selective_auth(request: Request, call_next):
            if "/device/verify" in request.url.path:
                request.state.user = admin_user
            return await call_next(request)

        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        # Step 1: Start device flow
        auth_resp = client.post("/api/v1/auth/device/authorize")
        device_code = auth_resp.json()["device_code"]
        user_code = auth_resp.json()["user_code"]

        # Step 2: Verify (as admin)
        verify_resp = client.post(
            "/api/v1/auth/device/verify",
            json={
                "user_code": user_code,
            },
        )
        assert verify_resp.status_code == 200

        # Step 3: Token should be issued
        token_resp = client.post(
            "/api/v1/auth/device/token",
            json={
                "device_code": device_code,
            },
        )
        assert token_resp.status_code == 200
        data = token_resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["email"] == "admin@test.com"
