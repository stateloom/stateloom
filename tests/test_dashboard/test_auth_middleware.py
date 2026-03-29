"""Tests for dashboard dual-mode auth middleware (JWT + API key)."""

import pytest

pytest.importorskip("jwt")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.auth.jwt import create_access_token
from stateloom.auth.models import User
from stateloom.auth.password import hash_password
from stateloom.core.types import Role
from stateloom.store.memory_store import MemoryStore


class FakeConfig:
    auth_enabled = True
    auth_jwt_algorithm = "HS256"
    auth_jwt_access_ttl = 900
    auth_jwt_refresh_ttl = 604800
    dashboard_api_key = ""
    dashboard_host = "127.0.0.1"
    dashboard_port = 4782
    max_request_body_mb = 10.0


class FakeGate:
    def __init__(self, api_key=""):
        self.store = MemoryStore()
        self.config = FakeConfig()
        self.config.dashboard_api_key = api_key


def _make_test_app(gate):
    """Build a minimal FastAPI app with the auth middleware from server.py."""
    import secrets

    from fastapi import Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    # Replicate the dual-mode auth middleware from server.py
    api_key = gate.config.dashboard_api_key

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path

        skip_prefixes = (
            "/api/health",
            "/api/v1/health",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
            "/api/v1/auth/bootstrap",
        )
        if any(path.startswith(p) for p in skip_prefixes):
            return await call_next(request)
        if not path.startswith(("/api/", "/ws", "/metrics")):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        query_key = request.query_params.get("api_key", "")
        provided_key = ""
        if auth_header.startswith("Bearer "):
            provided_key = auth_header[7:]
        elif query_key:
            provided_key = query_key

        # Try 1: JWT decode
        if gate.config.auth_enabled and provided_key:
            try:
                from stateloom.auth.jwt import _get_jwt_secret, decode_access_token

                jwt_secret = _get_jwt_secret(gate.store, gate.config)
                payload = decode_access_token(
                    provided_key,
                    jwt_secret,
                    algorithm=gate.config.auth_jwt_algorithm,
                )
                if payload and payload.sub:
                    user = gate.store.get_user(payload.sub)
                    if user and user.is_active:
                        request.state.user = user
                        request.state.team_roles = gate.store.get_user_team_roles(user.id)
                        return await call_next(request)
            except Exception:
                pass

        # Try 2: Legacy API key
        if api_key and provided_key:
            if secrets.compare_digest(provided_key, api_key):
                from stateloom.auth.models import User as UserModel

                request.state.user = UserModel(
                    id="usr-system",
                    email="system@stateloom.local",
                    display_name="System (API Key)",
                    org_role=Role.ORG_ADMIN,
                    email_verified=True,
                    is_active=True,
                )
                request.state.team_roles = []
                return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing credentials"},
        )

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/sessions")
    async def sessions(request: Request):
        user = getattr(request.state, "user", None)
        return {"email": user.email if user else "none"}

    @app.get("/api/v1/auth/login")
    async def login():
        return {"status": "login_page"}

    return app


class TestJWTAuth:
    def test_jwt_auth_succeeds(self):
        gate = FakeGate()
        user = User(
            email="dev@test.com",
            password_hash=hash_password("pass"),
            org_role=Role.ORG_ADMIN,
            is_active=True,
        )
        gate.store.save_user(user)

        from stateloom.auth.jwt import _get_jwt_secret

        secret = _get_jwt_secret(gate.store, gate.config)
        token = create_access_token(user, secret, ttl=60)

        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["email"] == "dev@test.com"

    def test_jwt_inactive_user_rejected(self):
        gate = FakeGate()
        user = User(
            email="inactive@test.com",
            password_hash=hash_password("pass"),
            org_role=Role.ORG_ADMIN,
            is_active=False,
        )
        gate.store.save_user(user)

        from stateloom.auth.jwt import _get_jwt_secret

        secret = _get_jwt_secret(gate.store, gate.config)
        token = create_access_token(user, secret, ttl=60)

        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_invalid_jwt_rejected(self):
        gate = FakeGate()
        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get("/api/v1/sessions", headers={"Authorization": "Bearer invalid-token"})
        assert resp.status_code == 401

    def test_no_auth_rejected(self):
        gate = FakeGate()
        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 401


class TestAPIKeyAuth:
    def test_api_key_via_bearer(self):
        gate = FakeGate(api_key="test-key-123")
        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
        assert resp.json()["email"] == "system@stateloom.local"

    def test_api_key_via_query(self):
        gate = FakeGate(api_key="test-key-123")
        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get("/api/v1/sessions?api_key=test-key-123")
        assert resp.status_code == 200

    def test_wrong_api_key_rejected(self):
        gate = FakeGate(api_key="test-key-123")
        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401


class TestSkipPaths:
    def test_health_skips_auth(self):
        gate = FakeGate()
        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_login_skips_auth(self):
        gate = FakeGate()
        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get("/api/v1/auth/login")
        assert resp.status_code == 200


class TestDualMode:
    def test_jwt_preferred_over_api_key(self):
        """When both JWT and API key are valid, JWT auth is tried first."""
        gate = FakeGate(api_key="test-key-123")
        user = User(
            email="jwt-user@test.com",
            password_hash=hash_password("pass"),
            org_role=Role.TEAM_VIEWER,
            is_active=True,
        )
        gate.store.save_user(user)

        from stateloom.auth.jwt import _get_jwt_secret

        secret = _get_jwt_secret(gate.store, gate.config)
        token = create_access_token(user, secret, ttl=60)

        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        # JWT user, not system user
        assert resp.json()["email"] == "jwt-user@test.com"

    def test_api_key_fallback_when_jwt_invalid(self):
        """When JWT fails, API key auth is tried as fallback."""
        gate = FakeGate(api_key="test-key-123")
        app = _make_test_app(gate)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
        assert resp.json()["email"] == "system@stateloom.local"
