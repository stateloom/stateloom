"""Tests for dashboard API versioning (URL prefix + headers)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from stateloom.dashboard.api import create_api_router


def _make_app(*, require_auth: bool = False, api_key: str = "test-key"):
    """Build a minimal FastAPI app with versioned API routes + middleware."""
    gate = MagicMock()
    gate.store = MagicMock()
    gate.store.list_sessions.return_value = []
    gate.config = MagicMock()
    gate.config.async_jobs_enabled = False

    # Health endpoint needs store check
    gate.store.get_session.return_value = None

    app = FastAPI()

    # Auth middleware (conditional)
    if require_auth:
        import secrets as _secrets

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            path = request.url.path
            if path in ("/api/health", "/api/v1/health") or not path.startswith(
                ("/api/", "/ws", "/metrics")
            ):
                return await call_next(request)
            auth_header = request.headers.get("Authorization", "")
            provided = auth_header[7:] if auth_header.startswith("Bearer ") else ""
            if not _secrets.compare_digest(provided, api_key):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
            return await call_next(request)

    # Version middleware (same as server.py)
    @app.middleware("http")
    async def api_version_middleware(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/api/"):
            response.headers["X-StateLoom-API-Version"] = "1"
            if not path.startswith("/api/v1/") and not path.startswith("/api/v1?"):
                response.headers["Deprecation"] = "true"
                versioned_path = "/api/v1" + path[4:]
                if request.url.query:
                    versioned_path += f"?{request.url.query}"
                response.headers["Link"] = f'<{versioned_path}>; rel="successor-version"'
        return response

    router = create_api_router(gate)

    # Canonical versioned paths
    app.include_router(router, prefix="/api/v1")
    # Legacy aliases (deprecated, hidden from OpenAPI docs)
    app.include_router(router, prefix="/api", include_in_schema=False)

    # Dummy proxy route to verify no version headers on /v1/
    @app.get("/v1/chat/completions")
    async def dummy_proxy():
        return {"object": "chat.completion"}

    return app


@pytest.fixture
def client():
    return TestClient(_make_app())


@pytest.fixture
def auth_client():
    return TestClient(_make_app(require_auth=True, api_key="secret"))


class TestVersionedHealth:
    """Versioned health endpoint returns api_version and version header."""

    def test_versioned_health_has_api_version(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_version"] == "1"

    def test_versioned_health_has_version_header(self, client):
        resp = client.get("/api/v1/health")
        assert resp.headers["X-StateLoom-API-Version"] == "1"

    def test_versioned_health_no_deprecation(self, client):
        resp = client.get("/api/v1/health")
        assert "Deprecation" not in resp.headers
        assert "Link" not in resp.headers


class TestLegacyAlias:
    """Unversioned /api/ routes still work but are deprecated."""

    def test_legacy_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_legacy_health_has_version_header(self, client):
        resp = client.get("/api/health")
        assert resp.headers["X-StateLoom-API-Version"] == "1"

    def test_legacy_health_has_deprecation_header(self, client):
        resp = client.get("/api/health")
        assert resp.headers["Deprecation"] == "true"

    def test_legacy_health_has_link_header(self, client):
        resp = client.get("/api/health")
        link = resp.headers["Link"]
        assert "</api/v1/health>" in link
        assert 'rel="successor-version"' in link

    def test_legacy_and_versioned_same_body(self, client):
        legacy = client.get("/api/health").json()
        versioned = client.get("/api/v1/health").json()
        assert legacy == versioned

    def test_legacy_sessions_has_deprecation(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.headers["Deprecation"] == "true"
        assert "</api/v1/sessions>" in resp.headers["Link"]


class TestLinkQueryParams:
    """Link header preserves query parameters."""

    def test_link_preserves_query_params(self, client):
        resp = client.get("/api/sessions?limit=10&offset=5")
        link = resp.headers.get("Link", "")
        assert "</api/v1/sessions?limit=10&offset=5>" in link


class TestProxyRoutesExcluded:
    """Proxy routes (/v1/) do NOT get version headers."""

    def test_proxy_no_version_header(self, client):
        resp = client.get("/v1/chat/completions")
        assert resp.status_code == 200
        assert "X-StateLoom-API-Version" not in resp.headers
        assert "Deprecation" not in resp.headers


class TestAuthExemption:
    """Both /api/health and /api/v1/health are exempt from auth."""

    def test_versioned_health_no_auth_needed(self, auth_client):
        resp = auth_client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_legacy_health_no_auth_needed(self, auth_client):
        resp = auth_client.get("/api/health")
        assert resp.status_code == 200

    def test_other_api_routes_require_auth(self, auth_client):
        resp = auth_client.get("/api/v1/sessions")
        assert resp.status_code == 401

    def test_auth_works_with_bearer(self, auth_client):
        resp = auth_client.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200

    def test_legacy_routes_also_require_auth(self, auth_client):
        resp = auth_client.get("/api/sessions")
        assert resp.status_code == 401

    def test_legacy_401_has_version_header(self, auth_client):
        """Even 401 responses from /api/ get version + deprecation headers."""
        resp = auth_client.get("/api/sessions")
        assert resp.status_code == 401
        assert resp.headers["X-StateLoom-API-Version"] == "1"
        assert resp.headers["Deprecation"] == "true"
