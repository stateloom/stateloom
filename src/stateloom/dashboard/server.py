"""Dashboard server — FastAPI on a background daemon thread."""

from __future__ import annotations

import logging
import secrets
import socket
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from stateloom.dashboard.api import create_api_router
from stateloom.dashboard.ws import create_websocket_route

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.dashboard")

STATIC_DIR = Path(__file__).parent / "static"


class _ReuseAddrServer(uvicorn.Server):
    """Uvicorn server that sets SO_REUSEADDR so the port is freed immediately on stop."""

    def _bind_socket(self) -> socket.socket:
        """Create a socket with SO_REUSEADDR before binding."""
        config = self.config
        sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((config.host, config.port))
        sock.set_inheritable(True)
        return sock

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        """Override startup to inject our pre-bound socket."""
        sock = self._bind_socket()
        await super().startup(sockets=[sock])


class DashboardServer:
    """Serves the StateLoom dashboard on a background daemon thread."""

    def __init__(self, gate: Gate) -> None:
        self.gate = gate
        self._passthrough: Any = None  # PassthroughProxy instance
        self.app = self._create_app()
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

    def _requires_auth(self) -> bool:
        """Determine if authentication is required.

        Auth is required when an API key is configured or when the dashboard
        is bound to a non-loopback address (network-accessible).
        """
        if self.gate.config.dashboard_config.api_key:
            return True
        host = self.gate.config.dashboard_config.host
        return host not in ("127.0.0.1", "localhost", "::1")

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="StateLoom Dashboard",
            version="0.1.0",
            docs_url="/api/v1/docs",
        )

        # CORS middleware (L3)
        loopback = ("127.0.0.1", "localhost", "::1")
        origins = ["*"] if self.gate.config.dashboard_config.host in loopback else []
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request body size limit (M8)
        max_body_bytes = int(self.gate.config.max_request_body_mb * 1024 * 1024)

        @app.middleware("http")
        async def size_limit_middleware(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"},
                )
            return await call_next(request)

        # HTTP access log middleware — errors and slow requests at INFO,
        # routine polling suppressed entirely, other traffic at DEBUG.
        _POLL_PREFIXES = (
            "/api/sessions",
            "/api/organizations",
            "/api/teams",
            "/api/v1/stats",
            "/api/v1/license",
            "/api/debug",
        )

        @app.middleware("http")
        async def access_log_middleware(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            start = time.monotonic()
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            path = request.url.path

            if response.status_code >= 400 or elapsed_ms >= 1000:
                # Errors and slow requests always logged
                logger.info(
                    "%s %s %d %.0fms",
                    request.method, path, response.status_code, elapsed_ms,
                )
            elif not path.startswith(_POLL_PREFIXES):
                # Non-polling traffic at DEBUG
                logger.debug(
                    "%s %s %d %.0fms",
                    request.method, path, response.status_code, elapsed_ms,
                )
            # Successful polling requests: silent

            return response

        # API key / JWT dual-mode authentication middleware
        if self._requires_auth() or self.gate.config.auth.enabled:
            api_key = self.gate.config.dashboard_config.api_key
            if not api_key and not self.gate.config.auth.enabled:
                # Auto-generate a key when binding to non-loopback without explicit key
                api_key = secrets.token_urlsafe(32)
                self.gate.config.dashboard_api_key = api_key
                logger.warning(
                    "[StateLoom] Dashboard bound to non-loopback address. "
                    "Auto-generated API key: %s",
                    api_key,
                )

            @app.middleware("http")
            async def auth_middleware(
                request: Request,
                call_next: Callable[[Request], Awaitable[Response]],
            ) -> Response:
                path = request.url.path

                # Skip auth for: health, static, auth login/refresh/bootstrap
                skip_prefixes = (
                    "/api/health",
                    "/api/v1/health",
                    "/api/v1/auth/login",
                    "/api/v1/auth/refresh",
                    "/api/v1/auth/bootstrap",
                    "/api/v1/auth/oidc/authorize",
                    "/api/v1/auth/oidc/callback",
                    "/api/v1/auth/oidc/providers",
                    "/api/v1/auth/device/",
                )
                if any(path.startswith(p) for p in skip_prefixes):
                    return await call_next(request)
                if path.startswith("/ws"):
                    # WebSocket upgrade requests bypass auth (browser can't set headers)
                    return await call_next(request)
                if not path.startswith(("/api/", "/metrics")):
                    return await call_next(request)

                auth_header = request.headers.get("Authorization", "")
                query_key = request.query_params.get("api_key", "")
                provided_key = ""
                if auth_header.startswith("Bearer "):
                    provided_key = auth_header[7:]
                elif query_key:
                    provided_key = query_key

                # Try 1: JWT decode (when auth_enabled)
                if self.gate.config.auth.enabled and provided_key:
                    try:
                        from stateloom.auth.jwt import (
                            _get_jwt_secret,
                            decode_access_token,
                        )

                        jwt_secret = _get_jwt_secret(self.gate.store, self.gate.config)
                        payload = decode_access_token(
                            provided_key,
                            jwt_secret,
                            algorithm=self.gate.config.auth.jwt_algorithm,
                        )
                        if payload and payload.sub:
                            user = self.gate.store.get_user(payload.sub)
                            if user and user.is_active:
                                request.state.user = user
                                request.state.team_roles = self.gate.store.get_user_team_roles(
                                    user.id
                                )
                                return await call_next(request)
                    except Exception as exc:
                        logger.debug("JWT decode failed: %s", exc)

                # Try 2: Legacy API key
                if api_key and provided_key:
                    if secrets.compare_digest(provided_key, api_key):
                        # Synthetic system user with org_admin
                        from stateloom.auth.models import User
                        from stateloom.core.types import Role

                        request.state.user = User(
                            id="usr-system",
                            email="system@stateloom.local",
                            display_name="System (API Key)",
                            org_role=Role.ORG_ADMIN,
                            email_verified=True,
                            is_active=True,
                        )
                        request.state.team_roles = []
                        return await call_next(request)

                if not provided_key:
                    detail = (
                        "No credentials provided. "
                        "Send API key or JWT in Authorization header."
                    )
                elif self.gate.config.auth.enabled:
                    detail = (
                        "Invalid credentials. "
                        "JWT token may be expired or malformed."
                    )
                else:
                    detail = (
                        "Invalid API key. "
                        "Check your dashboard_api_key configuration."
                    )
                return JSONResponse(
                    status_code=401,
                    content={"detail": detail},
                )

        # API version headers middleware — registered last = outermost,
        # so it runs on all /api/** responses including 401s from auth.
        @app.middleware("http")
        async def api_version_middleware(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
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

        # License watermark header (Guardrails 6+7)
        try:
            from stateloom.ee import is_restricted_dev_mode as _is_rdm

            if _is_rdm():

                @app.middleware("http")
                async def license_watermark_middleware(
                    request: Request,
                    call_next: Callable[[Request], Awaitable[Response]],
                ) -> Response:
                    response = await call_next(request)
                    response.headers["X-StateLoom-License"] = "unlicensed-dev-mode"
                    return response
        except ImportError:
            pass

        # Auth API routes (login, refresh, bootstrap, me, etc.)
        from stateloom.auth.endpoints import create_auth_router

        auth_router = create_auth_router(self.gate)
        app.include_router(auth_router, prefix="/api/v1")

        # User management API routes
        from stateloom.dashboard.user_api import create_user_api_router

        user_router = create_user_api_router(self.gate)
        app.include_router(user_router, prefix="/api/v1")
        app.include_router(user_router, prefix="/api", include_in_schema=False)

        # OIDC provider management API routes
        from stateloom.dashboard.oidc_api import create_oidc_api_router

        oidc_router = create_oidc_api_router(self.gate)
        app.include_router(oidc_router, prefix="/api/v1")
        app.include_router(oidc_router, prefix="/api", include_in_schema=False)

        # API routes
        api_router = create_api_router(self.gate)

        # Observability API routes
        from stateloom.dashboard.observability_api import (
            create_metrics_endpoint,
            create_observability_router,
        )

        obs_router = create_observability_router(self.gate)

        # Canonical versioned paths
        app.include_router(api_router, prefix="/api/v1")
        app.include_router(obs_router, prefix="/api/v1")

        # Legacy aliases (deprecated, hidden from OpenAPI docs)
        app.include_router(api_router, prefix="/api", include_in_schema=False)
        app.include_router(obs_router, prefix="/api", include_in_schema=False)

        # Root-level /metrics endpoint (Prometheus standard path)
        # Must be registered before the catch-all StaticFiles mount
        @app.get("/metrics")
        async def metrics_endpoint():
            return create_metrics_endpoint(self.gate)

        # WebSocket
        ws_route = create_websocket_route(self.gate)
        app.add_api_websocket_route("/ws", ws_route)

        # WebSocket for live log streaming (debug mode)
        from stateloom.dashboard.ws import create_log_websocket_route

        log_ws_route = create_log_websocket_route(self.gate)
        app.add_api_websocket_route("/ws/logs", log_ws_route)

        # Shared sticky session manager for all proxy routers
        from stateloom.proxy.sticky_session import StickySessionManager

        sticky = StickySessionManager()

        # Shared passthrough proxy for HTTP forwarding
        from stateloom.proxy.passthrough import PassthroughProxy

        self._passthrough = PassthroughProxy(
            timeout=self.gate.config.proxy.timeout,
        )

        # Proxy router (/v1) — always mounted with dashboard (agents UI needs it)
        from stateloom.proxy.router import create_proxy_router

        proxy_router = create_proxy_router(
            self.gate, sticky_session=sticky, passthrough=self._passthrough
        )
        app.include_router(proxy_router, prefix="/v1")

        # Anthropic-native proxy (/v1/messages)
        from stateloom.proxy.anthropic_native import create_anthropic_router

        anthropic_router = create_anthropic_router(
            self.gate, sticky_session=sticky, passthrough=self._passthrough
        )
        app.include_router(anthropic_router, prefix="/v1")

        # Gemini-native proxy (/v1beta/models/{model}:generateContent)
        from stateloom.proxy.gemini_native import create_gemini_router

        gemini_router = create_gemini_router(
            self.gate, sticky_session=sticky, passthrough=self._passthrough
        )
        app.include_router(gemini_router, prefix="/v1beta")

        # Code Assist proxy (/code-assist)
        from stateloom.proxy.code_assist import create_code_assist_router

        code_assist_router = create_code_assist_router(
            self.gate, sticky_session=sticky, passthrough=self._passthrough
        )
        app.include_router(code_assist_router, prefix="/code-assist")

        # OpenAI Responses API proxy (/v1/responses) — Codex CLI support
        from stateloom.proxy.responses import create_responses_router

        responses_router = create_responses_router(
            self.gate, sticky_session=sticky, passthrough=self._passthrough
        )
        app.include_router(responses_router, prefix="/v1")

        # Static files (frontend)
        if STATIC_DIR.exists():
            app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

        return app

    def start(self) -> None:
        """Start the dashboard on a background daemon thread."""
        host = self.gate.config.dashboard_config.host
        port = self.gate.config.dashboard_config.port

        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level=(
                self.gate.config.log_level.lower() if self.gate.config.log_level else "warning"
            ),
            access_log=False,  # We use our own access log middleware
        )
        # Suppress noisy WebSocket accept/open/close messages from Uvicorn.
        # These fire on every dashboard page load and clutter the server log.
        # They remain visible at DEBUG level for troubleshooting.
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        self._server = _ReuseAddrServer(config)

        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
            name="stateloom-dashboard",
        )
        self._thread.start()
        logger.info(f"[StateLoom] Dashboard at http://{host}:{port}")

    def stop(self) -> None:
        """Stop the dashboard server and wait for it to release the port."""
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._server = None
        # Clean up passthrough proxy
        if self._passthrough is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                loop.create_task(self._passthrough.close())
            else:
                try:
                    asyncio.run(self._passthrough.close())
                except Exception as exc:
                    logger.debug("Passthrough close error: %s", exc)
            self._passthrough = None
