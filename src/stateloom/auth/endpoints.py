"""Auth API endpoints — login, refresh, logout, bootstrap, me."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("stateloom.auth.endpoints")


def create_auth_router(gate: Any) -> APIRouter:
    """Create the auth API router.

    Provides login, refresh, logout, bootstrap, me, and change-password.
    All endpoints are unauthenticated except logout, me, and change-password.
    """
    from stateloom.auth.jwt import (
        _get_jwt_secret,
        create_access_token,
        create_refresh_token,
        decode_refresh_token,
    )
    from stateloom.auth.models import User
    from stateloom.auth.password import hash_password, verify_password
    from stateloom.core.types import Role

    router = APIRouter(prefix="/auth", tags=["auth"])

    def _get_secret() -> str:
        return _get_jwt_secret(gate.store, gate.config)

    @router.post("/bootstrap")
    async def bootstrap(request: Request) -> JSONResponse:
        """First-user setup. Only works when zero users exist."""
        users = gate.store.list_users(limit=1)
        if users:
            return JSONResponse(
                status_code=409,
                content={"detail": "System already bootstrapped"},
            )

        body = await request.json()
        email = body.get("email", "").strip()
        password = body.get("password", "")

        if not email or not password:
            return JSONResponse(
                status_code=422,
                content={"detail": "email and password are required"},
            )

        if len(password) < 8:
            return JSONResponse(
                status_code=422,
                content={"detail": "Password must be at least 8 characters"},
            )

        # Create default org if needed
        org_id = _ensure_default_org(gate)

        user = User(
            email=email,
            display_name=body.get("display_name", ""),
            password_hash=hash_password(password),
            email_verified=True,
            org_id=org_id,
            org_role=Role.ORG_ADMIN,
        )
        gate.store.save_user(user)

        secret = _get_secret()
        access_token = create_access_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_access_ttl,
        )
        refresh_token, refresh_hash = create_refresh_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_refresh_ttl,
        )
        expires_at = datetime.now(timezone.utc).isoformat()
        gate.store.save_refresh_token(refresh_hash, user.id, expires_at)

        return JSONResponse(
            status_code=201,
            content={
                "user_id": user.id,
                "email": user.email,
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
        )

    @router.post("/login")
    async def login(request: Request) -> JSONResponse:
        """Email + password login. Returns JWT tokens."""
        body = await request.json()
        email = body.get("email", "").strip()
        password = body.get("password", "")

        if not email or not password:
            return JSONResponse(
                status_code=422,
                content={"detail": "email and password are required"},
            )

        user = gate.store.get_user_by_email(email)
        if not user or not user.is_active:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid credentials"},
            )

        if not verify_password(password, user.password_hash):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid credentials"},
            )

        # Update last_login
        user.last_login = datetime.now(timezone.utc)
        gate.store.save_user(user)

        secret = _get_secret()
        access_token = create_access_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_access_ttl,
        )
        refresh_token, refresh_hash = create_refresh_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_refresh_ttl,
        )
        expires_at = datetime.now(timezone.utc).isoformat()
        gate.store.save_refresh_token(refresh_hash, user.id, expires_at)

        return JSONResponse(
            content={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user_id": user.id,
                "email": user.email,
            }
        )

    @router.post("/refresh")
    async def refresh(request: Request) -> JSONResponse:
        """Rotate a refresh token. Returns new access + refresh tokens."""
        body = await request.json()
        old_token = body.get("refresh_token", "")

        if not old_token:
            return JSONResponse(
                status_code=422,
                content={"detail": "refresh_token is required"},
            )

        secret = _get_secret()
        payload = decode_refresh_token(
            old_token,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
        )
        if not payload:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired refresh token"},
            )

        old_hash = hashlib.sha256(old_token.encode()).hexdigest()
        stored = gate.store.get_refresh_token(old_hash)
        if not stored or stored["revoked"]:
            return JSONResponse(
                status_code=401,
                content={"detail": "Refresh token revoked or not found"},
            )

        # Revoke old token
        gate.store.revoke_refresh_token(old_hash)

        user = gate.store.get_user(payload["sub"])
        if not user or not user.is_active:
            return JSONResponse(
                status_code=401,
                content={"detail": "User not found or inactive"},
            )

        access_token = create_access_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_access_ttl,
        )
        new_refresh, new_hash = create_refresh_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_refresh_ttl,
        )
        expires_at = datetime.now(timezone.utc).isoformat()
        gate.store.save_refresh_token(new_hash, user.id, expires_at)

        return JSONResponse(
            content={
                "access_token": access_token,
                "refresh_token": new_refresh,
            }
        )

    @router.post("/logout")
    async def logout(request: Request) -> JSONResponse:
        """Revoke a refresh token."""
        body = await request.json()
        refresh_token = body.get("refresh_token", "")
        if refresh_token:
            token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            gate.store.revoke_refresh_token(token_hash)
        return JSONResponse(content={"status": "ok"})

    @router.get("/me")
    async def me(request: Request) -> JSONResponse:
        """Get current user info + permissions."""
        user = getattr(request.state, "user", None)
        if not user:
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

        from stateloom.auth.permissions import resolve_permissions

        team_roles = getattr(request.state, "team_roles", [])
        perms = resolve_permissions(user, team_roles)

        return JSONResponse(
            content={
                "user_id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "org_id": user.org_id,
                "org_role": user.org_role.value if user.org_role else None,
                "email_verified": user.email_verified,
                "team_roles": [
                    {
                        "team_id": tr.team_id,
                        "role": tr.role.value,
                    }
                    for tr in team_roles
                ],
                "permissions": sorted(p.value for p in perms),
            }
        )

    @router.post("/change-password")
    async def change_password(request: Request) -> JSONResponse:
        """Change password. Requires current password. Revokes all refresh tokens."""
        user = getattr(request.state, "user", None)
        if not user:
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

        body = await request.json()
        current = body.get("current_password", "")
        new_password = body.get("new_password", "")

        if not current or not new_password:
            return JSONResponse(
                status_code=422,
                content={"detail": "current_password and new_password are required"},
            )

        if len(new_password) < 8:
            return JSONResponse(
                status_code=422,
                content={"detail": "New password must be at least 8 characters"},
            )

        if not verify_password(current, user.password_hash):
            return JSONResponse(
                status_code=401,
                content={"detail": "Current password is incorrect"},
            )

        user.password_hash = hash_password(new_password)
        gate.store.save_user(user)
        gate.store.revoke_all_refresh_tokens(user.id)

        return JSONResponse(content={"status": "ok"})

    # --- OIDC endpoints (enterprise-gated) ---

    # In-memory state storage for OIDC flows (state -> flow data)
    _oidc_states: dict[str, dict[str, Any]] = {}

    # Feature gate: OIDC requires enterprise license
    _oidc_deps: list[Any] = []
    _registry = getattr(gate, "_feature_registry", None)
    if _registry is not None:
        from stateloom.auth.dependencies import require_feature

        _oidc_deps = [Depends(require_feature(_registry, "oidc"))]

    @router.get("/oidc/providers", dependencies=_oidc_deps)
    async def oidc_providers() -> JSONResponse:
        """List active OIDC providers (public, for login UI)."""
        providers = gate.store.list_oidc_providers()
        return JSONResponse(
            content=[
                {
                    "id": p.id,
                    "name": p.name,
                    "issuer_url": p.issuer_url,
                }
                for p in providers
                if p.is_active
            ]
        )

    @router.get("/oidc/authorize/{provider_id}", dependencies=_oidc_deps)
    async def oidc_authorize(provider_id: str, request: Request) -> JSONResponse:
        """Get authorization URL for OIDC login."""
        provider = gate.store.get_oidc_provider(provider_id)
        if not provider or not provider.is_active:
            return JSONResponse(
                status_code=404,
                content={"detail": "OIDC provider not found"},
            )

        from stateloom.auth.oidc import OIDCClient

        oidc = OIDCClient(
            issuer_url=provider.issuer_url,
            client_id=provider.client_id,
            client_secret=provider.client_secret_encrypted,
            scopes=provider.scopes,
        )

        redirect_uri = request.query_params.get(
            "redirect_uri", str(request.base_url) + "api/v1/auth/oidc/callback"
        )

        url, state, nonce, code_verifier = oidc.build_authorization_url(
            redirect_uri=redirect_uri,
        )

        # Store flow state
        _oidc_states[state] = {
            "provider_id": provider_id,
            "nonce": nonce,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
        }

        return JSONResponse(
            content={
                "authorization_url": url,
                "state": state,
            }
        )

    @router.post("/oidc/callback", dependencies=_oidc_deps)
    async def oidc_callback(request: Request) -> JSONResponse:
        """Exchange OIDC authorization code for StateLoom JWT tokens."""
        body = await request.json()
        code = body.get("code", "")
        state = body.get("state", "")

        if not code or not state:
            return JSONResponse(
                status_code=422,
                content={"detail": "code and state are required"},
            )

        flow = _oidc_states.pop(state, None)
        if not flow:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid or expired state"},
            )

        provider = gate.store.get_oidc_provider(flow["provider_id"])
        if not provider:
            return JSONResponse(
                status_code=404,
                content={"detail": "OIDC provider not found"},
            )

        from stateloom.auth.oidc import OIDCClient, OIDCError

        oidc = OIDCClient(
            issuer_url=provider.issuer_url,
            client_id=provider.client_id,
            client_secret=provider.client_secret_encrypted,
            scopes=provider.scopes,
        )

        try:
            token_response = await oidc.exchange_code(
                code=code,
                redirect_uri=flow["redirect_uri"],
                code_verifier=flow["code_verifier"],
            )
        except OIDCError as exc:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Token exchange failed: {exc}"},
            )

        id_token = token_response.get("id_token", "")
        if not id_token:
            return JSONResponse(
                status_code=400,
                content={"detail": "No id_token in response"},
            )

        try:
            claims = oidc.validate_id_token(
                id_token,
                nonce=flow["nonce"],
            )
        except OIDCError as exc:
            return JSONResponse(
                status_code=400,
                content={"detail": f"ID token validation failed: {exc}"},
            )

        oidc_sub = claims.get("sub", "")
        oidc_email = claims.get("email", "")

        if not oidc_sub:
            return JSONResponse(
                status_code=400,
                content={"detail": "No sub claim in ID token"},
            )

        # TOFU flow: lookup by OIDC identity
        user = gate.store.get_user_by_oidc(provider.id, oidc_sub)
        if user:
            # Known user — log in
            return await _oidc_login(user, provider, claims)

        # Check if email matches an existing user
        if oidc_email:
            existing = gate.store.get_user_by_email(oidc_email)
            if existing and existing.email_verified:
                # TOFU: requires local password proof before merge
                return JSONResponse(
                    status_code=200,
                    content={
                        "requires_linking": True,
                        "email": oidc_email,
                        "oidc_provider_id": provider.id,
                        "oidc_subject": oidc_sub,
                        "state": state,
                    },
                )

        # JIT provision new user
        org_id = _ensure_default_org(gate)
        groups = OIDCClient.extract_groups(claims, provider.group_claim)

        new_user = User(
            email=oidc_email or f"oidc-{oidc_sub}@{provider.issuer_url}",
            email_verified=bool(claims.get("email_verified", False)),
            oidc_provider_id=provider.id,
            oidc_subject=oidc_sub,
            org_id=org_id,
            display_name=claims.get("name", ""),
            is_active=True,
        )

        # Apply group-role mapping
        _apply_group_mapping(new_user, groups, provider)

        gate.store.save_user(new_user)
        return await _oidc_login(new_user, provider, claims)

    @router.post("/oidc/link", dependencies=_oidc_deps)
    async def oidc_link(request: Request) -> JSONResponse:
        """TOFU: verify local password, merge OIDC identity."""
        body = await request.json()
        email = body.get("email", "")
        password = body.get("password", "")
        oidc_provider_id = body.get("oidc_provider_id", "")
        oidc_subject = body.get("oidc_subject", "")

        if not email or not password or not oidc_provider_id or not oidc_subject:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "email, password, oidc_provider_id, and oidc_subject are required"
                },
            )

        user = gate.store.get_user_by_email(email)
        if not user or not user.is_active:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid credentials"},
            )

        if not verify_password(password, user.password_hash):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid credentials"},
            )

        # Link OIDC identity
        user.oidc_provider_id = oidc_provider_id
        user.oidc_subject = oidc_subject
        user.last_login = datetime.now(timezone.utc)
        gate.store.save_user(user)

        secret = _get_secret()
        access_token = create_access_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_access_ttl,
        )
        refresh_token, refresh_hash = create_refresh_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_refresh_ttl,
        )
        expires_at = datetime.now(timezone.utc).isoformat()
        gate.store.save_refresh_token(refresh_hash, user.id, expires_at)

        return JSONResponse(
            content={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user_id": user.id,
                "email": user.email,
                "linked": True,
            }
        )

    async def _oidc_login(user: Any, provider: Any, claims: dict[str, Any]) -> JSONResponse:
        """Issue JWT tokens for an OIDC-authenticated user."""
        from stateloom.auth.oidc import OIDCClient

        user.last_login = datetime.now(timezone.utc)

        # Apply group-role mapping on every login
        groups = OIDCClient.extract_groups(claims, provider.group_claim)
        _apply_group_mapping(user, groups, provider)

        gate.store.save_user(user)

        secret = _get_secret()
        access_token = create_access_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_access_ttl,
        )
        refresh_token, refresh_hash = create_refresh_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_refresh_ttl,
        )
        expires_at = datetime.now(timezone.utc).isoformat()
        gate.store.save_refresh_token(refresh_hash, user.id, expires_at)

        return JSONResponse(
            content={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user_id": user.id,
                "email": user.email,
            }
        )

    def _apply_group_mapping(user: Any, groups: list[str], provider: Any) -> None:
        """Apply OIDC group-to-role mapping to a user."""
        if not groups or not provider.group_role_mapping:
            return
        for group in groups:
            mapping = provider.group_role_mapping.get(group)
            if mapping:
                team_id = mapping.get("team_id", "")
                role_str = mapping.get("role", "")
                if team_id and role_str:
                    try:
                        role = Role(role_str)
                        from stateloom.auth.models import UserTeamRole

                        # Check if role already exists
                        existing = gate.store.get_user_team_roles(user.id)
                        already_assigned = any(r.team_id == team_id for r in existing)
                        if not already_assigned:
                            utr = UserTeamRole(
                                user_id=user.id,
                                team_id=team_id,
                                role=role,
                                granted_by="oidc",
                            )
                            gate.store.save_user_team_role(utr)
                    except (ValueError, Exception):
                        pass

    # --- Device authorization grant (CLI login) ---

    _device_codes: dict[str, dict[str, Any]] = {}

    @router.post("/device/authorize")
    async def device_authorize(request: Request) -> JSONResponse:
        """Device authorization grant — start device flow."""
        import secrets as _secrets

        device_code = _secrets.token_urlsafe(32)
        user_code = _secrets.token_hex(4).upper()

        base_url = str(request.base_url).rstrip("/")
        verification_uri = f"{base_url}/api/v1/auth/device/verify?user_code={user_code}"

        _device_codes[device_code] = {
            "user_code": user_code,
            "user_id": None,
            "approved": False,
            "denied": False,
        }

        return JSONResponse(
            content={
                "device_code": device_code,
                "user_code": user_code,
                "verification_uri": verification_uri,
                "expires_in": 600,
                "interval": 5,
            }
        )

    @router.post("/device/token")
    async def device_token(request: Request) -> JSONResponse:
        """Device authorization grant — poll for token."""
        body = await request.json()
        device_code = body.get("device_code", "")

        record = _device_codes.get(device_code)
        if not record:
            return JSONResponse(
                status_code=400,
                content={"error": "expired_token"},
            )

        if record["denied"]:
            _device_codes.pop(device_code, None)
            return JSONResponse(
                status_code=403,
                content={"error": "access_denied"},
            )

        if not record["approved"] or not record["user_id"]:
            return JSONResponse(
                status_code=428,
                content={"error": "authorization_pending"},
            )

        # Approved — issue tokens
        user = gate.store.get_user(record["user_id"])
        if not user or not user.is_active:
            return JSONResponse(
                status_code=401,
                content={"error": "User not found or inactive"},
            )

        _device_codes.pop(device_code, None)

        secret = _get_secret()
        access_token = create_access_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_access_ttl,
        )
        refresh_token, refresh_hash = create_refresh_token(
            user,
            secret,
            algorithm=gate.config.auth_jwt_algorithm,
            ttl=gate.config.auth_jwt_refresh_ttl,
        )
        expires_at = datetime.now(timezone.utc).isoformat()
        gate.store.save_refresh_token(refresh_hash, user.id, expires_at)

        return JSONResponse(
            content={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user_id": user.id,
                "email": user.email,
            }
        )

    @router.post("/device/verify")
    async def device_verify(request: Request) -> JSONResponse:
        """Device authorization grant — user approves the device."""
        user = getattr(request.state, "user", None)
        if not user:
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

        body = await request.json()
        user_code = body.get("user_code", "")

        # Find matching device code
        for dc, record in _device_codes.items():
            if record["user_code"] == user_code:
                record["user_id"] = user.id
                record["approved"] = True
                return JSONResponse(content={"status": "approved"})

        return JSONResponse(
            status_code=404,
            content={"detail": "Invalid user code"},
        )

    return router


def _ensure_default_org(gate: Any) -> str:
    """Ensure a 'Default Organization' exists. Returns its ID."""
    orgs = gate.store.list_organizations()
    if orgs:
        return str(orgs[0].id)

    from stateloom.core.organization import Organization

    org = Organization(name="Default Organization")
    gate.store.save_organization(org)
    return org.id
