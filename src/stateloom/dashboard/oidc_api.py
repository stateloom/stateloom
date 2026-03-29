"""OIDC provider management API endpoints for the dashboard."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("stateloom.dashboard.oidc_api")


def create_oidc_api_router(gate: Any) -> APIRouter:
    """Create the OIDC provider management API router."""
    from stateloom.auth.oidc_models import OIDCProvider

    router = APIRouter(prefix="/oidc/providers", tags=["oidc"])

    @router.post("")
    async def create_provider(request: Request) -> JSONResponse:
        """Create an OIDC provider. Requires ADMIN_OIDC permission."""
        body = await request.json()
        issuer_url = body.get("issuer_url", "").strip()
        client_id = body.get("client_id", "").strip()

        if not issuer_url or not client_id:
            return JSONResponse(
                status_code=422,
                content={"detail": "issuer_url and client_id are required"},
            )

        # Check for duplicate issuer
        existing = gate.store.get_oidc_provider_by_issuer(issuer_url)
        if existing:
            return JSONResponse(
                status_code=409,
                content={"detail": "Provider with this issuer_url already exists"},
            )

        provider = OIDCProvider(
            name=body.get("name", ""),
            issuer_url=issuer_url,
            client_id=client_id,
            client_secret_encrypted=body.get("client_secret", ""),
            scopes=body.get("scopes", "openid email profile"),
            group_claim=body.get("group_claim", ""),
            group_role_mapping=body.get("group_role_mapping", {}),
            is_active=body.get("is_active", True),
        )
        gate.store.save_oidc_provider(provider)

        return JSONResponse(
            status_code=201,
            content={
                "id": provider.id,
                "name": provider.name,
                "issuer_url": provider.issuer_url,
                "client_id": provider.client_id,
                "is_active": provider.is_active,
            },
        )

    @router.get("")
    async def list_providers() -> JSONResponse:
        """List all OIDC providers. Requires ADMIN_OIDC permission."""
        providers = gate.store.list_oidc_providers()
        return JSONResponse(
            content=[
                {
                    "id": p.id,
                    "name": p.name,
                    "issuer_url": p.issuer_url,
                    "client_id": p.client_id,
                    "scopes": p.scopes,
                    "group_claim": p.group_claim,
                    "is_active": p.is_active,
                    "created_at": p.created_at.isoformat(),
                }
                for p in providers
            ]
        )

    @router.get("/{provider_id}")
    async def get_provider(provider_id: str) -> JSONResponse:
        """Get an OIDC provider by ID. Requires ADMIN_OIDC permission."""
        provider = gate.store.get_oidc_provider(provider_id)
        if not provider:
            return JSONResponse(
                status_code=404,
                content={"detail": "Provider not found"},
            )
        return JSONResponse(
            content={
                "id": provider.id,
                "name": provider.name,
                "issuer_url": provider.issuer_url,
                "client_id": provider.client_id,
                "scopes": provider.scopes,
                "group_claim": provider.group_claim,
                "group_role_mapping": provider.group_role_mapping,
                "is_active": provider.is_active,
                "created_at": provider.created_at.isoformat(),
            }
        )

    @router.patch("/{provider_id}")
    async def update_provider(provider_id: str, request: Request) -> JSONResponse:
        """Update an OIDC provider. Requires ADMIN_OIDC permission."""
        provider = gate.store.get_oidc_provider(provider_id)
        if not provider:
            return JSONResponse(
                status_code=404,
                content={"detail": "Provider not found"},
            )

        body = await request.json()
        if "name" in body:
            provider.name = body["name"]
        if "client_id" in body:
            provider.client_id = body["client_id"]
        if "client_secret" in body:
            provider.client_secret_encrypted = body["client_secret"]
        if "scopes" in body:
            provider.scopes = body["scopes"]
        if "group_claim" in body:
            provider.group_claim = body["group_claim"]
        if "group_role_mapping" in body:
            provider.group_role_mapping = body["group_role_mapping"]
        if "is_active" in body:
            provider.is_active = body["is_active"]

        gate.store.save_oidc_provider(provider)
        return JSONResponse(content={"status": "ok", "id": provider.id})

    @router.delete("/{provider_id}")
    async def delete_provider(provider_id: str) -> JSONResponse:
        """Delete an OIDC provider. Requires ADMIN_OIDC permission."""
        if not gate.store.delete_oidc_provider(provider_id):
            return JSONResponse(
                status_code=404,
                content={"detail": "Provider not found"},
            )
        return JSONResponse(content={"status": "ok"})

    return router
