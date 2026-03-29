"""User management API endpoints for the dashboard."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("stateloom.dashboard.user_api")


def create_user_api_router(gate: Any) -> APIRouter:
    """Create the user management API router."""
    from stateloom.auth.models import User, UserTeamRole
    from stateloom.auth.password import hash_password
    from stateloom.core.types import Role

    router = APIRouter(prefix="/users", tags=["users"])

    @router.post("")
    async def create_user(request: Request) -> JSONResponse:
        """Create a new user. Requires ADMIN_USERS permission."""
        try:
            from stateloom.ee.setup import _check_dev_scale_cap

            _check_dev_scale_cap(gate, "users")
        except Exception as exc:
            if "LICENSE_REQUIRED" in str(exc):
                return JSONResponse(status_code=403, content={"detail": str(exc)})
            raise
        body = await request.json()
        email = body.get("email", "").strip()
        if not email:
            return JSONResponse(status_code=422, content={"detail": "email is required"})

        existing = gate.store.get_user_by_email(email)
        if existing:
            return JSONResponse(status_code=409, content={"detail": "Email already exists"})

        password = body.get("password", "")
        password_hash = hash_password(password) if password else ""

        org_role = None
        org_role_str = body.get("org_role")
        if org_role_str:
            try:
                org_role = Role(org_role_str)
            except ValueError:
                return JSONResponse(
                    status_code=422,
                    content={"detail": f"Invalid org_role: {org_role_str}"},
                )

        user = User(
            email=email,
            display_name=body.get("display_name", ""),
            password_hash=password_hash,
            email_verified=body.get("email_verified", False),
            org_id=body.get("org_id", ""),
            org_role=org_role,
        )
        gate.store.save_user(user)

        return JSONResponse(
            status_code=201,
            content={
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "org_id": user.org_id,
                "org_role": user.org_role.value if user.org_role else None,
                "is_active": user.is_active,
            },
        )

    @router.get("")
    async def list_users(request: Request) -> JSONResponse:
        """List users. Requires ADMIN_USERS permission."""
        org_id = request.query_params.get("org_id")
        limit = int(request.query_params.get("limit", "100"))
        offset = int(request.query_params.get("offset", "0"))
        users = gate.store.list_users(org_id=org_id, limit=limit, offset=offset)
        return JSONResponse(
            content=[
                {
                    "id": u.id,
                    "email": u.email,
                    "display_name": u.display_name,
                    "org_id": u.org_id,
                    "org_role": u.org_role.value if u.org_role else None,
                    "is_active": u.is_active,
                    "email_verified": u.email_verified,
                    "created_at": u.created_at.isoformat(),
                }
                for u in users
            ]
        )

    @router.get("/{user_id}")
    async def get_user(user_id: str) -> JSONResponse:
        """Get a user by ID. Requires ADMIN_USERS permission."""
        user = gate.store.get_user(user_id)
        if not user:
            return JSONResponse(status_code=404, content={"detail": "User not found"})

        team_roles = gate.store.get_user_team_roles(user_id)
        return JSONResponse(
            content={
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "org_id": user.org_id,
                "org_role": user.org_role.value if user.org_role else None,
                "is_active": user.is_active,
                "email_verified": user.email_verified,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "team_roles": [
                    {
                        "id": tr.id,
                        "team_id": tr.team_id,
                        "role": tr.role.value,
                        "granted_at": tr.granted_at.isoformat(),
                        "granted_by": tr.granted_by,
                    }
                    for tr in team_roles
                ],
            }
        )

    @router.patch("/{user_id}")
    async def update_user(user_id: str, request: Request) -> JSONResponse:
        """Update a user. Requires ADMIN_USERS permission."""
        user = gate.store.get_user(user_id)
        if not user:
            return JSONResponse(status_code=404, content={"detail": "User not found"})

        body = await request.json()
        if "display_name" in body:
            user.display_name = body["display_name"]
        if "email_verified" in body:
            user.email_verified = body["email_verified"]
        if "is_active" in body:
            user.is_active = body["is_active"]
        if "org_role" in body:
            if body["org_role"] is None:
                user.org_role = None
            else:
                try:
                    user.org_role = Role(body["org_role"])
                except ValueError:
                    return JSONResponse(
                        status_code=422,
                        content={"detail": f"Invalid org_role: {body['org_role']}"},
                    )
        if "password" in body and body["password"]:
            user.password_hash = hash_password(body["password"])

        gate.store.save_user(user)
        return JSONResponse(content={"status": "ok", "user_id": user.id})

    @router.delete("/{user_id}")
    async def delete_user(user_id: str) -> JSONResponse:
        """Soft-delete a user. Requires ADMIN_USERS permission."""
        if not gate.store.delete_user(user_id):
            return JSONResponse(status_code=404, content={"detail": "User not found"})
        return JSONResponse(content={"status": "ok"})

    @router.post("/{user_id}/team-roles")
    async def assign_team_role(user_id: str, request: Request) -> JSONResponse:
        """Assign a team role to a user."""
        user = gate.store.get_user(user_id)
        if not user:
            return JSONResponse(status_code=404, content={"detail": "User not found"})

        body = await request.json()
        team_id = body.get("team_id", "")
        role_str = body.get("role", "")

        if not team_id or not role_str:
            return JSONResponse(
                status_code=422,
                content={"detail": "team_id and role are required"},
            )

        try:
            role = Role(role_str)
        except ValueError:
            return JSONResponse(
                status_code=422,
                content={"detail": f"Invalid role: {role_str}"},
            )

        # Get granter from request state (if authenticated)
        granter_id = ""
        req_user = getattr(request.state, "user", None)
        if req_user:
            granter_id = req_user.id

        utr = UserTeamRole(
            user_id=user_id,
            team_id=team_id,
            role=role,
            granted_by=granter_id,
        )
        gate.store.save_user_team_role(utr)

        return JSONResponse(
            status_code=201,
            content={"id": utr.id, "user_id": user_id, "team_id": team_id, "role": role.value},
        )

    @router.delete("/{user_id}/team-roles/{team_id}")
    async def remove_team_role(user_id: str, team_id: str) -> JSONResponse:
        """Remove a user's team role."""
        roles = gate.store.get_user_team_roles(user_id)
        for role in roles:
            if role.team_id == team_id:
                gate.store.delete_user_team_role(role.id)
                return JSONResponse(content={"status": "ok"})
        return JSONResponse(status_code=404, content={"detail": "Role not found"})

    return router
