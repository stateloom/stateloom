"""FastAPI dependencies for permission-based authorization."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import JSONResponse

from stateloom.auth.permissions import Permission, check_permission

if TYPE_CHECKING:
    from stateloom.core.feature_registry import FeatureRegistry


def require_feature(
    registry: FeatureRegistry,
    feature_name: str,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """FastAPI dependency — returns 403 if the feature is not licensed.

    Args:
        registry: The feature registry instance.
        feature_name: The feature to check.

    Returns:
        A FastAPI Depends-compatible callable.
    """

    async def _check(request: Request) -> None:
        if registry is not None and not registry.is_available(feature_name):
            from fastapi import HTTPException

            raise HTTPException(
                status_code=403,
                detail=f"Feature '{feature_name}' requires an enterprise license.",
            )

    return _check


def require_permission(
    permission: Permission,
    team_id_param: str | None = None,
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """FastAPI dependency that checks if the current user has a permission.

    Args:
        permission: The required permission.
        team_id_param: If set, extract team_id from request path/query params
            and scope the check to that team.

    Returns:
        A FastAPI Depends-compatible callable that returns the user or raises 403.
    """

    async def _check(request: Request) -> Any:
        user = getattr(request.state, "user", None)
        if user is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

        team_roles = getattr(request.state, "team_roles", [])

        # Resolve team_id from path or query params
        team_id = None
        if team_id_param:
            team_id = request.path_params.get(team_id_param) or request.query_params.get(
                team_id_param
            )

        if not check_permission(user, team_roles, permission, team_id=team_id):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": f"Permission denied: requires '{permission.value}'",
                    "permission": permission.value,
                },
            )

        return user

    return _check
