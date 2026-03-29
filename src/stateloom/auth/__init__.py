"""Authentication and authorization for StateLoom.

Note: Enterprise auth features are now in stateloom.ee.auth.
This module provides backward compatibility.
"""

from stateloom.auth.models import User, UserTeamRole
from stateloom.auth.password import hash_password, verify_password
from stateloom.auth.permissions import (
    ROLE_PERMISSIONS,
    Permission,
    check_permission,
    resolve_permissions,
)

__all__ = [
    "User",
    "UserTeamRole",
    "Permission",
    "ROLE_PERMISSIONS",
    "check_permission",
    "resolve_permissions",
    "hash_password",
    "verify_password",
]
