"""
FastAPI dependencies for RBAC auth.

Usage in route handlers:
    @router.get("/...")
    async def endpoint(user=Depends(require_role("ADMIN", "MANAGER"))):
        ...
"""

from fastapi import Depends, HTTPException, Header, status
from typing import Optional

from app.auth.jwt import decode_access_token, verify_role


async def extract_token(authorization: Optional[str] = Header(default=None)) -> dict:
    """Extract and decode JWT from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Bearer <token>",
        )

    try:
        payload = decode_access_token(parts[1])
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return payload


def require_role(*allowed_roles: str):
    """Dependency factory that requires the user to have one of the given roles."""
    async def _check(user: dict = Depends(extract_token)):
        if not verify_role(user, list(allowed_roles)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.get('role')}' not allowed. Required: {list(allowed_roles)}",
            )
        return user
    return _check


# Convenience dependencies
require_admin = require_role("ADMIN")
require_manager = require_role("ADMIN", "MANAGER")
require_waiter = require_role("ADMIN", "MANAGER", "WAITER")
require_kitchen = require_role("ADMIN", "MANAGER", "KITCHEN")
require_cashier = require_role("ADMIN", "MANAGER", "CASHIER")
require_driver = require_role("ADMIN", "MANAGER", "DRIVER")
