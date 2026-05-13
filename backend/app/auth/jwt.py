"""
JWT authentication for Dzukku vNext staff portal.

Generates and verifies HS256 JWTs for staff login.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

# Config
JWT_SECRET = os.getenv("JWT_SECRET", "dzukku-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "480"))  # 8 hours default


def create_access_token(
    user_id: int,
    restaurant_id: int,
    role: str,
    email: Optional[str] = None,
) -> str:
    """Generate a JWT access token for a staff user."""
    payload = {
        "sub": str(user_id),
        "user_id": user_id,
        "restaurant_id": restaurant_id,
        "role": role,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT token.

    Returns the payload dict, or raises jwt exceptions on failure.
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def verify_role(token_payload: dict, allowed_roles: list[str]) -> bool:
    """Check if the token's role is in the allowed list."""
    return token_payload.get("role") in allowed_roles
