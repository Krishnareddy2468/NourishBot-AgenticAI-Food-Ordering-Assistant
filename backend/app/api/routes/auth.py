"""
Auth routes — staff login.
"""

import bcrypt
import hmac
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import User
from app.auth.jwt import create_access_token
from app.auth.deps import extract_token
from app.core.config import settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash, or fall back to plain comparison for seeds."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        # Legacy plain-text seeds are tolerated only in local development.
        return not settings.is_production and hmac.compare_digest(plain, hashed)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    user_id: int
    email: str
    role: str
    restaurant_id: int


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, response: Response):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.email == body.email.strip().lower(), User.active == True)
        )
        user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(
        user_id=user.id,
        restaurant_id=user.restaurant_id,
        role=user.role,
        email=user.email,
    )
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=token,
        max_age=settings.JWT_EXPIRY_MINUTES * 60,
        httponly=True,
        secure=settings.is_production,
        samesite=settings.AUTH_COOKIE_SAMESITE,
        path="/",
    )

    return LoginResponse(
        user_id=user.id,
        email=user.email or "",
        role=user.role,
        restaurant_id=user.restaurant_id,
    )


@router.get("/me", response_model=LoginResponse)
async def me(user=Depends(extract_token)):
    return LoginResponse(
        user_id=int(user["user_id"]),
        email=str(user.get("email") or ""),
        role=str(user["role"]),
        restaurant_id=int(user["restaurant_id"]),
    )


@router.post("/logout", status_code=204)
async def logout(response: Response):
    response.delete_cookie(
        key=settings.AUTH_COOKIE_NAME,
        path="/",
        secure=settings.is_production,
        httponly=True,
        samesite=settings.AUTH_COOKIE_SAMESITE,
    )
