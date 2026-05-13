"""
Auth routes — staff login.
"""

import bcrypt
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import User
from app.auth.jwt import create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash, or fall back to plain comparison for seeds."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        # Fallback for plain-text seeded passwords (dev only)
        return plain == hashed


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    role: str
    restaurant_id: int


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.email == body.email, User.active == True)
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

    return LoginResponse(
        access_token=token,
        user_id=user.id,
        role=user.role,
        restaurant_id=user.restaurant_id,
    )
