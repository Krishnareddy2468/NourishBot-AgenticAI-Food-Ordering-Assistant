"""
Auth routes — staff login, token refresh.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import User
from app.auth.jwt import create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


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
    """Staff login — returns JWT on success."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.email == body.email, User.active == True)
        )
        user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # TODO: Replace with proper password hashing (bcrypt/argon2)
    if user.password_hash != body.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

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
