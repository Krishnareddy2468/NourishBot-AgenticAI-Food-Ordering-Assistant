"""
Staff routes — list and manage restaurant staff/users.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models.core import User
from app.auth.deps import extract_token, require_admin
from app.api.routes.auth import hash_password

router = APIRouter(prefix="/api/v1/staff", tags=["staff"])


class StaffCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    role: str  # ADMIN | MANAGER | CASHIER | WAITER | KITCHEN | DRIVER
    password: str


class StaffToggle(BaseModel):
    active: bool


@router.get("")
async def list_staff(user=Depends(extract_token)):
    rid = user.get("restaurant_id", 1)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User)
            .where(User.restaurant_id == rid)
            .order_by(User.role, User.name)
        )
        staff = result.scalars().all()
        return [
            {
                "id": s.id,
                "name": s.name,
                "email": s.email,
                "phone": s.phone,
                "role": s.role,
                "active": s.active,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in staff
        ]


@router.post("", status_code=201)
async def create_staff(body: StaffCreate, user=Depends(require_admin)):
    rid = user.get("restaurant_id", 1)
    async with AsyncSessionLocal() as session:
        new_user = User(
            restaurant_id=rid,
            name=body.name,
            email=body.email,
            phone=body.phone,
            role=body.role,
            password_hash=hash_password(body.password),
            active=True,
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        return {"id": new_user.id, "name": new_user.name, "role": new_user.role}


@router.patch("/{staff_id}/active")
async def toggle_staff_active(staff_id: int, body: StaffToggle, user=Depends(require_admin)):
    rid = user.get("restaurant_id", 1)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.id == staff_id, User.restaurant_id == rid)
        )
        staff = result.scalar_one_or_none()
        if not staff:
            raise HTTPException(404, "Staff member not found")
        staff.active = body.active
        await session.commit()
        return {"ok": True, "staff_id": staff_id, "active": body.active}
