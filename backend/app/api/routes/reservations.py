"""
Reservation routes — list, confirm, cancel.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models.reservations import Reservation
from app.db.models.core import Customer
from app.auth.deps import extract_token, require_manager

router = APIRouter(prefix="/api/v1/reservations", tags=["reservations"])


class ReservationStatusUpdate(BaseModel):
    status: str  # CONFIRMED | CANCELLED | NO_SHOW


@router.get("")
async def list_reservations(
    status: Optional[str] = None,
    limit: int = 100,
    user=Depends(extract_token),
):
    rid = user.get("restaurant_id", 1)
    async with AsyncSessionLocal() as session:
        query = (
            select(Reservation)
            .where(Reservation.restaurant_id == rid)
            .order_by(Reservation.date.desc(), Reservation.time.desc())
            .limit(limit)
        )
        if status:
            query = query.where(Reservation.status == status)
        result = await session.execute(query)
        reservations = result.scalars().all()

        out = []
        for r in reservations:
            cust_name = None
            cust_phone = None
            if r.customer_id:
                cust_result = await session.execute(select(Customer).where(Customer.id == r.customer_id))
                cust = cust_result.scalar_one_or_none()
                if cust:
                    cust_name = cust.name
                    cust_phone = cust.phone
            out.append({
                "id": r.id,
                "reservation_ref": r.reservation_ref,
                "customer_name": cust_name,
                "customer_phone": cust_phone,
                "date": r.date.isoformat() if r.date else None,
                "time": r.time.isoformat() if r.time else None,
                "guests": r.guests,
                "special_request": r.special_request,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })
        return out


@router.patch("/{reservation_id}")
async def update_reservation_status(
    reservation_id: int,
    body: ReservationStatusUpdate,
    user=Depends(require_manager),
):
    rid = user.get("restaurant_id", 1)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Reservation).where(Reservation.id == reservation_id, Reservation.restaurant_id == rid)
        )
        reservation = result.scalar_one_or_none()
        if not reservation:
            raise HTTPException(404, "Reservation not found")
        reservation.status = body.status
        await session.commit()
        return {"ok": True, "reservation_id": reservation_id, "status": body.status}
