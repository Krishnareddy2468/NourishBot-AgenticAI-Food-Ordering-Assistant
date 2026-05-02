"""
Delivery routes — assign driver, location updates, tracking.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Delivery, Driver, DeliveryLocationEvent
from app.auth.deps import require_manager

router = APIRouter(prefix="/api/v1/deliveries", tags=["deliveries"])


class AssignDriverRequest(BaseModel):
    order_id: int
    driver_id: int


class LocationUpdateRequest(BaseModel):
    lat: float
    lng: float
    accuracy_m: Optional[int] = None


@router.post("/assign", status_code=201)
async def assign_driver(body: AssignDriverRequest, user=Depends(require_manager)):
    async with AsyncSessionLocal() as session:
        # Check driver exists and is active
        drv_result = await session.execute(select(Driver).where(Driver.id == body.driver_id))
        driver = drv_result.scalar_one_or_none()
        if not driver or not driver.active:
            raise HTTPException(400, "Driver not found or inactive")

        # Find or create delivery for the order
        del_result = await session.execute(
            select(Delivery).where(Delivery.order_id == body.order_id)
        )
        delivery = del_result.scalar_one_or_none()

        if delivery:
            delivery.driver_id = body.driver_id
            delivery.status = "ASSIGNED"
            delivery.assigned_at = datetime.now(timezone.utc)
        else:
            delivery = Delivery(
                restaurant_id=1,
                order_id=body.order_id,
                driver_id=body.driver_id,
                status="ASSIGNED",
                address_json={},
                assigned_at=datetime.now(timezone.utc),
            )
            session.add(delivery)

        await session.commit()

        # Publish event
        from app.realtime.events import delivery_status_changed
        from app.realtime.ws_manager import ws_manager
        evt = delivery_status_changed(delivery.id, body.order_id, "ASSIGNED")
        await ws_manager.broadcast(1, evt.to_dict())

        return {"delivery_id": delivery.id, "driver_id": body.driver_id, "status": "ASSIGNED"}


@router.post("/{delivery_id}/location")
async def update_location(delivery_id: int, body: LocationUpdateRequest):
    """Driver GPS ping — update delivery location."""
    async with AsyncSessionLocal() as session:
        event = DeliveryLocationEvent(
            restaurant_id=1,
            delivery_id=delivery_id,
            lat=body.lat,
            lng=body.lng,
            accuracy_m=body.accuracy_m,
        )
        session.add(event)
        await session.commit()

        # Publish location event
        from app.realtime.events import delivery_location_updated
        from app.realtime.ws_manager import ws_manager
        evt = delivery_location_updated(delivery_id, body.lat, body.lng)
        await ws_manager.broadcast(1, evt.to_dict())

        return {"ok": True}


@router.get("/{delivery_id}/track")
async def track_delivery(delivery_id: int):
    """Get delivery tracking info with last known location."""
    async with AsyncSessionLocal() as session:
        del_result = await session.execute(select(Delivery).where(Delivery.id == delivery_id))
        delivery = del_result.scalar_one_or_none()
        if not delivery:
            raise HTTPException(404, "Delivery not found")

        # Get latest location
        loc_result = await session.execute(
            select(DeliveryLocationEvent)
            .where(DeliveryLocationEvent.delivery_id == delivery_id)
            .order_by(DeliveryLocationEvent.recorded_at.desc())
            .limit(1)
        )
        last_loc = loc_result.scalar_one_or_none()

        return {
            "delivery_id": delivery.id,
            "status": delivery.status,
            "driver_id": delivery.driver_id,
            "last_location": {
                "lat": float(last_loc.lat) if last_loc else None,
                "lng": float(last_loc.lng) if last_loc else None,
                "recorded_at": last_loc.recorded_at.isoformat() if last_loc else None,
            } if last_loc else None,
        }
