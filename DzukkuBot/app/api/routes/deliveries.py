"""
Delivery routes — assign driver, location updates, tracking, proof of delivery.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Delivery, Driver, DeliveryLocationEvent, Order, User
from app.auth.deps import require_manager, extract_token

router = APIRouter(prefix="/api/v1/deliveries", tags=["deliveries"])


class AssignDriverRequest(BaseModel):
    order_id: int
    driver_id: int


class LocationUpdateRequest(BaseModel):
    lat: float
    lng: float
    accuracy_m: Optional[int] = None


class DeliveryStatusUpdateRequest(BaseModel):
    status: str  # ASSIGNED | PICKED_UP | EN_ROUTE | DELIVERED | FAILED


class ProofOfDeliveryRequest(BaseModel):
    proof_url: str
    proof_type: str  # PHOTO | SIGNATURE


# ── List active deliveries ───────────────────────────────────────────────────

@router.get("")
async def list_deliveries(status: Optional[str] = None, user=Depends(extract_token)):
    """List all deliveries, optionally filtered by status."""
    async with AsyncSessionLocal() as session:
        query = select(Delivery).order_by(Delivery.id.desc())
        if status:
            query = query.where(Delivery.status == status)
        result = await session.execute(query.limit(200))
        deliveries = result.scalars().all()

        out = []
        for d in deliveries:
            driver_name = None
            if d.driver_id:
                drv = (await session.execute(select(Driver).where(Driver.id == d.driver_id))).scalar_one_or_none()
                if drv and drv.user_id:
                    usr = (await session.execute(select(User).where(User.id == drv.user_id))).scalar_one_or_none()
                    driver_name = usr.name if usr else None

            order_ref = None
            order_type = None
            order_total_cents = 0
            customer_phone = None
            if d.order_id:
                order = (await session.execute(select(Order).where(Order.id == d.order_id))).scalar_one_or_none()
                if order:
                    order_ref = order.order_ref
                    order_type = order.order_type
                    order_total_cents = order.total_cents

            out.append({
                "id": d.id,
                "order_id": d.order_id,
                "order_ref": order_ref,
                "order_type": order_type,
                "order_total_cents": order_total_cents,
                "driver_id": d.driver_id,
                "driver_name": driver_name,
                "status": d.status,
                "address_json": d.address_json,
                "customer_phone": d.customer_phone or (d.address_json or {}).get("phone"),
                "proof_url": d.proof_url,
                "proof_type": d.proof_type,
                "assigned_at": d.assigned_at.isoformat() if d.assigned_at else None,
                "picked_up_at": d.picked_up_at.isoformat() if d.picked_up_at else None,
                "delivered_at": d.delivered_at.isoformat() if d.delivered_at else None,
            })
        return out


# ── List available drivers ────────────────────────────────────────────────────

@router.get("/drivers")
async def list_drivers(user=Depends(extract_token)):
    """List all drivers with their availability status."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Driver).order_by(Driver.id)
        )
        drivers = result.scalars().all()

        out = []
        for drv in drivers:
            name = None
            phone = None
            if drv.user_id:
                usr = (await session.execute(select(User).where(User.id == drv.user_id))).scalar_one_or_none()
                name = usr.name if usr else None
                phone = usr.phone if usr else None

            # Check if driver has an active delivery
            active_delivery = (await session.execute(
                select(Delivery).where(
                    Delivery.driver_id == drv.id,
                    Delivery.status.in_(["ASSIGNED", "PICKED_UP", "EN_ROUTE"]),
                )
            )).scalar_one_or_none()

            out.append({
                "id": drv.id,
                "name": name,
                "phone": phone,
                "vehicle_type": drv.vehicle_type,
                "vehicle_no": drv.vehicle_no,
                "active": drv.active,
                "on_delivery": active_delivery is not None,
                "current_delivery_id": active_delivery.id if active_delivery else None,
            })
        return out


# ── Assign driver ─────────────────────────────────────────────────────────────

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

        # Get order info for address
        order = (await session.execute(
            select(Order).where(Order.id == body.order_id)
        )).scalar_one_or_none()
        order_ref = order.order_ref if order else None

        rid = user.get("restaurant_id", 1)
        if delivery:
            delivery.driver_id = body.driver_id
            delivery.status = "ASSIGNED"
            delivery.assigned_at = datetime.now(timezone.utc)
        else:
            delivery = Delivery(
                restaurant_id=rid,
                order_id=body.order_id,
                driver_id=body.driver_id,
                status="ASSIGNED",
                address_json={},
                assigned_at=datetime.now(timezone.utc),
            )
            session.add(delivery)

        await session.commit()
        await session.refresh(delivery)

        # Publish event
        from app.realtime.events import delivery_status_changed
        from app.realtime.ws_manager import ws_manager
        evt = delivery_status_changed(delivery.id, body.order_id, "ASSIGNED")
        await ws_manager.broadcast(rid, evt.to_dict())

        # Notify customer
        if order_ref:
            try:
                from app.realtime.notifications import notify_delivery_status
                await notify_delivery_status(delivery.id, order_ref, "ASSIGNED", rid)
            except Exception:
                pass

        return {"delivery_id": delivery.id, "driver_id": body.driver_id, "status": "ASSIGNED"}


# ── Delivery status update ────────────────────────────────────────────────────

@router.patch("/{delivery_id}/status")
async def update_delivery_status(delivery_id: int, body: DeliveryStatusUpdateRequest, user=Depends(extract_token)):
    """Update delivery status (ASSIGNED → PICKED_UP → EN_ROUTE → DELIVERED/FAILED)."""
    VALID = {"ASSIGNED", "PICKED_UP", "EN_ROUTE", "DELIVERED", "FAILED"}
    if body.status not in VALID:
        raise HTTPException(400, f"Invalid status. Use: {', '.join(sorted(VALID))}")

    async with AsyncSessionLocal() as session:
        del_result = await session.execute(select(Delivery).where(Delivery.id == delivery_id))
        delivery = del_result.scalar_one_or_none()
        if not delivery:
            raise HTTPException(404, "Delivery not found")

        delivery.status = body.status
        if body.status == "PICKED_UP":
            delivery.picked_up_at = datetime.now(timezone.utc)
        elif body.status == "DELIVERED":
            delivery.delivered_at = datetime.now(timezone.utc)
            # Also mark order as DELIVERED
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id)
            )).scalar_one_or_none()
            if order:
                order.status = "DELIVERED"
        elif body.status == "FAILED":
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id)
            )).scalar_one_or_none()
            if order:
                order.status = "CANCELLED"

        await session.commit()

        # Publish events
        from app.realtime.events import delivery_status_changed, order_status_changed
        from app.realtime.ws_manager import ws_manager
        evt = delivery_status_changed(delivery_id, delivery.order_id, body.status)
        await ws_manager.broadcast(delivery.restaurant_id, evt.to_dict())

        if body.status == "DELIVERED":
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id)
            )).scalar_one_or_none()
            if order:
                evt2 = order_status_changed(order.id, order.order_ref, "DELIVERED")
                await ws_manager.broadcast(delivery.restaurant_id, evt2.to_dict())

        # Notify customer
        order_ref = None
        if delivery.order_id:
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id)
            )).scalar_one_or_none()
            order_ref = order.order_ref if order else None

        if order_ref:
            try:
                from app.realtime.notifications import notify_delivery_status
                await notify_delivery_status(delivery_id, order_ref, body.status, delivery.restaurant_id)
            except Exception:
                pass

        return {"delivery_id": delivery_id, "status": body.status}


# ── Proof of delivery ─────────────────────────────────────────────────────────

@router.post("/{delivery_id}/proof")
async def submit_proof_of_delivery(delivery_id: int, body: ProofOfDeliveryRequest, user=Depends(extract_token)):
    """Submit proof of delivery (photo URL or signature)."""
    async with AsyncSessionLocal() as session:
        del_result = await session.execute(select(Delivery).where(Delivery.id == delivery_id))
        delivery = del_result.scalar_one_or_none()
        if not delivery:
            raise HTTPException(404, "Delivery not found")

        delivery.proof_url = body.proof_url
        delivery.proof_type = body.proof_type
        await session.commit()

        return {"delivery_id": delivery_id, "proof_url": body.proof_url, "proof_type": body.proof_type}


# ── Driver GPS ping ───────────────────────────────────────────────────────────

@router.post("/{delivery_id}/location")
async def update_location(delivery_id: int, body: LocationUpdateRequest):
    """Driver GPS ping — update delivery location."""
    async with AsyncSessionLocal() as session:
        del_result = await session.execute(select(Delivery).where(Delivery.id == delivery_id))
        delivery = del_result.scalar_one_or_none()
        if not delivery:
            raise HTTPException(404, "Delivery not found")
        rid = delivery.restaurant_id
        event = DeliveryLocationEvent(
            restaurant_id=rid,
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
        await ws_manager.broadcast(rid, evt.to_dict())

        return {"ok": True}


# ── Delivery tracking ─────────────────────────────────────────────────────────

@router.get("/{delivery_id}/track")
async def track_delivery(delivery_id: int):
    """Get delivery tracking info with order details, driver info, and last known location."""
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

        # Get recent locations (last 20)
        locs_result = await session.execute(
            select(DeliveryLocationEvent)
            .where(DeliveryLocationEvent.delivery_id == delivery_id)
            .order_by(DeliveryLocationEvent.recorded_at.desc())
            .limit(20)
        )
        recent_locs = locs_result.scalars().all()

        # Driver info
        driver_info = None
        if delivery.driver_id:
            drv = (await session.execute(select(Driver).where(Driver.id == delivery.driver_id))).scalar_one_or_none()
            if drv and drv.user_id:
                usr = (await session.execute(select(User).where(User.id == drv.user_id))).scalar_one_or_none()
                driver_info = {
                    "id": drv.id,
                    "name": usr.name if usr else "Driver",
                    "phone": usr.phone if usr else None,
                    "vehicle_type": drv.vehicle_type,
                    "vehicle_no": drv.vehicle_no,
                }

        # Order info
        order_info = None
        if delivery.order_id:
            order = (await session.execute(select(Order).where(Order.id == delivery.order_id))).scalar_one_or_none()
            if order:
                order_info = {
                    "order_ref": order.order_ref,
                    "status": order.status,
                    "total_cents": order.total_cents,
                    "order_type": order.order_type,
                }

        return {
            "delivery_id": delivery.id,
            "status": delivery.status,
            "driver_id": delivery.driver_id,
            "driver": driver_info,
            "order": order_info,
            "address": delivery.address_json,
            "proof_url": delivery.proof_url,
            "proof_type": delivery.proof_type,
            "assigned_at": delivery.assigned_at.isoformat() if delivery.assigned_at else None,
            "picked_up_at": delivery.picked_up_at.isoformat() if delivery.picked_up_at else None,
            "delivered_at": delivery.delivered_at.isoformat() if delivery.delivered_at else None,
            "last_location": {
                "lat": float(last_loc.lat),
                "lng": float(last_loc.lng),
                "recorded_at": last_loc.recorded_at.isoformat(),
            } if last_loc else None,
            "location_history": [
                {
                    "lat": float(loc.lat),
                    "lng": float(loc.lng),
                    "recorded_at": loc.recorded_at.isoformat(),
                }
                for loc in reversed(recent_locs)
            ],
        }


# ── Track by order reference (customer-facing) ────────────────────────────────

@router.get("/orders/{order_ref}/track")
async def track_delivery_by_order(order_ref: str):
    """Customer-facing: track delivery by order reference."""
    async with AsyncSessionLocal() as session:
        order = (await session.execute(
            select(Order).where(Order.order_ref == order_ref)
        )).scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")

        delivery = (await session.execute(
            select(Delivery).where(Delivery.order_id == order.id)
        )).scalar_one_or_none()
        if not delivery:
            return {
                "order_ref": order.order_ref,
                "order_status": order.status,
                "delivery": None,
                "message": "Delivery not yet assigned. Your order is being prepared.",
            }

        # Reuse track_delivery logic
        return await track_delivery(delivery.id)
