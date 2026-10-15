"""
Delivery routes — assign driver, location updates, tracking, proof of delivery.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Delivery, Driver, DeliveryLocationEvent, Order, User
from app.auth.deps import require_manager, require_driver, extract_token, tenant_id

router = APIRouter(prefix="/api/v1/deliveries", tags=["deliveries"])

DELIVERY_TRANSITIONS = {
    "PENDING_ASSIGNMENT": {"ASSIGNED", "FAILED"},
    "ASSIGNED": {"PICKED_UP", "FAILED"},
    "PICKED_UP": {"EN_ROUTE", "DELIVERED", "FAILED"},
    "EN_ROUTE": {"DELIVERED", "FAILED"},
}


class AssignDriverRequest(BaseModel):
    order_id: int
    driver_id: int


class LocationUpdateRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    accuracy_m: Optional[int] = Field(default=None, ge=0, le=10000)


class DeliveryStatusUpdateRequest(BaseModel):
    status: str  # ASSIGNED | PICKED_UP | EN_ROUTE | DELIVERED | FAILED


class ProofOfDeliveryRequest(BaseModel):
    proof_url: HttpUrl
    proof_type: str  # PHOTO | SIGNATURE


# ── List active deliveries ───────────────────────────────────────────────────

@router.get("")
async def list_deliveries(status: Optional[str] = None, user=Depends(extract_token)):
    """List all deliveries, optionally filtered by status."""
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        query = select(Delivery).where(Delivery.restaurant_id == rid).order_by(Delivery.id.desc())
        if status:
            query = query.where(Delivery.status == status)
        result = await session.execute(query.limit(200))
        deliveries = result.scalars().all()

        out = []
        for d in deliveries:
            driver_name = None
            if d.driver_id:
                drv = (await session.execute(select(Driver).where(Driver.id == d.driver_id, Driver.restaurant_id == rid))).scalar_one_or_none()
                if drv and drv.user_id:
                    usr = (await session.execute(select(User).where(User.id == drv.user_id, User.restaurant_id == rid))).scalar_one_or_none()
                    driver_name = usr.name if usr else None

            order_ref = None
            order_type = None
            order_status = None
            scheduled_for = None
            order_total_cents = 0
            customer_phone = None
            if d.order_id:
                order = (await session.execute(select(Order).where(Order.id == d.order_id, Order.restaurant_id == rid))).scalar_one_or_none()
                if order:
                    order_ref = order.order_ref
                    order_type = order.order_type
                    order_status = order.status
                    scheduled_for = order.scheduled_for.isoformat() if order.scheduled_for else None
                    order_total_cents = order.total_cents

            out.append({
                "id": d.id,
                "order_id": d.order_id,
                "order_ref": order_ref,
                "order_type": order_type,
                "order_status": order_status,
                "scheduled_for": scheduled_for,
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
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Driver).where(Driver.restaurant_id == rid).order_by(Driver.id)
        )
        drivers = result.scalars().all()

        out = []
        for drv in drivers:
            name = None
            phone = None
            if drv.user_id:
                usr = (await session.execute(select(User).where(User.id == drv.user_id, User.restaurant_id == rid))).scalar_one_or_none()
                name = usr.name if usr else None
                phone = usr.phone if usr else None

            # Check if driver has an active delivery
            active_delivery = (await session.execute(
                select(Delivery).where(
                    Delivery.driver_id == drv.id,
                    Delivery.restaurant_id == rid,
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
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        # Check driver exists and is active
        drv_result = await session.execute(select(Driver).where(Driver.id == body.driver_id, Driver.restaurant_id == rid))
        driver = drv_result.scalar_one_or_none()
        if not driver or not driver.active:
            raise HTTPException(400, "Driver not found or inactive")
        active_delivery = (await session.execute(
            select(Delivery).where(
                Delivery.driver_id == driver.id,
                Delivery.restaurant_id == rid,
                Delivery.status.in_(["ASSIGNED", "PICKED_UP", "EN_ROUTE"]),
            )
        )).scalars().first()
        if active_delivery and active_delivery.order_id != body.order_id:
            raise HTTPException(409, "Driver already has an active delivery")

        # Find or create delivery for the order
        del_result = await session.execute(
            select(Delivery).where(Delivery.order_id == body.order_id, Delivery.restaurant_id == rid)
        )
        delivery = del_result.scalar_one_or_none()

        # Get order info for address
        order = (await session.execute(
            select(Order).where(Order.id == body.order_id, Order.restaurant_id == rid)
        )).scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")
        if order.order_type != "DELIVERY":
            raise HTTPException(400, "Drivers can only be assigned to home-delivery orders")
        if (not delivery or delivery.status == "PENDING_ASSIGNMENT") and order.status != "READY":
            raise HTTPException(409, "The kitchen must mark the order ready before driver assignment")
        order_ref = order.order_ref if order else None

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
        evt = delivery_status_changed(delivery.id, body.order_id, "ASSIGNED", rid)
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

    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        del_result = await session.execute(select(Delivery).where(Delivery.id == delivery_id, Delivery.restaurant_id == rid))
        delivery = del_result.scalar_one_or_none()
        if not delivery:
            raise HTTPException(404, "Delivery not found")
        await _require_delivery_access(session, delivery, user)
        allowed = DELIVERY_TRANSITIONS.get(delivery.status, set())
        if body.status not in allowed:
            raise HTTPException(409, f"Cannot move delivery from {delivery.status} to {body.status}")

        delivery.status = body.status
        if body.status == "PICKED_UP":
            delivery.picked_up_at = datetime.now(timezone.utc)
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id, Order.restaurant_id == rid)
            )).scalar_one_or_none()
            if order:
                order.status = "OUT_FOR_DELIVERY"
        elif body.status == "EN_ROUTE":
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id, Order.restaurant_id == rid)
            )).scalar_one_or_none()
            if order:
                order.status = "OUT_FOR_DELIVERY"
        elif body.status == "DELIVERED":
            delivery.delivered_at = datetime.now(timezone.utc)
            # Also mark order as DELIVERED
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id, Order.restaurant_id == rid)
            )).scalar_one_or_none()
            if order:
                order.status = "DELIVERED"
        elif body.status == "FAILED":
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id, Order.restaurant_id == rid)
            )).scalar_one_or_none()
            if order:
                order.status = "CANCELLED"

        await session.commit()

        # Publish events
        from app.realtime.events import delivery_status_changed, order_status_changed
        from app.realtime.ws_manager import ws_manager
        evt = delivery_status_changed(delivery_id, delivery.order_id, body.status, rid)
        await ws_manager.broadcast(delivery.restaurant_id, evt.to_dict())

        if body.status in ("PICKED_UP", "EN_ROUTE", "DELIVERED"):
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id, Order.restaurant_id == rid)
            )).scalar_one_or_none()
            if order:
                public_status = "DELIVERED" if body.status == "DELIVERED" else "OUT_FOR_DELIVERY"
                evt2 = order_status_changed(order.id, order.order_ref, public_status, rid)
                await ws_manager.broadcast(delivery.restaurant_id, evt2.to_dict())

        # Notify customer
        order_ref = None
        if delivery.order_id:
            order = (await session.execute(
                select(Order).where(Order.id == delivery.order_id, Order.restaurant_id == rid)
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
async def submit_proof_of_delivery(delivery_id: int, body: ProofOfDeliveryRequest, user=Depends(require_driver)):
    """Submit proof of delivery (photo URL or signature)."""
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        del_result = await session.execute(select(Delivery).where(Delivery.id == delivery_id, Delivery.restaurant_id == rid))
        delivery = del_result.scalar_one_or_none()
        if not delivery:
            raise HTTPException(404, "Delivery not found")
        await _require_delivery_access(session, delivery, user)
        if delivery.status != "DELIVERED":
            raise HTTPException(409, "Proof can be submitted only after delivery")

        delivery.proof_url = str(body.proof_url)
        delivery.proof_type = body.proof_type
        await session.commit()

        return {"delivery_id": delivery_id, "proof_url": str(body.proof_url), "proof_type": body.proof_type}


# ── Driver GPS ping ───────────────────────────────────────────────────────────

@router.post("/{delivery_id}/location")
async def update_location(delivery_id: int, body: LocationUpdateRequest, user=Depends(require_driver)):
    """Driver GPS ping — update delivery location."""
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        del_result = await session.execute(select(Delivery).where(Delivery.id == delivery_id, Delivery.restaurant_id == rid))
        delivery = del_result.scalar_one_or_none()
        if not delivery:
            raise HTTPException(404, "Delivery not found")
        await _require_delivery_access(session, delivery, user)
        if delivery.status not in {"ASSIGNED", "PICKED_UP", "EN_ROUTE"}:
            raise HTTPException(409, "Location updates are allowed only for an active delivery")
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
        evt = delivery_location_updated(delivery_id, body.lat, body.lng, rid)
        await ws_manager.broadcast(rid, evt.to_dict())

        return {"ok": True}


# ── Delivery tracking ─────────────────────────────────────────────────────────

@router.get("/{delivery_id}/track")
async def track_delivery(delivery_id: int, user=Depends(extract_token)):
    """Get delivery tracking info with order details, driver info, and last known location."""
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        del_result = await session.execute(select(Delivery).where(Delivery.id == delivery_id, Delivery.restaurant_id == rid))
        delivery = del_result.scalar_one_or_none()
        if not delivery:
            raise HTTPException(404, "Delivery not found")

        # Get latest location
        loc_result = await session.execute(
            select(DeliveryLocationEvent)
            .where(DeliveryLocationEvent.delivery_id == delivery_id, DeliveryLocationEvent.restaurant_id == rid)
            .order_by(DeliveryLocationEvent.recorded_at.desc())
            .limit(1)
        )
        last_loc = loc_result.scalar_one_or_none()

        # Get recent locations (last 20)
        locs_result = await session.execute(
            select(DeliveryLocationEvent)
            .where(DeliveryLocationEvent.delivery_id == delivery_id, DeliveryLocationEvent.restaurant_id == rid)
            .order_by(DeliveryLocationEvent.recorded_at.desc())
            .limit(20)
        )
        recent_locs = locs_result.scalars().all()

        # Driver info
        driver_info = None
        if delivery.driver_id:
            drv = (await session.execute(select(Driver).where(Driver.id == delivery.driver_id, Driver.restaurant_id == rid))).scalar_one_or_none()
            if drv and drv.user_id:
                usr = (await session.execute(select(User).where(User.id == drv.user_id, User.restaurant_id == rid))).scalar_one_or_none()
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
            order = (await session.execute(select(Order).where(Order.id == delivery.order_id, Order.restaurant_id == rid))).scalar_one_or_none()
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

@router.get("/public/{tracking_token}")
async def track_delivery_public(tracking_token: str):
    """Customer-facing minimal status view, authorized by an unguessable token."""
    if len(tracking_token) < 32 or len(tracking_token) > 128:
        raise HTTPException(404, "Order not found")
    async with AsyncSessionLocal() as session:
        order = (await session.execute(
            select(Order).where(Order.tracking_token == tracking_token)
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
                "provider": order.provider,
                "source_channel": order.source_channel,
                "order_type": order.order_type,
                "scheduled_for": order.scheduled_for.isoformat() if order.scheduled_for else None,
                "delivery": None,
                "message": (
                    "Your meal is scheduled and will enter preparation near its selected time."
                    if order.status == "SCHEDULED"
                    else "Your order is in the live preparation queue."
                ),
            }

        return {
            "order_ref": order.order_ref,
            "order_status": order.status,
            "provider": order.provider,
            "source_channel": order.source_channel,
            "order_type": order.order_type,
            "scheduled_for": order.scheduled_for.isoformat() if order.scheduled_for else None,
            "delivery": {
                "status": delivery.status,
                "assigned": delivery.driver_id is not None,
                "picked_up_at": delivery.picked_up_at.isoformat() if delivery.picked_up_at else None,
                "delivered_at": delivery.delivered_at.isoformat() if delivery.delivered_at else None,
            },
        }


async def _require_delivery_access(session, delivery: Delivery, user: dict) -> None:
    """Managers may operate tenant deliveries; drivers only their own assignment."""
    role = str(user.get("role") or "").upper()
    if role in {"ADMIN", "MANAGER"}:
        return
    if role != "DRIVER":
        raise HTTPException(403, "Delivery operator role required")
    driver = (await session.execute(
        select(Driver).where(
            Driver.user_id == int(user.get("user_id") or 0),
            Driver.restaurant_id == delivery.restaurant_id,
        )
    )).scalar_one_or_none()
    if not driver or delivery.driver_id != driver.id:
        raise HTTPException(403, "This delivery is not assigned to you")
