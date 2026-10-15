"""
Order routes — listing, state transitions, item status updates.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import secrets
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Order, OrderItem
from app.auth.deps import require_role, require_kitchen, require_cashier, extract_token, tenant_id

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])
require_order_operator = require_role("ADMIN", "MANAGER", "KITCHEN", "WAITER")


def _serialize_order(order: Order, items: list) -> dict:
    return {
        "id": order.id,
        "order_ref": order.order_ref,
        "tracking_token": order.tracking_token,
        "order_state": order.status,
        "status": order.status,
        "order_type": order.order_type,
        "provider": order.provider,
        "source_channel": order.source_channel,
        "scheduled_for": order.scheduled_for.isoformat() if order.scheduled_for else None,
        "activated_at": order.activated_at.isoformat() if order.activated_at else None,
        "customer": order.customer.name if order.customer else "Guest",
        "phone": order.customer.phone if order.customer else "",
        "total_cents": order.total_cents,
        "amount": order.total_cents / 100,
        "created_at": order.created_at.isoformat() if order.created_at else None,
        "items": [
            {
                "id": i.id,
                "item_id": i.item_id,
                "name": i.item_name_snapshot,
                "item_name_snapshot": i.item_name_snapshot,
                "qty": i.qty,
                "unit_price_cents": i.unit_price_cents,
                "unit_price": i.unit_price_cents / 100,
                "status": i.status,
                "modifiers_json": i.modifiers_json,
            }
            for i in items
        ],
        "price_breakdown": {"grand_total": order.total_cents / 100},
    }


@router.get("")
async def list_orders(limit: int = 200, status: Optional[str] = None, user=Depends(extract_token)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        query = select(Order).where(Order.restaurant_id == rid).order_by(Order.id.desc()).limit(limit)
        if status:
            query = query.where(Order.status == status)
        result = await session.execute(query)
        orders = result.scalars().all()

        out = []
        for order in orders:
            items_result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id, OrderItem.restaurant_id == rid)
            )
            items = items_result.scalars().all()
            # Eagerly load customer
            if order.customer_id:
                from app.db.models import Customer
                cust_result = await session.execute(
                    select(Customer).where(Customer.id == order.customer_id, Customer.restaurant_id == rid)
                )
                order.customer = cust_result.scalar_one_or_none()
            else:
                order.customer = None
            out.append(_serialize_order(order, items))
        return out


@router.get("/{order_id}")
async def get_order(order_id: int, user=Depends(extract_token)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id, Order.restaurant_id == rid))
        order = result.scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")
        items_result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id, OrderItem.restaurant_id == rid)
        )
        items = items_result.scalars().all()
        if order.customer_id:
            from app.db.models import Customer
            cust_result = await session.execute(
                select(Customer).where(Customer.id == order.customer_id, Customer.restaurant_id == rid)
            )
            order.customer = cust_result.scalar_one_or_none()
        else:
            order.customer = None
        return _serialize_order(order, items)


class OrderStateUpdate(BaseModel):
    order_state: str


ALLOWED_STATE_TRANSITIONS = {
    "SCHEDULED": {"CANCELLED"},
    "CREATED": {"ACCEPTED", "CONFIRMED_BY_RESTAURANT", "PREPARING", "CANCELLED"},
    "PAID": {"ACCEPTED", "CONFIRMED_BY_RESTAURANT", "PREPARING", "CANCELLED"},
    "ACCEPTED": {"PREPARING", "CANCELLED"},
    "CONFIRMED_BY_RESTAURANT": {"PREPARING", "CANCELLED"},
    "PREPARING": {"READY", "CANCELLED"},
    "READY": {"OUT_FOR_DELIVERY", "COMPLETED", "DELIVERED"},
    "OUT_FOR_DELIVERY": {"DELIVERED", "CANCELLED"},
}


@router.patch("/{order_id}/state")
async def update_order_state(order_id: int, body: OrderStateUpdate, user=Depends(require_order_operator)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id, Order.restaurant_id == rid))
        order = result.scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")

        requested_state = body.order_state.strip().upper()
        allowed = ALLOWED_STATE_TRANSITIONS.get(order.status, set())
        if requested_state not in allowed:
            raise HTTPException(
                409,
                f"Cannot move order from {order.status} to {requested_state}",
            )

        # Capture before commit to avoid async expiry issues
        restaurant_id = order.restaurant_id
        order_ref = order.order_ref
        customer_id = order.customer_id

        order.status = requested_state
        await session.commit()
        await session.refresh(order)

        from app.realtime.events import order_status_changed
        from app.realtime.ws_manager import ws_manager
        evt = order_status_changed(order_id, order_ref, requested_state, rid)
        await ws_manager.broadcast(restaurant_id, evt.to_dict())

        # Notify customer on status change
        try:
            from app.realtime.notifications import notify_order_status
            await notify_order_status(order_ref, requested_state, order.order_type if hasattr(order, 'order_type') else "DELIVERY")
        except Exception:
            pass

        items_result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id, OrderItem.restaurant_id == rid)
        )
        items = items_result.scalars().all()
        if customer_id:
            from app.db.models import Customer
            cust_result = await session.execute(
                select(Customer).where(Customer.id == customer_id, Customer.restaurant_id == rid)
            )
            order.customer = cust_result.scalar_one_or_none()
        else:
            order.customer = None
        return _serialize_order(order, items)


@router.post("/{order_id}/mark-paid")
async def mark_order_paid(order_id: int, user=Depends(require_cashier)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id, Order.restaurant_id == rid))
        order = result.scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")

        restaurant_id = order.restaurant_id
        order_ref = order.order_ref
        customer_id = order.customer_id

        if order.status != "SCHEDULED":
            order.status = "PAID"
        await session.commit()
        await session.refresh(order)

        from app.realtime.events import order_status_changed
        from app.realtime.ws_manager import ws_manager
        evt = order_status_changed(order_id, order_ref, order.status, rid)
        await ws_manager.broadcast(restaurant_id, evt.to_dict())

        items_result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id, OrderItem.restaurant_id == rid)
        )
        items = items_result.scalars().all()
        if customer_id:
            from app.db.models import Customer
            cust_result = await session.execute(
                select(Customer).where(Customer.id == customer_id, Customer.restaurant_id == rid)
            )
            order.customer = cust_result.scalar_one_or_none()
        else:
            order.customer = None
        return _serialize_order(order, items)


@router.post("/{order_id}/tracking-token/rotate")
async def rotate_tracking_token(order_id: int, user=Depends(require_order_operator)):
    """Revoke the existing customer tracking link and issue a new token."""
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        order = (await session.execute(
            select(Order).where(Order.id == order_id, Order.restaurant_id == rid)
        )).scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")
        order.tracking_token = secrets.token_urlsafe(32)
        await session.commit()
        return {"order_id": order.id, "tracking_token": order.tracking_token}


class ItemStatusUpdate(BaseModel):
    status: str


@router.patch("/{order_id}/items/{item_id}/status")
async def update_order_item_status(
    order_id: int,
    item_id: int,
    body: ItemStatusUpdate,
    user=Depends(require_kitchen),
):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        parent = (await session.execute(
            select(Order).where(Order.id == order_id, Order.restaurant_id == rid)
        )).scalar_one_or_none()
        if not parent:
            raise HTTPException(404, "Order not found")
        result = await session.execute(
            select(OrderItem).where(
                OrderItem.order_id == order_id,
                OrderItem.id == item_id,
                OrderItem.restaurant_id == rid,
            )
        )
        item = result.scalar_one_or_none()
        if not item:
            raise HTTPException(404, "Order item not found")

        item.status = body.status
        await session.commit()

        from app.realtime.events import order_item_status_changed
        from app.realtime.ws_manager import ws_manager
        event = order_item_status_changed(order_id, item_id, body.status, rid)
        await ws_manager.broadcast(rid, event.to_dict())

        return {"ok": True, "order_id": order_id, "item_id": item_id, "status": body.status}
