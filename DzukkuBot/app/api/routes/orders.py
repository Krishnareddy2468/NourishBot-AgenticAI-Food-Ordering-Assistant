"""
Order routes — listing, state transitions, item status updates.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Order, OrderItem
from app.auth.deps import require_manager, require_kitchen, extract_token

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])


def _serialize_order(order: Order, items: list) -> dict:
    return {
        "id": order.id,
        "order_ref": order.order_ref,
        "order_state": order.status,
        "status": order.status,
        "order_type": order.order_type,
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
    async with AsyncSessionLocal() as session:
        query = select(Order).order_by(Order.id.desc()).limit(limit)
        if status:
            query = query.where(Order.status == status)
        result = await session.execute(query)
        orders = result.scalars().all()

        out = []
        for order in orders:
            items_result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = items_result.scalars().all()
            # Eagerly load customer
            if order.customer_id:
                from app.db.models import Customer
                cust_result = await session.execute(
                    select(Customer).where(Customer.id == order.customer_id)
                )
                order.customer = cust_result.scalar_one_or_none()
            else:
                order.customer = None
            out.append(_serialize_order(order, items))
        return out


@router.get("/{order_id}")
async def get_order(order_id: int, user=Depends(extract_token)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")
        items_result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id)
        )
        items = items_result.scalars().all()
        if order.customer_id:
            from app.db.models import Customer
            cust_result = await session.execute(
                select(Customer).where(Customer.id == order.customer_id)
            )
            order.customer = cust_result.scalar_one_or_none()
        else:
            order.customer = None
        return _serialize_order(order, items)


class OrderStateUpdate(BaseModel):
    order_state: str


@router.patch("/{order_id}/state")
async def update_order_state(order_id: int, body: OrderStateUpdate, user=Depends(extract_token)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")

        order.status = body.order_state
        await session.commit()

        from app.realtime.events import order_status_changed
        from app.realtime.ws_manager import ws_manager
        evt = order_status_changed(order_id, order.order_ref, body.order_state)
        await ws_manager.broadcast(order.restaurant_id, evt.to_dict())

        return {"ok": True, "order_id": order_id, "status": body.order_state}


class ItemStatusUpdate(BaseModel):
    status: str


@router.patch("/{order_id}/items/{item_id}/status")
async def update_order_item_status(
    order_id: int,
    item_id: int,
    body: ItemStatusUpdate,
    user=Depends(extract_token),
):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(OrderItem).where(
                OrderItem.order_id == order_id,
                OrderItem.id == item_id,
            )
        )
        item = result.scalar_one_or_none()
        if not item:
            raise HTTPException(404, "Order item not found")

        item.status = body.status
        await session.commit()

        from app.realtime.events import order_item_status_changed
        from app.realtime.ws_manager import ws_manager
        event = order_item_status_changed(order_id, item_id, body.status)
        await ws_manager.broadcast(1, event.to_dict())

        return {"ok": True, "order_id": order_id, "item_id": item_id, "status": body.status}
