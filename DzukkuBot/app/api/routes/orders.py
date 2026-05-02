"""
Order routes — order status updates, order listing.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Order
from app.auth.deps import require_manager, require_cashier

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])


class OrderStatusUpdate(BaseModel):
    status: str


@router.patch("/{order_id}/status")
async def update_order_status(order_id: int, body: OrderStatusUpdate, user=Depends(require_manager)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")

        order.status = body.status
        await session.commit()

        # Publish event
        from app.realtime.events import order_status_changed
        from app.realtime.ws_manager import ws_manager
        evt = order_status_changed(order_id, order.order_ref, body.status)
        await ws_manager.broadcast(1, evt.to_dict())

        return {"ok": True, "order_id": order_id, "status": body.status}
