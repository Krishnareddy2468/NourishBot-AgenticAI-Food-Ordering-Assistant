"""
Kitchen routes — KDS item-level status updates.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import OrderItem
from app.auth.deps import require_kitchen

router = APIRouter(prefix="/api/v1/orders", tags=["kitchen"])


class ItemStatusUpdate(BaseModel):
    status: str  # IN_PROGRESS | DONE | CANCELLED


@router.patch("/{order_id}/items/{item_id}/status")
async def update_order_item_status(
    order_id: int,
    item_id: int,
    body: ItemStatusUpdate,
    user=Depends(require_kitchen),
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

        # Publish real-time event
        from app.realtime.events import order_item_status_changed
        from app.realtime.ws_manager import ws_manager
        event = order_item_status_changed(order_id, item_id, body.status)
        await ws_manager.broadcast(1, event.to_dict())

        return {"ok": True, "order_id": order_id, "item_id": item_id, "status": body.status}
