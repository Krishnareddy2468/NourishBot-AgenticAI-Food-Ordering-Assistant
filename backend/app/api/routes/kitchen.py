"""
Kitchen routes — KDS order listing and item-level status updates.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Order, OrderItem
from app.auth.deps import extract_token

router = APIRouter(prefix="/api/v1/kitchen", tags=["kitchen"])


@router.get("/orders")
async def get_kitchen_orders(user=Depends(extract_token)):
    """Return all active orders for the KDS (ACCEPTED or PREPARING status)."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Order)
            .where(Order.status.in_(["ACCEPTED", "PREPARING", "CREATED"]))
            .order_by(Order.created_at)
        )
        orders = result.scalars().all()

        out = []
        for order in orders:
            items_result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = items_result.scalars().all()

            if order.customer_id:
                from app.db.models import Customer
                cust_result = await session.execute(
                    select(Customer).where(Customer.id == order.customer_id)
                )
                customer = cust_result.scalar_one_or_none()
                customer_name = customer.name if customer else "Guest"
            else:
                customer_name = "Guest"

            out.append({
                "id": order.id,
                "order_ref": order.order_ref,
                "orderRef": order.order_ref,
                "status": order.status,
                "order_type": order.order_type,
                "customer": customer_name,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "items": [
                    {
                        "id": i.id,
                        "name": i.item_name_snapshot,
                        "item_name_snapshot": i.item_name_snapshot,
                        "qty": i.qty,
                        "unit_price_cents": i.unit_price_cents,
                        "status": i.status or "PENDING",
                        "modifiers_json": i.modifiers_json,
                        "category": "",
                    }
                    for i in items
                ],
            })
        return out
