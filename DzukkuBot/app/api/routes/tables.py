"""
Table / dine-in routes — sessions, orders, fire-to-kitchen, invoice.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.db.models import DiningTable, TableSession, TableSessionOrder, Order, OrderItem
from app.auth.deps import require_waiter, require_manager

router = APIRouter(prefix="/api/v1/tables", tags=["tables"])


class OpenSessionRequest(BaseModel):
    table_id: int
    waiter_user_id: Optional[int] = None
    guests: int


class AddTableOrderRequest(BaseModel):
    items: list[dict]  # [{"item_id": int, "qty": int, "modifiers_json": ...}]


@router.post("/sessions", status_code=201)
async def open_table_session(body: OpenSessionRequest, user=Depends(require_waiter)):
    async with AsyncSessionLocal() as session:
        ts = TableSession(
            restaurant_id=1,
            table_id=body.table_id,
            waiter_user_id=body.waiter_user_id,
            guests=body.guests,
            status="OPEN",
        )
        session.add(ts)
        await session.commit()
        await session.refresh(ts)

        # Publish event
        from app.realtime.events import table_session_opened
        from app.realtime.ws_manager import ws_manager
        event = table_session_opened(ts.id, body.table_id)
        await ws_manager.broadcast(1, event.to_dict())

        return {"session_id": ts.id, "table_id": body.table_id, "status": "OPEN"}


@router.post("/sessions/{session_id}/orders", status_code=201)
async def add_table_order(session_id: int, body: AddTableOrderRequest, user=Depends(require_waiter)):
    import uuid
    async with AsyncSessionLocal() as session:
        # Verify session exists and is open
        ts_result = await session.execute(select(TableSession).where(TableSession.id == session_id))
        ts = ts_result.scalar_one_or_none()
        if not ts or ts.status != "OPEN":
            raise HTTPException(400, "Table session not found or not open")

        # Create order
        order_ref = f"DZK-{uuid.uuid4().hex[:6].upper()}"
        total = 0
        order = Order(
            restaurant_id=1,
            order_ref=order_ref,
            order_type="DINE_IN",
            status="CREATED",
            subtotal_cents=0,
            total_cents=0,
        )
        session.add(order)
        await session.flush()

        for item_data in body.items:
            item_id = item_data.get("item_id")
            qty = item_data.get("qty", 1)
            # Look up price
            mi_result = await session.execute(select(MenuItem).where(MenuItem.id == item_id))
            from app.db.models import MenuItem
            mi = mi_result.scalar_one_or_none()
            if mi:
                unit_cents = mi.price_cents
                total += unit_cents * qty
                oi = OrderItem(
                    restaurant_id=1,
                    order_id=order.id,
                    item_id=item_id,
                    item_name_snapshot=mi.name,
                    qty=qty,
                    unit_price_cents=unit_cents,
                    modifiers_json=item_data.get("modifiers_json"),
                )
                session.add(oi)

        order.subtotal_cents = total
        order.total_cents = total

        # Link order to table session
        tso = TableSessionOrder(
            restaurant_id=1,
            table_session_id=session_id,
            order_id=order.id,
        )
        session.add(tso)
        await session.commit()

        return {"order_id": order.id, "order_ref": order_ref, "total_cents": total}


@router.post("/sessions/{session_id}/fire")
async def fire_to_kitchen(session_id: int, user=Depends(require_waiter)):
    """Mark all CREATED orders in this session as ACCEPTED (sent to kitchen)."""
    async with AsyncSessionLocal() as session:
        tso_result = await session.execute(
            select(TableSessionOrder).where(TableSessionOrder.table_session_id == session_id)
        )
        links = tso_result.scalars().all()

        fired = []
        for link in links:
            order_result = await session.execute(select(Order).where(Order.id == link.order_id))
            order = order_result.scalar_one_or_none()
            if order and order.status == "CREATED":
                order.status = "ACCEPTED"
                fired.append(order.order_ref)

        await session.commit()
        return {"fired_orders": fired}


@router.post("/sessions/{session_id}/invoice")
async def generate_invoice(session_id: int, user=Depends(require_waiter)):
    """Generate invoice for a table session and close it."""
    import uuid
    async with AsyncSessionLocal() as session:
        ts_result = await session.execute(select(TableSession).where(TableSession.id == session_id))
        ts = ts_result.scalar_one_or_none()
        if not ts:
            raise HTTPException(404, "Session not found")

        # Sum all orders
        tso_result = await session.execute(
            select(TableSessionOrder).where(TableSessionOrder.table_session_id == session_id)
        )
        links = tso_result.scalars().all()

        subtotal = 0
        for link in links:
            order_result = await session.execute(select(Order).where(Order.id == link.order_id))
            order = order_result.scalar_one_or_none()
            if order:
                subtotal += order.total_cents

        from app.db.models import Invoice
        invoice = Invoice(
            restaurant_id=1,
            invoice_no=f"INV-{uuid.uuid4().hex[:6].upper()}",
            entity_type="TABLE_SESSION",
            entity_id=session_id,
            subtotal_cents=subtotal,
            tax_cents=0,
            total_cents=subtotal,
        )
        session.add(invoice)

        # Close session
        from datetime import datetime, timezone
        ts.status = "CLOSED"
        ts.closed_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(invoice)

        # Publish event
        from app.realtime.events import table_session_closed
        from app.realtime.ws_manager import ws_manager
        event = table_session_closed(session_id, ts.table_id)
        await ws_manager.broadcast(1, event.to_dict())

        return {
            "invoice_no": invoice.invoice_no,
            "total_cents": subtotal,
            "session_status": "CLOSED",
        }
