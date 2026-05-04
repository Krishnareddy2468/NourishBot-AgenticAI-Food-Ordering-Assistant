"""
Table / dine-in routes — tables list, sessions, orders, fire-to-kitchen, invoice.
"""

import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import (
    DiningTable, TableSession, TableSessionOrder,
    Order, OrderItem, MenuItem, Invoice,
)
from app.auth.deps import require_waiter, extract_token

router = APIRouter(prefix="/api/v1/tables", tags=["tables"])


# ── Tables ────────────────────────────────────────────────────────────────────

@router.get("")
async def list_tables(user=Depends(extract_token)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(DiningTable).where(DiningTable.restaurant_id == 1).order_by(DiningTable.id)
        )
        tables = result.scalars().all()
        return [
            {
                "id": t.id,
                "table_number": t.name,
                "name": t.name,
                "capacity": t.capacity,
                "status": "AVAILABLE" if t.active else "INACTIVE",
                "active": t.active,
            }
            for t in tables
        ]


# ── Sessions ──────────────────────────────────────────────────────────────────

class OpenSessionRequest(BaseModel):
    table_id: int
    guests: int
    waiter_user_id: Optional[int] = None


@router.get("/sessions")
async def list_sessions(status: Optional[str] = None, user=Depends(extract_token)):
    async with AsyncSessionLocal() as session:
        query = select(TableSession).where(TableSession.restaurant_id == 1)
        if status:
            query = query.where(TableSession.status == status)
        result = await session.execute(query.order_by(TableSession.id.desc()))
        sessions = result.scalars().all()
        return [
            {
                "id": s.id,
                "table_id": s.table_id,
                "guests": s.guests,
                "status": s.status,
                "waiter_user_id": s.waiter_user_id,
                "opened_at": s.opened_at.isoformat() if s.opened_at else None,
                "closed_at": s.closed_at.isoformat() if s.closed_at else None,
            }
            for s in sessions
        ]


@router.post("/sessions", status_code=201)
async def open_table_session(body: OpenSessionRequest, user=Depends(extract_token)):
    async with AsyncSessionLocal() as session:
        ts = TableSession(
            restaurant_id=1,
            table_id=body.table_id,
            waiter_user_id=body.waiter_user_id,
            guests=body.guests,
            status="OPEN",
            opened_at=datetime.now(timezone.utc),
        )
        session.add(ts)
        await session.commit()
        await session.refresh(ts)

        from app.realtime.events import table_session_opened
        from app.realtime.ws_manager import ws_manager
        await ws_manager.broadcast(1, table_session_opened(ts.id, body.table_id).to_dict())

        return {
            "id": ts.id,
            "session_id": ts.id,
            "table_id": body.table_id,
            "guests": body.guests,
            "status": "OPEN",
        }


class SessionStatusUpdate(BaseModel):
    status: str


@router.patch("/sessions/{session_id}")
async def update_session(session_id: int, body: SessionStatusUpdate, user=Depends(extract_token)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TableSession).where(TableSession.id == session_id)
        )
        ts = result.scalar_one_or_none()
        if not ts:
            raise HTTPException(404, "Session not found")
        ts.status = body.status
        if body.status == "CLOSED":
            ts.closed_at = datetime.now(timezone.utc)
        await session.commit()
        return {"ok": True, "session_id": session_id, "status": body.status}


@router.post("/sessions/{session_id}/orders", status_code=201)
async def add_table_order(session_id: int, body: dict, user=Depends(extract_token)):
    items_data = body.get("items", [])
    async with AsyncSessionLocal() as session:
        ts_result = await session.execute(
            select(TableSession).where(TableSession.id == session_id)
        )
        ts = ts_result.scalar_one_or_none()
        if not ts or ts.status != "OPEN":
            raise HTTPException(400, "Table session not found or not open")

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

        for item_data in items_data:
            item_id = item_data.get("item_id")
            qty = item_data.get("qty", 1)
            mi_result = await session.execute(
                select(MenuItem).where(MenuItem.id == item_id)
            )
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
                )
                session.add(oi)

        order.subtotal_cents = total
        order.total_cents = total

        tso = TableSessionOrder(
            restaurant_id=1,
            table_session_id=session_id,
            order_id=order.id,
        )
        session.add(tso)
        await session.commit()
        return {"order_id": order.id, "order_ref": order_ref, "total_cents": total}


@router.post("/sessions/{session_id}/fire")
async def fire_to_kitchen(session_id: int, user=Depends(extract_token)):
    async with AsyncSessionLocal() as session:
        tso_result = await session.execute(
            select(TableSessionOrder).where(TableSessionOrder.table_session_id == session_id)
        )
        links = tso_result.scalars().all()
        fired = []
        for link in links:
            order_result = await session.execute(
                select(Order).where(Order.id == link.order_id)
            )
            order = order_result.scalar_one_or_none()
            if order and order.status == "CREATED":
                order.status = "ACCEPTED"
                fired.append(order.order_ref)
        await session.commit()
        return {"fired_orders": fired}


@router.post("/sessions/{session_id}/invoice")
async def generate_invoice(session_id: int, user=Depends(extract_token)):
    async with AsyncSessionLocal() as session:
        ts_result = await session.execute(
            select(TableSession).where(TableSession.id == session_id)
        )
        ts = ts_result.scalar_one_or_none()
        if not ts:
            raise HTTPException(404, "Session not found")

        tso_result = await session.execute(
            select(TableSessionOrder).where(TableSessionOrder.table_session_id == session_id)
        )
        links = tso_result.scalars().all()
        subtotal = 0
        for link in links:
            order_result = await session.execute(
                select(Order).where(Order.id == link.order_id)
            )
            order = order_result.scalar_one_or_none()
            if order:
                subtotal += order.total_cents

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
        ts.status = "CLOSED"
        ts.closed_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(invoice)

        from app.realtime.events import table_session_closed
        from app.realtime.ws_manager import ws_manager
        await ws_manager.broadcast(1, table_session_closed(session_id, ts.table_id).to_dict())

        return {
            "invoice_no": invoice.invoice_no,
            "total_cents": subtotal,
            "session_status": "CLOSED",
        }
