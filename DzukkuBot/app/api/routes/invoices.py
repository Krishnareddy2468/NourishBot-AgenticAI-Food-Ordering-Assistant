"""
Invoice routes — list invoices for settlement and billing.
"""

from fastapi import APIRouter, Depends
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models.invoices import Invoice
from app.auth.deps import extract_token

router = APIRouter(prefix="/api/v1/invoices", tags=["invoices"])


@router.get("")
async def list_invoices(limit: int = 100, user=Depends(extract_token)):
    rid = user.get("restaurant_id", 1)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Invoice)
            .where(Invoice.restaurant_id == rid)
            .order_by(Invoice.id.desc())
            .limit(limit)
        )
        invoices = result.scalars().all()
        return [
            {
                "id": inv.id,
                "invoice_no": inv.invoice_no,
                "entity_type": inv.entity_type,
                "entity_id": inv.entity_id,
                "subtotal_cents": inv.subtotal_cents,
                "tax_cents": inv.tax_cents,
                "total_cents": inv.total_cents,
                "created_at": inv.created_at.isoformat() if inv.created_at else None,
            }
            for inv in invoices
        ]
