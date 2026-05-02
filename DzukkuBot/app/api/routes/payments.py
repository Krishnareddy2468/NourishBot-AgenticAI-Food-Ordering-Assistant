"""
Payment routes — Razorpay intents and webhooks.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.db.session import AsyncSessionLocal
from app.db.models import Payment
from app.payments.razorpay import create_payment_order, verify_webhook_signature
from app.auth.deps import extract_token

router = APIRouter(prefix="/api/v1/payments", tags=["payments"])


@router.post("/intents", status_code=201)
async def create_intent(order_id: int, user=Depends(extract_token)):
    """Create a Razorpay payment order (intent) for the given order."""
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        from app.db.models import Order
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")

    try:
        rzp_order = await create_payment_order(
            amount_cents=order.total_cents,
            order_ref=order.order_ref,
        )
    except Exception as e:
        raise HTTPException(500, f"Razorpay error: {e}")

    # Save payment record
    async with AsyncSessionLocal() as session:
        payment = Payment(
            restaurant_id=order.restaurant_id,
            order_id=order.id,
            provider="RAZORPAY",
            status="CREATED",
            amount_cents=order.total_cents,
            currency="INR",
            provider_order_id=rzp_order.get("id"),
        )
        session.add(payment)
        await session.commit()
        await session.refresh(payment)

        return {
            "payment_id": payment.id,
            "razorpay_order_id": rzp_order.get("id"),
            "amount_cents": order.total_cents,
            "currency": "INR",
            "key_id": rzp_order.get("key_id", ""),
        }


@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    """Handle Razorpay webhook events (payment.captured, payment.failed, etc.)."""
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not verify_webhook_signature(body, signature):
        raise HTTPException(400, "Invalid webhook signature")

    payload = await request.json()
    event = payload.get("event", "")
    payment_data = payload.get("payload", {}).get("payment", {}).get("entity", {})

    provider_payment_id = payment_data.get("id")
    provider_order_id = payment_data.get("order_id")
    new_status = "CAPTURED" if event == "payment.captured" else "FAILED"

    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(Payment).where(Payment.provider_order_id == provider_order_id)
        )
        payment = result.scalar_one_or_none()
        if payment:
            payment.status = new_status
            payment.provider_payment_id = provider_payment_id
            payment.provider_signature = payment_data.get("signature")
            await session.commit()

            # Publish event
            from app.realtime.events import payment_status_changed
            from app.realtime.ws_manager import ws_manager
            evt = payment_status_changed(payment.id, payment.order_id, new_status)
            await ws_manager.broadcast(1, evt.to_dict())

    return {"ok": True}
