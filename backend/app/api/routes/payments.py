"""
Payment routes — Razorpay intents and webhooks.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.db.session import AsyncSessionLocal
from app.db.models import Payment
from app.payments.razorpay import create_payment_order, verify_webhook_signature
from app.auth.deps import require_staff, tenant_id

router = APIRouter(prefix="/api/v1/payments", tags=["payments"])


@router.post("/intents", status_code=201)
async def create_intent(order_id: int, user=Depends(require_staff)):
    """Create a Razorpay payment order (intent) for the given order."""
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        from app.db.models import Order
        result = await session.execute(select(Order).where(Order.id == order_id, Order.restaurant_id == rid))
        order = result.scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")
        existing = (await session.execute(
            select(Payment).where(
                Payment.order_id == order_id,
                Payment.restaurant_id == rid,
                Payment.status.in_(["CREATED", "AUTHORIZED", "CAPTURED"]),
            ).order_by(Payment.id.desc())
        )).scalars().first()
        if existing:
            return {
                "payment_id": existing.id,
                "razorpay_order_id": existing.provider_order_id,
                "amount_cents": existing.amount_cents,
                "currency": existing.currency,
                "key_id": "",
                "idempotent_replay": True,
            }

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
            restaurant_id=rid,
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
        from app.db.models import Order
        result = await session.execute(
            select(Payment).where(Payment.provider_order_id == provider_order_id)
        )
        payment = result.scalar_one_or_none()
        if payment:
            payment.status = new_status
            payment.provider_payment_id = provider_payment_id
            payment.provider_signature = payment_data.get("signature")

            # Payment never activates a future meal. Only the scheduler may
            # move SCHEDULED orders into the live kitchen queue.
            order_id = payment.order_id
            restaurant_id = payment.restaurant_id
            order_advanced = False
            if new_status == "CAPTURED":
                order_result = await session.execute(
                    select(Order).where(Order.id == order_id)
                )
                order = order_result.scalar_one_or_none()
                if order and order.status in ("CREATED",):
                    order.status = "ACCEPTED"
                    order_advanced = True

            await session.commit()

            # Publish payment event
            from app.realtime.events import payment_status_changed, order_status_changed
            from app.realtime.ws_manager import ws_manager
            evt = payment_status_changed(payment.id, payment.order_id, new_status, restaurant_id)
            await ws_manager.broadcast(restaurant_id, evt.to_dict())

            # Publish order status event if auto-advanced
            if new_status == "CAPTURED":
                order_result2 = await session.execute(
                    select(Order).where(Order.id == order_id)
                )
                order = order_result2.scalar_one_or_none()
                if order:
                    if order_advanced:
                        evt2 = order_status_changed(order_id, order.order_ref, order.status, restaurant_id)
                        await ws_manager.broadcast(restaurant_id, evt2.to_dict())

                        from app.realtime.events import order_sent_to_kitchen
                        evt3 = order_sent_to_kitchen(order_id, order.order_ref, restaurant_id)
                        await ws_manager.broadcast(restaurant_id, evt3.to_dict())

                    # Notify customer via Telegram
                    try:
                        from app.realtime.notifications import notify_customer
                        await notify_customer(
                            restaurant_id=restaurant_id,
                            order_ref=order.order_ref,
                            event_type="order.status_changed",
                            message=(
                                f"Payment confirmed for {order.order_ref}. Your meal remains scheduled "
                                f"for {order.scheduled_for.isoformat()}."
                                if order.status == "SCHEDULED" and order.scheduled_for
                                else f"Payment confirmed for order {order.order_ref}! Your order is now {order.status}."
                            ),
                        )
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning("Customer notification failed: %s", e)

    return {"ok": True}
