"""
Customer notification module — sends Telegram messages on order/delivery events.

Called from:
  - Payment webhook (payment.captured → order ACCEPTED)
  - Delivery status changes (ASSIGNED, PICKED_UP, EN_ROUTE, DELIVERED)
  - Order status changes (PREPARING, READY, OUT_FOR_DELIVERY)
"""

import logging
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Order, Customer, Channel

logger = logging.getLogger(__name__)

# Status → customer-friendly message templates
_STATUS_MESSAGES = {
    "ACCEPTED": "Your order {order_ref} has been accepted! The kitchen is preparing your food.",
    "PREPARING": "Your order {order_ref} is being prepared. Almost there!",
    "READY": "Your order {order_ref} is ready! {delivery_hint}",
    "OUT_FOR_DELIVERY": "Your order {order_ref} is out for delivery! Track it live.",
    "DELIVERED": "Your order {order_ref} has been delivered. Enjoy your meal! ❤️",
    "CANCELLED": "Your order {order_ref} has been cancelled. Please contact us if this was unexpected.",
}

_DELIVERY_HINTS = {
    "DELIVERY": "A driver will pick it up shortly.",
    "PICKUP": "You can pick it up at the restaurant now!",
    "DINE_IN": "Your server will bring it to you.",
}

_DELIVERY_STATUS_MESSAGES = {
    "ASSIGNED": "A driver has been assigned to your order {order_ref}. They'll pick up your food shortly.",
    "PICKED_UP": "Your order {order_ref} has been picked up by the driver! On the way.",
    "EN_ROUTE": "Your delivery is nearby! Get ready to receive your order {order_ref}.",
    "DELIVERED": "Your order {order_ref} has been delivered. Enjoy! ❤️",
    "FAILED": "There was an issue delivering your order {order_ref}. Our team will contact you shortly.",
}


async def notify_customer(
    restaurant_id: int,
    order_ref: str,
    event_type: str,
    message: Optional[str] = None,
) -> bool:
    """
    Send a Telegram notification to the customer associated with an order.
    Returns True if notification was sent successfully.
    """
    async with AsyncSessionLocal() as db:
        # Find order → customer → channel → chat_id
        order = (await db.execute(
            select(Order).where(Order.order_ref == order_ref)
        )).scalar_one_or_none()
        if not order or not order.customer_id:
            logger.warning("notify_customer: order %s not found or no customer", order_ref)
            return False

        customer = (await db.execute(
            select(Customer).where(Customer.id == order.customer_id)
        )).scalar_one_or_none()
        if not customer:
            return False

        # Find the Telegram channel for this customer
        channel = (await db.execute(
            select(Channel).where(
                Channel.customer_id == customer.id,
                Channel.type == "TELEGRAM",
            )
        )).scalar_one_or_none()
        if not channel:
            logger.warning("notify_customer: no Telegram channel for customer %s", customer.id)
            return False

        chat_id = int(channel.external_id)

    # Build message if not provided
    if not message:
        message = _build_message(order_ref, event_type, order.order_type if order else "DELIVERY")

    # Send via Telegram bot
    return await _send_telegram_message(chat_id, message)


def _build_message(order_ref: str, event_type: str, order_type: str = "DELIVERY") -> str:
    """Build a customer-friendly notification message."""
    if event_type == "delivery.status_changed":
        # Extract delivery status from event context — use generic message
        return f"Update on your order {order_ref}! Check status with /cart or ask me."

    if event_type == "order.status_changed":
        # We'd need the actual status — use a generic update message
        return f"Your order {order_ref} has been updated! Ask me for the latest status."

    return f"Update on your order {order_ref}!"


async def notify_order_status(
    order_ref: str,
    new_status: str,
    order_type: str = "DELIVERY",
) -> bool:
    """Convenience: notify customer about an order status change."""
    template = _STATUS_MESSAGES.get(new_status)
    if not template:
        return False

    delivery_hint = _DELIVERY_HINTS.get(order_type, "")
    message = template.format(order_ref=order_ref, delivery_hint=delivery_hint)

    async with AsyncSessionLocal() as db:
        order = (await db.execute(
            select(Order).where(Order.order_ref == order_ref)
        )).scalar_one_or_none()
        if not order:
            return False

    return await notify_customer(
        restaurant_id=order.restaurant_id,
        order_ref=order_ref,
        event_type="order.status_changed",
        message=message,
    )


async def notify_delivery_status(
    delivery_id: int,
    order_ref: str,
    new_status: str,
    restaurant_id: int = 1,
) -> bool:
    """Convenience: notify customer about a delivery status change."""
    template = _DELIVERY_STATUS_MESSAGES.get(new_status)
    if not template:
        return False

    message = template.format(order_ref=order_ref)
    return await notify_customer(
        restaurant_id=restaurant_id,
        order_ref=order_ref,
        event_type="delivery.status_changed",
        message=message,
    )


async def _send_telegram_message(chat_id: int, message: str) -> bool:
    """Send a message via the Telegram bot instance."""
    try:
        from app.bot.telegram import get_bot_instance
        bot = get_bot_instance()
        if not bot:
            logger.warning("Telegram bot instance not available for notification")
            return False
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown",
        )
        return True
    except Exception as e:
        logger.error("Failed to send Telegram notification to chat %s: %s", chat_id, e)
        return False
