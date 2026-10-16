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

from app.db.session import AsyncSessionLocal, get_session_factory
from app.db.models import Order, Customer, Channel
from app.core.config import settings

logger = logging.getLogger(__name__)

# Status → customer-friendly message templates
_STATUS_MESSAGES = {
    "CREATED": "Your scheduled meal {order_ref} is now in the live preparation queue.",
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
    "DRIVE_THRU": "Your drive-through meal is ready at the collection lane!",
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
    async with get_session_factory()() as db:
        # Find order → customer → channel → chat_id
        order = (await db.execute(
            select(Order).where(
                Order.order_ref == order_ref,
                Order.restaurant_id == restaurant_id,
            )
        )).scalar_one_or_none()
        if not order or not order.customer_id:
            logger.warning("notify_customer: order %s not found or no customer", order_ref)
            return False

        customer = (await db.execute(
            select(Customer).where(Customer.id == order.customer_id)
        )).scalar_one_or_none()
        if not customer:
            return False

        # Prefer the channel that created the order, then fall back to any
        # supported customer channel.
        channel = (await db.execute(
            select(Channel).where(
                Channel.customer_id == customer.id,
                Channel.type == order.source_channel,
            )
        )).scalar_one_or_none()
        if not channel:
            channel = (await db.execute(
                select(Channel).where(
                    Channel.customer_id == customer.id,
                    Channel.type.in_(["TELEGRAM", "WHATSAPP"]),
                )
            )).scalars().first()
        if not channel:
            logger.warning("notify_customer: no supported channel for customer %s", customer.id)
            return False
        channel_type = channel.type
        external_id = channel.external_id

    # Build message if not provided
    if not message:
        message = _build_message(order_ref, event_type, order.order_type if order else "DELIVERY")

    if channel_type == "WHATSAPP":
        from app.api.routes.whatsapp import _send_text
        try:
            return await _send_text(str(external_id), message)
        except Exception as exc:
            logger.error("Failed to send WhatsApp notification to %s: %s", external_id, exc)
            return False
    return await _send_telegram_message(int(external_id), message)


async def notify_customer_id(
    *,
    restaurant_id: int,
    customer_id: int,
    message: str,
    preferred_channel: str | None = None,
) -> bool:
    """Deliver a planning message over the customer's Telegram or WhatsApp channel."""
    async with get_session_factory()() as db:
        query = select(Channel).where(
            Channel.restaurant_id == restaurant_id,
            Channel.customer_id == customer_id,
            Channel.type.in_(["TELEGRAM", "WHATSAPP"]),
        )
        channels = (await db.execute(query.order_by(Channel.id.desc()))).scalars().all()
    if preferred_channel:
        preferred = preferred_channel.upper()
        channels.sort(key=lambda channel: channel.type != preferred)
    for channel in channels:
        if channel.type == "WHATSAPP":
            from app.api.routes.whatsapp import _send_text
            try:
                if await _send_text(str(channel.external_id), message):
                    return True
            except Exception as exc:
                logger.error("Failed to send WhatsApp planning notification: %s", exc)
        else:
            try:
                if await _send_telegram_message(int(channel.external_id), message):
                    return True
            except (TypeError, ValueError):
                logger.error("Telegram planning channel has an invalid identifier")
    logger.warning("No planning notification channel delivered for customer %s", customer_id)
    return False


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
        if not bot and settings.TELEGRAM_TOKEN:
            from telegram import Bot
            bot = Bot(token=settings.TELEGRAM_TOKEN)
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
