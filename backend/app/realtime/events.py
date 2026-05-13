"""
Real-time event publishing for Dzukku vNext.

Publishes events to the WebSocket/SSE manager so connected clients
(KDS, waiter view, admin dashboard) receive live updates.
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class Event:
    """Structured real-time event."""

    __slots__ = ("event_type", "payload", "restaurant_id", "timestamp")

    def __init__(
        self,
        event_type: str,
        payload: dict[str, Any],
        restaurant_id: int = 1,
        timestamp: datetime | None = None,
    ):
        self.event_type = event_type
        self.payload = payload
        self.restaurant_id = restaurant_id
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "restaurant_id": self.restaurant_id,
            "timestamp": self.timestamp.isoformat(),
        }


# ── Convenience constructors for domain events ───────────────────────────────

def order_created(order_id: int, order_ref: str, restaurant_id: int = 1) -> Event:
    return Event("order.created", {"order_id": order_id, "order_ref": order_ref}, restaurant_id)


def order_status_changed(order_id: int, order_ref: str, new_status: str, restaurant_id: int = 1) -> Event:
    return Event("order.status_changed", {"order_id": order_id, "order_ref": order_ref, "new_status": new_status}, restaurant_id)


def order_item_status_changed(order_id: int, item_id: int, new_status: str, restaurant_id: int = 1) -> Event:
    return Event("order.item_status_updated", {"order_id": order_id, "item_id": item_id, "new_status": new_status}, restaurant_id)


def delivery_status_changed(delivery_id: int, order_id: int, new_status: str, restaurant_id: int = 1) -> Event:
    return Event("delivery.status_changed", {"delivery_id": delivery_id, "order_id": order_id, "new_status": new_status}, restaurant_id)


def delivery_location_updated(delivery_id: int, lat: float, lng: float, restaurant_id: int = 1) -> Event:
    return Event("delivery.location_updated", {"delivery_id": delivery_id, "lat": lat, "lng": lng}, restaurant_id)


def table_session_opened(session_id: int, table_id: int, restaurant_id: int = 1) -> Event:
    return Event("table_session.opened", {"session_id": session_id, "table_id": table_id}, restaurant_id)


def table_session_closed(session_id: int, table_id: int, restaurant_id: int = 1) -> Event:
    return Event("table_session.closed", {"session_id": session_id, "table_id": table_id}, restaurant_id)


def payment_status_changed(payment_id: int, order_id: int, new_status: str, restaurant_id: int = 1) -> Event:
    return Event("payment.status_changed", {"payment_id": payment_id, "order_id": order_id, "new_status": new_status}, restaurant_id)


def menu_item_availability_changed(item_id: int, available: bool, restaurant_id: int = 1) -> Event:
    return Event("menu_item.availability_changed", {"item_id": item_id, "available": available}, restaurant_id)


def order_sent_to_kitchen(order_id: int, order_ref: str, restaurant_id: int = 1) -> Event:
    return Event("order.sent_to_kitchen", {"order_id": order_id, "order_ref": order_ref}, restaurant_id)
