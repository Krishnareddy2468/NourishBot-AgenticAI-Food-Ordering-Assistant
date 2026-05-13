"""
Agent v2 operational policies for Dzukku Restaurant.

Enforces business rules that the LLM cannot violate:
  - Delivery radius limits
  - Prep time estimates based on kitchen load
  - Cancellation rules
  - Stock availability
  - Order amount limits
  - Operating hours
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from typing import Optional


@dataclass
class RestaurantPolicy:
    """Operational constraints for the restaurant agent."""

    # Delivery
    max_delivery_radius_km: float = 10.0
    delivery_available: bool = True
    min_order_for_delivery_cents: int = 20000  # ₹200

    # Pickup
    pickup_available: bool = True

    # Dine-in
    dine_in_available: bool = True
    max_guests_per_reservation: int = 20

    # Operating hours
    opens_at: time = time(6, 0)
    closes_at: time = time(23, 0)
    timezone: str = "Asia/Kolkata"

    # Kitchen
    max_concurrent_orders: int = 50
    base_prep_time_sec: int = 900  # 15 min
    busy_multiplier: float = 1.5

    # Cancellation
    can_cancel_before: str = "PREPARING"  # Status after which cancellation is denied

    # Ordering
    max_items_per_order: int = 30
    max_qty_per_item: int = 10

    # Upsell
    max_upsells_per_session: int = 1

    # Payment
    supported_payment_providers: list[str] = field(default_factory=lambda: ["RAZORPAY"])

    def is_within_operating_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if the restaurant is currently open."""
        import pytz
        tz = pytz.timezone(self.timezone)
        now = dt or datetime.now(tz)
        current_time = now.time()
        return self.opens_at <= current_time <= self.closes_at

    def can_cancel_order(self, current_status: str) -> bool:
        """Check if an order can be cancelled given its current status."""
        status_order = ["CREATED", "ACCEPTED", "PREPARING", "READY", "OUT_FOR_DELIVERY", "DELIVERED"]
        try:
            idx = status_order.index(current_status)
            cancel_idx = status_order.index(self.can_cancel_before)
            return idx < cancel_idx
        except ValueError:
            return False

    def estimate_prep_time(self, current_kitchen_load: int = 0) -> int:
        """Estimate prep time in seconds based on kitchen load."""
        if current_kitchen_load >= self.max_concurrent_orders:
            return -1  # Kitchen at capacity, cannot accept
        multiplier = self.busy_multiplier if current_kitchen_load > self.max_concurrent_orders // 2 else 1.0
        return int(self.base_prep_time_sec * multiplier)

    def validate_order_type(self, order_type: str) -> tuple[bool, str]:
        """Check if the requested order type is available."""
        if order_type == "DELIVERY" and not self.delivery_available:
            return False, "Delivery is currently unavailable. Would you like pickup or dine-in instead?"
        if order_type == "PICKUP" and not self.pickup_available:
            return False, "Pickup is currently unavailable."
        if order_type == "DINE_IN" and not self.dine_in_available:
            return False, "Dine-in is currently unavailable. Would you like delivery or pickup?"
        return True, ""


# Default policy instance
default_policy = RestaurantPolicy()
