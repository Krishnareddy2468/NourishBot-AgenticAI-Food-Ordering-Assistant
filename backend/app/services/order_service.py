"""
Order Service - Handles order placement, tracking, and status updates
Simulates Zomato order placement and tracking API
"""
import uuid
import json
import os
import asyncio
import random
from typing import Optional, List, Dict
from datetime import datetime
from app.models.schemas import Order, OrderStatus, CartItem
DELIVERY_PARTNERS = [
    {"name": "Ramesh Kumar", "phone": "9876543210"},
    {"name": "Suresh Reddy", "phone": "9876543211"},
]
import logging

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "data")
ORDERS_FILE = os.path.join(DATA_DIR, "orders.json")


class OrderService:
    """Manages orders - placement, tracking, and status simulation"""

    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._status_callbacks: Dict[str, list] = {}
        self._active_simulations: set[str] = set()
        self._load_orders()

    def _load_orders(self):
        """Load orders from file"""
        try:
            if os.path.exists(ORDERS_FILE):
                with open(ORDERS_FILE, "r") as f:
                    data = json.load(f)
                    for order_data in data:
                        order = Order(**order_data)
                        self._orders[order.order_id] = order
                logger.info(f"Loaded {len(self._orders)} orders from file")
        except Exception as e:
            logger.error(f"Error loading orders: {e}")

    def _save_orders(self):
        """Save orders to file"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(ORDERS_FILE, "w") as f:
                orders_data = [order.model_dump() for order in self._orders.values()]
                json.dump(orders_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving orders: {e}")

    def place_order(
        self,
        user_id: str,
        restaurant_id: str,
        restaurant_name: str,
        items: List[CartItem],
        address: str,
        delivery_fee: int,
    ) -> Order:
        """
        Place a new order. Simulates Zomato order placement API.
        """
        # Generate order ID like Zomato
        order_id = f"ZOM{uuid.uuid4().hex[:10].upper()}"

        # Calculate totals
        subtotal = sum(item.price * item.quantity for item in items)
        tax = round(subtotal * 0.05)  # 5% GST
        total = subtotal + tax + delivery_fee

        # Assign delivery partner
        partner = random.choice(DELIVERY_PARTNERS)

        order = Order(
            order_id=order_id,
            user_id=user_id,
            restaurant_id=restaurant_id,
            restaurant_name=restaurant_name,
            items=items,
            subtotal=subtotal,
            tax=tax,
            delivery_fee=delivery_fee,
            total=total,
            address=address,
            status=OrderStatus.CONFIRMED,
            estimated_delivery="30-40 minutes",
            delivery_partner=partner["name"],
            delivery_partner_phone=partner["phone"],
            created_at=datetime.now().isoformat(),
        )

        self._orders[order_id] = order
        self._save_orders()
        logger.info(f"Order placed: {order_id} for user {user_id}")

        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self._orders.get(order_id)

    def get_user_orders(self, user_id: str) -> List[Order]:
        """Get all orders for a user"""
        return [
            order for order in self._orders.values()
            if order.user_id == user_id
        ]

    def get_latest_order(self, user_id: str) -> Optional[Order]:
        """Get the latest order for a user"""
        user_orders = self.get_user_orders(user_id)
        if not user_orders:
            return None
        return max(user_orders, key=lambda o: o.created_at)

    def update_status(self, order_id: str, status: OrderStatus) -> Optional[Order]:
        """Update order status"""
        order = self._orders.get(order_id)
        if order:
            order.status = status
            self._save_orders()
            logger.info(f"Order {order_id} status updated to: {status.value}")
            return order
        return None

    def register_status_callback(self, order_id: str, callback):
        """Register a callback for status updates"""
        if order_id not in self._status_callbacks:
            self._status_callbacks[order_id] = []
        self._status_callbacks[order_id].append(callback)

    async def simulate_order_progress(self, order_id: str, send_update_callback=None):
        """
        Simulate order progress through stages.
        This mimics real-time Zomato order tracking.
        """
        if order_id in self._active_simulations:
            logger.info("Order simulation already running for %s", order_id)
            return
        self._active_simulations.add(order_id)

        status_timeline = [
            (OrderStatus.CONFIRMED, "✅ Your order has been confirmed by the restaurant!", 5),
            (OrderStatus.PREPARING, "👨‍🍳 Your food is being prepared with love!", 15),
            (OrderStatus.OUT_FOR_DELIVERY, None, 20),  # Special message with partner name
            (OrderStatus.DELIVERED, "🎉 Your order has been delivered! Enjoy your meal! 🍽️", 25),
        ]

        try:
            order = self.get_order(order_id)
            if not order:
                return

            for status, message, delay in status_timeline:
                await asyncio.sleep(delay)

                order = self.update_status(order_id, status)
                if not order:
                    break

                # Special message for out_for_delivery
                if status == OrderStatus.OUT_FOR_DELIVERY:
                    message = (
                        f"🚴 Your order is out for delivery!\n"
                        f"🏍️ Delivery Partner: {order.delivery_partner}\n"
                        f"📞 Contact: {order.delivery_partner_phone}"
                    )

                if send_update_callback and message:
                    try:
                        await send_update_callback(order.user_id, message)
                    except Exception as e:
                        logger.error(f"Error sending status update: {e}")

                if status == OrderStatus.DELIVERED:
                    break
        finally:
            self._active_simulations.discard(order_id)

    def get_all_orders(self) -> List[Order]:
        """Get all orders (for admin/dashboard)"""
        return list(self._orders.values())

    def cancel_order(self, order_id: str) -> Optional[Order]:
        """Cancel an order"""
        order = self._orders.get(order_id)
        if order and order.status in [OrderStatus.PLACED, OrderStatus.CONFIRMED]:
            order.status = OrderStatus.CANCELLED
            self._save_orders()
            logger.info(f"Order {order_id} cancelled")
            return order
        return None


# Singleton instance
order_service = OrderService()
