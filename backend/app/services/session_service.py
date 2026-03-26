"""
Session Service - Manages user conversation sessions and state
Persists sessions to disk so they survive backend restarts.
"""
import os
import json
from typing import List
from app.models.schemas import UserSession, ConversationState, CartItem
import logging

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "data")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")


class SessionService:
    """Manages user sessions with conversation context"""

    def __init__(self):
        self._sessions: dict[str, UserSession] = {}
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from file on startup"""
        try:
            if os.path.exists(SESSIONS_FILE):
                with open(SESSIONS_FILE, "r") as f:
                    data = json.load(f)
                    for session_data in data:
                        try:
                            session = UserSession(**session_data)
                            self._sanitize_session(session)
                            self._sessions[session.user_id] = session
                        except Exception as e:
                            logger.warning("Skipping invalid session: %s", e)
                logger.info("Loaded %d sessions from file", len(self._sessions))
        except Exception as e:
            logger.error("Error loading sessions: %s", e)

    def _looks_like_invalid_location(self, location: str | None) -> bool:
        text = (location or "").strip().lower()
        if not text:
            return False
        if text.startswith("lat="):
            return False
        comma_parts = [part.strip() for part in text.split(",") if part.strip()]
        if "," in text and len(comma_parts) >= 2:
            if all(part not in {"is the address", "the address", "address"} for part in comma_parts):
                return False
        food_markers = {
            "biryani", "pizza", "burger", "fries", "fried", "chicken", "butter",
            "garlic", "tandoori", "wrap", "sandwich", "popcorn", "dum", "dumb",
            "meal", "momo", "noodles", "pasta", "shawarma", "roll", "bucket",
            "eat", "hungry", "food",
        }
        order_markers = {
            "order", "one", "two", "three", "quantity", "person", "for one",
            "address", "location", "restaurant", "restaurants", "menu",
        }
        words = {w for w in text.replace("-", " ").split() if w}
        return bool(words & food_markers) and bool(words & order_markers or " and " in text)

    def _sanitize_session(self, session: UserSession):
        if not self._looks_like_invalid_location(session.current_location):
            return
        logger.warning(
            "Clearing suspicious saved location for user %s: %s",
            session.user_id,
            session.current_location,
        )
        session.current_location = None
        session.address_id = None
        session.search_results = []
        session.selected_restaurant_id = None
        session.selected_restaurant_name = None
        session.menu_items_map = {}
        if session.state in {
            ConversationState.SEARCHING,
            ConversationState.BROWSING_MENU,
            ConversationState.ORDERING,
        }:
            session.state = ConversationState.IDLE

    def _save_sessions(self):
        """Save sessions to file"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(SESSIONS_FILE, "w") as f:
                sessions_data = [s.model_dump() for s in self._sessions.values()]
                json.dump(sessions_data, f, indent=2, default=str)
        except Exception as e:
            logger.error("Error saving sessions: %s", e)

    def get_session(self, user_id: str, user_name: str = None) -> UserSession:
        """Get or create a user session"""
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(
                user_id=user_id,
                user_name=user_name,
            )
            logger.info(f"Created new session for user: {user_id}")
        else:
            if user_name:
                self._sessions[user_id].user_name = user_name
        self._sanitize_session(self._sessions[user_id])
        return self._sessions[user_id]

    def update_state(self, user_id: str, state: ConversationState):
        """Update the conversation state"""
        session = self.get_session(user_id)
        session.state = state
        logger.info(f"User {user_id} state changed to: {state.value}")

    def set_search_results(self, user_id: str, results: list):
        """Store search results in session"""
        session = self.get_session(user_id)
        session.search_results = results
        session.selected_restaurant_id = None
        session.selected_restaurant_name = None
        session.menu_items_map = {}
        if results:
            session.state = ConversationState.SEARCHING

    def set_selected_restaurant(self, user_id: str, restaurant_id: str):
        """Set the selected restaurant"""
        session = self.get_session(user_id)
        session.selected_restaurant_id = restaurant_id

    def reset_browsing_context(self, user_id: str):
        """Clear restaurant/menu/cart state while keeping user identity and location."""
        session = self.get_session(user_id)
        session.search_results = []
        session.selected_restaurant_id = None
        session.selected_restaurant_name = None
        session.menu_items_map = {}
        session.cart = []
        session.zomato_cart_id = None
        session.payment_type = None
        if session.state != ConversationState.IDLE:
            session.state = ConversationState.IDLE

    def add_to_cart(self, user_id: str, item: CartItem):
        """Add an item to the cart"""
        session = self.get_session(user_id)
        # Check if item already exists in cart
        for existing in session.cart:
            if existing.item_id == item.item_id:
                existing.quantity += item.quantity
                return
        session.cart.append(item)

    def remove_from_cart(self, user_id: str, item_name: str) -> bool:
        """Remove an item from the cart by name"""
        session = self.get_session(user_id)
        for i, item in enumerate(session.cart):
            if item_name.lower() in item.name.lower():
                session.cart.pop(i)
                return True
        return False

    def update_cart_quantity(self, user_id: str, item_name: str, quantity: int) -> bool:
        """Update quantity of a cart item"""
        session = self.get_session(user_id)
        for item in session.cart:
            if item_name.lower() in item.name.lower():
                if quantity <= 0:
                    return self.remove_from_cart(user_id, item_name)
                item.quantity = quantity
                return True
        return False

    def clear_cart(self, user_id: str):
        """Clear the cart"""
        session = self.get_session(user_id)
        session.cart = []

    def get_cart(self, user_id: str) -> List[CartItem]:
        """Get current cart items"""
        session = self.get_session(user_id)
        return session.cart

    def get_cart_total(self, user_id: str) -> int:
        """Calculate cart subtotal"""
        session = self.get_session(user_id)
        return sum(item.price * item.quantity for item in session.cart)

    def set_address(self, user_id: str, address: str):
        """Set delivery address"""
        session = self.get_session(user_id)
        session.pending_address = address

    def set_current_order(self, user_id: str, order_id: str):
        """Set the current order ID"""
        session = self.get_session(user_id)
        session.current_order_id = order_id

    def add_to_history(self, user_id: str, role: str, content: str):
        """Add a message to conversation history"""
        session = self.get_session(user_id)
        session.conversation_history.append({
            "role": role,
            "content": content,
        })
        # Keep last 20 messages for context
        if len(session.conversation_history) > 20:
            session.conversation_history = session.conversation_history[-20:]

    def set_last_bot_message(self, user_id: str, message: str):
        """Store the last bot message for context"""
        session = self.get_session(user_id)
        session.last_bot_message = message

    def set_preferences(self, user_id: str, cuisine: str = None, budget: int = None, veg: bool = None):
        """Update inferred user preferences for this session"""
        session = self.get_session(user_id)
        if cuisine:
            session.preferred_cuisine = cuisine
        if budget is not None:
            session.budget = budget
        if veg is not None:
            session.veg_preference = veg

    def record_past_order(self, user_id: str, restaurant_name: str, cuisine: str, total: int):
        """Store a lightweight summary of a completed order for personalisation"""
        session = self.get_session(user_id)
        entry = {"restaurant": restaurant_name, "cuisine": cuisine, "total": total}
        # Keep only the last 5 orders
        session.past_orders = (session.past_orders + [entry])[-5:]

    def set_address_id(self, user_id: str, address_id: str):
        """Cache the resolved Zomato address_id for MCP calls"""
        session = self.get_session(user_id)
        session.address_id = address_id

    def apply_location_change(self, user_id: str, location: str):
        """Update location and clear state that depends on the previous area/address."""
        session = self.get_session(user_id)
        session.current_location = location
        session.address_id = None
        session.search_results = []
        session.selected_restaurant_id = None
        session.selected_restaurant_name = None
        session.menu_items_map = {}
        session.cart = []
        session.zomato_cart_id = None
        session.payment_type = None
        if session.state != ConversationState.IDLE:
            session.state = ConversationState.IDLE
        logger.info("Applied location change for user %s -> %s", user_id, location)

    def set_delivery_context(self, user_id: str, location: str, address_id: str | None = None):
        """Update delivery location/address while preserving active order flow and cart."""
        session = self.get_session(user_id)
        session.current_location = location
        session.address_id = address_id
        logger.info("Updated delivery context for user %s -> %s", user_id, location)

    def reset_session(self, user_id: str):
        """Reset session to initial state (keep user info and past orders)"""
        session = self.get_session(user_id)
        past = session.past_orders  # preserve across resets
        saved = session.saved_addresses  # preserve fetched addresses
        user_name = session.user_name
        session.state = ConversationState.IDLE
        session.user_name = user_name
        session.current_location = None
        session.search_results = []
        session.selected_restaurant_id = None
        session.selected_restaurant_name = None
        session.cart = []
        session.pending_address = None
        session.current_order_id = None
        session.conversation_history = []
        session.last_bot_message = None
        session.menu_items_map = {}
        session.preferred_cuisine = None
        session.budget = None
        session.veg_preference = None
        session.past_orders = past
        session.saved_addresses = saved
        session.address_id = None
        session.zomato_cart_id = None
        session.payment_type = None
        logger.info(f"Session reset for user: {user_id}")

    def get_all_sessions(self) -> dict:
        """Get all active sessions (for admin/debugging)"""
        return {
            uid: {
                "user_name": s.user_name,
                "state": s.state.value,
                "cart_items": len(s.cart),
                "current_order": s.current_order_id,
            }
            for uid, s in self._sessions.items()
        }

    def save(self):
        """Persist current sessions to disk. Call at end of each request."""
        self._save_sessions()


# Singleton instance
session_service = SessionService()
