"""
Pydantic models for the restaurant order bot
"""
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class ConversationState(str, Enum):
    """State machine for conversation flow"""
    IDLE = "idle"
    SEARCHING = "searching"
    RESTAURANT_SELECTED = "restaurant_selected"
    BROWSING_MENU = "browsing_menu"
    ORDERING = "ordering"
    AWAITING_ADDRESS = "awaiting_address"
    AWAITING_PAYMENT = "awaiting_payment"
    CONFIRMING_ORDER = "confirming_order"
    ORDER_PLACED = "order_placed"
    TRACKING = "tracking"


class OrderStatus(str, Enum):
    """Order status progression"""
    PLACED = "placed"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class CartItem(BaseModel):
    """Item in the shopping cart"""
    item_id: str
    name: str
    variant_id: Optional[str] = None
    size: Optional[str] = None
    price: int
    quantity: int
    is_veg: bool


class Order(BaseModel):
    """Order model"""
    order_id: str
    user_id: str
    restaurant_id: str
    restaurant_name: str
    items: List[CartItem]
    subtotal: int
    tax: int
    delivery_fee: int
    total: int
    address: str
    payment_method: str = "Cash on Delivery"
    status: OrderStatus = OrderStatus.PLACED
    estimated_delivery: str = ""
    delivery_partner: Optional[str] = None
    delivery_partner_phone: Optional[str] = None
    created_at: str = ""


class UserSession(BaseModel):
    """User session for maintaining conversation context"""
    user_id: str
    user_name: Optional[str] = None
    state: ConversationState = ConversationState.IDLE
    current_location: Optional[str] = None
    search_results: list = []
    selected_restaurant_id: Optional[str] = None
    selected_restaurant_name: Optional[str] = None
    cart: List[CartItem] = []
    pending_address: Optional[str] = None
    current_order_id: Optional[str] = None
    conversation_history: list = []
    last_bot_message: Optional[str] = None
    menu_items_map: dict = {}  # maps number -> menu item for quick selection
    # ── Zomato address resolution ──────────────────────────────────────────────
    address_id: Optional[str] = None        # Zomato address_id for MCP calls
    saved_addresses: List[dict] = []        # from get_saved_addresses_for_user
    # ── Zomato cart/order ─────────────────────────────────────────────────────
    zomato_cart_id: Optional[str] = None    # from create_cart response
    payment_type: Optional[str] = None      # 'upi_qr' or 'pay_later'
    # ── In-session user preferences (extracted from their messages) ──
    preferred_cuisine: Optional[str] = None  # e.g. "biryani", "pizza"
    budget: Optional[int] = None             # e.g. 300 (rupees)
    veg_preference: Optional[bool] = None   # True=veg, False=non-veg, None=any
    past_orders: List[dict] = []            # lightweight order summaries for personalisation


class SearchFilters(BaseModel):
    """Active search filters from the user UI"""
    veg_only: bool = False          # True = only vegetarian restaurants/items
    non_veg_only: bool = False      # True = only non-veg restaurants
    min_rating: float = 0.0         # 0 = no filter, 3.0 / 4.0 / 4.5
    max_distance_km: int = 25       # search radius in kilometres


class ChatMessage(BaseModel):
    """Chat message model for API"""
    message: str
    user_id: str = "web_user"
    user_name: Optional[str] = "Guest"
    user_location: Optional[str] = None  # city/address from client GPS or manual input
    filters: Optional[SearchFilters] = None  # active UI filter state


class ChatResponse(BaseModel):
    """Chat response model for API"""
    response: str
    state: str
    cart_items: list = []
    restaurant: Optional[dict] = None
    order: Optional[dict] = None
    thinking_steps: List[str] = []  # what the agent did, shown in the UI


class VoiceChatMessage(BaseModel):
    """Voice chat payload when transcript is already available from client."""
    transcript: str
    user_id: str = "web_user"
    user_name: Optional[str] = "Guest"
    user_location: Optional[str] = None
    filters: Optional[SearchFilters] = None


class AIIntent(BaseModel):
    """Parsed intent from AI"""
    intent: str
    entities: dict = {}
    confidence: float = 0.0
