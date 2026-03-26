import os
import json
import copy
import logging
import asyncio
import traceback
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI  # type: ignore
from app.models.schemas import SearchFilters, ConversationState, CartItem
from app.services.session_service import session_service
from app.services.zomato_mcp import global_zomato_mcp

logger = logging.getLogger(__name__)

# Model defaults (override in `.env` as needed)
# GEMINI_MODELS="model-a,model-b"
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_CALL_TIMEOUT_SECONDS = 14
MAX_TOOL_LOOPS = 4
MAX_COMPLETION_TOKENS = 512
MAX_TOOL_RESULT_CHARS = 700
MAX_HISTORY_MESSAGES = 5
MAX_HISTORY_CHARS = 500

# ── City → coordinates mapping (for MCP tools that need lat/lon) ─────────────
CITY_COORDS: Dict[str, Dict[str, float]] = {
    "hyderabad":       {"lat": 17.3850, "lon": 78.4867},
    "bangalore":       {"lat": 12.9716, "lon": 77.5946},
    "bengaluru":       {"lat": 12.9716, "lon": 77.5946},
    "mumbai":          {"lat": 19.0760, "lon": 72.8777},
    "delhi":           {"lat": 28.6139, "lon": 77.2090},
    "new delhi":       {"lat": 28.6139, "lon": 77.2090},
    "chennai":         {"lat": 13.0827, "lon": 80.2707},
    "kolkata":         {"lat": 22.5726, "lon": 88.3639},
    "pune":            {"lat": 18.5204, "lon": 73.8567},
    "ahmedabad":       {"lat": 23.0225, "lon": 72.5714},
    "jaipur":          {"lat": 26.9124, "lon": 75.7873},
    "lucknow":         {"lat": 26.8467, "lon": 80.9462},
    "vijayawada":      {"lat": 16.5062, "lon": 80.6480},
    "visakhapatnam":   {"lat": 17.6868, "lon": 83.2185},
    "vizag":           {"lat": 17.6868, "lon": 83.2185},
    "guntur":          {"lat": 16.3067, "lon": 80.4365},
    "amaravati":       {"lat": 16.5131, "lon": 80.5150},
    "tirupati":        {"lat": 13.6288, "lon": 79.4192},
    "kochi":           {"lat":  9.9312, "lon": 76.2673},
    "thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366},
    "coimbatore":      {"lat": 11.0168, "lon": 76.9558},
    "madurai":         {"lat":  9.9252, "lon": 78.1198},
    "chandigarh":      {"lat": 30.7333, "lon": 76.7794},
    "goa":             {"lat": 15.2993, "lon": 74.1240},
    "indore":          {"lat": 22.7196, "lon": 75.8577},
    "bhopal":          {"lat": 23.2599, "lon": 77.4126},
    "nagpur":          {"lat": 21.1458, "lon": 79.0882},
    "surat":           {"lat": 21.1702, "lon": 72.8311},
    "vadodara":        {"lat": 22.3072, "lon": 73.1812},
    "gurgaon":         {"lat": 28.4595, "lon": 77.0266},
    "gurugram":        {"lat": 28.4595, "lon": 77.0266},
    "noida":           {"lat": 28.5355, "lon": 77.3910},
    "ghaziabad":       {"lat": 28.6692, "lon": 77.4538},
    "faridabad":       {"lat": 28.4089, "lon": 77.3178},
    "mysore":          {"lat": 12.2958, "lon": 76.6394},
    "mysuru":          {"lat": 12.2958, "lon": 76.6394},
    "mangalore":       {"lat": 12.9141, "lon": 74.8560},
    "hubli":           {"lat": 15.3647, "lon": 75.1240},
    "patna":           {"lat": 25.6093, "lon": 85.1376},
    "ranchi":          {"lat": 23.3441, "lon": 85.3096},
    "bhubaneswar":     {"lat": 20.2961, "lon": 85.8245},
    "dehradun":        {"lat": 30.3165, "lon": 78.0322},
    "agra":            {"lat": 27.1767, "lon": 78.0081},
    "varanasi":        {"lat": 25.3176, "lon": 83.0064},
    "kanpur":          {"lat": 26.4499, "lon": 80.3319},
    # Hyderabad neighbourhoods
    "madhapur":        {"lat": 17.4484, "lon": 78.3908},
    "hitech city":     {"lat": 17.4435, "lon": 78.3772},
    "kondapur":        {"lat": 17.4577, "lon": 78.3653},
    "gachibowli":      {"lat": 17.4401, "lon": 78.3489},
    "jubilee hills":   {"lat": 17.4325, "lon": 78.4073},
    "banjara hills":   {"lat": 17.4156, "lon": 78.4347},
    "kukatpally":      {"lat": 17.4849, "lon": 78.3900},
    "ameerpet":        {"lat": 17.4375, "lon": 78.4483},
    "secunderabad":    {"lat": 17.4399, "lon": 78.4983},
    # Bangalore neighbourhoods
    "koramangala":     {"lat": 12.9352, "lon": 77.6245},
    "indiranagar":     {"lat": 12.9784, "lon": 77.6408},
    "whitefield":      {"lat": 12.9698, "lon": 77.7500},
    "hsr layout":      {"lat": 12.9116, "lon": 77.6389},
    "electronic city": {"lat": 12.8450, "lon": 77.6625},
    "jp nagar":        {"lat": 12.9063, "lon": 77.5857},
    "marathahalli":    {"lat": 12.9591, "lon": 77.6974},
}

# Human-readable labels for tool calls shown in the UI
TOOL_LABELS = {
    "get_restaurants_for_keyword": "🔍 Searching restaurants",
    "get_saved_addresses_for_user": "📍 Fetching saved addresses",
    "get_menu_items_listing": "📋 Loading menu categories",
    "get_restaurant_menu_by_categories": "📋 Loading menu items",
    "search_restaurants": "🔍 Searching restaurants",
    "get_restaurant_details": "🏪 Fetching restaurant details",
    "add_item_to_cart": "🛒 Adding to cart",
    "place_order": "📦 Placing order",
    "get_order_status": "📦 Checking order status",
    "track_order": "🚴 Tracking delivery",
}


class GeminiAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        raw_models = os.getenv("GEMINI_MODELS", "")
        self._models: List[str] = [m.strip() for m in raw_models.split(",") if m.strip()] or GEMINI_MODELS[:]
        self._disabled_models = set()
        self.automation_only = (os.getenv("AUTOMATION_ONLY", "true").strip().lower() != "false")
        logger.info(
            "LLM provider: gemini | base_url: %s | models: %s | automation_only: %s",
            GEMINI_BASE_URL,
            ",".join(self._models),
            self.automation_only,
        )

    def _sanitize_schema(self, schema: dict) -> dict:
        """
        Convert a Zomato MCP JSON-schema to an OpenAI-compatible function-parameters schema.
        We enforce:
          • top-level { "type": "object", "properties": {...} }
          • no additionalProperties, no anyOf/oneOf at nested level (simplify to single type)
          • no null types — make those fields optional instead
        """
        schema = copy.deepcopy(schema)

        def _clean(s):
            if not isinstance(s, dict):
                return s

            # Resolve anyOf / oneOf: pick the first non-null variant
            for key in ("anyOf", "oneOf"):
                if key in s:
                    variants = s.pop(key)
                    non_null = [v for v in variants if isinstance(v, dict) and v.get("type") != "null"]
                    chosen = non_null[0] if non_null else (variants[0] if variants else {})
                    for k, v in chosen.items():
                        if k not in s:
                            s[k] = v

            # Remove fields that bloat prompts / break strict parsers
            for bad in ("additionalProperties", "default", "$schema", "title"):
                s.pop(bad, None)

            # Truncate description to avoid token bloat
            if "description" in s and isinstance(s["description"], str):
                s["description"] = s["description"][:120]

            # Recurse into properties
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_val in list(s["properties"].items()):
                    s["properties"][prop_name] = _clean(prop_val)

            # Recurse into array items
            if "items" in s:
                s["items"] = _clean(s["items"])

            return s

        schema = _clean(schema)

        # Function calling expects object params at the top level
        if "properties" in schema and "type" not in schema:
            schema["type"] = "object"

        # Remove required list entries for address_id (needs logged-in Zomato session)
        if "required" in schema and isinstance(schema["required"], list):
            schema["required"] = [r for r in schema["required"] if r != "address_id"]

        # Remove address_id from properties entirely — we won't use it
        if "properties" in schema:
            schema["properties"].pop("address_id", None)

        return schema

    async def get_tools(self):
        mcp_tools = await global_zomato_mcp.get_tools()
        openai_tools = []
        for t in mcp_tools:
            raw_schema = t.inputSchema if t.inputSchema else {}
            clean_schema = self._sanitize_schema(raw_schema)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "No description")[:180],
                    "parameters": clean_schema,
                }
            })
        return openai_tools

    def _detect_location_needed(self, message: str) -> bool:
        """Return True if user is asking for nearby/nearest results without specifying a place."""
        patterns = [
            r"\b(near(?:est)?|nearby|around|close to)\s+(me|my location|here|us)\b",
            r"\bnearest\s+to\s+me\b",
            r"\bnear\s+me\b",
            r"\bmy\s+(area|location|place|city|neighbourhood)\b",
        ]
        lowered = message.lower()
        return any(re.search(p, lowered) for p in patterns)

    def _build_filter_context(self, filters: Optional["SearchFilters"]) -> str:
        if not filters:
            return ""
        parts = []
        radius = filters.max_distance_km or 25
        parts.append(f"Search radius: {radius} km from the user's location.")
        if filters.veg_only:
            parts.append("FILTER: Show ONLY vegetarian restaurants and items (is_veg = true). Exclude all non-veg options.")
        elif filters.non_veg_only:
            parts.append("FILTER: Show ONLY non-vegetarian restaurants. Exclude pure-veg restaurants.")
        if filters.min_rating and filters.min_rating > 0:
            parts.append(f"FILTER: Show ONLY restaurants rated {filters.min_rating}★ or above. Exclude lower-rated ones.")
        parts.append("Sort all restaurant results by distance (closest first).")
        return "\n".join(parts)

    def _build_system_prompt(self, user_location: Optional[str] = None, filters: Optional["SearchFilters"] = None) -> str:
        location_hint = ""
        if user_location:
            location_hint = (
                f"\nThe user's current location is: {user_location}. "
                "Use this as the search location when they ask for restaurants near them."
            )
        filter_hint = ""
        radius = 25
        if filters:
            radius = filters.max_distance_km or 25
            fc = self._build_filter_context(filters)
            if fc:
                filter_hint = f"\n\nACTIVE SEARCH FILTERS (apply strictly):\n{fc}"
        return (
            "You are Zomato AI — a smart, friendly food-ordering assistant. "
            "Help users discover restaurants, browse menus, add items to cart, and place orders. "
            "Use the provided tools for all data lookups — never make up restaurants, menus, or prices.\n\n"
            "Guidelines:\n"
            f"• Search radius: Always search within {radius} km of the user's location. "
            "If the tool supports a radius or distance parameter, set it accordingly.\n"
            "• To search restaurants: call get_restaurants_for_keyword directly with the user's city/area "
            "as the location string. DO NOT call get_saved_addresses_for_user — the user is not logged in to Zomato.\n"
            "• Example search: get_restaurants_for_keyword(keyword='pizza', location='Amaravati, Andhra Pradesh')\n"
            "• Sort restaurant listings by distance (nearest first). Show estimated distance next to each result.\n"
            "• If the user asks 'nearest to me' or 'near me' and no location is known, "
            "politely ask them to share their delivery address or city.\n"
            "• IMPORTANT FOR MENUS: If the user asks for a restaurant's menu, you MUST pass its exact `restaurant_id`. "
            "If you don't know the exact `id`, you MUST call `get_restaurants_for_keyword` FIRST to search for the restaurant and find its `id`, "
            "and THEN call the menu tool using that `id`.\n"
            "• Present results as a numbered list with: name (bold), cuisine, rating (⭐), "
            "delivery time, and a one-line description.\n"
            "• Respect all active filters — veg/non-veg and minimum rating — and never show results "
            "that violate the active filters.\n"
            "• Always confirm with the user before placing an order or charging payment.\n"
            "• Keep responses concise and action-oriented — avoid long paragraphs.\n"
            "• If a tool call fails or returns no results, tell the user and suggest searching with a nearby major city name."
            + location_hint
            + filter_hint
        )

    def _is_location_update_only(self, message: str) -> bool:
        """Detect plain location updates so we can skip expensive LLM/tool calls."""
        text = (message or "").strip().lower()
        if not text:
            return False

        action_words = (
            "show", "find", "search", "order", "menu", "biryani", "pizza",
            "restaurant", "restaurants", "track", "checkout", "cart", "help",
        )
        if any(w in text for w in action_words):
            return False

        if text.startswith("my location is") or text.startswith("location is"):
            return True

        # Handles simple location text like "Amaravati, Andhra Pradesh"
        if "," in text and 3 <= len(text) <= 80:
            return True

        return False

    def _is_first_menu_request(self, message: str) -> bool:
        text = (message or "").strip().lower()
        patterns = (
            "show menu of first restaurant",
            "menu of first restaurant",
            "show first restaurant menu",
            "menu of 1st restaurant",
            "show menu of 1st restaurant",
        )
        return any(p in text for p in patterns)

    def _is_cancel_order_request(self, text: str) -> bool:
        """Detect order cancellation requests."""
        if not text:
            return False
        cancel_patterns = [
            r"\bcancel\b.*\border\b",
            r"\border\b.*\bcancel\b",
            r"\bcancel\b.*\bmy\b",
        ]
        exact_matches = {"cancel order", "cancel my order", "cancel the order",
                         "cancel this order", "i want to cancel", "cancel it"}
        if text in exact_matches:
            return True
        return any(re.search(p, text) for p in cancel_patterns)

    async def _cancel_order(self, user_id: str) -> str:
        """Attempt to cancel the current order.
        
        Zomato MCP has no cancel tool, so we inform the user
        and provide the Zomato app/support link.
        """
        session = session_service.get_session(user_id)
        # Fetch current order info for context
        order_info = ""
        try:
            result = await global_zomato_mcp.call_tool("get_order_tracking_info", {})
            for chunk in result or []:
                if isinstance(chunk, str):
                    try:
                        data = json.loads(chunk)
                        orders = self._extract_tracking_items(data)
                        if orders:
                            o = orders[0]
                            oid = o.get("order_id") or "N/A"
                            rest = o.get("restaurant_name") or ""
                            status = o.get("order_status") or o.get("status") or ""
                            order_info = (
                                f"\n\n📦 Your active order:\n"
                                f"• Order ID: **{oid}**\n"
                                f"• Restaurant: {rest}\n"
                                f"• Status: {status}"
                            )
                    except Exception:
                        pass
        except Exception:
            pass

        # Reset session state
        if session.state == ConversationState.ORDER_PLACED:
            session_service.update_state(user_id, ConversationState.IDLE)
            session.current_order_id = None
            session.cart = []
            session.zomato_cart_id = None

        return (
            "❌ **Order cancellation is not available through the bot.**\n\n"
            "To cancel your order, please:\n"
            "1. Open the **Zomato app** → Go to your active order → Tap **Cancel**\n"
            "2. Or contact Zomato support in the app\n"
            + order_info
            + "\n\nI've reset your session. Say **start over** to place a new order."
        )

    def _is_order_tracking_request(self, text: str) -> bool:
        """Detect order tracking/status requests flexibly."""
        if not text:
            return False
        tracking_patterns = [
            r"\btrack\b.*\border\b",
            r"\border\b.*\btrack\b",
            r"\border\b.*\bstatus\b",
            r"\bstatus\b.*\border\b",
            r"\bwhere\b.*\bmy\s+order\b",
            r"\bmy\s+order\b",
            r"\bcheck\b.*\border\b",
        ]
        exact_matches = {"track my order", "track order", "order status", "status",
                         "track my current order", "where is my order", "check order",
                         "check my order", "order tracking", "my order status"}
        if text in exact_matches:
            return True
        return any(re.search(p, text) for p in tracking_patterns)

    def _is_restaurant_search_request(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        # Don't treat order-tracking messages as restaurant searches
        if self._is_order_tracking_request(text):
            return False
        search_markers = (
            "restaurant", "restaurants", "near me", "nearby",
            "biryani", "pizza", "burger", "chinese", "south indian",
            "north indian", "italian", "mexican", "thai", "continental",
            "kerala", "andhra", "punjabi", "mughlai", "bengali",
            "cafe", "healthy", "veg", "vegetarian", "non veg",
            "noodles", "pasta", "thali", "rolls", "sandwich",
            "sushi", "dessert", "ice cream", "shawarma", "kebab",
            "food", "eat", "hungry",
        )
        return any(m in text for m in search_markers)

    def _is_smalltalk_request(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False

        # Handle greeting typos like "hhi", "hiii", "heyy"
        if re.fullmatch(r"h+i+|h+e+y+|hello+", text):
            return True

        smalltalk = {
            "hi", "hii", "hiii", "hello", "hey", "hey there", "yo",
            "help", "start", "menu", "what can you do",
        }
        return text in smalltalk

    def _smalltalk_reply(self, location: Optional[str], session=None) -> str:
        loc_line = f"\n📍 Current location: **{location}**" if location else ""
        if session and session.past_orders:
            last = session.past_orders[-1]
            return (
                f"👋 Welcome back! Last time you ordered from **{last['restaurant']}**."
                f"\n\nWould you like to reorder, or try something new today?{loc_line}"
            )
        return (
            "Hello! I can help you order food from Zomato. 🍽️"
            "\n\nWhat are you craving today? You can say:"
            "\n• `show me biryani places`"
            "\n• `find veg pizza under 300`"
            "\n• `top rated restaurants near me`"
            + loc_line
        )

    def _extract_intent(self, message: str) -> dict:
        """
        Extract structured intent from a user message.
        Returns dict with: cuisine, budget, veg (bool|None)
        """
        text = message.lower()
        intent: dict = {"cuisine": None, "budget": None, "veg": None}

        # Cuisine detection
        cuisines = ["biryani", "pizza", "burger", "chinese", "south indian",
                    "north indian", "cafe", "coffee", "noodles", "pasta",
                    "sushi", "mexican", "dessert", "ice cream", "sandwich",
                    "rolls", "wraps", "healthy", "salad", "thali"]
        for c in cuisines:
            if c in text:
                intent["cuisine"] = c
                break

        # Budget detection: "under 300", "less than 500", "below 400"
        budget_m = re.search(r'(?:under|below|less\s+than|within|max|upto|up\s+to)\s*[₹rs\.]*\s*(\d+)', text)
        if not budget_m:
            budget_m = re.search(r'[₹rs\.]*\s*(\d+)\s*(?:rupees|rs|/-)', text)
        if budget_m:
            intent["budget"] = int(budget_m.group(1))

        # Veg preference
        if re.search(r'\bveg(etarian)?\b', text) and 'non.?veg' not in text:
            intent["veg"] = True
        elif re.search(r'\bnon.?veg\b', text):
            intent["veg"] = False

        return intent

    def _format_restaurant_card(self, idx: int, r: dict, veg_filter: Optional[bool] = None, budget: Optional[int] = None) -> str:
        """Format a single restaurant as a rich text card."""
        name = r.get("name", "Unknown")
        rating = r.get("rating", "")
        delivery = r.get("delivery_time", "")
        cuisine = r.get("cuisines", "") or r.get("cuisine", "")
        cost = r.get("cost", "")  # cost for two / avg price

        rating_str  = f" ⭐ {rating}" if rating else ""
        delivery_str = f" • 🚚 {delivery} min" if delivery else ""
        cuisine_str  = f" • {cuisine}" if cuisine else ""
        cost_str     = f" • 💰 ₹{cost}" if cost else ""
        veg_badge    = " 🟢" if veg_filter else ""

        return f"{idx}. **{name}**{veg_badge}{rating_str}{delivery_str}{cuisine_str}{cost_str}"

    def _format_restaurant_list(self, restaurants: List[dict], keyword: str, location: str,
                                 veg_filter: Optional[bool] = None, budget: Optional[int] = None) -> str:
        """Build the full numbered restaurant list reply."""
        lines = []
        for idx, r in enumerate(restaurants[:8], start=1):
            lines.append(self._format_restaurant_card(idx, r, veg_filter, budget))

        filters_applied = []
        if veg_filter:
            filters_applied.append("vegetarian only")
        if budget:
            filters_applied.append(f"under ₹{budget}")
        filter_note = f" *(filtered: {', '.join(filters_applied)})*" if filters_applied else ""

        title = f"{keyword} restaurants" if keyword else "restaurants"
        return (
            f"Here are the top **{title}** in {location}{filter_note}:\n\n"
            + "\n".join(lines)
            + "\n\nWhich one would you like to order from? Reply with a number or name."
        )

    def _format_menu_list(self, menu_items: List[dict], restaurant_name: str) -> str:
        """Build a readable menu display."""
        if not menu_items:
            return f"I couldn't load the menu for **{restaurant_name}** right now."
        lines = []
        for item in menu_items[:10]:
            size_part = f" ({item['size']})" if item.get("size") else ""
            price = item.get("price", "?")
            lines.append(f"• {item['name']}{size_part} — ₹{price}")
        return (
            f"🍽️ **Menu — {restaurant_name}**\n\n"
            + "\n".join(lines)
            + "\n\nJust tell me what you'd like, e.g. *1 Margherita and 2 Garlic Breads*."
        )

    async def _resolve_address_id(self, user_id: str, session) -> tuple:
        """
        Fetch saved Zomato addresses and pick the best match for session.current_location.
        Returns (address_id: str | None, display_location: str).
        Result is cached on the session to avoid repeated API calls.
        """
        if session.address_id:
            return session.address_id, session.current_location or ""

        result = await global_zomato_mcp.call_tool("get_saved_addresses_for_user", {})
        if not result:
            return None, ""
        try:
            data = json.loads(result[0])
        except Exception:
            logger.warning("Failed to parse saved addresses: %s", result)
            return None, ""

        addresses = data.get("addresses", [])
        if not addresses:
            return None, ""

        session.saved_addresses = addresses

        # Match session.current_location against saved address names
        loc_lower = (session.current_location or "").lower().strip()
        # Strip punctuation from location words for matching
        loc_words = [re.sub(r'[^\w]', '', w) for w in loc_lower.split()]
        loc_words = [w for w in loc_words if w and len(w) > 2]

        # Generic state/country/landmark words — these appear in many addresses
        # and should NOT drive the match
        generic_words = {"india", "andhra", "pradesh", "telangana", "karnataka",
                        "tamil", "nadu", "maharashtra", "kerala", "gujarat",
                        "rajasthan", "uttar", "madhya", "west", "bengal",
                        "station", "railway", "airport", "road", "nagar",
                        "colony", "district", "state", "city", "town"}

        # Separate words into meaningful (city/area names) vs generic
        meaningful_words = [w for w in loc_words if w not in generic_words]
        generic_in_query = [w for w in loc_words if w in generic_words]

        best = addresses[0]
        best_score = 0.0
        for addr in addresses:
            name = addr.get("location_name", "").lower()
            name_clean = re.sub(r'[^\w\s]', ' ', name)  # strip punctuation
            score = 0.0

            # Meaningful words (city/area names) get high score
            for word in meaningful_words:
                if len(word) >= 4 and re.search(r'\b' + re.escape(word) + r'\b', name_clean):
                    score += 5.0  # strong match for city/area names
                elif len(word) >= 3 and word in name_clean:
                    score += 2.0

            # Generic words only get a small tiebreaker boost
            for word in generic_in_query:
                if word in name_clean:
                    score += 0.1

            logger.debug("Address match: %s score=%.1f for location=%s", addr.get('address_id'), score, loc_lower)
            if score > best_score:
                best_score = score
                best = addr

        address_id = best["address_id"]
        session_service.set_address_id(user_id, address_id)
        logger.info("Resolved address_id=%s (%s) for location=%s (score=%.1f)",
                    address_id, best.get("location_name", ""), session.current_location, best_score)
        return address_id, best.get("location_name", "")

    async def _fetch_restaurant_menu(self, res_id_str: str, address_id: str) -> List[Dict[str, str]]:
        """
        Two-step menu fetch using the correct Zomato MCP tools:
          1. get_menu_items_listing  → extract all categories
          2. get_restaurant_menu_by_categories → fetch items with prices
        Falls back to raw parse of listing result if step 2 fails.
        """
        try:
            res_id = int(res_id_str)
        except (ValueError, TypeError):
            logger.error("Cannot convert res_id to int: %s", res_id_str)
            return []

        # Step 1: get dish → category mapping
        listing = await global_zomato_mcp.call_tool(
            "get_menu_items_listing",
            {"res_id": res_id, "address_id": address_id},
        )
        logger.info("get_menu_items_listing raw: %s", str(listing)[:300])

        # Extract unique categories from listing
        categories: List[str] = []
        variant_id_map: Dict[str, str] = {}  # item_name_lower → variant_id
        for chunk in listing or []:
            if isinstance(chunk, str):
                try:
                    data = json.loads(chunk)
                    def _collect_cats_and_variants(node):
                        if isinstance(node, dict):
                            # Collect categories from nested arrays
                            cats = node.get("categories")
                            if isinstance(cats, list):
                                for cat in cats:
                                    if isinstance(cat, str) and cat not in categories:
                                        categories.append(cat)
                            cat = node.get("category") or node.get("category_name")
                            if cat and isinstance(cat, str) and cat not in categories:
                                categories.append(cat)
                            # Collect variant_id from item_mappings
                            item_name = node.get("item_name") or node.get("name")
                            vid = node.get("variant_id")
                            if item_name and vid:
                                variant_id_map[str(item_name).lower()] = str(vid)
                            for v in node.values():
                                _collect_cats_and_variants(v)
                        elif isinstance(node, list):
                            for v in node:
                                _collect_cats_and_variants(v)
                    _collect_cats_and_variants(data)
                except Exception:
                    pass

        # Step 2: fetch menu by categories (all of them to get full menu)
        menu_result = await global_zomato_mcp.call_tool(
            "get_restaurant_menu_by_categories",
            {"res_id": res_id, "categories": categories, "address_id": address_id},
        )
        logger.info("get_restaurant_menu_by_categories raw: %s", str(menu_result)[:300])

        items = self._extract_menu_items_from_tool_result(menu_result)
        # Fall back: parse items from the listing itself if categories step fails
        if not items:
            items = self._extract_menu_items_from_tool_result(listing)
        # Supplement missing variant_ids from the listing's item_mappings
        for item in items:
            if not item.get("variant_id"):
                vid = variant_id_map.get(item.get("name", "").lower())
                if vid:
                    item["variant_id"] = vid
        return items

    def _normalize_location_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip(" .,-"))
        return cleaned.title()

    def _resolve_coords(self, location: str) -> Optional[Dict[str, float]]:
        """
        Look up lat/lon for a location string from CITY_COORDS.
        Tries exact match first, then checks if any known city/area appears in the text.
        Returns {"lat": ..., "lon": ...} or None.
        """
        text = (location or "").strip().lower()
        if not text:
            return None
        # Exact match
        if text in CITY_COORDS:
            return CITY_COORDS[text]
        # Check if any known city name appears in the location string
        # Sort by longest key first so "electronic city" matches before "city"
        for city in sorted(CITY_COORDS, key=len, reverse=True):
            if city in text:
                return CITY_COORDS[city]
        return None

    def _is_plain_location_message(self, message: str) -> bool:
        """
        Detect location-only inputs like:
        - "vijayawada"
        - "hyderabad"
        - "kondapur hyderabad"
        """
        text = (message or "").strip().lower()
        if not text:
            return False
        if any(ch.isdigit() for ch in text):
            return False
        non_location_replies = {
            "yes", "yeah", "yep", "y", "ok", "okay", "sure", "confirm", "go ahead",
            "no", "nope", "nah", "n", "cancel", "stop",
            "hi", "hello", "hey", "thanks", "thank you",
        }
        if text in non_location_replies:
            return False

        # If it contains action words, it's not a pure location update
        action_words = (
            "find", "show", "search", "order", "menu", "track", "cart",
            "restaurant", "restaurants", "pizza", "biryani", "burger",
            "chinese", "south indian", "cafe", "healthy", "veg", "non veg",
            "near", "in ", "of ", "checkout", "status", "start over",
        )
        if any(w in text for w in action_words):
            return False

        # Allow alpha-only location strings up to 4 words
        if re.fullmatch(r"[a-z\s]{3,60}", text):
            words = [w for w in text.split() if w]
            return 1 <= len(words) <= 4
        return False

    def _render_cart(self, user_id: str) -> str:
        session = session_service.get_session(user_id)
        if not session.cart:
            return "🛒 Your cart is empty. Ask for restaurants and add items to start an order."
        lines = []
        subtotal = 0
        for item in session.cart:
            line_total = item.price * item.quantity
            subtotal += line_total
            lines.append(f"- {item.name} x{item.quantity} - {line_total} rupees")
        delivery_fee = 40
        total = subtotal + delivery_fee
        return (
            "🛒 Your current cart:\n"
            + "\n".join(lines)
            + f"\n\nSubtotal: {subtotal} rupees\nDelivery Fee: {delivery_fee} rupees\nTotal: {total} rupees"
        )

    def _render_order_status(self, user_id: str) -> str:
        session = session_service.get_session(user_id)
        # Try real Zomato order tracking
        return "📦 Fetching order status from Zomato... Use the track command after placing an order."

    def _extract_tracking_items(self, data: Any) -> List[Dict]:
        """Walk a parsed JSON response to find order tracking items."""
        # Zomato nests at data.order_tracking.order_tracking_items
        if isinstance(data, dict):
            ot = data.get("order_tracking")
            if isinstance(ot, dict):
                items = ot.get("order_tracking_items")
                if isinstance(items, list) and items:
                    return items
            # Fallback: common alternative keys
            for key in ("orders", "active_orders", "order_tracking_items"):
                items = data.get(key)
                if isinstance(items, list) and items:
                    return items
            # Single order dict
            if data.get("order_id") or data.get("order_status"):
                return [data]
        elif isinstance(data, list):
            return data
        return []

    def _format_tracking_item(self, order: Dict) -> str:
        """Format one tracking item into a user-friendly string."""
        oid = order.get("order_id") or order.get("id") or "N/A"
        status = order.get("order_status") or order.get("status") or "Unknown"
        rest = order.get("restaurant_name") or ""
        msg = order.get("message") or ""
        paid = order.get("is_order_paid")
        rider = order.get("rider")

        lines = [f"📦 Order: **{oid}**"]
        if rest:
            lines.append(f"🍽️ Restaurant: {rest}")
        lines.append(f"📋 Status: {status}")
        if msg:
            lines.append(f"💬 {msg}")
        if paid is not None:
            lines.append(f"💳 Paid: {'Yes' if paid else 'No'}")
        if rider and isinstance(rider, dict):
            rname = rider.get("name") or rider.get("rider_name") or ""
            rphone = rider.get("phone") or rider.get("phone_number") or ""
            if rname:
                lines.append(f"🏍️ Delivery Partner: {rname}")
            if rphone:
                lines.append(f"📞 Contact: {rphone}")
        return "\n".join(lines)

    async def _render_order_status_async(self, user_id: str) -> str:
        """Fetch real order tracking from Zomato MCP."""
        try:
            result = await global_zomato_mcp.call_tool("get_order_tracking_info", {})
            logger.info("get_order_tracking_info result: %s", str(result)[:500])

            for chunk in result or []:
                if isinstance(chunk, str):
                    try:
                        data = json.loads(chunk)
                        if isinstance(data, dict) and (data.get("error_message") or data.get("error_code")):
                            return f"📦 {data.get('error_message', 'No active orders found.')}"

                        orders = self._extract_tracking_items(data)
                        if orders:
                            formatted = [self._format_tracking_item(o) for o in orders]
                            return "\n\n".join(formatted)
                        return "📦 No active orders found."
                    except Exception:
                        pass
            return "📦 No order tracking data available."
        except Exception as e:
            logger.error("Order tracking failed: %s", e)
            return f"📦 Could not fetch order status: {str(e)}"

    def _automation_fallback_reply(self, location: Optional[str]) -> str:
        loc_line = f"\n📍 Current location: **{location}**" if location else ""
        loc_example = location or "Hyderabad"
        return (
            "Hello! I can help you order food from Zomato. 🍽️\n\n"
            "Here's what you can ask me:\n"
            f"• `find biryani in {loc_example}` — search by cuisine & city\n"
            f"• `show pizza near {loc_example}` — find nearby restaurants\n"
            "• `show menu of first restaurant` — browse a menu\n"
            "• `show cart` — view your current cart\n"
            "• `track my order` — check order status\n"
            "• `start over` — reset the conversation"
            + loc_line
        )

    def _state_aware_fallback(self, session, user_id: str) -> str:
        """
        Context-aware fallback that responds based on the current conversation state
        instead of always showing the generic welcome message.
        """
        state = session.state
        location = session.current_location

        if state == ConversationState.BROWSING_MENU:
            rest_name = session.selected_restaurant_name or "the restaurant"
            menu_items_list = list(session.menu_items_map.values()) if session.menu_items_map else []
            if menu_items_list:
                available = [item.get("name", "") for item in menu_items_list[:6]]
                available_str = "\n".join(f"• {name}" for name in available if name)
                return (
                    f"You're browsing the menu for **{rest_name}**.\n\n"
                    f"Available items:\n{available_str}\n\n"
                    "Tell me what you'd like to order, e.g. *2 Plain Phulka and 1 Butter Phulka*.\n"
                    "Or say `show cart` to view your cart, or `start over` to reset."
                )
            return (
                f"You're viewing **{rest_name}**, but the menu failed to load.\n\n"
                "Try saying `show menu of first restaurant` to reload, or search for another restaurant."
            )

        if state == ConversationState.ORDERING:
            cart_text = self._render_cart(user_id)
            return (
                f"{cart_text}\n\n"
                "Say **checkout** to place your order, or keep adding items from the menu."
            )

        if state == ConversationState.AWAITING_ADDRESS:
            return "📍 I need your delivery address to proceed. Please share your address or location."

        if state == ConversationState.CONFIRMING_ORDER:
            return "Please reply **yes** to confirm your order, or **no** to cancel."

        if state == ConversationState.AWAITING_PAYMENT:
            return (
                "💳 **How would you like to pay?**\n"
                "1️⃣ **UPI QR** — Reply **1** or **upi**\n"
                "2️⃣ **Pay Later (COD)** — Reply **2** or **cod**"
            )

        if state == ConversationState.ORDER_PLACED:
            return "📦 Your order has been placed! Say **track my order** to check status, or **start over** to begin a new order."

        if state in (ConversationState.SEARCHING, ConversationState.RESTAURANT_SELECTED):
            if session.search_results:
                lines = []
                for idx, r in enumerate(session.search_results[:5], start=1):
                    lines.append(f"{idx}. **{r.get('name', 'Unknown')}**")
                return (
                    "Here are the restaurants I found:\n\n"
                    + "\n".join(lines)
                    + "\n\nReply with a number to see the menu, or search for something else."
                )

        # Default: generic help with location context
        return self._automation_fallback_reply(location)

    def _is_restaurant_selection_request(self, message: str, search_results: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        text = (message or "").strip().lower()
        if not text or not search_results:
            return None

        # Number-based selection: "1", "2", etc.
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(search_results):
                return search_results[idx]

        # Name-based selection: "Domino's"
        for rest in search_results:
            name = (rest.get("name") or "").lower()
            if name and (text == name or text in name or name in text):
                return rest
        return None

    def _extract_search_keyword(self, message: str) -> str:
        text = (message or "").strip().lower()
        # If user asks for generic nearby options, empty keyword gives broad results.
        generic_markers = ("near me", "nearby", "restaurants near", "find restaurants")
        if any(m in text for m in generic_markers):
            for specific in ("pizza", "biryani", "burger", "chinese", "south indian", "cafe", "vegetarian", "veg", "non veg", "healthy"):
                if specific in text:
                    return specific
            return ""

        m = re.search(r"(?:show|find|search)\s+(?:me\s+)?(.+?)\s+restaurants?", text)
        if m:
            return m.group(1).strip()

        for specific in ("pizza", "biryani", "burger", "chinese", "south indian", "cafe", "vegetarian", "veg", "non veg", "healthy"):
            if specific in text:
                return specific
        return ""

    def _extract_location_override(self, message: str) -> Optional[str]:
        """
        Extract explicit location from user text.
        Examples:
        - "find biryani in vijayawada"
        - "show pizza near koramangala"
        """
        text = (message or "").strip()
        if not text:
            return None

        lower = text.lower()
        if "near me" in lower or "nearby" in lower:
            return None

        patterns = [
            r"\bin\s+([a-zA-Z][a-zA-Z\s,.-]{2,60})$",
            r"\bnear\s+([a-zA-Z][a-zA-Z\s,.-]{2,60})$",
        ]
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                loc = m.group(1).strip(" .,-")
                if loc:
                    return loc
        return None

    def _extract_restaurants_from_tool_result(self, tool_result: List[Any]) -> List[Dict[str, str]]:
        """
        Extract minimal restaurant records from MCP tool output.
        Works across different payload shapes returned by the remote MCP server.
        """
        extracted: List[Dict[str, str]] = []

        def _walk(node):
            if isinstance(node, dict):
                rid = (
                    node.get("restaurant_id")
                    or node.get("id")
                    or node.get("res_id")
                    or node.get("restaurantId")
                    or node.get("resId")
                    or node.get("entity_id")
                )
                name = (
                    node.get("name")
                    or node.get("restaurant_name")
                    or node.get("title")
                    or node.get("display_name")
                )
                if rid and name:
                    rating = node.get("rating") or node.get("avg_rating") or node.get("aggregate_rating")
                    delivery = (
                        node.get("delivery_time")
                        or node.get("eta")
                        or node.get("delivery_eta")
                        or node.get("delivery_time_in_minutes")
                        or node.get("sla")
                    )
                    cuisines = (
                        node.get("cuisines")
                        or node.get("cuisine_string")
                        or node.get("cuisine")
                        or node.get("cuisine_name")
                        or ""
                    )
                    if isinstance(cuisines, list):
                        cuisines = ", ".join(str(c) for c in cuisines)
                    extracted.append({
                        "id": str(rid),
                        "name": str(name),
                        "rating": str(rating) if rating is not None else "",
                        "delivery_time": str(delivery) if delivery is not None else "",
                        "cuisines": str(cuisines),
                    })
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        for chunk in tool_result or []:
            if isinstance(chunk, str):
                try:
                    _walk(json.loads(chunk))
                except Exception:
                    continue
            else:
                _walk(chunk)

        # Deduplicate by id while preserving order
        seen = set()
        uniq = []
        for r in extracted:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            uniq.append(r)
        return uniq

    def _extract_menu_items_from_tool_result(self, tool_result: List[Any]) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []

        def _walk(node):
            if isinstance(node, dict):
                name = node.get("name") or node.get("item_name")
                price = node.get("price") or node.get("final_price") or node.get("display_price")
                size = node.get("size") or node.get("variant_name") or ""
                variant_id = node.get("variant_id") or ""
                item_id = node.get("item_id") or ""
                is_veg = "veg" in str(node.get("item_tags", node.get("description", ""))).lower()
                if name and price is not None:
                    items.append({
                        "name": str(name),
                        "price": str(price),
                        "size": str(size) if size else "",
                        "variant_id": str(variant_id) if variant_id else "",
                        "item_id": str(item_id) if item_id else "",
                        "is_veg": is_veg,
                    })
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        for chunk in tool_result or []:
            if isinstance(chunk, str):
                try:
                    _walk(json.loads(chunk))
                except Exception:
                    continue
            else:
                _walk(chunk)

        dedup: List[Dict[str, str]] = []
        seen = set()
        for item in items:
            key = (item["name"].lower(), item["price"], item["size"].lower())
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)
        return dedup

    # ── Ordering helpers ────────────────────────────────────────────────────────

    def _is_checkout_request(self, message: str) -> bool:
        text = (message or "").strip().lower()
        return text in {"checkout", "place order", "confirm order", "order now", "proceed", "buy now"}

    def _is_confirm_yes(self, message: str) -> bool:
        text = (message or "").strip().lower()
        return text in {"yes", "yeah", "yep", "confirm", "place", "ok", "okay", "sure", "go ahead", "y"}

    def _is_confirm_no(self, message: str) -> bool:
        text = (message or "").strip().lower()
        return text in {"no", "nope", "cancel", "n", "nah", "stop"}

    def _parse_order_items(self, message: str, menu_items: List[Dict[str, str]]) -> List[CartItem]:
        """Parse natural language like '1 farmhouse medium and 2 garlic breadsticks' into CartItems."""
        cart_items: List[CartItem] = []
        if not menu_items:
            return cart_items

        def _match_item(query: str) -> tuple:
            """Find best matching menu item for a query string. Returns (menu_item, score)."""
            query_lower = query.strip().lower()
            if not query_lower:
                return None, 0.0
            query_words = set(re.split(r'\s+', query_lower))
            best_match = None
            best_score = 0.0
            for menu_item in menu_items:
                name_lower = (menu_item.get('name') or '').lower()
                size_lower = (menu_item.get('size') or '').lower()
                full_text = f"{name_lower} {size_lower}".strip()
                full_words = set(re.split(r'\s+', full_text))
                word_score = len(query_words & full_words) / max(len(query_words), 1)
                substr_score = 0.0
                if query_lower in name_lower or name_lower in query_lower:
                    substr_score = 0.8
                else:
                    for qw in query_words:
                        if len(qw) >= 4 and qw in name_lower:
                            substr_score = max(substr_score, 0.5)
                        elif len(qw) >= 4 and any(qw in w for w in full_words):
                            substr_score = max(substr_score, 0.4)
                score = max(word_score, substr_score)
                if score > best_score and score >= 0.3:
                    best_score = score
                    best_match = menu_item
            return best_match, best_score

        parts = re.split(r'\s+and\s+|,\s*', message.strip(), flags=re.IGNORECASE)
        for part in parts:
            part = part.strip().rstrip('.')
            if not part:
                continue

            qty_match = re.match(r'^(\d+)\s+(.*)', part.strip())
            if qty_match:
                # Try matching the FULL part first (e.g. "2 poori" is a menu item name)
                full_match, full_score = _match_item(part)
                # Also try the split version (qty=2, query="poori")
                split_qty = int(qty_match.group(1))
                split_query = qty_match.group(2).strip()
                split_match, split_score = _match_item(split_query)

                # If full match finds an item whose name starts with the number,
                # prefer it (e.g. "2 Poori" is the actual item name, not qty=2 of "Poori")
                if full_match and full_score >= split_score:
                    full_name_lower = (full_match.get('name') or '').lower()
                    if full_name_lower.startswith(str(split_qty)):
                        # The number is part of the item name, not a quantity
                        best_match = full_match
                        qty = 1
                    else:
                        # Number is a quantity prefix
                        best_match = split_match or full_match
                        qty = split_qty
                elif split_match:
                    best_match = split_match
                    qty = split_qty
                else:
                    best_match = full_match
                    qty = 1
            else:
                qty = 1
                best_match, _ = _match_item(part)

            if best_match:
                try:
                    price = int(float(
                        str(best_match.get('price', 0))
                        .replace('\u20b9', '').replace('Rs', '').replace(',', '').strip()
                    ))
                except (ValueError, TypeError):
                    price = 0
                item_id = best_match.get('item_id') or f"item_{abs(hash(best_match['name'])) % 100000}"
                variant_id = best_match.get('variant_id') or ""
                is_veg = best_match.get('is_veg', False)
                if isinstance(is_veg, str):
                    is_veg = is_veg.lower() == 'true'
                cart_items.append(CartItem(
                    item_id=item_id,
                    name=best_match['name'],
                    variant_id=variant_id if variant_id else None,
                    size=best_match.get('size') or None,
                    price=price,
                    quantity=qty,
                    is_veg=bool(is_veg),
                ))
        return cart_items

    def _is_add_to_cart_request(self, message: str, session) -> bool:
        """Return True when user appears to be ordering items from the current menu."""
        if not session.menu_items_map:
            return False
        text = (message or "").strip().lower()
        # "2 margherita", "1 plain phulka and 2 butter phulka"
        if re.search(r'\b\d+\s+\w', text):
            return True
        if any(text.startswith(p) for p in ("order ", "add ", "i want ", "give me ", "get me ", "i'll have ", "i'll take ")):
            return True
        # If user is in BROWSING_MENU state and their message partially matches a menu item name,
        # treat it as an order attempt (e.g. user just types "plain phulka")
        if session.state == ConversationState.BROWSING_MENU:
            for item in session.menu_items_map.values():
                item_name = (item.get("name") or "").lower()
                if item_name and (text in item_name or item_name in text):
                    return True
                # Check if any significant word from user text matches menu item words
                text_words = set(w for w in text.split() if len(w) >= 4)
                item_words = set(w for w in item_name.split() if len(w) >= 4)
                if text_words and item_words and text_words & item_words:
                    return True
        return False

    def _build_cart_with_address_prompt(self, session) -> str:
        """Render cart summary and ask for delivery address."""
        lines = []
        subtotal = 0
        for item in session.cart:
            line_total = item.price * item.quantity
            subtotal += line_total
            size_part = f" ({item.size})" if item.size else ""
            lines.append(f"- {item.name}{size_part} x{item.quantity} - {line_total} rupees")
        delivery_fee = 40
        total = subtotal + delivery_fee
        return (
            "Perfect! Your order:\n"
            + "\n".join(lines)
            + f"\n\nSubtotal: {subtotal} rupees\nDelivery Fee: {delivery_fee} rupees\nTotal: {total} rupees"
            + "\n\n📍 Please share your delivery address."
        )

    def _tool_result_to_text(self, tool_result: List[Any]) -> str:
        parts = []
        for chunk in tool_result or []:
            if isinstance(chunk, str):
                parts.append(chunk)
            else:
                parts.append(json.dumps(chunk))
        text = "\n".join(parts).strip()
        return text[:1800] if text else "No data returned."

    async def _call_with_retry(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ):
        """Single model attempt with 2 internal retries on transient errors."""
        last_err = None
        for attempt in range(2):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools or None,
                    temperature=0.2,
                    max_tokens=MAX_COMPLETION_TOKENS,
                    timeout=MODEL_CALL_TIMEOUT_SECONDS,
                )
            except Exception as e:
                last_err = e
                err_str = str(e)
                # Don't retry auth/quota errors — they won't recover
                if "401" in err_str or "API_KEY" in err_str or "429" in err_str or "rate limit" in err_str.lower() or "quota" in err_str.lower():
                    raise
                if attempt == 0:
                    await asyncio.sleep(1.5)
        raise last_err

    async def process_message(
        self,
        user_id: str,
        message: str,
        user_name: Optional[str] = None,
        user_location: Optional[str] = None,
        filters: Optional["SearchFilters"] = None,
    ) -> Tuple[str, List[str]]:
        """
        Returns (bot_reply, thinking_steps) where thinking_steps is a list of
        short human-readable strings describing what the agent did.
        """
        session = session_service.get_session(user_id, user_name)
        session_service.add_to_history(user_id, "user", message)

        thinking_steps: List[str] = []
        bot_reply = ""

        # If user asks "nearest to me" but we have no location, ask for it
        if self._detect_location_needed(message) and not user_location and not session.current_location:
            bot_reply = (
                "📍 To find restaurants near you, I need your location.\n\n"
                "Please share your **delivery address** or **city name** "
                "(e.g. *Madhapur, Hyderabad* or *Koramangala, Bangalore*).\n\n"
                "You can also click **📍 Share Location** below to use your device GPS."
            )
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["📍 Location required — asked user to share their address"]

        # Persist location in session if provided this turn
        # If location changed, clear cached address_id so it gets re-resolved
        if user_location:
            if session.current_location and user_location != session.current_location:
                session.address_id = None
                logger.info("Location changed from '%s' to '%s' — clearing cached address_id",
                            session.current_location, user_location)
            session.current_location = user_location

        text = (message or "").strip().lower()

        # ── CONFIRMING ORDER: intercept yes/no before other handlers misidentify them ──
        if session.state == ConversationState.CONFIRMING_ORDER:
            _passthrough = {"show cart", "view cart", "cart", "start over", "reset", "new chat"}
            if text not in _passthrough and not self._is_order_tracking_request(text):
                if self._is_confirm_yes(message):
                    if session.cart and session.selected_restaurant_id:
                        rest_name = next(
                            (r["name"] for r in (session.search_results or []) if r.get("id") == session.selected_restaurant_id),
                            "the restaurant"
                        )
                        address = session.pending_address or user_location or session.current_location or "your address"
                        thinking_steps = ["🛒 Creating Zomato cart"]
                        try:
                            # Build items array for create_cart (needs variant_id)
                            cart_items_for_zomato = []
                            for item in session.cart:
                                if item.variant_id:
                                    cart_items_for_zomato.append({
                                        "variant_id": item.variant_id,
                                        "quantity": item.quantity,
                                    })
                                else:
                                    logger.warning("Cart item '%s' missing variant_id — cannot order via Zomato", item.name)

                            if not cart_items_for_zomato:
                                bot_reply = "⚠️ Cart items are missing Zomato variant IDs. Please start over and re-add items from the menu."
                                session_service.add_to_history(user_id, "assistant", bot_reply)
                                session_service.set_last_bot_message(user_id, bot_reply)
                                return bot_reply, ["⚠️ Missing variant IDs"]

                            # Resolve address_id
                            addr_id = session.address_id
                            if not addr_id:
                                addr_id, _ = await self._resolve_address_id(user_id, session)
                            if not addr_id:
                                bot_reply = "⚠️ Could not resolve your delivery address. Please try updating your location and retry."
                                session_service.add_to_history(user_id, "assistant", bot_reply)
                                session_service.set_last_bot_message(user_id, bot_reply)
                                return bot_reply, ["⚠️ No address_id"]

                            payment_type = session.payment_type or "pay_later"
                            res_id = int(session.selected_restaurant_id)

                            # Step 1: Create cart on Zomato
                            thinking_steps.append("📤 Sending cart to Zomato")
                            create_result = await global_zomato_mcp.call_tool("create_cart", {
                                "res_id": res_id,
                                "items": cart_items_for_zomato,
                                "address_id": addr_id,
                                "payment_type": payment_type,
                            })
                            logger.info("create_cart result: %s", str(create_result)[:500])

                            # Parse cart_id from response
                            cart_id = None
                            create_error = None
                            for chunk in create_result or []:
                                if isinstance(chunk, str):
                                    try:
                                        data = json.loads(chunk)
                                        if isinstance(data, dict):
                                            cart_id = data.get("cart_id") or data.get("id")
                                            if data.get("error_message") or data.get("error_code"):
                                                create_error = data.get("error_message") or data.get("error_code")
                                            if not cart_id:
                                                # Walk to find cart_id
                                                def _find_cart_id(node):
                                                    if isinstance(node, dict):
                                                        for k, v in node.items():
                                                            if k in ("cart_id", "id") and isinstance(v, str):
                                                                return v
                                                            r = _find_cart_id(v)
                                                            if r:
                                                                return r
                                                    elif isinstance(node, list):
                                                        for v in node:
                                                            r = _find_cart_id(v)
                                                            if r:
                                                                return r
                                                    return None
                                                cart_id = _find_cart_id(data)
                                    except Exception:
                                        pass

                            if create_error:
                                bot_reply = f"⚠️ Zomato couldn't create the cart: {create_error}\n\nPlease try again or start over."
                                session_service.add_to_history(user_id, "assistant", bot_reply)
                                session_service.set_last_bot_message(user_id, bot_reply)
                                return bot_reply, thinking_steps + [f"⚠️ create_cart error: {create_error}"]

                            if not cart_id:
                                # Cart creation may have returned data without explicit cart_id
                                # Try to use the raw response as cart info
                                bot_reply = (
                                    "⚠️ Cart was created but I couldn't get the cart ID from Zomato.\n\n"
                                    "Response: " + str(create_result)[:300] + "\n\n"
                                    "Please try again or start over."
                                )
                                session_service.add_to_history(user_id, "assistant", bot_reply)
                                session_service.set_last_bot_message(user_id, bot_reply)
                                return bot_reply, thinking_steps + ["⚠️ No cart_id in response"]

                            session.zomato_cart_id = cart_id
                            thinking_steps.append(f"✅ Cart created: {cart_id}")

                            # Step 2: Checkout the cart
                            thinking_steps.append("💳 Checking out on Zomato")
                            checkout_result = await global_zomato_mcp.call_tool("checkout_cart", {
                                "cart_id": cart_id,
                            })
                            logger.info("checkout_cart result: %s", str(checkout_result)[:500])

                            # Parse checkout response
                            checkout_error = None
                            order_info = {}
                            for chunk in checkout_result or []:
                                if isinstance(chunk, str):
                                    try:
                                        data = json.loads(chunk)
                                        if isinstance(data, dict):
                                            if data.get("error_message") or data.get("error_code"):
                                                checkout_error = data.get("error_message") or data.get("error_code")
                                            else:
                                                order_info = data
                                    except Exception:
                                        pass

                            if checkout_error:
                                bot_reply = f"⚠️ Zomato checkout failed: {checkout_error}\n\nYour cart (ID: {cart_id}) is saved. Try again."
                                session_service.add_to_history(user_id, "assistant", bot_reply)
                                session_service.set_last_bot_message(user_id, bot_reply)
                                return bot_reply, thinking_steps + [f"⚠️ checkout error: {checkout_error}"]

                            # Order placed successfully!
                            session_service.update_state(user_id, ConversationState.ORDER_PLACED)
                            order_id = order_info.get("order_id") or order_info.get("id") or cart_id
                            session_service.set_current_order(user_id, str(order_id))
                            session_service.record_past_order(
                                user_id,
                                restaurant_name=rest_name,
                                cuisine=session.preferred_cuisine or "",
                                total=sum(i.price * i.quantity for i in session.cart),
                            )

                            # Build success message from checkout response
                            eta = order_info.get("estimated_delivery") or order_info.get("eta") or "30-40 minutes"
                            total = order_info.get("total") or order_info.get("amount") or sum(i.price * i.quantity for i in session.cart)
                            bot_reply = (
                                f"✅ **Order placed successfully on Zomato!**\n\n"
                                f"📦 Order ID: **{order_id}**\n"
                                f"🍽️ Restaurant: {rest_name}\n"
                                f"💳 Payment: {'UPI QR' if payment_type == 'upi_qr' else 'Pay Later'}\n"
                                f"⏱️ Estimated Delivery: {eta}\n"
                                f"💵 Total: {total} rupees\n\n"
                                "Say **track my order** to check the status. 🛵"
                            )
                            session_service.add_to_history(user_id, "assistant", bot_reply)
                            session_service.set_last_bot_message(user_id, bot_reply)
                            return bot_reply, thinking_steps + ["✅ Order placed on Zomato"]

                        except Exception as e:
                            logger.error("Real Zomato order failed: %s", e, exc_info=True)
                            bot_reply = f"⚠️ Failed to place order on Zomato: {str(e)}\n\nPlease try again or start over."
                            session_service.add_to_history(user_id, "assistant", bot_reply)
                            session_service.set_last_bot_message(user_id, bot_reply)
                            return bot_reply, [f"⚠️ Order error: {e}"]
                    else:
                        bot_reply = "\u26a0\ufe0f Something went wrong — cart or restaurant missing. Please start over."
                        session_service.add_to_history(user_id, "assistant", bot_reply)
                        session_service.set_last_bot_message(user_id, bot_reply)
                        return bot_reply, ["\u26a0\ufe0f Order failed"]
                elif self._is_confirm_no(message):
                    session_service.update_state(user_id, ConversationState.ORDERING)
                    bot_reply = "\u274c Order cancelled. Your cart is still saved — say **checkout** to try again or **start over** to reset."
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, ["\u274c Order cancelled"]
                else:
                    bot_reply = "Please reply **yes** to confirm the order or **no** to cancel."
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, ["\u23f3 Awaiting order confirmation"]

        # ── AWAITING ADDRESS: capture address before plain_location_message misidentifies it ──
        if session.state == ConversationState.AWAITING_ADDRESS:
            _addr_skip = {"show cart", "view cart", "cart", "start over", "reset", "new chat",
                          "checkout", "yes", "no", "__gps__"}

            # Don't capture plain numbers as addresses — they're likely restaurant selections
            if text.isdigit() and session.search_results:
                # Fall through to restaurant selection handler below
                pass
            elif text in _addr_skip or self._is_restaurant_search_request(message) or self._is_order_tracking_request(text):
                # Fall through to other handlers
                pass
            else:
                # Detect GPS or explicit address input
                gps_address = user_location if (user_location and user_location != session.current_location) else None
                # Extract actual address from "My location is: X" format
                address_text = message.strip()
                loc_prefix = re.match(r'^(?:my\s+)?location\s+is[:\s]+(.+)', address_text, re.IGNORECASE)
                if loc_prefix:
                    address_text = loc_prefix.group(1).strip()
                address = gps_address or address_text
                if address:
                    session_service.set_address(user_id, address)
                    session_service.update_state(user_id, ConversationState.AWAITING_PAYMENT)
                    subtotal = session_service.get_cart_total(user_id)
                    delivery_fee = 40
                    total = subtotal + delivery_fee
                    bot_reply = (
                        f"📍 Address confirmed: **{address}**\n\n"
                        f"💵 Subtotal: {subtotal} rupees + {delivery_fee} delivery = **{total} rupees**\n\n"
                        "💳 **How would you like to pay?**\n"
                        "1️⃣ **UPI QR** — Pay via UPI\n"
                        "2️⃣ **Pay Later** — Cash on delivery\n\n"
                        "Reply **1** or **upi** for UPI, or **2** or **cod** for Pay Later."
                    )
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, ["📍 Address received, asking payment method"]

        # ── AWAITING PAYMENT: capture payment method selection ──
        if session.state == ConversationState.AWAITING_PAYMENT:
            _pay_skip = {"show cart", "view cart", "cart", "start over", "reset", "new chat"}
            if text not in _pay_skip and not self._is_order_tracking_request(text):
                payment_type = None
                if text in {"1", "upi", "upi qr", "upi_qr"}:
                    payment_type = "upi_qr"
                elif text in {"2", "cod", "cash", "pay later", "pay_later", "cash on delivery"}:
                    payment_type = "pay_later"

                if payment_type:
                    session.payment_type = payment_type
                    session_service.update_state(user_id, ConversationState.CONFIRMING_ORDER)
                    subtotal = session_service.get_cart_total(user_id)
                    delivery_fee = 40
                    total = subtotal + delivery_fee
                    pay_label = "UPI QR" if payment_type == "upi_qr" else "Pay Later (COD)"
                    address = session.pending_address or user_location or session.current_location or "your address"
                    bot_reply = (
                        f"📋 **Order Summary**\n\n"
                        f"📍 Address: {address}\n"
                        f"💳 Payment: {pay_label}\n"
                        f"💵 Total: {total} rupees\n\n"
                        "Shall I place this order? Reply **yes** to confirm or **no** to cancel."
                    )
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, [f"💳 Payment: {pay_label}, awaiting confirmation"]
                else:
                    bot_reply = (
                        "Please choose a payment method:\n"
                        "1️⃣ **UPI QR** — Reply **1** or **upi**\n"
                        "2️⃣ **Pay Later (COD)** — Reply **2** or **cod**"
                    )
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, ["⏳ Awaiting payment selection"]

        if user_location and self._is_location_update_only(message):
            bot_reply = (
                f"📍 Got it — I’ll search near **{user_location}**.\n\n"
                "Tell me what you want, for example:\n"
                "• `show me biryani restaurants`\n"
                "• `find top rated pizza places`\n"
                "• `show vegetarian options`"
            )
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["📍 Location saved"]

        if self._is_plain_location_message(message):
            new_loc = self._normalize_location_text(message)
            session.current_location = new_loc
            bot_reply = (
                f"📍 Location updated to **{new_loc}**.\n\n"
                "Now tell me what you want to find, for example:\n"
                "• `find biryani restaurants`\n"
                "• `show pizza places near me`"
            )
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["📍 Location updated from user message"]

        if self._is_smalltalk_request(message):
            bot_reply = self._smalltalk_reply(session.current_location, session)
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["💬 Instant response mode"]

        text = (message or "").strip().lower()
        if text in {"show cart", "view cart", "cart"}:
            bot_reply = self._render_cart(user_id)
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["🛒 Cart status"]

        if self._is_order_tracking_request(text):
            bot_reply = await self._render_order_status_async(user_id)
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["📦 Order status"]

        if self._is_cancel_order_request(text):
            bot_reply = await self._cancel_order(user_id)
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["❌ Cancel order"]

        if text in {"start over", "reset", "new chat"}:
            session_service.reset_session(user_id)
            bot_reply = "✅ Conversation reset. Share a location or ask for restaurants to begin."
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["♻️ Session reset"]

        if self._is_first_menu_request(message):
            if session.selected_restaurant_id:
                try:
                    thinking_steps.append("📋 Loading menu")
                    # Resolve address_id for menu fetch
                    addr_id = session.address_id
                    if not addr_id:
                        addr_id, _ = await self._resolve_address_id(user_id, session)
                    if not addr_id:
                        bot_reply = (
                            "📍 I need a delivery address to load the menu.\n\n"
                            "Please share your location first."
                        )
                        session_service.add_to_history(user_id, "assistant", bot_reply)
                        session_service.set_last_bot_message(user_id, bot_reply)
                        return bot_reply, ["📍 Address needed for menu"]
                    menu_items = await self._fetch_restaurant_menu(
                        session.selected_restaurant_id, addr_id
                    )
                    if menu_items:
                        session.menu_items_map = {str(i): item for i, item in enumerate(menu_items[:20])}
                        session_service.update_state(user_id, ConversationState.BROWSING_MENU)
                        rest_name = session.selected_restaurant_name or "your selected restaurant"
                        bot_reply = self._format_menu_list(menu_items, rest_name)
                    else:
                        bot_reply = (
                            "📋 **Menu for your selected restaurant**\n\n"
                            "I couldn't load menu items right now. The restaurant may be closed or the menu is unavailable.\n\n"
                            "Try selecting a different restaurant."
                        )
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, thinking_steps + ["✅ Menu loaded (direct mode)"]
                except Exception as menu_err:
                    logger.error("Direct menu fetch failed: %s", menu_err)
                    bot_reply = (
                        "⚠️ I couldn't fetch the menu right now.\n\n"
                        "Please try searching restaurants again, then ask for the menu."
                    )
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, thinking_steps + ["⚠️ Menu fetch failed"]
            if not session.selected_restaurant_id:
                bot_reply = (
                    "📋 I don’t have a selected restaurant yet.\n\n"
                    "Please search first, for example:\n"
                    "• `show me biryani restaurants`\n"
                    "Then ask: `show menu of first restaurant`."
                )
                session_service.add_to_history(user_id, "assistant", bot_reply)
                session_service.set_last_bot_message(user_id, bot_reply)
                return bot_reply, ["📋 Menu request needs a selected restaurant"]

        selected = self._is_restaurant_selection_request(message, session.search_results or [])
        if selected:
            session_service.set_selected_restaurant(user_id, selected["id"])
            session.selected_restaurant_name = selected.get("name", "")
            try:
                thinking_steps.append("📋 Loading menu")
                # Resolve address_id for menu fetch
                addr_id = session.address_id
                if not addr_id:
                    addr_id, _ = await self._resolve_address_id(user_id, session)
                if addr_id:
                    menu_items = await self._fetch_restaurant_menu(selected["id"], addr_id)
                else:
                    menu_items = []
                if menu_items:
                    # Store menu items in session for later cart operations
                    session.menu_items_map = {str(i): item for i, item in enumerate(menu_items[:20])}
                    session_service.update_state(user_id, ConversationState.BROWSING_MENU)
                    bot_reply = self._format_menu_list(menu_items, selected["name"])
                else:
                    bot_reply = (
                        f"Great choice! I selected **{selected['name']}**.\n\n"
                        "I couldn't load the menu right now. The restaurant may be closed.\n"
                        "Try selecting a different restaurant."
                    )
                session_service.add_to_history(user_id, "assistant", bot_reply)
                session_service.set_last_bot_message(user_id, bot_reply)
                return bot_reply, thinking_steps + ["✅ Menu loaded for selected restaurant"]
            except Exception as selection_err:
                logger.error("Selection menu fetch failed: %s", selection_err)
                bot_reply = (
                    f"I selected **{selected['name']}**, but couldn't load the menu right now.\n\n"
                    "Please try again in a few seconds."
                )
                session_service.add_to_history(user_id, "assistant", bot_reply)
                session_service.set_last_bot_message(user_id, bot_reply)
                return bot_reply, thinking_steps + ["⚠️ Failed to load menu for selected restaurant"]

        # ── CHECKOUT intent ───────────────────────────────────────────────────────
        if self._is_checkout_request(message):
            if not session.cart:
                bot_reply = "🛒 Your cart is empty. Browse a restaurant's menu and add some items first."
                session_service.add_to_history(user_id, "assistant", bot_reply)
                session_service.set_last_bot_message(user_id, bot_reply)
                return bot_reply, ["🛒 Empty cart"]
            session_service.update_state(user_id, ConversationState.AWAITING_ADDRESS)
            session = session_service.get_session(user_id)
            bot_reply = self._build_cart_with_address_prompt(session)
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["🛒 Checkout initiated"]

        # ── ADD TO CART ────────────────────────────────────────────────────────────
        menu_items_list = list(session.menu_items_map.values()) if session.menu_items_map else []
        if menu_items_list and self._is_add_to_cart_request(message, session):
            cart_items = self._parse_order_items(message, menu_items_list)
            if cart_items:
                for item in cart_items:
                    session_service.add_to_cart(user_id, item)
                session_service.update_state(user_id, ConversationState.AWAITING_ADDRESS)
                session = session_service.get_session(user_id)
                bot_reply = self._build_cart_with_address_prompt(session)
                session_service.add_to_history(user_id, "assistant", bot_reply)
                session_service.set_last_bot_message(user_id, bot_reply)
                return bot_reply, ["🛒 Items added to cart"]
            else:
                # User tried to order but items didn't match the current menu
                rest_name = session.selected_restaurant_name or "the restaurant"
                available = [item.get("name", "") for item in menu_items_list[:8]]
                available_str = "\n".join(f"• {name}" for name in available if name)
                bot_reply = (
                    f"😕 I couldn't find those items on the **{rest_name}** menu.\n\n"
                    f"Here's what's available:\n{available_str}\n\n"
                    "Please pick from the menu above, e.g. *1 Plain Phulka and 2 Butter Phulka*."
                )
                session_service.add_to_history(user_id, "assistant", bot_reply)
                session_service.set_last_bot_message(user_id, bot_reply)
                return bot_reply, ["⚠️ Items not found on menu"]

        explicit_location = self._extract_location_override(message)
        if explicit_location:
            session.current_location = self._normalize_location_text(explicit_location)
            thinking_steps.append(f"📍 Switched search location to {session.current_location}")

        # Extract intent (cuisine, budget, veg preference) and save to session
        intent = self._extract_intent(message)
        session_service.set_preferences(
            user_id,
            cuisine=intent["cuisine"],
            budget=intent["budget"],
            veg=intent["veg"],
        )
        # Apply UI-level filters on top of intent
        effective_veg = (
            True if (filters and filters.veg_only) else
            False if (filters and filters.non_veg_only) else
            session.veg_preference
        )
        effective_budget = intent["budget"] or session.budget

        # If user wants restaurants but hasn't shared a location, ask for one
        if not session.current_location and self._is_restaurant_search_request(message):
            keyword = self._extract_search_keyword(message)
            cuisine_hint = f" **{keyword}**" if keyword else ""
            bot_reply = (
                f"📍 To find{cuisine_hint} restaurants, I need your location.\n\n"
                "Please share your **city or area** (e.g. *Koramangala, Bangalore* or *Madhapur, Hyderabad*), "
                "or tap **📍 Share My Location** below."
            )
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["📍 Location required for restaurant search"]

        if session.current_location and self._is_restaurant_search_request(message):
            try:
                keyword = intent.get("cuisine") or self._extract_search_keyword(message)

                # Log available MCP tool names once — helps debug tool-name mismatches
                tool_names = global_zomato_mcp.get_tool_names()
                if tool_names:
                    logger.info("Available MCP tools: %s", tool_names)

                # Resolve Zomato address_id (required by MCP — not a plain location string)
                address_id, addr_display = await self._resolve_address_id(user_id, session)
                if not address_id:
                    bot_reply = (
                        "📍 I couldn't find any delivery addresses in your Zomato account.\n\n"
                        "Please add at least one delivery address in the Zomato app, then try again."
                    )
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, ["📍 No Zomato address found"]

                thinking_steps.append("⚡ Fast search mode")
                thinking_steps.append("🔍 Searching restaurants")

                extracted: List[Dict[str, str]] = []

                # Zomato MCP only accepts: keyword, address_id, page_size
                # (lat/lon/location are NOT accepted — address_id handles geolocation)
                search_attempts = [
                    {"keyword": keyword or None, "address_id": address_id, "page_size": 10},
                    {"keyword": None,            "address_id": address_id, "page_size": 10},
                ]
                mcp_error_msg = None
                for attempt_args in search_attempts:
                    result = await global_zomato_mcp.call_tool("get_restaurants_for_keyword", attempt_args)
                    logger.info("MCP raw result preview: %s", str(result)[:400])
                    # Detect MCP connection/auth failure — surface as an error, not silent empty
                    if result and isinstance(result[0], str):
                        first = result[0].lower()
                        if "unavailable" in first or "timed out" in first or "failed:" in first or "validation error" in first:
                            mcp_error_msg = result[0]
                            logger.warning("MCP returned error string: %s", result[0])
                            break
                    extracted = self._extract_restaurants_from_tool_result(result)
                    logger.info("Extracted %d restaurants from attempt %s", len(extracted), attempt_args)
                    if extracted:
                        break

                # Rank results by keyword relevance so cuisine-matching restaurants appear first
                if extracted and keyword:
                    kw = keyword.lower()
                    def _relevance(r: dict) -> int:
                        name_lower = r.get("name", "").lower()
                        cuisines_lower = r.get("cuisines", "").lower()
                        if kw in name_lower:
                            return 0  # best: keyword in restaurant name
                        if kw in cuisines_lower:
                            return 1  # good: keyword in cuisines
                        return 2  # no match
                    extracted.sort(key=_relevance)
                    logger.info("Sorted %d restaurants by keyword '%s' relevance", len(extracted), keyword)
                if mcp_error_msg:
                    raise Exception(mcp_error_msg)
                if extracted:
                    session_service.set_search_results(user_id, extracted)
                    bot_reply = self._format_restaurant_list(
                        extracted, keyword, session.current_location, effective_veg, effective_budget
                    )
                    session_service.add_to_history(user_id, "assistant", bot_reply)
                    session_service.set_last_bot_message(user_id, bot_reply)
                    return bot_reply, thinking_steps + ["✅ Search complete (direct mode)"]

                # No results even with broad search
                addr_note = f"\n\n_(Searched near: {addr_display})_" if addr_display else ""
                bot_reply = (
                    f"😕 I couldn't find matching restaurants near your saved address.{addr_note}\n\n"
                    "Try a different cuisine, or add a delivery address closer to where you want to order from in your Zomato app."
                )
                session_service.add_to_history(user_id, "assistant", bot_reply)
                session_service.set_last_bot_message(user_id, bot_reply)
                return bot_reply, thinking_steps + ["⚠️ No results"]
            except Exception as search_err:
                logger.error("Direct search failed: %s", search_err)
                bot_reply = (
                    f"⚠️ I had trouble reaching the Zomato service right now.\n\n"
                    "Please try again in a moment, or check that your Zomato account has a saved delivery address."
                )
                session_service.add_to_history(user_id, "assistant", bot_reply)
                session_service.set_last_bot_message(user_id, bot_reply)
                return bot_reply, ["⚠️ MCP search error"]

        # ── Catch-all: user is trying to order items but menu session was lost ────
        if re.search(r'\b\d+\s+\w', text) and not session.menu_items_map:
            loc = session.current_location or "your area"
            bot_reply = (
                "😕 It looks like you're trying to order items, but your session has expired "
                "and the menu is no longer loaded.\n\n"
                f"Please search for a restaurant again (e.g. *show me restaurants in {loc}*), "
                "select one, and then place your order."
            )
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["⚠️ Session expired — menu data lost"]

        if self.automation_only:
            bot_reply = self._state_aware_fallback(session, user_id)
            session_service.add_to_history(user_id, "assistant", bot_reply)
            session_service.set_last_bot_message(user_id, bot_reply)
            return bot_reply, ["🤖 Automation-only mode"]

        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set in your backend `.env` file.")

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=GEMINI_BASE_URL,
                max_retries=0,
            )

            tools = []
            try:
                tools = await asyncio.wait_for(self.get_tools(), timeout=13)
                thinking_steps.append(f"🤖 {len(tools)} Zomato tools available" if tools else "⚠️ Zomato MCP unavailable — will answer from knowledge")
            except asyncio.TimeoutError:
                thinking_steps.append("⚠️ Zomato MCP timed out — answering without live data")

            location_ctx = user_location or session.current_location
            messages = [{"role": "system", "content": self._build_system_prompt(location_ctx, filters)}]

            # Limit history to avoid hitting token limits
            for msg in session.conversation_history[-MAX_HISTORY_MESSAGES:-1]:
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"][:MAX_HISTORY_CHARS]})

            messages.append({"role": "user", "content": message[:1200]})

            # Try models in fallback order
            last_model_err = None
            for model in self._models:
                if model in self._disabled_models:
                    continue
                try:
                    thinking_steps.append(f"⚡ Using model: {model}")
                    loop_count = 0

                    while loop_count < MAX_TOOL_LOOPS:  # prevent runaway tool loops and excess API usage
                        loop_count += 1
                        response = await self._call_with_retry(client, model, messages, tools)
                        resp_msg = response.choices[0].message

                        if not resp_msg.tool_calls:
                            bot_reply = resp_msg.content or "Done."
                            break

                        # Build assistant turn with tool calls
                        msg_dict = {"role": "assistant", "content": resp_msg.content}
                        msg_dict["tool_calls"] = [
                            {
                                "id": t.id,
                                "type": "function",
                                "function": {
                                    "name": t.function.name,
                                    "arguments": t.function.arguments,
                                }
                            }
                            for t in resp_msg.tool_calls
                        ]
                        messages.append(msg_dict)

                        # Execute each tool call
                        for tcall in resp_msg.tool_calls:
                            name = tcall.function.name
                            label = TOOL_LABELS.get(name, f"🔧 Running {name}")
                            thinking_steps.append(label)
                            try:
                                args = json.loads(tcall.function.arguments or "{}")
                                if not isinstance(args, dict):
                                    args = {}
                                logger.info("Tool call: %s %s", name, args)
                                result = await global_zomato_mcp.call_tool(name, args)
                                thinking_steps.append(f"✅ {label} — done")
                                if name == "get_restaurants_for_keyword":
                                    extracted = self._extract_restaurants_from_tool_result(result)
                                    if extracted:
                                        session_service.set_search_results(user_id, extracted)
                                        thinking_steps.append(f"📌 Stored {len(extracted)} restaurants for quick follow-up")
                            except Exception as tool_err:
                                logger.error("Tool %s failed: %s", name, tool_err)
                                result = [f"Tool error: {tool_err}"]
                                thinking_steps.append(f"⚠️ {label} — failed, continuing")

                            res_str = json.dumps(result)
                            if len(res_str) > MAX_TOOL_RESULT_CHARS:
                                res_str = res_str[:MAX_TOOL_RESULT_CHARS] + "... (truncated)"

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tcall.id,
                                "name": name,
                                "content": res_str,
                            })

                    # Reached here without exception → model worked
                    last_model_err = None
                    break

                except Exception as model_err:
                    err_str = str(model_err)
                    logger.warning("Model %s failed: %s", model, err_str)
                    last_model_err = model_err
                    thinking_steps.append(f"⚠️ {model} unavailable, trying next model…")

                    if "401" in err_str or "API_KEY" in err_str:
                        break  # no point trying other models with a bad key
                    if "model_decommissioned" in err_str or "decommissioned" in err_str.lower():
                        self._disabled_models.add(model)
                        thinking_steps.append(f"⚠️ Disabled deprecated model: {model}")

                    await asyncio.sleep(0.5)

            if last_model_err:
                raise last_model_err

        except Exception as e:
            err_msg = str(e)
            logger.error("Agent error: %s\n%s", e, traceback.format_exc())

            if "401" in err_msg or "API_KEY" in err_msg or "invalid_api_key" in err_msg.lower():
                bot_reply = (
                    "🔑 **API Key Error** — The API key is missing or invalid.\n\n"
                    "Please add your key to `backend/.env`:\n"
                    "`GEMINI_API_KEY=your_key_here`\n\n"
                    "For Gemini keys, create one in **Google AI Studio**."
                )
            elif "api key not valid" in err_msg.lower() or "invalid argument" in err_msg.lower():
                bot_reply = (
                    "🔑 **Gemini key issue** — The key or request config looks invalid.\n\n"
                    "Check `GEMINI_API_KEY` in `backend/.env`, and keep `GEMINI_MODELS=gemini-2.0-flash,gemini-1.5-flash`."
                )
            elif "429" in err_msg or "rate limit" in err_msg.lower() or "quota" in err_msg.lower():
                bot_reply = (
                    "⏳ **Rate limit reached** — The API is temporarily busy.\n\n"
                    "Please wait a few seconds and try again. "
                    "If this keeps happening, the free tier quota may be exhausted."
                )
            elif (
                "model_decommissioned" in err_msg
                or "decommissioned" in err_msg.lower()
                or ("model" in err_msg.lower() and "not found" in err_msg.lower())
            ):
                bot_reply = (
                    "🧩 **Model config issue** — One or more configured models are deprecated.\n\n"
                    "Update your model list in `backend/.env` (`GEMINI_MODELS`) "
                    "with currently supported model IDs."
                )
            elif "connection" in err_msg.lower() or "timeout" in err_msg.lower():
                bot_reply = (
                    "🌐 **Connection issue** — Could not reach the AI service.\n\n"
                    "Please check your internet connection and try again."
                )
            elif "GEMINI_API_KEY is not set" in err_msg:
                bot_reply = (
                    "⚙️ **Setup required** — No API key found.\n\n"
                    "Add your API key to `backend/.env`:\n"
                    "`GEMINI_API_KEY=your_key_here`"
                )
            else:
                bot_reply = (
                    "😕 **Something went wrong** — I hit an unexpected error.\n\n"
                    "Please try rephrasing your request, or click **🔄 Retry** to try again."
                )
            thinking_steps.append(f"❌ Error: {err_msg[:120]}")

        session_service.add_to_history(user_id, "assistant", bot_reply)
        session_service.set_last_bot_message(user_id, bot_reply)
        return bot_reply, thinking_steps


# Singleton used across routes/services
gemini_agent = GeminiAgent()
