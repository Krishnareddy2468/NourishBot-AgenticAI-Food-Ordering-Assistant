import json
import logging
import os
import re
import base64
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, create_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import app_config
from app.models.schemas import CartItem, ConversationState, SearchFilters
from app.services.session_service import session_service
from app.services.zomato_mcp import global_zomato_mcp

logger = logging.getLogger(__name__)
_active_tool_user_id: ContextVar[str | None] = ContextVar("_active_tool_user_id", default=None)
QR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "..",
    "data",
    "qrcodes",
)


class _EmptyArgs(BaseModel):
    pass


class LangGraphFoodAgent:
    def __init__(self):
        self._compiled_graphs: dict[str, Any] = {}
        self._tool_cache: list[StructuredTool] = []

    def _build_system_prompt(self, user_location: str | None, filters: SearchFilters | None) -> str:
        location_text = user_location or "unknown"
        filter_text = "none"
        if filters:
            filter_text = (
                f"veg_only={filters.veg_only}, non_veg_only={filters.non_veg_only}, "
                f"min_rating={filters.min_rating}, max_distance_km={filters.max_distance_km}"
            )
        return (
            "You are a food ordering assistant.\n"
            "Rules:\n"
            "1) Use MCP tools for restaurants/menu/order data. Never invent data.\n"
            "2) Ask follow-up questions when required fields are missing.\n"
            "3) Keep responses short and actionable.\n"
            "4) Confirm before order placement.\n"
            "5) Never include debug/internal reasoning or assumptions in user-visible responses.\n"
            f"Current user location context: {location_text}\n"
            f"Current active filters: {filter_text}\n"
        )

    def _extract_output_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
            return "\n".join(c for c in chunks if c).strip()
        return str(content or "").strip()

    def _extract_thinking_steps(self, messages: list[Any]) -> list[str]:
        steps: list[str] = []
        for msg in messages:
            tool_name = getattr(msg, "name", None)
            if tool_name:
                steps.append(f"Used tool: {tool_name}")
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                if isinstance(call, dict) and call.get("name"):
                    steps.append(f"Planned tool call: {call['name']}")
        deduped = []
        seen = set()
        for step in steps:
            if step in seen:
                continue
            seen.add(step)
            deduped.append(step)
        return deduped[:8]

    def _friendly_provider_error(self, err: Exception) -> tuple[str, list[str]]:
        text = str(err)
        retry_seconds = None
        delay_match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+)s['\"]", text)
        if delay_match:
            retry_seconds = delay_match.group(1)
        lowered = text.lower()
        if "api_key_invalid" in lowered or "api key not found" in lowered or "invalid api key" in lowered:
            return (
                "The Gemini API key is invalid or missing for this backend. Please update `LLM_API_KEY` in `backend/.env` and restart the server.",
                ["Provider API key invalid"],
            )
        if "402" in lowered or "more credits" in lowered or "can only afford" in lowered:
            return (
                "Your LLM provider account does not have enough credits for this request. Add credits or lower `LLM_MAX_TOKENS` in `backend/.env`, then retry.",
                ["Provider credits exhausted"],
            )
        if "503" in lowered or "unavailable" in lowered or "high demand" in lowered or "temporarily overloaded" in lowered:
            if retry_seconds:
                return (
                    f"The LLM provider is under high demand right now. Please retry in about {retry_seconds} seconds.",
                    [f"Provider temporarily unavailable (retry in ~{retry_seconds}s)"],
                )
            return (
                "The LLM provider is under high demand right now. Please try again in a few moments.",
                ["Provider temporarily unavailable"],
            )
        if "rate" in lowered or "quota" in lowered or "resource_exhausted" in lowered or "retryinfo" in lowered:
            if retry_seconds:
                return (
                    f"I'm getting rate-limited by the LLM provider right now. Please retry in about {retry_seconds} seconds.",
                    [f"Provider rate-limited request (retry in ~{retry_seconds}s)"],
                )
            return (
                "I'm getting rate-limited by the LLM provider right now. Please retry in a few moments.",
                ["Provider rate-limited request"],
            )
        return (
            "I couldn't process that request right now due to a provider error. Please try again.",
            ["Provider request failed"],
        )

    def _normalize_location_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip(" .,-"))
        return cleaned.title()

    def _clean_location_candidate(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^(?:for\s+)?(?:the\s+)?(?:address|location)\s+(?:is|to|as)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:is\s+the\s+address|is\s+my\s+address|address)\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(?:my\s+)?(?:delivery\s+)?address\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" .,-")

    def _is_list_saved_addresses_request(self, message: str) -> bool:
        text = (message or "").strip().lower()
        phrases = (
            "list saved addresses",
            "show saved addresses",
            "saved addresses",
            "list addresses",
            "show addresses",
            "my addresses",
            "zomato addresses",
            "list the saved addresses from zomato",
        )
        return any(phrase in text for phrase in phrases)

    def _extract_address_selection(self, message: str) -> Optional[str]:
        text = (message or "").strip()
        if not text:
            return None
        patterns = [
            r"^(?:choose|select|use|set)\s+(?:the\s+)?address\s+(?:as|to)?\s*(.+)$",
            r"^(?:choose|select|use)\s+(.+)\s+as\s+(?:the\s+)?address$",
            r"^(?:deliver\s+to|ship\s+to)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._clean_location_candidate(match.group(1))
        return None

    def _format_saved_addresses(self, addresses: List[Dict[str, Any]]) -> str:
        if not addresses:
            return "I couldn't find any saved Zomato addresses."
        lines = []
        for idx, address in enumerate(addresses[:10], start=1):
            label = address.get("location_name") or address.get("address") or "Unknown address"
            lines.append(f"{idx}. {label}")
        return (
            "Here are your saved Zomato addresses:\n\n"
            + "\n".join(lines)
            + "\n\nReply with `choose address as <name>` or send the address number."
        )

    def _pick_best_saved_address_match(
        self,
        requested: str,
        addresses: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        query = (requested or "").strip().lower()
        if not query:
            return None

        for address in addresses:
            name = str(address.get("location_name") or "").strip().lower()
            if name == query:
                return address

        for address in addresses:
            name = str(address.get("location_name") or "").strip().lower()
            if query in name or name in query:
                return address

        query_words = {w for w in re.split(r"\W+", query) if len(w) >= 3}
        best = None
        best_score = 0
        for address in addresses:
            name_words = {
                w for w in re.split(r"\W+", str(address.get("location_name") or "").lower())
                if len(w) >= 3
            }
            score = len(query_words & name_words)
            if score > best_score:
                best_score = score
                best = address
        return best if best_score >= 1 else None

    def _is_restaurant_search_request(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        search_markers = (
            "restaurant", "restaurants", "near me", "nearby", "nearest",
            "biryani", "pizza", "burger", "chinese", "south indian",
            "north indian", "italian", "mexican", "thai", "continental",
            "cafe", "healthy", "veg", "vegetarian", "non veg",
            "food", "eat", "hungry",
        )
        return any(m in text for m in search_markers)

    def _build_empty_response_fallback(self, session, message: str) -> str:
        if self._is_restaurant_search_request(message):
            if not session.current_location:
                return (
                    "📍 To find nearby restaurants, I need your location.\n\n"
                    "Share your city/area or tap Share My Location."
                )
            return (
                "Tell me the cuisine or restaurant name you want, for example: "
                "`show me pizza restaurants` or `menu from Pizza Hut`."
            )
        if self._is_generic_order_intent(message):
            if not session.current_location:
                return (
                    "📍 First share your location or area, then tell me what you want to eat."
                )
            return (
                "Tell me what you want to order, for example: `show me biryani restaurants`."
            )
        return "I couldn't understand that clearly. Please try a shorter message."

    def _should_handle_as_ordering_message(self, session) -> bool:
        ordering_states = {
            ConversationState.BROWSING_MENU,
            ConversationState.ORDERING,
            ConversationState.AWAITING_ADDRESS,
            ConversationState.AWAITING_PAYMENT,
            ConversationState.CONFIRMING_ORDER,
        }
        return session.state in ordering_states or bool(session.menu_items_map)

    def _is_confirm_yes(self, message: str) -> bool:
        text = (message or "").strip().lower()
        return text in {
            "yes", "yeah", "yep", "confirm", "confirm order", "confirm an order",
            "place", "place order", "ok", "okay", "sure", "go ahead", "y",
        }

    def _is_confirm_no(self, message: str) -> bool:
        text = (message or "").strip().lower()
        return text in {"no", "nope", "cancel", "n", "nah", "stop"}

    def _is_tracking_request(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        if text in {"track my order", "track order", "order status", "track", "track my current order"}:
            return True
        return ("track" in text and "order" in text) or ("order" in text and "status" in text)

    def _is_show_all_active_orders_request(self, message: str) -> bool:
        text = (message or "").strip().lower()
        triggers = {
            "show all active orders",
            "show active orders",
            "list active orders",
            "list my active orders",
            "show my active orders",
            "all active orders",
        }
        return text in triggers

    def _extract_orders_from_tracking_data(self, data: Any) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            order_tracking = data.get("order_tracking")
            if isinstance(order_tracking, dict):
                items = order_tracking.get("order_tracking_items")
                if isinstance(items, list):
                    orders = [o for o in items if isinstance(o, dict)]
            if not orders:
                for key in ("orders", "active_orders", "order_tracking_items"):
                    value = data.get(key)
                    if isinstance(value, list) and value:
                        orders = [o for o in value if isinstance(o, dict)]
                        break
            if not orders and (data.get("order_id") or data.get("status") or data.get("order_status")):
                orders = [data]
        elif isinstance(data, list):
            orders = [o for o in data if isinstance(o, dict)]
        return orders

    def _is_ordering_flow_active(self, session) -> bool:
        active_states = {
            ConversationState.BROWSING_MENU,
            ConversationState.ORDERING,
            ConversationState.AWAITING_ADDRESS,
            ConversationState.AWAITING_PAYMENT,
            ConversationState.CONFIRMING_ORDER,
        }
        return session.state in active_states or bool(session.cart)

    def _normalize_tracking_status(self, order: Dict[str, Any]) -> str:
        status = str(order.get("order_status") or order.get("status") or "placed")
        paid_flag = order.get("is_order_paid")
        paid = paid_flag is True or str(paid_flag).strip().lower() == "true"
        lowered = status.lower()
        if paid and "payment is incomplete" in lowered:
            return "Order placed and payment confirmed"
        return status

    def _pick_tracked_order(self, orders: List[Dict[str, Any]], session) -> Optional[Dict[str, Any]]:
        if not orders:
            return None
        current_id = str(session.current_order_id or "").strip()
        if current_id:
            for order in orders:
                oid = str(order.get("order_id") or order.get("id") or "").strip()
                if oid and oid == current_id:
                    return order

        selected_restaurant = (session.selected_restaurant_name or "").strip().lower()
        if selected_restaurant:
            for order in orders:
                rname = str(order.get("restaurant_name") or "").strip().lower()
                if rname and rname == selected_restaurant:
                    return order

        return orders[0]

    def _parse_payment_type(self, message: str) -> Optional[str]:
        text = (message or "").strip().lower()
        if text in {"1", "upi", "upi qr", "upi_qr", "gpay", "phonepe", "paytm"}:
            return "upi_qr"
        if text in {"2", "cod", "cash", "cash on delivery", "pay later", "pay_later"}:
            return "pay_later"
        return None

    def _parse_order_items(self, message: str, menu_items: List[Dict[str, str]]) -> List[CartItem]:
        cart_items: List[CartItem] = []
        if not menu_items:
            return cart_items

        def _normalize_food_text(value: str) -> str:
            text = (value or "").lower()
            text = text.replace("&", " and ")
            text = re.sub(r"\b(add|order|please|get|give|me|to|cart|item|items)\b", " ", text)
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def _split_item_parts(raw_message: str) -> List[str]:
            # Protect menu names that contain "and" so we don't split them incorrectly.
            protected = raw_message
            replacements: list[tuple[str, str]] = []
            for idx, item in enumerate(menu_items):
                name = (item.get("name") or "").strip().lower()
                if " and " not in name and "&" not in name:
                    continue
                token = f"__ITEM_AND_{idx}__"
                normalized_name = _normalize_food_text(name)
                if not normalized_name:
                    continue
                words = [re.escape(w) for w in normalized_name.split()]
                # Match either '&' or 'and' between words when users speak/type.
                pattern = r"\b" + r"\s*(?:&|and)?\s*".join(words) + r"\b"
                if re.search(pattern, protected, flags=re.IGNORECASE):
                    protected = re.sub(pattern, token, protected, flags=re.IGNORECASE)
                    replacements.append((token, name))

            parts = re.split(r"\s+and\s+|\s+plus\s+|,\s*", protected, flags=re.IGNORECASE)
            restored_parts: List[str] = []
            for part in parts:
                current = part
                for token, original in replacements:
                    current = current.replace(token, original)
                current = current.strip().rstrip(".")
                if current:
                    restored_parts.append(current)
            return restored_parts

        def _match_item(query: str) -> tuple:
            query_lower = _normalize_food_text(query.strip())
            if not query_lower:
                return None, 0.0
            query_words = set(re.split(r"\s+", query_lower))
            best_match = None
            best_score = 0.0
            for menu_item in menu_items:
                name_lower = _normalize_food_text(menu_item.get("name") or "")
                size_lower = _normalize_food_text(menu_item.get("size") or "")
                full_text = f"{name_lower} {size_lower}".strip()
                full_words = set(re.split(r"\s+", full_text))
                word_score = len(query_words & full_words) / max(len(query_words), 1)
                substr_score = 0.0
                if query_lower in name_lower or name_lower in query_lower:
                    substr_score = 0.8
                else:
                    for query_word in query_words:
                        if len(query_word) >= 4 and query_word in name_lower:
                            substr_score = max(substr_score, 0.5)
                        elif len(query_word) >= 4 and any(query_word in word for word in full_words):
                            substr_score = max(substr_score, 0.4)
                score = max(word_score, substr_score)
                if score > best_score and score >= 0.3:
                    best_score = score
                    best_match = menu_item
            return best_match, best_score

        parts = _split_item_parts(message.strip().lower())
        for part in parts:
            index_match = re.match(r"^(?:item\s*)?(\d+)$", part, flags=re.IGNORECASE)
            if index_match:
                idx = int(index_match.group(1)) - 1
                qty = 1
                best_match = menu_items[idx] if 0 <= idx < len(menu_items) else None
            else:
                qty_match = re.match(r"^(?:order\s+)?(\d+|one|two|three)\s+(.*)", part, flags=re.IGNORECASE)
                if qty_match:
                    qty_token = qty_match.group(1).lower()
                    qty = {"one": 1, "two": 2, "three": 3}.get(qty_token, int(qty_token) if qty_token.isdigit() else 1)
                    query = qty_match.group(2).strip()
                    best_match, _ = _match_item(query)
                else:
                    qty = 1
                    best_match, _ = _match_item(part)

            if best_match:
                try:
                    price = int(float(str(best_match.get("price", 0)).replace("₹", "").replace(",", "").strip()))
                except (ValueError, TypeError):
                    price = 0
                item_id = best_match.get("item_id") or f"item_{abs(hash(best_match['name'])) % 100000}"
                variant_id = best_match.get("variant_id") or ""
                if not variant_id:
                    target_name = _normalize_food_text(best_match.get("name") or "")
                    for candidate in menu_items:
                        candidate_variant = candidate.get("variant_id") or ""
                        if not candidate_variant:
                            continue
                        candidate_name = _normalize_food_text(candidate.get("name") or "")
                        if candidate_name == target_name:
                            variant_id = candidate_variant
                            break
                is_veg = best_match.get("is_veg", False)
                if isinstance(is_veg, str):
                    is_veg = is_veg.lower() == "true"
                cart_items.append(CartItem(
                    item_id=item_id,
                    name=best_match["name"],
                    variant_id=variant_id if variant_id else None,
                    size=best_match.get("size") or None,
                    price=price,
                    quantity=qty,
                    is_veg=bool(is_veg),
                ))
        return cart_items

    def _is_add_to_cart_request(self, message: str, session) -> bool:
        if not session.menu_items_map:
            return False
        text = (message or "").strip().lower()
        if re.search(r"\b\d+\s+\w", text):
            return True
        if session.state in {ConversationState.BROWSING_MENU, ConversationState.AWAITING_PAYMENT}:
            for item in session.menu_items_map.values():
                item_name = (item.get("name") or "").lower()
                if item_name and (text in item_name or item_name in text):
                    return True
                text_words = set(word for word in text.split() if len(word) >= 4)
                item_words = set(word for word in item_name.split() if len(word) >= 4)
                if text_words and item_words and text_words & item_words:
                    return True
        return False

    def _is_generic_order_intent(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        generic_phrases = {
            "order me a food",
            "order food",
            "i need food",
            "i want food",
            "get me food",
            "order me something",
            "i want to order",
            "help me order",
        }
        if text in generic_phrases:
            return True
        return bool(re.fullmatch(r"(?:please\s+)?order(?:\s+me)?(?:\s+some(?:thing)?|\s+a)?\s+food", text))

    def _render_cart(self, session) -> str:
        if not session.cart:
            return "🛒 Your cart is empty."
        lines = []
        subtotal = 0
        for item in session.cart:
            line_total = item.price * item.quantity
            subtotal += line_total
            size_part = f" ({item.size})" if item.size else ""
            lines.append(f"- {item.name}{size_part} x{item.quantity} - {line_total} rupees")
        return "🛒 Your cart:\n" + "\n".join(lines) + f"\n\nSubtotal: {subtotal} rupees"

    def _build_payment_prompt(self, session) -> str:
        subtotal = sum(item.price * item.quantity for item in session.cart)
        return (
            self._render_cart(session)
            + f"\n\n📍 Delivery location: {session.current_location or 'your saved address'}"
            + "\n\n💳 Choose payment:\n"
            + "1️⃣ UPI QR — reply `1` or `upi`\n"
            + "2️⃣ Pay Later (COD) — reply `2` or `cod`"
        )

    def _build_confirm_prompt(self, session) -> str:
        total = sum(item.price * item.quantity for item in session.cart)
        payment_label = "UPI QR" if session.payment_type == "upi_qr" else "Pay Later (COD)"
        return (
            "📋 Order Summary\n\n"
            + self._render_cart(session)
            + f"\n\n📍 Delivery location: {session.current_location or 'your saved address'}"
            + f"\n💳 Payment: {payment_label}"
            + f"\n💵 Total: {total} rupees"
            + "\n\nReply `yes` to confirm or `no` to cancel."
        )

    def _find_first_value(self, node: Any, keys: set[str]) -> Optional[Any]:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in keys and value:
                    return value
                found = self._find_first_value(value, keys)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = self._find_first_value(item, keys)
                if found:
                    return found
        return None

    def _save_checkout_image(self, payload_b64: str, cart_id: str) -> Optional[str]:
        try:
            os.makedirs(QR_DIR, exist_ok=True)
            path = os.path.join(QR_DIR, f"checkout_{cart_id}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(payload_b64))
            return path
        except Exception as exc:
            logger.error("Failed to save checkout QR image for cart %s: %s", cart_id, exc)
            return None

    def _parse_checkout_content(self, content_chunks: list[Any], cart_id: str) -> tuple[dict[str, Any], Optional[str], Optional[str]]:
        order_info: dict[str, Any] = {}
        checkout_error = None
        qr_path = None
        for chunk in content_chunks or []:
            text = getattr(chunk, "text", None)
            if text:
                try:
                    data = json.loads(text)
                except Exception:
                    data = None
                if isinstance(data, dict):
                    checkout_error = data.get("error_message") or data.get("error_code") or checkout_error
                    if not checkout_error:
                        order_info = data
                continue

            image_data = getattr(chunk, "data", None)
            if image_data and not qr_path:
                qr_path = self._save_checkout_image(image_data, cart_id)

        if not qr_path:
            qr_candidate = self._find_first_value(
                order_info,
                {"qr_code", "qrCode", "payment_link", "payment_url", "upi_link", "deeplink"},
            )
            if isinstance(qr_candidate, str) and qr_candidate.startswith("data:image"):
                try:
                    _, encoded = qr_candidate.split(",", 1)
                except ValueError:
                    encoded = None
                if encoded:
                    qr_path = self._save_checkout_image(encoded, cart_id)

        return order_info, checkout_error, qr_path

    def _is_plain_location_message(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        if any(ch.isdigit() for ch in text):
            return False
        if "address" in text or "location" in text:
            return False
        non_location_replies = {
            "yes", "yeah", "yep", "y", "ok", "okay", "sure", "confirm", "go ahead",
            "no", "nope", "nah", "n", "cancel", "stop",
            "hi", "hello", "hey", "thanks", "thank you",
        }
        if text in non_location_replies:
            return False
        action_words = (
            "find", "show", "search", "order", "menu", "track", "cart",
            "restaurant", "restaurants", "pizza", "biryani", "burger",
            "chinese", "south indian", "cafe", "healthy", "veg", "non veg",
            "near", "in ", "of ", "checkout", "status", "start over",
        )
        if any(w in text for w in action_words):
            return False
        if re.fullmatch(r"[a-z\s,]{3,80}", text):
            words = [w for w in re.split(r"[\s,]+", text) if w]
            return 1 <= len(words) <= 6
        return False

    def _should_treat_message_as_location(self, session, message: str) -> bool:
        active_ordering_states = {
            ConversationState.BROWSING_MENU,
            ConversationState.ORDERING,
            ConversationState.AWAITING_ADDRESS,
            ConversationState.AWAITING_PAYMENT,
            ConversationState.CONFIRMING_ORDER,
        }
        if session.state in active_ordering_states:
            return False
        if session.menu_items_map:
            return False
        return self._is_plain_location_message(message)

    def _extract_location_override(self, message: str) -> Optional[str]:
        text = (message or "").strip()
        if not text:
            return None
        lower = text.lower()
        if "near me" in lower or "nearby" in lower:
            return None
        patterns = [
            r"^(?:my\s+)?location\s+is[:\s]+(.+)$",
            r"^(?:update|change|set)\s+(?:the\s+)?location\s+(?:to|as)\s+(.+)$",
            r"^(?:update|change|set)\s+(?:the\s+)?address\s+(?:to|as)\s+(.+)$",
            r"^(?:my\s+)?address\s+is[:\s]+(.+)$",
            r"^(.+?)\s+is\s+the\s+address$",
            r"^(?:move|switch)\s+(?:me\s+)?to\s+(.+)$",
        ]
        for pattern in patterns:
            prefix = re.match(pattern, text, flags=re.IGNORECASE)
            if prefix:
                return self._clean_location_candidate(prefix.group(1))
        patterns = [
            r"\bin\s+([a-zA-Z][a-zA-Z\s,.-]{2,80})$",
            r"\bnear\s+([a-zA-Z][a-zA-Z\s,.-]{2,80})$",
        ]
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip(" .,-")
        return None

    def _extract_search_keyword(self, message: str) -> str:
        text = (message or "").strip().lower()
        stop_phrases = (
            "find", "show", "search", "me", "near me", "nearby", "nearest",
            "restaurant", "restaurants", "shops", "shop", "places", "place",
            "top", "best", "to me", "around me",
        )
        cleaned = text
        for phrase in stop_phrases:
            cleaned = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", cleaned)
        # Rating-only requests (e.g. "restaurants rated 4 stars and above")
        # should not become a fake cuisine keyword.
        cleaned = re.sub(r"\b(?:rated?|rating|stars?|above|and|or|only|minimum|min|at least|over|under)\b", " ", cleaned)
        cleaned = re.sub(r"\b\d(?:\.\d+)?\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        generic_markers = ("near me", "nearby", "restaurants near", "find restaurants", "show restaurants")
        if any(m in text for m in generic_markers):
            for specific in (
                "pizza", "biryani", "burger", "chinese", "south indian", "north indian",
                "cafe", "vegetarian", "veg", "non veg", "healthy", "dessert",
            ):
                if specific in text:
                    return specific
            return ""
        m = re.search(r"(?:show|find|search)\s+(?:me\s+)?(.+?)\s+restaurants?", text)
        if m:
            candidate = m.group(1).strip()
            candidate = re.sub(r"\bnearest\b", " ", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip(" ,.-")
            return candidate
        if cleaned:
            for specific in (
                "pizza", "biryani", "burger", "chinese", "south indian", "north indian",
                "cafe", "vegetarian", "veg", "non veg", "healthy", "dessert",
            ):
                if specific in cleaned:
                    return specific
            return cleaned
        for specific in (
            "pizza", "biryani", "burger", "chinese", "south indian", "north indian",
            "cafe", "vegetarian", "veg", "non veg", "healthy", "dessert",
        ):
            if specific in text:
                return specific
        return ""

    def _extract_min_rating_filter(self, message: str) -> Optional[float]:
        text = (message or "").strip().lower()
        if not text:
            return None

        patterns = [
            r"rated\s*(\d(?:\.\d)?)\s*(?:stars?)?\s*(?:and above|or above|\+)?",
            r"(\d(?:\.\d)?)\s*\+?\s*stars?",
            r"above\s*(\d(?:\.\d)?)",
            r"at least\s*(\d(?:\.\d)?)",
            r"minimum\s*(\d(?:\.\d)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= 5.0:
                return value
        return None

    def _parse_rating_value(self, rating_text: str) -> Optional[float]:
        if rating_text is None:
            return None
        match = re.search(r"\d(?:\.\d+)?", str(rating_text))
        if not match:
            return None
        try:
            value = float(match.group(0))
        except ValueError:
            return None
        if 0.0 <= value <= 5.0:
            return value
        return None

    def _filter_restaurants_by_rating(self, restaurants: List[Dict[str, str]], min_rating: Optional[float]) -> List[Dict[str, str]]:
        if min_rating is None:
            return restaurants
        filtered = []
        for restaurant in restaurants:
            rating_value = self._parse_rating_value(restaurant.get("rating", ""))
            if rating_value is not None and rating_value >= min_rating:
                filtered.append(restaurant)
        return filtered

    def _extract_restaurant_name_from_menu_request(self, message: str) -> Optional[str]:
        text = (message or "").strip()
        if not text:
            return None
        patterns = [
            r"(?:i\s+need|show|get|open|give\s+me)\s+(?:the\s+)?menu\s+(?:from|for|of)\s+(.+)$",
            r"menu\s+(?:from|for|of)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" .,-")
        return None

    def _extract_restaurant_name_from_order_request(self, message: str) -> Optional[str]:
        text = (message or "").strip()
        if not text:
            return None
        patterns = [
            r"(?:i\s+want\s+to\s+order|i\s+need\s+to\s+order|order\s+from|i\s+want\s+food\s+from)\s+(.+)$",
            r"(?:from)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                name = match.group(1).strip(" .,-")
                if name and len(name.split()) <= 6:
                    return name
        return None

    def _pick_best_restaurant_match(
        self,
        requested_name: str,
        restaurants: List[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        query = (requested_name or "").strip().lower()
        if not query or not restaurants:
            return None

        for restaurant in restaurants:
            name = (restaurant.get("name") or "").strip().lower()
            if name == query:
                return restaurant

        for restaurant in restaurants:
            name = (restaurant.get("name") or "").strip().lower()
            if query in name or name in query:
                return restaurant

        query_words = {w for w in re.split(r"\s+", query) if len(w) >= 3}
        best = None
        best_score = 0
        for restaurant in restaurants:
            name_words = {
                w for w in re.split(r"\s+", (restaurant.get("name") or "").strip().lower())
                if len(w) >= 3
            }
            score = len(query_words & name_words)
            if score > best_score:
                best_score = score
                best = restaurant
        return best if best_score >= 2 else None

    def _extract_restaurants_from_tool_result(self, tool_result: List[Any]) -> List[Dict[str, str]]:
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
                    preview_items: list[str] = []
                    raw_items = node.get("items") or node.get("menu_items") or []
                    if isinstance(raw_items, list):
                        for preview in raw_items[:5]:
                            if isinstance(preview, dict):
                                preview_name = (
                                    preview.get("name")
                                    or preview.get("item_name")
                                    or preview.get("title")
                                )
                                if preview_name:
                                    preview_items.append(str(preview_name))
                    extracted.append({
                        "id": str(rid),
                        "name": str(name),
                        "rating": str(rating) if rating is not None else "",
                        "delivery_time": str(delivery) if delivery is not None else "",
                        "cuisines": str(cuisines),
                        "preview_text": ", ".join(preview_items),
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

        seen = set()
        uniq = []
        for r in extracted:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            uniq.append(r)
        return uniq

    def _restaurant_matches_keyword(self, restaurant: Dict[str, str], keyword: str) -> bool:
        query = (keyword or "").strip().lower()
        if not query:
            return True

        query_aliases = {
            "tiffin": {"tiffin", "tiffins", "idli", "dosa", "breakfast", "upma", "vada"},
            "breakfast": {"breakfast", "tiffin", "idli", "dosa", "upma", "vada"},
        }
        aliases = {
            "pizza": {"pizza", "pizzeria", "margherita", "farmhouse", "pepperoni"},
            "biryani": {"biryani", "dum", "dum biryani", "kunda biryani"},
            "burger": {"burger", "burgers", "whopper", "zinger"},
            "chinese": {"chinese", "noodles", "fried rice", "manchurian", "schezwan"},
            "dessert": {"dessert", "cake", "brownie", "pastry", "ice cream"},
            "cafe": {"cafe", "coffee", "tea", "latte", "espresso"},
        }
        terms = set(aliases.get(query, {query}))
        for token in re.split(r"\s+", query):
            terms.update(query_aliases.get(token, set()))
            if len(token) >= 3:
                terms.add(token)

        haystack = " ".join(
            [
                restaurant.get("name", ""),
                restaurant.get("cuisines", ""),
                restaurant.get("preview_text", ""),
            ]
        ).lower()
        haystack = re.sub(r"[^a-z0-9\s]", " ", haystack)

        for term in terms:
            if term in haystack:
                return True
        return False

    def _filter_restaurants_for_keyword(self, restaurants: List[Dict[str, str]], keyword: str) -> List[Dict[str, str]]:
        query = (keyword or "").strip().lower()
        if not query:
            return restaurants
        filtered = [restaurant for restaurant in restaurants if self._restaurant_matches_keyword(restaurant, query)]
        return filtered

    def _format_relaxed_match_list(self, restaurants: List[Dict[str, str]], keyword: str, location: str) -> str:
        return (
            f"I couldn't find strong **{keyword.title()}** matches, but here are nearby popular options near **{location}**:\n\n"
            + "\n".join(
                [
                    f"{idx}. {r.get('name', 'Unknown')}"
                    + (f" | Rating: {r.get('rating')}" if r.get("rating") else "")
                    + (f" | ETA: {r.get('delivery_time')}" if r.get("delivery_time") else "")
                    for idx, r in enumerate(restaurants[:10], start=1)
                ]
            )
            + "\n\nReply with a number to open menu, or try a more specific cuisine/restaurant name."
        )

    def _format_restaurant_list(self, restaurants: List[Dict[str, str]], keyword: str, location: str, min_rating: Optional[float] = None) -> str:
        title = f"{keyword.title()} restaurants" if keyword else "restaurants"
        if min_rating is not None:
            title = f"{title} (rating {min_rating:.1f}+)"
        lines = []
        for idx, r in enumerate(restaurants[:10], start=1):
            name = r.get("name", "Unknown")
            rating = r.get("rating", "")
            delivery = r.get("delivery_time", "")
            cuisines = r.get("cuisines", "")
            rating_str = f" | Rating: {rating}" if rating else ""
            delivery_str = f" | ETA: {delivery}" if delivery else ""
            cuisine_str = f" | {cuisines}" if cuisines else ""
            lines.append(f"{idx}. {name}{rating_str}{delivery_str}{cuisine_str}")
        return (
            f"Here are the top {title} near {location}:\n\n"
            + "\n".join(lines)
            + "\n\nReply with a number or restaurant name to continue."
        )

    def _format_menu_list(self, menu_items: List[dict], restaurant_name: str) -> str:
        if not menu_items:
            return f"I couldn't load the menu for {restaurant_name} right now."
        lines = []
        for idx, item in enumerate(menu_items[:10], start=1):
            size_part = f" ({item['size']})" if item.get("size") else ""
            price = item.get("price", "?")
            lines.append(f"{idx}. {item['name']}{size_part} - Rs {price}")
        return (
            f"Menu - {restaurant_name}\n\n"
            + "\n".join(lines)
            + "\n\nReply with item names or menu numbers, for example: `1 and 2` or `1 Margherita and 2 Garlic Breads`."
        )

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

    def _is_restaurant_selection_request(self, message: str, search_results: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        text = (message or "").strip().lower()
        if not text or not search_results:
            return None
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(search_results):
                return search_results[idx]
        for rest in search_results:
            name = (rest.get("name") or "").lower()
            if name and (text == name or text in name or name in text):
                return rest
        return None

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

    async def _fetch_restaurant_menu(self, user_id: str, res_id_str: str, address_id: str) -> List[Dict[str, str]]:
        try:
            res_id = int(res_id_str)
        except (ValueError, TypeError):
            logger.error("Cannot convert res_id to int: %s", res_id_str)
            return []

        prepared_listing_args = await self._prepare_tool_arguments(
            "get_menu_items_listing",
            {"res_id": res_id, "address_id": address_id},
            user_id,
        )
        listing = await global_zomato_mcp.call_tool("get_menu_items_listing", prepared_listing_args)
        logger.info("get_menu_items_listing raw: %s", str(listing)[:300])

        listing_items = self._extract_menu_items_from_tool_result(listing)
        if listing_items:
            return listing_items

        categories: List[str] = []
        variant_id_map: Dict[str, str] = {}
        for chunk in listing or []:
            if isinstance(chunk, str):
                try:
                    data = json.loads(chunk)
                except Exception:
                    continue

                def _collect(node):
                    if isinstance(node, dict):
                        cats = node.get("categories")
                        if isinstance(cats, list):
                            for cat in cats:
                                if isinstance(cat, str) and cat not in categories:
                                    categories.append(cat)
                        cat = node.get("category") or node.get("category_name")
                        if isinstance(cat, str) and cat not in categories:
                            categories.append(cat)
                        item_name = node.get("item_name") or node.get("name")
                        variant_id = node.get("variant_id")
                        if item_name and variant_id:
                            variant_id_map[str(item_name).lower()] = str(variant_id)
                        for value in node.values():
                            _collect(value)
                    elif isinstance(node, list):
                        for value in node:
                            _collect(value)

                _collect(data)

        prepared_menu_args = await self._prepare_tool_arguments(
            "get_restaurant_menu_by_categories",
            {"res_id": res_id, "categories": categories, "address_id": address_id},
            user_id,
        )
        menu_result = await global_zomato_mcp.call_tool("get_restaurant_menu_by_categories", prepared_menu_args)
        logger.info("get_restaurant_menu_by_categories raw: %s", str(menu_result)[:300])

        items = self._extract_menu_items_from_tool_result(menu_result)
        for item in items:
            if not item.get("variant_id"):
                variant_id = variant_id_map.get(item.get("name", "").lower())
                if variant_id:
                    item["variant_id"] = variant_id
        return items

    async def _resolve_address_id(self, user_id: str, session, force_refresh: bool = False) -> Tuple[Optional[str], str]:
        if session.address_id and not force_refresh:
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

        loc_lower = (session.current_location or "").lower().strip()
        loc_words = [re.sub(r"[^\w]", "", w) for w in loc_lower.split()]
        loc_words = [w for w in loc_words if w and len(w) > 2]
        generic_words = {
            "india", "andhra", "pradesh", "telangana", "karnataka",
            "tamil", "nadu", "maharashtra", "kerala", "gujarat",
            "rajasthan", "uttar", "madhya", "west", "bengal",
            "station", "railway", "airport", "road", "nagar",
            "colony", "district", "state", "city", "town",
            "address", "location", "near", "nearest",
            "the", "is", "for",
        }
        meaningful_words = [w for w in loc_words if w not in generic_words]
        generic_in_query = [w for w in loc_words if w in generic_words]

        best = addresses[0]
        best_score = 0.0
        for addr in addresses:
            name = addr.get("location_name", "").lower()
            name_clean = re.sub(r"[^\w\s]", " ", name)
            score = 0.0
            for word in meaningful_words:
                if len(word) >= 4 and re.search(r"\b" + re.escape(word) + r"\b", name_clean):
                    score += 5.0
                elif len(word) >= 3 and word in name_clean:
                    score += 2.0
            for word in generic_in_query:
                if word in name_clean:
                    score += 0.1
            if score > best_score:
                best_score = score
                best = addr

        if session.current_location:
            location_text = (session.current_location or "").strip().lower()
            if location_text and not location_text.startswith("lat=") and best_score <= 0:
                logger.warning(
                    "No saved Zomato address matched requested location '%s' for user %s",
                    session.current_location,
                    user_id,
                )
                return None, ""

        address_id = best.get("address_id")
        if address_id:
            session_service.set_address_id(user_id, address_id)
            return address_id, best.get("location_name", "")
        return None, ""

    def _sanitize_tool_schema(self, schema: dict[str, Any] | None) -> dict[str, Any]:
        cleaned = deepcopy(schema or {})
        if isinstance(cleaned.get("required"), list):
            cleaned["required"] = [field for field in cleaned["required"] if field != "address_id"]
        if isinstance(cleaned.get("properties"), dict):
            cleaned["properties"].pop("address_id", None)
        return cleaned

    def _tool_uses_address_id(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name == "checkout_cart":
            return False
        if "address_id" in arguments:
            return True
        return tool_name in {
            "get_restaurants_for_keyword",
            "get_menu_items_listing",
            "get_restaurant_menu_by_categories",
            "create_cart",
        }

    def _normalize_payment_type(self, payment_type: Any) -> str:
        text = str(payment_type or "").strip().lower()
        if text in {"upi", "upi_qr", "qr", "gpay", "phonepe", "paytm"}:
            return "upi_qr"
        if text in {"cod", "cash", "cash on delivery", "pay later", "pay_later"}:
            return "pay_later"
        return "pay_later"

    def _strip_none_values(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._strip_none_values(val)
                for key, val in value.items()
                if val is not None
            }
        if isinstance(value, list):
            return [self._strip_none_values(item) for item in value if item is not None]
        return value

    def _prepare_create_cart_items(self, session, items: list[Any]) -> list[dict[str, Any]]:
        menu_items = list((session.menu_items_map or {}).values())
        by_name = {
            (item.get("name") or "").strip().lower(): item
            for item in menu_items
            if item.get("name")
        }
        prepared: list[dict[str, Any]] = []
        for raw in items or []:
            if not isinstance(raw, dict):
                continue
            quantity = int(raw.get("quantity") or 1)
            if raw.get("variant_id"):
                prepared.append({
                    "variant_id": str(raw["variant_id"]),
                    "quantity": quantity,
                })
                continue
            item_name = str(raw.get("item_name") or raw.get("name") or "").strip().lower()
            if not item_name:
                continue
            matched = by_name.get(item_name)
            if not matched:
                matched = next(
                    (
                        item for name, item in by_name.items()
                        if item_name in name or name in item_name
                    ),
                    None,
                )
            if matched and matched.get("variant_id"):
                prepared.append({
                    "variant_id": str(matched["variant_id"]),
                    "quantity": quantity,
                })
        return prepared

    async def _prepare_tool_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: str | None,
        force_refresh_address: bool = False,
    ) -> dict[str, Any]:
        prepared = self._strip_none_values(dict(arguments))
        if tool_name == "checkout_cart":
            prepared.pop("address_id", None)
            return prepared
        if not user_id or not self._tool_uses_address_id(tool_name, prepared):
            return prepared

        session = session_service.get_session(user_id)
        if not session.current_location and not session.address_id:
            pass

        if tool_name == "create_cart" and session.selected_restaurant_id:
            try:
                prepared["res_id"] = int(session.selected_restaurant_id)
            except (TypeError, ValueError):
                prepared["res_id"] = session.selected_restaurant_id

        address_id, _ = await self._resolve_address_id(user_id, session, force_refresh=force_refresh_address)
        if address_id:
            prepared["address_id"] = address_id
        if tool_name == "create_cart":
            prepared["payment_type"] = self._normalize_payment_type(prepared.get("payment_type"))
            prepared_items = self._prepare_create_cart_items(session, prepared.get("items") or [])
            if prepared_items:
                prepared["items"] = prepared_items
        return self._strip_none_values(prepared)

    def _is_invalid_address_result(self, result: list[Any]) -> bool:
        for part in result or []:
            if not isinstance(part, str):
                continue
            try:
                data = json.loads(part)
            except Exception:
                continue
            if isinstance(data, dict) and data.get("error_code") == "INVALID_ADDRESS_ID":
                return True
        return False

    def _json_schema_to_args_model(self, name: str, schema: dict[str, Any] | None) -> type[BaseModel]:
        if not schema:
            return _EmptyArgs

        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
        fields = {}

        type_map: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list[Any],
            "object": dict[str, Any],
        }

        for key, details in properties.items():
            json_type = "string"
            if isinstance(details, dict):
                json_type = details.get("type", "string")
            py_type = type_map.get(json_type, Any)
            default = ... if key in required else None
            fields[key] = (py_type, default)

        if not fields:
            return _EmptyArgs
        return create_model(f"{name.title().replace('_', '')}Args", **fields)

    def _is_retryable_model_error(self, err: Exception) -> bool:
        text = str(err).lower()
        return (
            "503" in text
            or "unavailable" in text
            or "high demand" in text
            or "temporarily overloaded" in text
            or "429" in text
            or "resource_exhausted" in text
            or "quota" in text
            or "rate limit" in text
            or "retryinfo" in text
        )

    async def _ensure_graph(self, model_name: str):
        if model_name in self._compiled_graphs:
            return self._compiled_graphs[model_name]
        if not app_config.llm_api_key:
            logger.warning("LLM API key is missing. Set LLM_API_KEY or GEMINI_API_KEY.")
            return None

        if not self._tool_cache:
            mcp_tools = await global_zomato_mcp.get_tools()
            tool_defs: list[StructuredTool] = []
            for tool in mcp_tools:
                args_schema = self._json_schema_to_args_model(
                    tool.name,
                    self._sanitize_tool_schema(tool.inputSchema or {}),
                )

                async def _runner(_tool_name: str = tool.name, **kwargs):
                    active_user_id = _active_tool_user_id.get()
                    prepared_args = await self._prepare_tool_arguments(_tool_name, kwargs, active_user_id)
                    result = await global_zomato_mcp.call_tool(_tool_name, prepared_args)
                    if active_user_id and self._is_invalid_address_result(result):
                        logger.warning("Tool %s returned INVALID_ADDRESS_ID, refreshing address for user %s", _tool_name, active_user_id)
                        session = session_service.get_session(active_user_id)
                        session.address_id = None
                        prepared_args = await self._prepare_tool_arguments(
                            _tool_name,
                            kwargs,
                            active_user_id,
                            force_refresh_address=True,
                        )
                        result = await global_zomato_mcp.call_tool(_tool_name, prepared_args)
                    text_parts = [part for part in result if isinstance(part, str)]
                    if text_parts:
                        return "\n".join(text_parts)
                    return json.dumps(result, default=str)

                tool_defs.append(
                    StructuredTool.from_function(
                        coroutine=_runner,
                        name=tool.name,
                        description=tool.description or tool.name,
                        args_schema=args_schema,
                    )
                )
            self._tool_cache = tool_defs

        llm = ChatOpenAI(
            model=model_name,
            base_url=app_config.llm_base_url,
            api_key=app_config.llm_api_key,
            temperature=app_config.llm_temperature,
            default_headers=app_config.llm_default_headers,
        )
        self._compiled_graphs[model_name] = create_react_agent(llm, self._tool_cache)
        logger.info("LangGraph agent initialized with model=%s and %d MCP tools", model_name, len(self._tool_cache))
        return self._compiled_graphs[model_name]

    async def process_message(
        self,
        user_id: str,
        message: str,
        user_name: str | None = None,
        user_location: str | None = None,
        filters: SearchFilters | None = None,
        channel: str = "web",
        input_mode: str = "text",
    ) -> tuple[str, list[str]]:
        session = session_service.get_session(user_id, user_name)
        if user_location:
            if session.current_location != user_location:
                session_service.apply_location_change(user_id, user_location)
                session = session_service.get_session(user_id, user_name)

        lower_msg = (message or "").strip().lower()

        if lower_msg in {"start over", "reset", "/reset", "restart", "clear chat"}:
            session_service.reset_session(user_id)
            reply = (
                "Everything is reset.\n\n"
                "Share your area/city or live location, then tell me what you want to order."
            )
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["Session reset"]

        if self._is_show_all_active_orders_request(message):
            tracking_result = await global_zomato_mcp.call_tool("get_order_tracking_info", {})
            all_orders: List[Dict[str, Any]] = []
            for chunk in tracking_result or []:
                if not isinstance(chunk, str):
                    continue
                try:
                    data = json.loads(chunk)
                except Exception:
                    continue
                orders = self._extract_orders_from_tracking_data(data)
                if orders:
                    all_orders = orders
                    break

            if not all_orders:
                reply = "I couldn't find any active orders right now."
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["No active orders found"]

            lines = []
            for idx, order in enumerate(all_orders[:10], start=1):
                oid = order.get("order_id") or order.get("id") or "unknown"
                restaurant = order.get("restaurant_name") or "Unknown restaurant"
                status = self._normalize_tracking_status(order)
                lines.append(f"{idx}. {restaurant} | ID: {oid} | Status: {status}")
            reply = "Here are your active orders:\n\n" + "\n".join(lines)
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["All active orders listed"]

        if self._is_tracking_request(message):
            tracking_result = await global_zomato_mcp.call_tool("get_order_tracking_info", {})
            tracking_reply = None
            for chunk in tracking_result or []:
                if not isinstance(chunk, str):
                    continue
                try:
                    data = json.loads(chunk)
                except Exception:
                    continue
                orders = self._extract_orders_from_tracking_data(data)
                if not orders:
                    continue
                picked = self._pick_tracked_order(orders, session)
                if not picked:
                    continue
                order = picked
                status = self._normalize_tracking_status(order)
                order_id = order.get("order_id") or order.get("id") or session.current_order_id or "unknown"
                rider = order.get("rider") or {}
                restaurant = order.get("restaurant_name") or session.selected_restaurant_name or "your restaurant"
                tracking_reply = (
                    f"📦 Order status: **{status}**\n"
                    f"🆔 Order ID: **{order_id}**\n"
                    f"🍽️ Restaurant: **{restaurant}**"
                )
                if isinstance(rider, dict):
                    rider_name = rider.get("name") or rider.get("rider_name")
                    rider_phone = rider.get("phone") or rider.get("mobile")
                    if rider_name:
                        tracking_reply += f"\n🚴 Rider: **{rider_name}**"
                    if rider_phone:
                        tracking_reply += f"\n📞 Contact: `{rider_phone}`"
                break
            if not tracking_reply:
                logger.warning("telemetry event=tracking_lookup_failed user_id=%s", user_id)
                tracking_reply = "I couldn't fetch a live Zomato tracking update right now. Please try again in a moment."
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", tracking_reply)
            session_service.set_last_bot_message(user_id, tracking_reply)
            return tracking_reply, ["Order tracking requested"]

        if self._is_list_saved_addresses_request(message):
            result = await global_zomato_mcp.call_tool("get_saved_addresses_for_user", {})
            addresses: List[Dict[str, Any]] = []
            for chunk in result or []:
                if not isinstance(chunk, str):
                    continue
                try:
                    data = json.loads(chunk)
                except Exception:
                    continue
                if isinstance(data, dict) and isinstance(data.get("addresses"), list):
                    addresses = data.get("addresses") or []
                    break
            session.saved_addresses = addresses
            reply = self._format_saved_addresses(addresses)
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["Saved addresses listed"]

        address_selection = self._extract_address_selection(message)
        if address_selection:
            addresses = session.saved_addresses
            if not addresses:
                _, _ = await self._resolve_address_id(user_id, session, force_refresh=True)
                session = session_service.get_session(user_id, user_name)
                addresses = session.saved_addresses

            chosen_address = None
            if address_selection.isdigit():
                idx = int(address_selection) - 1
                if 0 <= idx < len(addresses):
                    chosen_address = addresses[idx]
            if not chosen_address:
                chosen_address = self._pick_best_saved_address_match(address_selection, addresses)

            if not chosen_address:
                reply = (
                    "I couldn't match that to one of your saved Zomato addresses.\n\n"
                    "Say `list saved addresses` and then `choose address as <name>`."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Saved address selection failed"]

            chosen_label = str(chosen_address.get("location_name") or address_selection)
            chosen_id = str(chosen_address.get("address_id") or "")
            normalized_label = self._normalize_location_text(chosen_label)
            if self._is_ordering_flow_active(session):
                session_service.set_delivery_context(user_id, normalized_label, chosen_id)
            else:
                session_service.apply_location_change(user_id, normalized_label)
                session_service.set_address_id(user_id, chosen_id)
            session = session_service.get_session(user_id, user_name)
            session.saved_addresses = addresses
            if self._is_ordering_flow_active(session):
                reply = (
                    f"📍 Delivery address updated to **{chosen_label}** for this order.\n\n"
                    "Your current cart is preserved. Continue with payment or add more items."
                )
            else:
                reply = (
                    f"📍 Delivery address set to **{chosen_label}**.\n\n"
                    "Now tell me what you want, for example: `show me pizza restaurants`."
                )
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["Saved address selected"]

        if "add new address" in lower_msg or "add address" in lower_msg or "change address" in lower_msg:
            reply = (
                "I can't directly create addresses in Zomato from this bot.\n\n"
                "Please add/update the address in your Zomato app first, then share that area/city here and I will find nearby restaurants."
            )
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["Address management requested"]

        explicit_location = self._extract_location_override(message)
        if explicit_location:
            normalized = self._normalize_location_text(explicit_location)
            active_ordering_flow = self._is_ordering_flow_active(session)
            if session.current_location != normalized:
                if active_ordering_flow:
                    session_service.set_delivery_context(user_id, normalized, None)
                    session = session_service.get_session(user_id, user_name)
                    resolved_id, _ = await self._resolve_address_id(user_id, session, force_refresh=True)
                    if resolved_id:
                        session_service.set_address_id(user_id, resolved_id)
                        session = session_service.get_session(user_id, user_name)
                else:
                    session_service.apply_location_change(user_id, normalized)
                    session = session_service.get_session(user_id, user_name)

            # if this was just location sharing, acknowledge quickly
            if (
                lower_msg.startswith("my location is")
                or lower_msg.startswith("update")
                or lower_msg.startswith("change")
                or lower_msg.startswith("set")
                or lower_msg.startswith("location is")
                or self._should_treat_message_as_location(session, message)
            ):
                if active_ordering_flow:
                    reply = (
                        f"📍 Delivery location updated to **{normalized}** for this order.\n\n"
                        "Your current cart is preserved. Continue with payment or add more items."
                    )
                else:
                    reply = (
                        f"📍 Location updated to **{normalized}**.\n\n"
                        "Now tell me what you want, for example: `show me pizza places near me`."
                    )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Location updated"]
        elif self._should_treat_message_as_location(session, message):
            normalized = self._normalize_location_text(self._clean_location_candidate(message))
            if session.current_location != normalized:
                session_service.apply_location_change(user_id, normalized)
                session = session_service.get_session(user_id, user_name)
            reply = (
                f"📍 Location updated to **{normalized}**.\n\n"
                "Now tell me what you want, for example: `show me biryani restaurants`."
            )
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["Location updated"]

        if lower_msg in {"show cart", "view cart", "cart"}:
            reply = self._render_cart(session)
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["Cart rendered"]

        if session.state == ConversationState.AWAITING_PAYMENT:
            payment_type = self._parse_payment_type(message)
            if payment_type:
                session.payment_type = payment_type
                session_service.update_state(user_id, ConversationState.CONFIRMING_ORDER)
                reply = self._build_confirm_prompt(session)
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Payment selected"]

            # Allow users to keep adding items even after payment prompt.
            if self._is_add_to_cart_request(message, session):
                menu_items = list(session.menu_items_map.values())
                cart_items = self._parse_order_items(message, menu_items)
                if cart_items:
                    for item in cart_items:
                        session_service.add_to_cart(user_id, item)
                    reply = self._build_payment_prompt(session)
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Items added while awaiting payment"]

        if session.state == ConversationState.CONFIRMING_ORDER:
            if self._is_confirm_yes(message):
                if not session.cart or not session.selected_restaurant_id:
                    reply = "⚠️ I need a selected restaurant and cart items before I can place the order."
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Missing cart context"]

                payment_type = session.payment_type or "pay_later"
                addr_id = session.address_id
                if not addr_id:
                    addr_id, _ = await self._resolve_address_id(user_id, session)
                if not addr_id:
                    reply = "⚠️ Could not resolve your delivery address. Please share your location again and retry."
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Missing address"]

                cart_items_for_zomato = []
                for item in session.cart:
                    if item.variant_id:
                        cart_items_for_zomato.append({"variant_id": item.variant_id, "quantity": item.quantity})
                if not cart_items_for_zomato:
                    logger.warning("telemetry event=order_missing_variant_ids user_id=%s cart_items=%d", user_id, len(session.cart))
                    reply = "⚠️ Cart items are missing variant IDs. Please reopen the menu and add items again."
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Missing variant ids"]

                res_id = int(session.selected_restaurant_id)
                create_result = await global_zomato_mcp.call_tool(
                    "create_cart",
                    {
                        "res_id": res_id,
                        "items": cart_items_for_zomato,
                        "address_id": addr_id,
                        "payment_type": payment_type,
                    },
                )
                cart_id = None
                create_error = None
                for chunk in create_result or []:
                    if isinstance(chunk, str):
                        try:
                            data = json.loads(chunk)
                        except Exception:
                            continue
                        if isinstance(data, dict):
                            create_error = data.get("error_message") or data.get("error_code") or create_error
                            cart_id = (
                                data.get("cart_id")
                                or self._find_first_value(data, {"cart_id", "id"})
                                or cart_id
                            )
                if create_error:
                    logger.warning("telemetry event=order_create_cart_failed user_id=%s error=%s", user_id, create_error)
                    reply = f"⚠️ Zomato couldn't create the cart: {create_error}"
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Create cart failed"]
                if not cart_id:
                    logger.warning("telemetry event=order_missing_cart_id user_id=%s", user_id)
                    reply = "⚠️ I couldn't get a cart ID from Zomato. Please try again."
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Missing cart id"]

                session.zomato_cart_id = str(cart_id)
                checkout_result = await global_zomato_mcp.call_tool_raw("checkout_cart", {"cart_id": str(cart_id)})
                order_info, checkout_error, qr_path = self._parse_checkout_content(checkout_result, str(cart_id))
                if checkout_error:
                    logger.warning("telemetry event=order_checkout_failed user_id=%s error=%s", user_id, checkout_error)
                    reply = f"⚠️ Zomato checkout failed: {checkout_error}"
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Checkout failed"]

                order_id = (
                    order_info.get("order_id")
                    or self._find_first_value(order_info, {"order_id", "id"})
                    or str(cart_id)
                )
                session_service.set_current_order(user_id, str(order_id))
                session_service.update_state(user_id, ConversationState.ORDER_PLACED)

                if payment_type == "upi_qr":
                    qr_value = self._find_first_value(order_info, {"qr_code", "qrCode", "payment_link", "payment_url", "upi_link", "deeplink"})
                    if qr_value:
                        reply = (
                            "✅ Cart is ready.\n\n"
                            f"💳 UPI payment link / QR:\n{qr_value}\n\n"
                            "Complete the payment, then say `track my order`."
                        )
                    elif qr_path:
                        reply = (
                            "✅ Cart is ready for UPI payment.\n\n"
                            f"[QR Code Image Saved to {qr_path}]\n\n"
                            "Scan the QR and complete the payment, then say `track my order`."
                        )
                    else:
                        reply = (
                            "✅ Cart is ready for UPI payment.\n\n"
                            "Zomato did not return a QR code in the response. Please continue payment in the Zomato flow and then say `track my order`."
                        )
                else:
                    reply = (
                        "✅ **Order placed successfully on Zomato!**\n\n"
                        f"📦 Order ID: **{order_id}**\n"
                        f"💳 Payment: Pay Later (COD)\n\n"
                        "Say `track my order` to check the status."
                    )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Order placed directly"]

            if self._is_confirm_no(message):
                reply = "Okay, I won't place the order. You can add more items or say `show cart`."
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Order confirmation cancelled"]

        if self._is_add_to_cart_request(message, session):
            menu_items = list(session.menu_items_map.values())
            cart_items = self._parse_order_items(message, menu_items)
            if cart_items:
                for item in cart_items:
                    session_service.add_to_cart(user_id, item)
                session_service.update_state(user_id, ConversationState.AWAITING_PAYMENT)
                reply = self._build_payment_prompt(session)
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Items added to cart"]
            if session.menu_items_map:
                reply = "I couldn't match those items exactly from the current menu. Please try item names as shown in the menu."
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Item match failed"]

        if self._is_generic_order_intent(message):
            if session.menu_items_map and session.selected_restaurant_name:
                reply = (
                    f"You already have the menu for **{session.selected_restaurant_name}** open.\n\n"
                    "Reply with menu numbers or item names, for example: `1 and 2`."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Prompted for menu item selection"]
            if session.current_location:
                reply = (
                    "Tell me what you want to order, for example: `show me pizza places` or `show me biryani restaurants`."
                )
            else:
                reply = (
                    "📍 First share your location or area, then tell me what you want to eat.\n\n"
                    "Example: `Madhapur, Hyderabad` and then `show me pizza places`."
                )
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["Generic ordering intent handled"]

        requested_restaurant_name = self._extract_restaurant_name_from_menu_request(message)
        if requested_restaurant_name:
            session_service.reset_browsing_context(user_id)
            session = session_service.get_session(user_id, user_name)
            if not session.current_location:
                reply = (
                    "📍 To load a menu, I need your location first.\n\n"
                    "Share your city/area or tap Share My Location."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Location required for menu lookup"]

            address_id, address_label = await self._resolve_address_id(user_id, session)
            if not address_id:
                reply = (
                    "📍 I couldn't find a saved delivery address in your Zomato account.\n\n"
                    "Please add an address in the Zomato app first, then try again."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["No saved Zomato address"]

            result = await global_zomato_mcp.call_tool(
                "get_restaurants_for_keyword",
                {"keyword": requested_restaurant_name, "address_id": address_id, "page_size": 10},
            )
            extracted = self._extract_restaurants_from_tool_result(result)
            if not extracted:
                reply = (
                    f"I couldn't find **{requested_restaurant_name}** near **{address_label or session.current_location}**.\n\n"
                    "Try the exact restaurant name or search restaurants first."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Restaurant not found for menu request"]

            chosen = self._pick_best_restaurant_match(requested_restaurant_name, extracted)
            if not chosen:
                reply = (
                    f"I found some restaurants near {address_label or session.current_location}, but none matched '{requested_restaurant_name}' closely enough.\n\n"
                    "Please try the exact restaurant name."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Restaurant match too weak for menu request"]
            session_service.set_search_results(user_id, extracted)
            session_service.set_selected_restaurant(user_id, chosen["id"])
            session.selected_restaurant_name = chosen.get("name", "")
            menu_items = await self._fetch_restaurant_menu(user_id, chosen["id"], address_id)
            if menu_items:
                session.menu_items_map = {str(i + 1): item for i, item in enumerate(menu_items[:20])}
                session_service.update_state(user_id, ConversationState.BROWSING_MENU)
                reply = self._format_menu_list(menu_items, chosen["name"])
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Menu loaded from restaurant name"]

            reply = (
                f"I found **{chosen['name']}**, but I couldn't load its menu for **{address_label or session.current_location}** right now.\n\n"
                "Try another restaurant or search again."
            )
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", reply)
            session_service.set_last_bot_message(user_id, reply)
            return reply, ["Menu unavailable for matched restaurant"]

        requested_order_restaurant = self._extract_restaurant_name_from_order_request(message)
        if requested_order_restaurant:
            session_service.reset_browsing_context(user_id)
            session = session_service.get_session(user_id, user_name)
            if not session.current_location:
                reply = (
                    "📍 First share your location or area, then I can open restaurants near you.\n\n"
                    "Example: `Vijayawada`."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Location required for restaurant order request"]

            address_id, address_label = await self._resolve_address_id(user_id, session)
            if not address_id:
                reply = (
                    f"I couldn't match **{session.current_location}** to one of your saved Zomato addresses.\n\n"
                    "Please choose a location you already have saved in Zomato, then try again."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["No matching saved address for order request"]

            result = await global_zomato_mcp.call_tool(
                "get_restaurants_for_keyword",
                {"keyword": requested_order_restaurant, "address_id": address_id, "page_size": 10},
            )
            extracted = self._extract_restaurants_from_tool_result(result)
            if not extracted:
                reply = (
                    f"I couldn't find **{requested_order_restaurant}** near **{address_label or session.current_location}**.\n\n"
                    "Try another restaurant name or ask for nearby restaurants."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Restaurant not found for direct order request"]

            chosen = self._pick_best_restaurant_match(requested_order_restaurant, extracted)
            if not chosen:
                reply = (
                    f"I found restaurants near {address_label or session.current_location}, but none matched '{requested_order_restaurant}' closely enough.\n\n"
                    "Please say the exact restaurant name or ask to see nearby restaurants."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Restaurant match too weak for direct order request"]
            session_service.set_search_results(user_id, extracted)
            session_service.set_selected_restaurant(user_id, chosen["id"])
            session.selected_restaurant_name = chosen.get("name", "")
            menu_items = await self._fetch_restaurant_menu(user_id, chosen["id"], address_id)
            if menu_items:
                session.menu_items_map = {str(i + 1): item for i, item in enumerate(menu_items[:20])}
                session_service.update_state(user_id, ConversationState.BROWSING_MENU)
                reply = self._format_menu_list(menu_items, chosen["name"])
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Menu opened from direct order request"]

        if self._is_restaurant_search_request(message):
            session_service.reset_browsing_context(user_id)
            session = session_service.get_session(user_id, user_name)
            if not session.current_location:
                reply = (
                    "📍 To find nearby restaurants, I need your location.\n\n"
                    "Share your city/area (for example: `Madhapur, Hyderabad`) or tap Share My Location."
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Location required for search"]

            try:
                keyword = self._extract_search_keyword(message)
                min_rating = self._extract_min_rating_filter(message)
                address_id, address_label = await self._resolve_address_id(user_id, session)
                if not address_id:
                    reply = (
                        "📍 I couldn't find a saved delivery address in your Zomato account.\n\n"
                        "Please add an address in the Zomato app first, then try again."
                    )
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["No saved Zomato address"]

                attempts = [{"keyword": keyword or None, "address_id": address_id, "page_size": 10}]
                if keyword or min_rating is not None:
                    attempts.append({"keyword": None, "address_id": address_id, "page_size": 20})
                extracted: List[Dict[str, str]] = []
                broad_extracted: List[Dict[str, str]] = []
                filtered_results: List[Dict[str, str]] = []
                for args in attempts:
                    result = await global_zomato_mcp.call_tool("get_restaurants_for_keyword", args)
                    candidate = self._extract_restaurants_from_tool_result(result)
                    if not candidate:
                        continue
                    if args.get("keyword"):
                        extracted = candidate
                    else:
                        broad_extracted = candidate
                    if extracted:
                        break

                filtered_results = self._filter_restaurants_for_keyword(extracted, keyword)
                if keyword and not filtered_results and broad_extracted:
                    filtered_results = self._filter_restaurants_for_keyword(broad_extracted, keyword)
                if keyword and filtered_results:
                    extracted = filtered_results
                elif keyword and broad_extracted:
                    extracted = broad_extracted
                extracted = self._filter_restaurants_by_rating(extracted, min_rating)

                if not extracted:
                    near = address_label or session.current_location
                    if min_rating is not None:
                        reply = (
                            f"😕 I couldn't find restaurants rated **{min_rating:.1f}+** near **{near}**.\n\n"
                            "Try a lower rating filter or update to a nearby area."
                        )
                    else:
                        reply = (
                            f"😕 I couldn't find matching restaurants near **{near}**.\n\n"
                            "Try another cuisine or update to a nearby area."
                        )
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Restaurant search returned no results"]

                if keyword and not filtered_results:
                    near = address_label or session.current_location
                    reply = self._format_relaxed_match_list(extracted, keyword, near)
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Cuisine-specific search fallback shown"]

                session_service.set_search_results(user_id, extracted)
                reply = self._format_restaurant_list(
                    extracted,
                    keyword=keyword,
                    location=address_label or session.current_location,
                    min_rating=min_rating,
                )
                session_service.add_to_history(user_id, "user", message)
                session_service.add_to_history(user_id, "assistant", reply)
                session_service.set_last_bot_message(user_id, reply)
                return reply, ["Nearby restaurant search complete"]
            except Exception as search_err:
                logger.error("Deterministic search failed: %s", search_err)

        if self._is_first_menu_request(message) and session.search_results:
            session_service.set_selected_restaurant(user_id, session.search_results[0]["id"])
            session.selected_restaurant_name = session.search_results[0].get("name", "")

        selected = self._is_restaurant_selection_request(message, session.search_results or [])
        if selected or (self._is_first_menu_request(message) and session.selected_restaurant_id):
            chosen = selected or next(
                (r for r in (session.search_results or []) if r.get("id") == session.selected_restaurant_id),
                None,
            )
            if chosen:
                session_service.set_selected_restaurant(user_id, chosen["id"])
                session.selected_restaurant_name = chosen.get("name", "")
                try:
                    address_id, _ = await self._resolve_address_id(user_id, session)
                    if not address_id:
                        reply = (
                            "📍 I need a valid saved delivery address to load this menu.\n\n"
                            "Please check your Zomato saved addresses and try again."
                        )
                        session_service.add_to_history(user_id, "user", message)
                        session_service.add_to_history(user_id, "assistant", reply)
                        session_service.set_last_bot_message(user_id, reply)
                        return reply, ["Address needed for menu"]

                    menu_items = await self._fetch_restaurant_menu(user_id, chosen["id"], address_id)
                    if menu_items:
                        session.menu_items_map = {str(i + 1): item for i, item in enumerate(menu_items[:20])}
                        session_service.update_state(user_id, ConversationState.BROWSING_MENU)
                        reply = self._format_menu_list(menu_items, chosen["name"])
                        session_service.add_to_history(user_id, "user", message)
                        session_service.add_to_history(user_id, "assistant", reply)
                        session_service.set_last_bot_message(user_id, reply)
                        return reply, ["Menu loaded directly"]

                    reply = (
                        f"I couldn't load the menu for **{chosen['name']}** right now.\n\n"
                        "The restaurant may be unavailable for your current saved address. Try another restaurant or update your address in Zomato."
                    )
                    session_service.add_to_history(user_id, "user", message)
                    session_service.add_to_history(user_id, "assistant", reply)
                    session_service.set_last_bot_message(user_id, reply)
                    return reply, ["Menu unavailable"]
                except Exception as menu_err:
                    logger.error("Deterministic menu load failed for %s: %s", chosen.get("name", "restaurant"), menu_err)

        candidate_models = app_config.llm_models
        compiled_graph = None
        selected_model = None
        for model_name in candidate_models:
            compiled_graph = await self._ensure_graph(model_name)
            if compiled_graph is not None:
                selected_model = model_name
                break

        if compiled_graph is None:
            fallback = "LLM is not configured. Please set LLM_API_KEY/LLM_MODEL and retry."
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", fallback)
            session_service.set_last_bot_message(user_id, fallback)
            return fallback, ["LLM unavailable"]

        history_messages = []
        for item in session.conversation_history[-4:]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                history_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                history_messages.append(AIMessage(content=content))

        prompt = self._build_system_prompt(session.current_location, filters)
        runtime_context = (
            f"Channel={channel}; InputMode={input_mode}; "
            "For voice input, be concise and confirm key details before placing an order."
        )

        graph_input = {
            "messages": [
                SystemMessage(content=prompt),
                SystemMessage(content=runtime_context),
                *history_messages,
                HumanMessage(content=message),
            ]
        }

        final_text = ""
        thinking_steps: list[str] = []
        provider_error: Exception | None = None

        for idx, model_name in enumerate(candidate_models):
            graph = await self._ensure_graph(model_name)
            if graph is None:
                continue
            try:
                tool_context_token = _active_tool_user_id.set(user_id)
                try:
                    result = await graph.ainvoke(graph_input)
                finally:
                    _active_tool_user_id.reset(tool_context_token)
                response_messages = result.get("messages", [])
                if response_messages:
                    final_text = self._extract_output_text(response_messages[-1].content)
                if not final_text:
                    final_text = self._build_empty_response_fallback(session, message)
                thinking_steps = self._extract_thinking_steps(response_messages)
                if idx > 0 or model_name != selected_model:
                    thinking_steps = [f"Fallback model used: {model_name}", *thinking_steps]
                break
            except Exception as e:
                provider_error = e
                if self._is_retryable_model_error(e) and idx < len(candidate_models) - 1:
                    logger.warning("Model %s failed with retryable error, trying next model: %s", model_name, e)
                    continue
                logger.error("LangGraph provider call failed on model %s: %s", model_name, e)
                final_text, thinking_steps = self._friendly_provider_error(e)
                break

        if not final_text and provider_error is not None:
            final_text, thinking_steps = self._friendly_provider_error(provider_error)

        session_service.add_to_history(user_id, "user", message)
        session_service.add_to_history(user_id, "assistant", final_text)
        session_service.set_last_bot_message(user_id, final_text)
        return final_text, thinking_steps


langgraph_food_agent = LangGraphFoodAgent()
