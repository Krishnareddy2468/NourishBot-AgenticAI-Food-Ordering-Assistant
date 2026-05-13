"""
Dzukku Bot State Machine — §3.3 v2

States are stored in Session.state (DB) and propagated through ContextSnapshot.
Transitions are event-driven: each committed tool call emits an event that
the StateMachine maps to a next state.

State stored per Channel Session, linked to customer identity.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


# ── State enum ────────────────────────────────────────────────────────────────

class BotState(str, Enum):
    IDLE                  = "IDLE"
    BROWSING_MENU         = "BROWSING_MENU"
    BUILDING_CART         = "BUILDING_CART"
    COLLECTING_DETAILS    = "COLLECTING_DETAILS"   # name/phone/address/table
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    AWAITING_PAYMENT      = "AWAITING_PAYMENT"
    ORDER_PLACED          = "ORDER_PLACED"
    ORDER_IN_PROGRESS     = "ORDER_IN_PROGRESS"    # kitchen accepted
    OUT_FOR_DELIVERY      = "OUT_FOR_DELIVERY"
    COMPLETED             = "COMPLETED"            # delivered / session closed
    SUPPORT_CASE          = "SUPPORT_CASE"

    @classmethod
    def from_str(cls, value: str) -> "BotState":
        try:
            return cls(value)
        except ValueError:
            return cls.IDLE


# ── Events ────────────────────────────────────────────────────────────────────

class Event(str, Enum):
    # Cart / ordering
    MENU_BROWSED        = "MENU_BROWSED"
    ITEM_ADDED          = "ITEM_ADDED"
    CART_VIEWED         = "CART_VIEWED"
    ORDER_TYPE_SET      = "ORDER_TYPE_SET"
    ADDRESS_SET         = "ADDRESS_SET"
    DETAILS_COLLECTED   = "DETAILS_COLLECTED"     # all required slots filled
    CONFIRMATION_READY  = "CONFIRMATION_READY"    # cart + details complete
    ORDER_CONFIRMED     = "ORDER_CONFIRMED"       # customer said "yes, place it"
    PAYMENT_REQUIRED    = "PAYMENT_REQUIRED"      # order placed, payment pending
    PAYMENT_CAPTURED    = "PAYMENT_CAPTURED"      # Razorpay webhook / polling
    PAYMENT_FAILED      = "PAYMENT_FAILED"
    ORDER_PLACED        = "ORDER_PLACED"          # COD / no payment required
    KITCHEN_ACCEPTED    = "KITCHEN_ACCEPTED"
    DISPATCHED          = "DISPATCHED"            # driver assigned
    DELIVERED           = "DELIVERED"
    SESSION_CLOSED      = "SESSION_CLOSED"        # dine-in session closed
    ISSUE_RAISED        = "ISSUE_RAISED"
    RESET               = "RESET"


# ── Transition table  (current_state, event) → next_state ─────────────────────

_TRANSITIONS: dict[tuple[BotState, Event], BotState] = {
    # From IDLE
    (BotState.IDLE, Event.MENU_BROWSED):        BotState.BROWSING_MENU,
    (BotState.IDLE, Event.ITEM_ADDED):          BotState.BUILDING_CART,
    (BotState.IDLE, Event.ISSUE_RAISED):        BotState.SUPPORT_CASE,
    (BotState.IDLE, Event.RESET):               BotState.IDLE,

    # From BROWSING_MENU
    (BotState.BROWSING_MENU, Event.ITEM_ADDED):         BotState.BUILDING_CART,
    (BotState.BROWSING_MENU, Event.ISSUE_RAISED):       BotState.SUPPORT_CASE,
    (BotState.BROWSING_MENU, Event.RESET):              BotState.IDLE,

    # From BUILDING_CART
    (BotState.BUILDING_CART, Event.CART_VIEWED):            BotState.BUILDING_CART,
    (BotState.BUILDING_CART, Event.ITEM_ADDED):             BotState.BUILDING_CART,
    (BotState.BUILDING_CART, Event.ORDER_TYPE_SET):         BotState.COLLECTING_DETAILS,
    (BotState.BUILDING_CART, Event.DETAILS_COLLECTED):      BotState.AWAITING_CONFIRMATION,
    (BotState.BUILDING_CART, Event.CONFIRMATION_READY):     BotState.AWAITING_CONFIRMATION,
    (BotState.BUILDING_CART, Event.ORDER_PLACED):           BotState.ORDER_PLACED,
    (BotState.BUILDING_CART, Event.PAYMENT_REQUIRED):       BotState.AWAITING_PAYMENT,
    (BotState.BUILDING_CART, Event.ISSUE_RAISED):           BotState.SUPPORT_CASE,
    (BotState.BUILDING_CART, Event.RESET):                  BotState.IDLE,

    # From COLLECTING_DETAILS
    (BotState.COLLECTING_DETAILS, Event.ITEM_ADDED):            BotState.COLLECTING_DETAILS,
    (BotState.COLLECTING_DETAILS, Event.ADDRESS_SET):           BotState.COLLECTING_DETAILS,
    (BotState.COLLECTING_DETAILS, Event.DETAILS_COLLECTED):     BotState.AWAITING_CONFIRMATION,
    (BotState.COLLECTING_DETAILS, Event.CONFIRMATION_READY):    BotState.AWAITING_CONFIRMATION,
    (BotState.COLLECTING_DETAILS, Event.ORDER_PLACED):          BotState.ORDER_PLACED,
    (BotState.COLLECTING_DETAILS, Event.PAYMENT_REQUIRED):      BotState.AWAITING_PAYMENT,
    (BotState.COLLECTING_DETAILS, Event.RESET):                 BotState.IDLE,

    # From AWAITING_CONFIRMATION
    (BotState.AWAITING_CONFIRMATION, Event.ORDER_CONFIRMED):   BotState.AWAITING_PAYMENT,
    (BotState.AWAITING_CONFIRMATION, Event.PAYMENT_REQUIRED):  BotState.AWAITING_PAYMENT,
    (BotState.AWAITING_CONFIRMATION, Event.ORDER_PLACED):      BotState.ORDER_PLACED,
    (BotState.AWAITING_CONFIRMATION, Event.ITEM_ADDED):        BotState.BUILDING_CART,
    (BotState.AWAITING_CONFIRMATION, Event.RESET):             BotState.IDLE,

    # From AWAITING_PAYMENT
    (BotState.AWAITING_PAYMENT, Event.PAYMENT_CAPTURED):  BotState.ORDER_PLACED,
    (BotState.AWAITING_PAYMENT, Event.PAYMENT_FAILED):    BotState.AWAITING_CONFIRMATION,
    (BotState.AWAITING_PAYMENT, Event.RESET):             BotState.IDLE,

    # From ORDER_PLACED
    (BotState.ORDER_PLACED, Event.KITCHEN_ACCEPTED):  BotState.ORDER_IN_PROGRESS,
    (BotState.ORDER_PLACED, Event.ISSUE_RAISED):      BotState.SUPPORT_CASE,
    (BotState.ORDER_PLACED, Event.RESET):             BotState.IDLE,

    # From ORDER_IN_PROGRESS
    (BotState.ORDER_IN_PROGRESS, Event.DISPATCHED):        BotState.OUT_FOR_DELIVERY,
    (BotState.ORDER_IN_PROGRESS, Event.DELIVERED):         BotState.COMPLETED,
    (BotState.ORDER_IN_PROGRESS, Event.SESSION_CLOSED):    BotState.COMPLETED,
    (BotState.ORDER_IN_PROGRESS, Event.ISSUE_RAISED):      BotState.SUPPORT_CASE,

    # From OUT_FOR_DELIVERY
    (BotState.OUT_FOR_DELIVERY, Event.DELIVERED):      BotState.COMPLETED,
    (BotState.OUT_FOR_DELIVERY, Event.ISSUE_RAISED):   BotState.SUPPORT_CASE,

    # From COMPLETED
    (BotState.COMPLETED, Event.ITEM_ADDED):   BotState.BUILDING_CART,
    (BotState.COMPLETED, Event.MENU_BROWSED): BotState.BROWSING_MENU,
    (BotState.COMPLETED, Event.RESET):        BotState.IDLE,

    # From SUPPORT_CASE
    (BotState.SUPPORT_CASE, Event.RESET):         BotState.IDLE,
    (BotState.SUPPORT_CASE, Event.ITEM_ADDED):    BotState.BUILDING_CART,
    (BotState.SUPPORT_CASE, Event.MENU_BROWSED):  BotState.BROWSING_MENU,
}

# Tools → events they emit on success
TOOL_EVENTS: dict[str, Event] = {
    "get_menu":                 Event.MENU_BROWSED,
    "search_menu":              Event.MENU_BROWSED,
    "get_item_details":         Event.MENU_BROWSED,
    "add_to_cart":              Event.ITEM_ADDED,
    "update_cart_item":         Event.ITEM_ADDED,
    "remove_from_cart":         Event.ITEM_ADDED,
    "view_cart":                Event.CART_VIEWED,
    "set_order_type":           Event.ORDER_TYPE_SET,
    "set_delivery_address":     Event.ADDRESS_SET,
    "update_customer":          Event.DETAILS_COLLECTED,
    "place_order":              Event.ORDER_PLACED,
    "create_payment_intent":    Event.PAYMENT_REQUIRED,
    "check_payment_status":     Event.PAYMENT_CAPTURED,  # only if status=CAPTURED
    "cancel_order":             Event.RESET,
    "make_reservation":         Event.ORDER_PLACED,
    "open_table_session":       Event.ITEM_ADDED,
    "add_table_order":          Event.ITEM_ADDED,
    "close_table_session":      Event.SESSION_CLOSED,
    "generate_invoice":         Event.SESSION_CLOSED,
    "assign_driver":            Event.DISPATCHED,
    "update_order_status":      Event.KITCHEN_ACCEPTED,  # depends on status arg
    "update_delivery_status":   Event.DELIVERED,
}


class StateMachine:

    @staticmethod
    def transition(current: BotState, event: Event) -> BotState:
        """Return next state given current state and event. No-op if no transition defined."""
        return _TRANSITIONS.get((current, event), current)

    @staticmethod
    def next_from_tool(current: BotState, tool_name: str, tool_data: dict | None = None) -> BotState:
        """
        Derive event from tool name + result data, then transition.
        Some tools emit different events depending on their output
        (e.g. check_payment_status → CAPTURED vs FAILED).
        """
        event = TOOL_EVENTS.get(tool_name)
        if not event:
            return current

        # Special cases: refine event based on tool result data
        if tool_name == "check_payment_status" and tool_data:
            status = (tool_data.get("status") or "").upper()
            if status in ("CAPTURED", "AUTHORIZED"):
                event = Event.PAYMENT_CAPTURED
            elif status == "FAILED":
                event = Event.PAYMENT_FAILED
            else:
                return current  # pending — no transition

        if tool_name == "update_order_status" and tool_data:
            status = (tool_data.get("status") or "").upper()
            if status in ("ACCEPTED", "PREPARING"):
                event = Event.KITCHEN_ACCEPTED
            elif status in ("OUT_FOR_DELIVERY",):
                event = Event.DISPATCHED
            elif status in ("DELIVERED", "COMPLETED"):
                event = Event.DELIVERED

        if tool_name == "update_delivery_status" and tool_data:
            status = (tool_data.get("status") or "").upper()
            if status == "DELIVERED":
                event = Event.DELIVERED
            elif status in ("ASSIGNED", "PICKED_UP", "EN_ROUTE"):
                event = Event.DISPATCHED

        if tool_name == "place_order" and tool_data:
            # If requires payment → PAYMENT_REQUIRED; else → ORDER_PLACED
            if tool_data.get("payment_url") or tool_data.get("payment_required"):
                event = Event.PAYMENT_REQUIRED
            else:
                event = Event.ORDER_PLACED

        return StateMachine.transition(current, event)

    @staticmethod
    def needs_details(current: BotState) -> bool:
        return current in (BotState.BUILDING_CART, BotState.COLLECTING_DETAILS)

    @staticmethod
    def is_terminal(current: BotState) -> bool:
        return current in (BotState.COMPLETED, BotState.IDLE)

    @staticmethod
    def goal_for_state(current: BotState) -> Optional[str]:
        """Map a bot state back to a planner goal for context injection."""
        mapping = {
            BotState.BROWSING_MENU:         "MENU_BROWSE",
            BotState.BUILDING_CART:         "ORDER_ONLINE",
            BotState.COLLECTING_DETAILS:    "ORDER_ONLINE",
            BotState.AWAITING_CONFIRMATION: "ORDER_ONLINE",
            BotState.AWAITING_PAYMENT:      "ORDER_ONLINE",
            BotState.ORDER_PLACED:          "TRACK_ORDER",
            BotState.ORDER_IN_PROGRESS:     "TRACK_ORDER",
            BotState.OUT_FOR_DELIVERY:      "TRACK_ORDER",
            BotState.SUPPORT_CASE:          "SUPPORT",
        }
        return mapping.get(current)
