"""Canonical order, schedule, provider, and fulfillment state transitions.

The service mutates an already-loaded model and appends its audit event to the
caller's transaction. It never commits, so the state and history remain atomic.
"""

from datetime import datetime, timezone
from typing import Any

from app.db.models.analytics import SpendingLedgerEntry
from app.db.models.planning import OrderEvent, ScheduleEvent


class InvalidStateTransition(ValueError):
    """Raised when a caller asks for an unsupported lifecycle transition."""


LEGACY_ORDER_STATE_MAP = {
    "SCHEDULED": "CONFIRMED",
    "PAID": "CONFIRMED",
    "CONFIRMED_BY_RESTAURANT": "ACCEPTED",
}

ORDER_TRANSITIONS = {
    "CREATED": {"CONFIRMED", "ACCEPTED", "PREPARING", "CANCELLED", "FAILED", "EXPIRED"},
    "CONFIRMED": {"ACCEPTED", "PREPARING", "CANCELLED", "FAILED", "EXPIRED"},
    "ACCEPTED": {"PREPARING", "CANCELLED", "FAILED"},
    "PREPARING": {"READY", "CANCELLED", "FAILED"},
    "READY": {"OUT_FOR_DELIVERY", "DELIVERED", "COMPLETED", "CANCELLED", "FAILED"},
    "OUT_FOR_DELIVERY": {"DELIVERED", "FAILED"},
    "DELIVERED": {"COMPLETED"},
    "COMPLETED": set(),
    "CANCELLED": set(),
    "FAILED": set(),
    "EXPIRED": set(),
}

SCHEDULE_TRANSITIONS = {
    "NOT_SCHEDULED": {"SCHEDULED"},
    "SCHEDULED": {"ACTIVATED", "CANCELLED", "EXPIRED"},
    "ACTIVATED": {"FULFILLED", "CANCELLED", "FAILED"},
    "FULFILLED": set(),
    "CANCELLED": set(),
    "FAILED": set(),
    "EXPIRED": set(),
}

PROVIDER_TRANSITIONS = {
    "NOT_REQUIRED": set(),
    "PENDING": {"SUBMITTING", "CANCELLED", "EXPIRED"},
    "SUBMITTING": {"CONFIRMED", "REJECTED", "UNKNOWN"},
    "UNKNOWN": {"CONFIRMED", "REJECTED", "CANCELLED"},
    "CONFIRMED": {"CANCELLED"},
    "REJECTED": set(),
    "CANCELLED": set(),
    "EXPIRED": set(),
}

FULFILLMENT_BY_ORDER_STATE = {
    "CREATED": "NOT_STARTED",
    "CONFIRMED": "NOT_STARTED",
    "ACCEPTED": "ACCEPTED",
    "PREPARING": "PREPARING",
    "READY": "READY",
    "OUT_FOR_DELIVERY": "OUT_FOR_DELIVERY",
    "DELIVERED": "DELIVERED",
    "COMPLETED": "COMPLETED",
    "CANCELLED": "CANCELLED",
    "FAILED": "FAILED",
    "EXPIRED": "EXPIRED",
}


def canonical_order_state(value: str) -> str:
    state = str(value or "").strip().upper()
    return LEGACY_ORDER_STATE_MAP.get(state, state)


def record_order_event(
    session,
    order,
    *,
    event_type: str,
    to_state: str,
    actor_type: str,
    source: str,
    actor_id: str | int | None = None,
    from_state: str | None = None,
    reason_code: str | None = None,
    metadata: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> OrderEvent:
    audit = OrderEvent(
        restaurant_id=order.restaurant_id,
        order_id=order.id,
        event_type=event_type.upper(),
        from_state=from_state,
        to_state=to_state.upper(),
        actor_type=actor_type.upper(),
        actor_id=str(actor_id) if actor_id is not None else None,
        source=source.upper(),
        reason_code=reason_code,
        metadata_json=metadata or {},
        occurred_at=occurred_at or datetime.now(timezone.utc),
    )
    session.add(audit)
    return audit


def record_schedule_event(
    session,
    schedule,
    *,
    event_type: str,
    to_state: str,
    actor_type: str,
    source: str,
    actor_id: str | int | None = None,
    from_state: str | None = None,
    reason_code: str | None = None,
    metadata: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> ScheduleEvent:
    audit = ScheduleEvent(
        restaurant_id=schedule.restaurant_id,
        meal_schedule_id=schedule.id,
        order_id=schedule.order_id,
        event_type=event_type.upper(),
        from_state=from_state,
        to_state=to_state.upper(),
        actor_type=actor_type.upper(),
        actor_id=str(actor_id) if actor_id is not None else None,
        source=source.upper(),
        reason_code=reason_code,
        metadata_json=metadata or {},
        occurred_at=occurred_at or datetime.now(timezone.utc),
    )
    session.add(audit)
    return audit


def _validate(current: str, target: str, transitions: dict[str, set[str]], domain: str) -> None:
    if current == target:
        raise InvalidStateTransition(f"{domain} is already {target}")
    if current not in transitions:
        raise InvalidStateTransition(f"Unknown {domain.lower()} state: {current}")
    if target not in transitions[current]:
        raise InvalidStateTransition(f"Cannot move {domain.lower()} from {current} to {target}")


def transition_order(
    session,
    order,
    target_state: str,
    *,
    actor_type: str,
    actor_id: str | int | None = None,
    source: str,
    reason_code: str | None = None,
    metadata: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> OrderEvent:
    current = canonical_order_state(order.status)
    target = canonical_order_state(target_state)
    _validate(current, target, ORDER_TRANSITIONS, "Order")

    now = occurred_at or datetime.now(timezone.utc)
    order.status = target
    order.fulfillment_state = FULFILLMENT_BY_ORDER_STATE[target]
    if target in {"ACCEPTED", "PREPARING", "READY", "OUT_FOR_DELIVERY", "DELIVERED", "COMPLETED"}:
        if getattr(order, "submitted_at", None) is None:
            order.submitted_at = now
    if target in {"DELIVERED", "COMPLETED"} and getattr(order, "completed_at", None) is None:
        order.completed_at = now
        customer_id = getattr(order, "customer_id", None)
        amount_cents = int(getattr(order, "total_cents", 0) or 0)
        if customer_id and amount_cents:
            session.add(SpendingLedgerEntry(
                restaurant_id=order.restaurant_id,
                customer_id=customer_id,
                order_id=order.id,
                entry_type="COMPLETED_SPEND",
                amount_cents=amount_cents,
                currency="INR",
                metadata_json={"order_state": target, "source": source.upper()},
                occurred_at=now,
            ))
    if target == "CANCELLED":
        order.cancelled_at = now
        order.terminal_reason = reason_code or "CANCELLED"
    elif target == "FAILED":
        order.failed_at = now
        order.terminal_reason = reason_code or "ORDER_FAILED"
    elif target == "EXPIRED":
        order.expired_at = now
        order.terminal_reason = reason_code or "ORDER_EXPIRED"

    audit = OrderEvent(
        restaurant_id=order.restaurant_id,
        order_id=order.id,
        event_type="ORDER_STATE_CHANGED",
        from_state=current,
        to_state=target,
        actor_type=actor_type.upper(),
        actor_id=str(actor_id) if actor_id is not None else None,
        source=source.upper(),
        reason_code=reason_code,
        metadata_json=metadata or {},
        occurred_at=now,
    )
    session.add(audit)
    return audit


def transition_schedule(
    session,
    order,
    target_state: str,
    *,
    actor_type: str,
    source: str,
    actor_id: str | int | None = None,
    reason_code: str | None = None,
    metadata: dict[str, Any] | None = None,
    meal_schedule=None,
    occurred_at: datetime | None = None,
):
    current = str(order.schedule_state or "NOT_SCHEDULED").upper()
    target = str(target_state or "").strip().upper()
    _validate(current, target, SCHEDULE_TRANSITIONS, "Schedule")
    now = occurred_at or datetime.now(timezone.utc)
    order.schedule_state = target
    if target == "ACTIVATED":
        order.activated_at = now
    elif target == "CANCELLED":
        order.cancelled_at = order.cancelled_at or now
    elif target == "EXPIRED":
        order.expired_at = now

    if meal_schedule is not None:
        meal_schedule.state = target
        audit = ScheduleEvent(
            restaurant_id=order.restaurant_id,
            meal_schedule_id=meal_schedule.id,
            order_id=order.id,
            event_type="SCHEDULE_STATE_CHANGED",
            from_state=current,
            to_state=target,
            actor_type=actor_type.upper(),
            actor_id=str(actor_id) if actor_id is not None else None,
            source=source.upper(),
            reason_code=reason_code,
            metadata_json=metadata or {},
            occurred_at=now,
        )
        session.add(audit)
        return audit

    audit = OrderEvent(
        restaurant_id=order.restaurant_id,
        order_id=order.id,
        event_type="SCHEDULE_STATE_CHANGED",
        from_state=current,
        to_state=target,
        actor_type=actor_type.upper(),
        actor_id=str(actor_id) if actor_id is not None else None,
        source=source.upper(),
        reason_code=reason_code,
        metadata_json=metadata or {},
        occurred_at=now,
    )
    session.add(audit)
    return audit


def transition_provider(
    session,
    order,
    target_state: str,
    *,
    actor_type: str,
    source: str,
    actor_id: str | int | None = None,
    reason_code: str | None = None,
    metadata: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> OrderEvent:
    current = str(order.provider_state or "NOT_REQUIRED").upper()
    target = str(target_state or "").strip().upper()
    _validate(current, target, PROVIDER_TRANSITIONS, "Provider")
    now = occurred_at or datetime.now(timezone.utc)
    order.provider_state = target
    audit = OrderEvent(
        restaurant_id=order.restaurant_id,
        order_id=order.id,
        event_type="PROVIDER_STATE_CHANGED",
        from_state=current,
        to_state=target,
        actor_type=actor_type.upper(),
        actor_id=str(actor_id) if actor_id is not None else None,
        source=source.upper(),
        reason_code=reason_code,
        metadata_json=metadata or {},
        occurred_at=now,
    )
    session.add(audit)
    if target == "REJECTED" and canonical_order_state(order.status) not in {
        "COMPLETED", "CANCELLED", "FAILED", "EXPIRED"
    }:
        transition_order(
            session,
            order,
            "FAILED",
            actor_type=actor_type,
            actor_id=actor_id,
            source=source,
            reason_code=reason_code or "PROVIDER_REJECTED",
            metadata=metadata,
            occurred_at=now,
        )
    return audit


def sync_terminal_schedule(
    session,
    order,
    *,
    meal_schedule=None,
    actor_type: str,
    source: str,
    actor_id: str | int | None = None,
    reason_code: str | None = None,
    occurred_at: datetime | None = None,
):
    """Mirror an order terminal outcome into its independent schedule state."""
    order_state = canonical_order_state(order.status)
    schedule_state = str(order.schedule_state or "NOT_SCHEDULED").upper()
    target = None
    if order_state in {"DELIVERED", "COMPLETED"} and schedule_state == "ACTIVATED":
        target = "FULFILLED"
    elif order_state == "CANCELLED" and schedule_state in {"SCHEDULED", "ACTIVATED"}:
        target = "CANCELLED"
    elif order_state == "FAILED" and schedule_state == "ACTIVATED":
        target = "FAILED"
    elif order_state == "EXPIRED" and schedule_state == "SCHEDULED":
        target = "EXPIRED"
    if target is None:
        return None
    return transition_schedule(
        session,
        order,
        target,
        actor_type=actor_type,
        actor_id=actor_id,
        source=source,
        reason_code=reason_code,
        meal_schedule=meal_schedule,
        occurred_at=occurred_at,
    )
