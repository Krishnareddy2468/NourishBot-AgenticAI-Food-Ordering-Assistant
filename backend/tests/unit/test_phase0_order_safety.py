from types import SimpleNamespace

import pytest

from app.agent.context_builder import ContextSnapshot
from app.agent.order_confirmation import build_snapshot, is_explicit_confirmation, snapshot_hash
from app.agent.pipeline import _guard_order_plan
from app.agent.planner import PlannerOutput
from app.agent.state_machine import BotState
from app.auth.deps import tenant_id


def _ctx(**overrides):
    values = {
        "chat_id": "chat-1",
        "session_id": 7,
        "restaurant_id": 3,
        "customer_id": 9,
        "customer_name": "Asha",
        "customer_phone": "9999999999",
        "order_type": "DELIVERY",
        "delivery_address": "12 Lake View Road, Hyderabad",
        "scheduled_for": "2026-10-16T13:00:00+05:30",
        "cart": [{"item_id": 2, "item_name": "Millet Bowl", "qty": 1, "unit_price_cents": 25000}],
    }
    values.update(overrides)
    return ContextSnapshot(**values)


def _place_plan():
    return PlannerOutput(
        goal="ORDER_ONLINE",
        proposed_actions=[{"tool": "place_order", "args": {"order_type": "DELIVERY"}}],
        requires_confirmation=True,
    )


def test_only_short_unambiguous_affirmations_count_as_confirmation():
    assert is_explicit_confirmation("Yes") is True
    assert is_explicit_confirmation("go ahead") is True
    assert is_explicit_confirmation("yes, but change the address") is False
    assert is_explicit_confirmation("yesterday was fine") is False


def test_guard_prepares_snapshot_instead_of_placing_without_confirmation():
    guarded = _guard_order_plan(_place_plan(), _ctx(), "show me the summary")

    assert any(action["tool"] == "prepare_order_confirmation" for action in guarded.proposed_actions)
    assert all(action["tool"] != "place_order" for action in guarded.proposed_actions)


def test_guard_allows_exact_confirmed_snapshot_only_in_confirmation_state():
    snapshot = {
        "order_type": "DELIVERY",
        "delivery_address": "12 Lake View Road, Hyderabad",
        "scheduled_for": "2026-10-16T07:30:00+00:00",
        "provider": "NOURISHBOT",
    }
    ctx = _ctx(
        current_state=BotState.AWAITING_CONFIRMATION,
        pending_confirmation_id="challenge-1",
        confirmation_snapshot=snapshot,
    )

    guarded = _guard_order_plan(_place_plan(), ctx, "yes")
    action = next(action for action in guarded.proposed_actions if action["tool"] == "place_order")

    assert action["args"]["confirmation_id"] == "challenge-1"
    assert action["args"]["scheduled_for"] == snapshot["scheduled_for"]
    assert action["args"]["address"] == snapshot["delivery_address"]


def test_snapshot_hash_changes_when_price_or_address_changes():
    ctx = _ctx()
    args = {
        "order_type": "DELIVERY",
        "address": ctx.delivery_address,
        "scheduled_for": ctx.scheduled_for,
        "provider": "NOURISHBOT",
    }
    items = [{"item_id": 2, "item_name": "Millet Bowl", "qty": 1, "unit_price_cents": 25000}]
    original = build_snapshot(ctx=ctx, args=args, items=items)
    price_changed = build_snapshot(
        ctx=ctx,
        args=args,
        items=[{**items[0], "unit_price_cents": 26000}],
    )
    address_changed = build_snapshot(ctx=ctx, args={**args, "address": "99 New Road"}, items=items)

    assert snapshot_hash(original) != snapshot_hash(price_changed)
    assert snapshot_hash(original) != snapshot_hash(address_changed)


def test_tenant_scope_never_defaults_to_restaurant_one():
    assert tenant_id({"restaurant_id": 22}) == 22
    with pytest.raises(Exception):
        tenant_id({})
