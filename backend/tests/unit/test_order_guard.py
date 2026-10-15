from app.agent.context_builder import ContextSnapshot
from app.agent.pipeline import _extract_order_type, _guard_order_plan
from app.agent.planner import PlannerOutput


def _plan():
    return PlannerOutput(
        goal="ORDER_ONLINE",
        proposed_actions=[{"tool": "place_order", "args": {"order_type": "DELIVERY"}}],
        requires_confirmation=True,
    )


def test_guard_blocks_order_without_future_schedule():
    ctx = ContextSnapshot(
        chat_id="test-user",
        customer_name="Asha",
        customer_phone="9999999999",
        cart=[{"item_id": 1, "item_name": "Bowl", "qty": 1, "unit_price_cents": 25000}],
    )

    guarded = _guard_order_plan(_plan(), ctx)

    assert "scheduled_for" in guarded.missing_slots
    assert all(action["tool"] != "place_order" for action in guarded.proposed_actions)


def test_guard_preserves_delivery_and_injects_schedule_and_address():
    scheduled_for = "2026-10-16T13:00:00+05:30"
    ctx = ContextSnapshot(
        chat_id="test-user",
        customer_name="Asha",
        customer_phone="9999999999",
        scheduled_for=scheduled_for,
        delivery_address="12 Lake View Road, Hyderabad",
        cart=[{"item_id": 1, "item_name": "Bowl", "qty": 1, "unit_price_cents": 25000}],
    )

    guarded = _guard_order_plan(_plan(), ctx)
    action = next(item for item in guarded.proposed_actions if item["tool"] == "prepare_order_confirmation")

    assert action["args"]["order_type"] == "DELIVERY"
    assert action["args"]["scheduled_for"] == scheduled_for
    assert action["args"]["address"] == "12 Lake View Road, Hyderabad"


def test_guard_requires_address_for_home_delivery():
    ctx = ContextSnapshot(
        chat_id="delivery-user",
        customer_name="Asha",
        customer_phone="9999999999",
        scheduled_for="2026-10-16T19:30:00+05:30",
        cart=[{"item_id": 1, "item_name": "Bowl", "qty": 1, "unit_price_cents": 25000}],
    )

    guarded = _guard_order_plan(_plan(), ctx)

    assert "delivery_address" in guarded.missing_slots
    assert all(action["tool"] != "place_order" for action in guarded.proposed_actions)


def test_guard_preserves_scheduled_drive_through():
    scheduled_for = "2026-10-16T19:30:00+05:30"
    ctx = ContextSnapshot(
        chat_id="drive-user",
        customer_name="Asha",
        customer_phone="9999999999",
        scheduled_for=scheduled_for,
        cart=[{"item_id": 1, "item_name": "Bowl", "qty": 1, "unit_price_cents": 25000}],
    )
    plan = PlannerOutput(
        goal="ORDER_ONLINE",
        proposed_actions=[
            {"tool": "set_order_type", "args": {"order_type": "DRIVE_THRU"}},
            {"tool": "place_order", "args": {}},
        ],
        requires_confirmation=True,
    )

    guarded = _guard_order_plan(plan, ctx)

    assert all(
        action["args"].get("order_type") == "DRIVE_THRU"
        for action in guarded.proposed_actions
        if action["tool"] in ("set_order_type", "prepare_order_confirmation")
    )


def test_extracts_home_delivery_without_confusing_drive_through():
    assert _extract_order_type("Please deliver dinner to my home tomorrow") == "DELIVERY"
    assert _extract_order_type("Schedule drive-through tomorrow") == "DRIVE_THRU"
