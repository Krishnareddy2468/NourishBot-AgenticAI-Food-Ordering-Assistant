"""Phase 9 — direct planned-meal ordering from the customer web dashboard.

The dashboard must reach the same confirmation, provider-capability, and
idempotency controls as the chat handoff, and must refuse for the same reasons.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.agent.executor import ActionResult, Executor, _INTERNAL_TOOL_ARGS
from app.services import planned_meal_order
from app.services.planned_meal_order import (
    PlannedMealOrderError,
    _resolve_contact_phone,
    load_orderable_planned_meal,
    place_planned_meal_web_order,
    preview_planned_meal_web_order,
)


TOMORROW = datetime.now(timezone.utc) + timedelta(days=1)


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def first(self):
        return self._value


class _FakeDatabase:
    def __init__(self, values):
        self.values = list(values)
        self.added = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def execute(self, statement):
        return _FakeResult(self.values.pop(0) if self.values else None)

    def add(self, instance):
        self.added.append(instance)

    async def commit(self):
        self.commits += 1


def _fixtures(*, phone="9876543210", tags=None, price_cents=30000):
    meal = SimpleNamespace(
        id=17,
        meal_plan_id=4,
        restaurant_id=1,
        customer_id=7,
        order_id=None,
        state="CONFIRMED",
        desired_for=TOMORROW,
        menu_item_id=23,
        diet_tags=["veg", "allergen-verified"],
        allergen_tags=["allergen-verified"],
        selection_json={"menu_type": "VEG"},
        currency="INR",
        estimated_cost_cents=28000,
    )
    plan = SimpleNamespace(
        id=4,
        restaurant_id=1,
        customer_id=7,
        status="CONFIRMED",
        constraints_json={},
        timezone="Asia/Kolkata",
    )
    menu_item = SimpleNamespace(
        id=23,
        name="Balanced Bowl",
        type="VEG",
        tags=tags if tags is not None else ["allergen-verified"],
        price_cents=price_cents,
        available=True,
        stock_qty=None,
    )
    customer = SimpleNamespace(id=7, restaurant_id=1, name="Asha", phone=phone)
    return meal, plan, menu_item, customer


def _channel():
    return SimpleNamespace(id=55, restaurant_id=1, customer_id=7, external_id="account:9")


def _context(*, cart=None, ordering_platform="NOURISHBOT"):
    return SimpleNamespace(
        session_id=31,
        customer_id=7,
        channel_id=55,
        restaurant_id=1,
        ordering_platform=ordering_platform,
        cart=list(cart or []),
        cart_total_cents=0,
        order_type=None,
        delivery_address="",
        scheduled_for=None,
        policy=SimpleNamespace(min_order_for_delivery_cents=20000),
    )


def _install(
    monkeypatch,
    *,
    values,
    ctx,
    tool=None,
    tool_name="_tool_prepare_order_confirmation",
    replayed_order=None,
):
    """Point the service at fake storage, a fake context, and a fake executor tool."""
    databases = [_FakeDatabase(values), _FakeDatabase([SimpleNamespace(id=31)])]
    if tool_name == "_tool_place_order":
        # Confirming first asks whether this confirmation already placed an order.
        databases.insert(0, _FakeDatabase([replayed_order]))
    monkeypatch.setattr(planned_meal_order, "AsyncSessionLocal", lambda: databases.pop(0))

    from app.agent import context_builder, executor

    async def fake_build_context(*args, **kwargs):
        return ctx

    monkeypatch.setattr(context_builder, "build_context", fake_build_context)
    if tool is not None:
        monkeypatch.setattr(executor, tool_name, tool)
    return databases


# ── Shared guards ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_and_chat_share_one_validator_that_refuses_changed_menu_metadata():
    meal, plan, menu_item, _ = _fixtures(tags=["allergen-peanut"])
    db = _FakeDatabase([meal, plan, menu_item])

    with pytest.raises(PlannedMealOrderError) as refusal:
        await load_orderable_planned_meal(db, meal_id=17, restaurant_id=1, customer_id=7)

    assert "metadata changed" in str(refusal.value)
    assert "reconfirm" in str(refusal.value)


@pytest.mark.parametrize(
    ("stored", "supplied", "expected"),
    [
        ("web-account-pending-abc", "", "Add the mobile number"),
        ("web-account-pending-abc", "98765", "8 to 15 digits"),
        ("web-account-pending-abc", "98-76-543", "8 to 15 digits"),
        ("9876543210", "9123456780", "already has a contact number"),
    ],
)
def test_contact_phone_is_required_valid_and_never_silently_replaced(stored, supplied, expected):
    customer = SimpleNamespace(phone=stored)

    with pytest.raises(PlannedMealOrderError) as refusal:
        _resolve_contact_phone(customer, supplied)

    assert expected in str(refusal.value)
    assert customer.phone == stored


def test_contact_phone_is_accepted_once_for_an_account_without_one():
    customer = SimpleNamespace(phone="web-account-pending-abc")

    assert _resolve_contact_phone(customer, "9876543210") == "9876543210"
    # An account that already has a number keeps it and needs no write.
    assert _resolve_contact_phone(SimpleNamespace(phone="9876543210"), "9876543210") is None


@pytest.mark.asyncio
async def test_preview_refuses_a_contact_number_held_by_another_profile(monkeypatch):
    meal, plan, menu_item, customer = _fixtures(phone="web-account-pending-abc")
    _install(
        monkeypatch,
        values=[meal, plan, menu_item, customer, 99],
        ctx=_context(),
    )

    with pytest.raises(PlannedMealOrderError) as refusal:
        await preview_planned_meal_web_order(
            channel=_channel(),
            meal_id=17,
            order_type="DELIVERY",
            delivery_address="12 Park Road",
            contact_phone="9876543210",
        )

    assert "another NourishBot profile" in str(refusal.value)
    assert customer.phone == "web-account-pending-abc"


# ── Dashboard checkout ────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("order_type", "address", "expected"),
    [
        ("DINE_IN", "12 Park Road", "home delivery or pickup"),
        ("DELIVERY", "", "delivery address is required"),
    ],
)
async def test_preview_rejects_unsupported_fulfillment_before_any_lookup(order_type, address, expected):
    with pytest.raises(PlannedMealOrderError) as refusal:
        await preview_planned_meal_web_order(
            channel=_channel(),
            meal_id=17,
            order_type=order_type,
            delivery_address=address,
        )

    assert expected in str(refusal.value).lower()


@pytest.mark.asyncio
async def test_preview_refuses_while_another_cart_is_open(monkeypatch):
    meal, plan, menu_item, customer = _fixtures()
    _install(
        monkeypatch,
        values=[meal, plan, menu_item, customer],
        ctx=_context(cart=[{"item_id": 2, "item_name": "Chai", "qty": 1, "unit_price_cents": 5000}]),
    )

    with pytest.raises(PlannedMealOrderError) as refusal:
        await preview_planned_meal_web_order(
            channel=_channel(),
            meal_id=17,
            order_type="DELIVERY",
            delivery_address="12 Park Road",
        )

    assert "cart is already open" in str(refusal.value)
    assert "No order was placed" in str(refusal.value)


@pytest.mark.asyncio
async def test_preview_fails_closed_for_an_external_provider_session(monkeypatch):
    meal, plan, menu_item, customer = _fixtures()
    _install(
        monkeypatch,
        values=[meal, plan, menu_item, customer],
        ctx=_context(ordering_platform="SWIGGY"),
    )
    from app.providers import service as provider_service

    async def fake_experience(provider):
        return {"mode_label": "Mode A — safe discovery (transactional release gate closed)"}

    monkeypatch.setattr(provider_service, "provider_experience", fake_experience)

    with pytest.raises(PlannedMealOrderError) as refusal:
        await preview_planned_meal_web_order(
            channel=_channel(),
            meal_id=17,
            order_type="DELIVERY",
            delivery_address="12 Park Road",
        )

    message = str(refusal.value)
    assert "Mode A — safe discovery" in message
    assert "WhatsApp or Telegram" in message
    assert "No order was placed" in message


@pytest.mark.asyncio
async def test_preview_refuses_delivery_below_the_restaurant_minimum(monkeypatch):
    meal, plan, menu_item, customer = _fixtures(price_cents=9000)
    _install(monkeypatch, values=[meal, plan, menu_item, customer], ctx=_context())

    with pytest.raises(PlannedMealOrderError) as refusal:
        await preview_planned_meal_web_order(
            channel=_channel(),
            meal_id=17,
            order_type="DELIVERY",
            delivery_address="12 Park Road",
        )

    assert "Minimum order for delivery is ₹200" in str(refusal.value)


@pytest.mark.asyncio
async def test_preview_returns_the_stored_confirmation_and_the_current_price(monkeypatch):
    meal, plan, menu_item, customer = _fixtures()
    ctx = _context()
    captured = {}

    async def fake_prepare(args, context, result):
        captured["args"] = args
        captured["cart"] = list(context.cart)
        captured["scheduled_for"] = context.scheduled_for
        return ActionResult("prepare_order_confirmation", True, data={
            "confirmation_id": "challenge-1",
            "confirmation_required": True,
            "order_type": "DELIVERY",
            "delivery_address": "12 Park Road",
            "scheduled_for": meal.desired_for.isoformat(),
            "provider": "NOURISHBOT",
            "total_cents": 30000,
            "items": [{"item_id": 23, "item_name": "Balanced Bowl", "qty": 1, "unit_price_cents": 30000}],
            "expires_at": "2026-10-18T10:00:00+00:00",
        })

    _install(monkeypatch, values=[meal, plan, menu_item, customer], ctx=ctx, tool=fake_prepare)

    preview = await preview_planned_meal_web_order(
        channel=_channel(),
        meal_id=17,
        order_type="DELIVERY",
        delivery_address="12 Park Road",
    )

    assert preview["confirmation_id"] == "challenge-1"
    assert preview["confirmation_required"] is True
    assert "Nothing has been ordered yet." in preview["message"]
    # The plan estimate never replaces the live price in the reviewed summary.
    assert preview["current_price_cents"] == 30000
    assert preview["plan_estimate_cents"] == 28000
    assert preview["price_changed_since_plan"] is True
    assert captured["args"]["provider"] == "NOURISHBOT"
    assert captured["cart"] == [{
        "item_id": 23,
        "item_name": "Balanced Bowl",
        "qty": 1,
        "unit_price_cents": 30000,
        "available": True,
        "type": "VEG",
    }]
    assert captured["scheduled_for"] == meal.desired_for.astimezone(timezone.utc).isoformat()


@pytest.mark.asyncio
async def test_confirm_places_the_order_through_the_shared_confirmation_path(monkeypatch):
    meal, plan, menu_item, customer = _fixtures()
    captured = {}

    async def fake_place(args, context, result):
        captured["args"] = args
        return ActionResult("place_order", True, data={
            "order_ref": "NB-ABC123",
            "order_type": "DELIVERY",
            "scheduled_for": meal.desired_for.isoformat(),
            "provider": "NOURISHBOT",
            "tracking_token": "token",
        })

    _install(
        monkeypatch,
        values=[meal, plan, menu_item, customer],
        ctx=_context(),
        tool=fake_place,
        tool_name="_tool_place_order",
    )

    placed = await place_planned_meal_web_order(
        channel=_channel(),
        meal_id=17,
        confirmation_id="challenge-1",
        order_type="DELIVERY",
        delivery_address="12 Park Road",
    )

    assert placed["order_ref"] == "NB-ABC123"
    assert captured["args"]["confirmation_id"] == "challenge-1"
    # The confirmed planned meal becomes the order's schedule instead of a duplicate.
    assert captured["args"]["meal_schedule_id"] == 17
    assert captured["args"]["provider"] == "NOURISHBOT"


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_error", [
    "That confirmation expired or was already used. Please review the order again.",
    "The cart, price, address, provider, or meal time changed. Please review and confirm the updated summary.",
])
async def test_confirm_surfaces_stale_and_changed_confirmation_refusals(monkeypatch, executor_error):
    meal, plan, menu_item, customer = _fixtures()

    async def fake_place(args, context, result):
        return ActionResult("place_order", False, error=executor_error)

    _install(
        monkeypatch,
        values=[meal, plan, menu_item, customer],
        ctx=_context(),
        tool=fake_place,
        tool_name="_tool_place_order",
    )

    with pytest.raises(PlannedMealOrderError) as refusal:
        await place_planned_meal_web_order(
            channel=_channel(),
            meal_id=17,
            confirmation_id="challenge-1",
            order_type="DELIVERY",
            delivery_address="12 Park Road",
        )

    assert str(refusal.value) == executor_error


@pytest.mark.asyncio
async def test_confirm_revalidates_the_plan_before_reaching_the_order_tool(monkeypatch):
    meal, plan, menu_item, customer = _fixtures()
    meal.state = "DRAFT"
    plan.status = "DRAFT"

    async def fake_place(args, context, result):  # pragma: no cover - must not run
        raise AssertionError("place_order must not be reached for an unconfirmed meal")

    _install(
        monkeypatch,
        values=[meal, plan, menu_item, customer],
        ctx=_context(),
        tool=fake_place,
        tool_name="_tool_place_order",
    )

    with pytest.raises(PlannedMealOrderError) as refusal:
        await place_planned_meal_web_order(
            channel=_channel(),
            meal_id=17,
            confirmation_id="challenge-1",
            order_type="DELIVERY",
            delivery_address="12 Park Road",
        )

    assert "Confirm and review plan #4 first" in str(refusal.value)


@pytest.mark.asyncio
async def test_resubmitting_a_confirmation_returns_the_first_order(monkeypatch):
    meal, plan, menu_item, customer = _fixtures()
    placed = SimpleNamespace(
        order_ref="NB-ABC123",
        order_type="DELIVERY",
        scheduled_for=TOMORROW,
        provider="NOURISHBOT",
        tracking_token="token",
    )

    async def fake_place(args, context, result):  # pragma: no cover - must not run
        raise AssertionError("a replayed confirmation must not reach place_order again")

    _install(
        monkeypatch,
        values=[meal, plan, menu_item, customer],
        ctx=_context(),
        tool=fake_place,
        tool_name="_tool_place_order",
        replayed_order=placed,
    )

    replay = await place_planned_meal_web_order(
        channel=_channel(),
        meal_id=17,
        confirmation_id="challenge-1",
        order_type="DELIVERY",
        delivery_address="12 Park Road",
    )

    assert replay["order_ref"] == "NB-ABC123"
    assert replay["idempotent_replay"] is True
    assert replay["meal_id"] == 17


@pytest.mark.asyncio
async def test_an_already_ordered_planned_meal_is_refused_plainly():
    meal, plan, menu_item, _ = _fixtures()
    meal.order_id = 4321
    db = _FakeDatabase([meal, plan, menu_item])

    with pytest.raises(PlannedMealOrderError) as refusal:
        await load_orderable_planned_meal(db, meal_id=17, restaurant_id=1, customer_id=7)

    assert "already ordered" in str(refusal.value)


# ── Planner isolation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_proposals_cannot_set_the_internal_meal_schedule_argument(monkeypatch):
    assert "meal_schedule_id" in _INTERNAL_TOOL_ARGS
    seen = {}

    async def fake_place(args, ctx, result):
        seen.update(args)
        return ActionResult("place_order", True, data={"order_ref": "NB-TEST"})

    from app.agent import executor

    monkeypatch.setitem(executor._TOOL_REGISTRY, "place_order", fake_place)

    await Executor.run(
        [{"tool": "place_order", "args": {"order_type": "PICKUP", "meal_schedule_id": 999}}],
        _context(),
    )

    assert "meal_schedule_id" not in seen
    assert seen["order_type"] == "PICKUP"
