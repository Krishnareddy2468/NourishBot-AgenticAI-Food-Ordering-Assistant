"""End-to-end evidence for the customer web dashboard planned-meal order entry point.

Runs against a real PostgreSQL database so the confirmation challenge, idempotency
key, snapshot comparison, schedule linking, and durable jobs are exercised exactly
as deployed rather than through fakes.

Point DATABASE_URL at a disposable database migrated to head and drop it afterwards:
the order and schedule audit trails are append-only by database trigger, so a test
cannot and must not delete what it wrote. Every test seeds its own customer, menu
item, plan, and meal, so repeated runs in one disposable database stay independent.

    createdb nourishbot_phase9_check && alembic upgrade head
    PHASE9_WEB_ORDER_INTEGRATION=1 DATABASE_URL=...nourishbot_phase9_check \
        pytest tests/integration/test_phase9_web_order_end_to_end.py
"""

from __future__ import annotations

import os
import secrets
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.core.config import settings
from app.db.models import (
    Channel,
    ConfirmationChallenge,
    Customer,
    MealPlan,
    MealSchedule,
    MenuItem,
    Order,
    Restaurant,
    ScheduledJob,
)
from app.db.models.planning import ScheduleEvent
from app.db.session import AsyncSessionLocal
from app.services.planned_meal_order import (
    PlannedMealOrderError,
    place_planned_meal_web_order,
    preview_planned_meal_web_order,
)


# The application engine is created once at import, so its pooled connections
# belong to the loop that first used them: every test here shares one loop.
pytestmark = [
    pytest.mark.skipif(
        os.getenv("PHASE9_WEB_ORDER_INTEGRATION") != "1",
        reason="Set PHASE9_WEB_ORDER_INTEGRATION=1 with a disposable PostgreSQL database",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]

RESTAURANT_ID = settings.DEFAULT_RESTAURANT_ID
MEAL_PRICE_CENTS = 32000
PLAN_ESTIMATE_CENTS = 30000
DELIVERY_ADDRESS = "12 Park Road, Jubilee Hills, Hyderabad"
TEST_CUSTOMER_NAME = "Phase 9 Web Tester"


def _unique_phone() -> str:
    return "9" + "".join(secrets.choice("0123456789") for _ in range(9))


def _tomorrow_lunch() -> datetime:
    local = datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(days=1)
    return local.replace(hour=13, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


class _Fixture:
    """Seeded identities for one confirmed, orderable planned meal."""

    def __init__(self) -> None:
        self.customer_id: int | None = None
        self.channel_id: int | None = None
        self.menu_item_id: int | None = None
        self.plan_id: int | None = None
        self.meal_id: int | None = None
        self.external_id = f"account:test-{secrets.token_hex(6)}"
        self.contact_phone = _unique_phone()
        self.desired_for = _tomorrow_lunch()

    @property
    def channel(self):
        return Channel(
            id=self.channel_id,
            restaurant_id=RESTAURANT_ID,
            customer_id=self.customer_id,
            external_id=self.external_id,
            type="WEB",
        )


async def _ensure_restaurant(db) -> None:
    """A freshly migrated database carries no restaurant row."""
    existing = await db.get(Restaurant, RESTAURANT_ID)
    if existing:
        return
    db.add(Restaurant(id=RESTAURANT_ID, name="Phase 9 test kitchen", timezone="Asia/Kolkata"))
    await db.flush()


async def _seed() -> _Fixture:
    fixture = _Fixture()
    async with AsyncSessionLocal() as db:
        await _ensure_restaurant(db)
        customer = Customer(
            restaurant_id=RESTAURANT_ID,
            name=TEST_CUSTOMER_NAME,
            # The account signup placeholder: checkout must collect a real number.
            phone=f"web-account-pending-{secrets.token_hex(12)}",
            email=f"phase9-{secrets.token_hex(6)}@example.test",
        )
        db.add(customer)
        await db.flush()

        channel = Channel(
            restaurant_id=RESTAURANT_ID,
            type="WEB",
            external_id=fixture.external_id,
            customer_id=customer.id,
            verified_at=datetime.now(timezone.utc),
            verification_method="AUTHENTICATED_WEB_ACCOUNT",
            linked_at=datetime.now(timezone.utc),
        )
        menu_item = MenuItem(
            restaurant_id=RESTAURANT_ID,
            name=f"Phase 9 Balanced Bowl {secrets.token_hex(3)}",
            type="VEG",
            price_cents=MEAL_PRICE_CENTS,
            available=True,
            tags=["allergen-verified"],
        )
        db.add_all([channel, menu_item])
        await db.flush()

        plan = MealPlan(
            restaurant_id=RESTAURANT_ID,
            customer_id=customer.id,
            name="Phase 9 web handoff plan",
            status="CONFIRMED",
            scope="DAILY",
            starts_on=date.today(),
            ends_on=date.today() + timedelta(days=1),
            timezone="Asia/Kolkata",
            constraints_json={},
            estimated_total_cents=PLAN_ESTIMATE_CENTS,
            currency="INR",
            confirmed_at=datetime.now(timezone.utc),
        )
        db.add(plan)
        await db.flush()

        meal = MealSchedule(
            restaurant_id=RESTAURANT_ID,
            customer_id=customer.id,
            meal_plan_id=plan.id,
            menu_item_id=menu_item.id,
            title="Lunch",
            meal_period="LUNCH",
            state="CONFIRMED",
            desired_for=fixture.desired_for,
            timezone="Asia/Kolkata",
            source="WEB",
            selection_json={"menu_type": "VEG"},
            constraints_json={},
            estimated_cost_cents=PLAN_ESTIMATE_CENTS,
            currency="INR",
            diet_tags=["veg", "allergen-verified"],
            allergen_tags=["allergen-verified"],
            confirmed_at=datetime.now(timezone.utc),
        )
        db.add(meal)
        await db.flush()

        fixture.customer_id = customer.id
        fixture.channel_id = channel.id
        fixture.menu_item_id = menu_item.id
        fixture.plan_id = plan.id
        fixture.meal_id = meal.id
        await db.commit()
    return fixture


@pytest_asyncio.fixture(loop_scope="module")
async def seeded():
    return await _seed()


async def test_web_preview_then_confirmation_places_exactly_one_linked_order(seeded):
    preview = await preview_planned_meal_web_order(
        channel=seeded.channel,
        meal_id=seeded.meal_id,
        order_type="DELIVERY",
        delivery_address=DELIVERY_ADDRESS,
        contact_phone=seeded.contact_phone,
    )

    # The preview stores the exact summary and places nothing.
    assert preview["confirmation_required"] is True
    assert preview["total_cents"] == MEAL_PRICE_CENTS
    assert preview["current_price_cents"] == MEAL_PRICE_CENTS
    assert preview["plan_estimate_cents"] == PLAN_ESTIMATE_CENTS
    assert preview["price_changed_since_plan"] is True
    assert preview["provider"] == "NOURISHBOT"
    async with AsyncSessionLocal() as db:
        assert (await db.execute(select(Order).where(
            Order.customer_id == seeded.customer_id
        ))).scalars().first() is None
        challenge = (await db.execute(select(ConfirmationChallenge).where(
            ConfirmationChallenge.id == preview["confirmation_id"]
        ))).scalar_one()
        assert challenge.used_at is None
        assert challenge.customer_id == seeded.customer_id
        customer = await db.get(Customer, seeded.customer_id)
        assert customer.phone == seeded.contact_phone

    placed = await place_planned_meal_web_order(
        channel=seeded.channel,
        meal_id=seeded.meal_id,
        confirmation_id=preview["confirmation_id"],
        order_type="DELIVERY",
        delivery_address=DELIVERY_ADDRESS,
    )

    assert placed["order_ref"].startswith("NB-")
    async with AsyncSessionLocal() as db:
        orders = (await db.execute(select(Order).where(
            Order.customer_id == seeded.customer_id
        ))).scalars().all()
        assert len(orders) == 1
        order = orders[0]
        assert order.status == "CONFIRMED"
        assert order.provider == "NOURISHBOT"
        assert order.provider_state == "NOT_REQUIRED"
        assert order.source_channel == "WEB"
        assert order.confirmation_id == preview["confirmation_id"]
        assert order.scheduled_for == seeded.desired_for

        # The confirmed planned meal itself becomes the order's schedule.
        schedules = (await db.execute(select(MealSchedule).where(
            MealSchedule.customer_id == seeded.customer_id
        ))).scalars().all()
        assert len(schedules) == 1
        assert schedules[0].id == seeded.meal_id
        assert schedules[0].order_id == order.id
        assert schedules[0].state == "SCHEDULED"

        assert (await db.execute(select(ScheduleEvent).where(
            ScheduleEvent.meal_schedule_id == seeded.meal_id,
            ScheduleEvent.event_type == "SCHEDULE_LINKED_TO_ORDER",
        ))).scalars().first() is not None
        assert (await db.execute(select(ScheduledJob).where(
            ScheduledJob.order_id == order.id
        ))).scalars().first() is not None
        challenge = (await db.execute(select(ConfirmationChallenge).where(
            ConfirmationChallenge.id == preview["confirmation_id"]
        ))).scalar_one()
        assert challenge.used_at is not None


async def test_repeating_the_same_confirmation_returns_the_first_order(seeded):
    preview = await preview_planned_meal_web_order(
        channel=seeded.channel,
        meal_id=seeded.meal_id,
        order_type="DELIVERY",
        delivery_address=DELIVERY_ADDRESS,
        contact_phone=seeded.contact_phone,
    )
    first = await place_planned_meal_web_order(
        channel=seeded.channel,
        meal_id=seeded.meal_id,
        confirmation_id=preview["confirmation_id"],
        order_type="DELIVERY",
        delivery_address=DELIVERY_ADDRESS,
    )
    replay = await place_planned_meal_web_order(
        channel=seeded.channel,
        meal_id=seeded.meal_id,
        confirmation_id=preview["confirmation_id"],
        order_type="DELIVERY",
        delivery_address=DELIVERY_ADDRESS,
    )

    assert replay["order_ref"] == first["order_ref"]
    assert replay.get("idempotent_replay") is True
    async with AsyncSessionLocal() as db:
        orders = (await db.execute(select(Order).where(
            Order.customer_id == seeded.customer_id
        ))).scalars().all()
        assert len(orders) == 1


async def test_a_price_change_after_review_rejects_the_confirmation(seeded):
    preview = await preview_planned_meal_web_order(
        channel=seeded.channel,
        meal_id=seeded.meal_id,
        order_type="DELIVERY",
        delivery_address=DELIVERY_ADDRESS,
        contact_phone=seeded.contact_phone,
    )
    async with AsyncSessionLocal() as db:
        item = await db.get(MenuItem, seeded.menu_item_id)
        item.price_cents = MEAL_PRICE_CENTS + 5000
        await db.commit()

    with pytest.raises(PlannedMealOrderError) as refusal:
        await place_planned_meal_web_order(
            channel=seeded.channel,
            meal_id=seeded.meal_id,
            confirmation_id=preview["confirmation_id"],
            order_type="DELIVERY",
            delivery_address=DELIVERY_ADDRESS,
        )

    assert "changed" in str(refusal.value)
    async with AsyncSessionLocal() as db:
        assert (await db.execute(select(Order).where(
            Order.customer_id == seeded.customer_id
        ))).scalars().first() is None
        challenge = (await db.execute(select(ConfirmationChallenge).where(
            ConfirmationChallenge.id == preview["confirmation_id"]
        ))).scalar_one()
        assert challenge.invalidated_at is not None
        assert challenge.invalidation_reason == "SNAPSHOT_CHANGED"


async def test_an_unavailable_menu_item_blocks_the_preview(seeded):
    async with AsyncSessionLocal() as db:
        item = await db.get(MenuItem, seeded.menu_item_id)
        item.available = False
        await db.commit()

    with pytest.raises(PlannedMealOrderError) as refusal:
        await preview_planned_meal_web_order(
            channel=seeded.channel,
            meal_id=seeded.meal_id,
            order_type="DELIVERY",
            delivery_address=DELIVERY_ADDRESS,
            contact_phone=seeded.contact_phone,
        )

    assert "no longer available" in str(refusal.value)
    async with AsyncSessionLocal() as db:
        assert (await db.execute(select(ConfirmationChallenge).where(
            ConfirmationChallenge.customer_id == seeded.customer_id
        ))).scalars().first() is None


async def test_an_ordered_planned_meal_cannot_be_ordered_again(seeded):
    preview = await preview_planned_meal_web_order(
        channel=seeded.channel,
        meal_id=seeded.meal_id,
        order_type="DELIVERY",
        delivery_address=DELIVERY_ADDRESS,
        contact_phone=seeded.contact_phone,
    )
    await place_planned_meal_web_order(
        channel=seeded.channel,
        meal_id=seeded.meal_id,
        confirmation_id=preview["confirmation_id"],
        order_type="DELIVERY",
        delivery_address=DELIVERY_ADDRESS,
    )

    with pytest.raises(PlannedMealOrderError) as refusal:
        await preview_planned_meal_web_order(
            channel=seeded.channel,
            meal_id=seeded.meal_id,
            order_type="DELIVERY",
            delivery_address=DELIVERY_ADDRESS,
        )

    assert "already ordered" in str(refusal.value)
    async with AsyncSessionLocal() as db:
        orders = (await db.execute(select(Order).where(
            Order.customer_id == seeded.customer_id
        ))).scalars().all()
        assert len(orders) == 1
