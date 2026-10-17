import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi import HTTPException

from app.agent import meal_planner_commands
from app.agent.meal_planner_commands import _MEAL_EDIT_RE, _nutrition_targets
from app.api.routes.meal_plans import NutritionTargets, PlanCreate, _daily_nutrition_payload
from app.api.routes.menu import _nutrition_data
from app.domain.meal_planning import (
    MealPlanningError,
    MenuCandidate,
    PlanConstraints,
    normalize_nutrition_targets,
    recommend_complete_plan,
)
from app.services.meal_planner import update_planned_meal


TARGETS = {
    "energy_kcal": 1900,
    "protein_g": 100,
    "carbohydrates_g": 200,
    "fat_g": 60,
    "fiber_g": 30,
}


def verified_candidates():
    values = [
        (1, "Breakfast Bowl", "breakfast", 450, 25, 50, 15, 8),
        (2, "Lunch Plate", "lunch", 550, 30, 60, 20, 9),
        (3, "Fruit Snack", "snacks", 250, 15, 25, 8, 5),
        (4, "Dinner Plate", "dinner", 650, 30, 65, 17, 8),
    ]
    return [
        MenuCandidate(
            id=item_id,
            name=name,
            price_cents=300,
            menu_type="VEG",
            tags=(period, "allergen-verified"),
            nutrition_status="VERIFIED",
            nutrition_source="Restaurant nutrition sheet, rev 3",
            nutrition_serving_g=300,
            nutrition_per_serving={
                "energy_kcal": energy,
                "protein_g": protein,
                "carbohydrates_g": carbs,
                "fat_g": fat,
                "fiber_g": fiber,
            },
        )
        for item_id, name, period, energy, protein, carbs, fat, fiber in values
    ]


def test_verified_nutrition_targets_select_complete_day_and_preserve_verified_facts():
    plan = recommend_complete_plan(
        verified_candidates(),
        scope="DAILY",
        starts_on=date(2026, 10, 19),
        timezone_name="Asia/Kolkata",
        constraints=PlanConstraints.from_values(nutrition_targets=TARGETS),
    )

    assert len(plan) == 4
    assert {meal.candidate.nutrition_status for meal in plan} == {"VERIFIED"}
    assert {
        meal.candidate.nutrition_per_serving["energy_kcal"]
        for meal in plan
    } == {450, 550, 250, 650}
    assert sum(meal.candidate.nutrition_per_serving["protein_g"] for meal in plan) == 100


def test_nutrition_target_plan_fails_closed_without_complete_verified_menu_facts():
    candidates = verified_candidates()
    candidates[0] = MenuCandidate(
        id=1,
        name="Estimated Breakfast",
        price_cents=200,
        menu_type="VEG",
        nutrition_status="ESTIMATED",
        nutrition_source="Model estimate",
        nutrition_serving_g=300,
        nutrition_per_serving={
            "energy_kcal": 450,
            "protein_g": 25,
            "carbohydrates_g": 50,
            "fat_g": 15,
            "fiber_g": 8,
        },
    )

    with pytest.raises(MealPlanningError, match="verified nutrition data"):
        recommend_complete_plan(
            candidates[:1],
            scope="DAILY",
            starts_on=date(2026, 10, 19),
            timezone_name="Asia/Kolkata",
            constraints=PlanConstraints.from_values(nutrition_targets=TARGETS),
        )


@pytest.mark.parametrize(
    "targets",
    [
        {"protein_g": 100},
        {**TARGETS, "fiber_g": 0},
        {**TARGETS, "fiber_g": float("nan")},
        {**TARGETS, "fiber_g": True},
    ],
)
def test_nutrition_targets_require_all_positive_finite_values(targets):
    with pytest.raises(MealPlanningError):
        normalize_nutrition_targets(targets)


def test_manager_nutrition_entry_requires_provenance_and_complete_verified_serving():
    valid = _nutrition_data({
        "energy_kcal": 450,
        "protein_g": 25,
        "carbohydrates_g": 50,
        "fat_g": 15,
        "fiber_g": 8,
        "nutrition_serving_g": 300,
        "nutrition_source": "Restaurant nutrition sheet, rev 3",
        "nutrition_status": "VERIFIED",
    })
    assert valid["nutrition_status"] == "VERIFIED"

    with pytest.raises(HTTPException, match="all five values"):
        _nutrition_data({
            "protein_g": 25,
            "nutrition_serving_g": 300,
            "nutrition_source": "Restaurant nutrition sheet",
            "nutrition_status": "VERIFIED",
        })

    with pytest.raises(HTTPException, match="cannot include"):
        _nutrition_data({
            "energy_kcal": 450,
            "nutrition_status": "UNKNOWN",
        })


def test_nutrition_plan_api_accepts_user_targets_and_rejects_non_positive_values():
    target_model = NutritionTargets.model_validate(TARGETS)
    assert target_model.model_dump() == TARGETS
    with pytest.raises(ValueError):
        NutritionTargets.model_validate({**TARGETS, "protein_g": 0})

    request = PlanCreate.model_validate({
        "scope": "DAILY",
        "starts_on": "2026-10-19",
        "nutrition_targets": TARGETS,
    })
    assert request.nutrition_targets is not None


def test_chat_nutrition_target_parser_requires_all_target_components():
    targets = _nutrition_targets(
        "plan my day calories 1900 protein 100g carbs 200g fat 60g fiber 30g"
    )
    assert targets == TARGETS

    with pytest.raises(MealPlanningError, match="all five daily targets"):
        _nutrition_targets("plan my day protein 100g carbs 200g")


def test_chat_planned_meal_edit_parser_requires_identifiers_and_accepts_timezone():
    match = _MEAL_EDIT_RE.match(
        "change planned meal #17 to menu item #23 at 2026-10-20T08:30:00+05:30"
    )

    assert match is not None
    assert match.groups() == ("17", "23", "2026-10-20T08:30:00+05:30")
    assert _MEAL_EDIT_RE.match("change planned meal 17 to menu item 23")
    assert meal_planner_commands._ORDER_PLANNED_MEAL_RE.match("order planned meal #17")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("desired_for", "message"),
    [
        (datetime(2026, 10, 20, 8, 0), "include a timezone"),
        (datetime.now(timezone.utc) - timedelta(minutes=1), "in the future"),
    ],
)
async def test_planned_meal_edits_require_a_future_timezone_aware_time(desired_for, message):
    plan = SimpleNamespace(id=1, customer_id=7, constraints_json={})
    meal = SimpleNamespace(
        meal_plan_id=1,
        customer_id=7,
        meal_period="BREAKFAST",
        title="Current meal",
    )

    with pytest.raises(MealPlanningError, match=message):
        await update_planned_meal(
            None,
            plan=plan,
            schedule=meal,
            desired_for=desired_for,
        )


class _FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _FakeMealPlanDatabase:
    def __init__(self, values):
        self.values = list(values)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def execute(self, statement):
        return _FakeScalarResult(self.values.pop(0))


def _planned_order_fixtures(*, tags=None):
    meal = SimpleNamespace(
        id=17,
        meal_plan_id=4,
        restaurant_id=1,
        customer_id=7,
        order_id=None,
        state="CONFIRMED",
        desired_for=datetime.now(timezone.utc) + timedelta(days=1),
        menu_item_id=23,
        diet_tags=["veg", "allergen-verified"],
        allergen_tags=["allergen-verified"],
        selection_json={"menu_type": "VEG"},
        currency="INR",
        estimated_cost_cents=300,
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
        price_cents=300,
        available=True,
        stock_qty=None,
    )
    return meal, plan, menu_item


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["NourishBot", "SWIGGY"])
async def test_ordering_a_confirmed_plan_meal_routes_through_existing_chat_confirmation(
    monkeypatch,
    provider,
):
    meal, plan, menu_item = _planned_order_fixtures()
    fake_db = _FakeMealPlanDatabase([meal, plan, menu_item])
    monkeypatch.setattr(meal_planner_commands, "get_session_factory", lambda: lambda: fake_db)
    monkeypatch.setattr(
        meal_planner_commands,
        "get_session",
        lambda *args: asyncio.sleep(0, result={
            "ordering_platform": provider,
            "user_name": "Asha",
            "cart": [],
        }),
    )
    captured = {}

    async def fake_checkout(*args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return "Please review the individual order and confirm it explicitly."

    if provider == "NourishBot":
        from app.agent import pipeline
        monkeypatch.setattr(pipeline, "process_message", fake_checkout)
    else:
        from app.agent import mcp_agent
        monkeypatch.setattr(mcp_agent, "get_mcp_response", fake_checkout)

    reply = await meal_planner_commands._start_planned_meal_order(
        meal_id=17,
        chat_id="chat-1",
        channel_type="WHATSAPP",
        channel=SimpleNamespace(customer_id=7, restaurant_id=1),
    )

    assert "confirm it explicitly" in reply
    assert captured["planning_context"].startswith(
        "Meal #17 from confirmed plan #4"
    )
    assert "fulfillment DELIVERY" in captured["planning_context"]
    request = captured.get("user_message") or captured["args"][0]
    assert meal.desired_for.astimezone(ZoneInfo("Asia/Kolkata")).isoformat() in request


@pytest.mark.asyncio
async def test_plan_meal_order_refuses_nonempty_inhouse_cart(monkeypatch):
    meal, plan, menu_item = _planned_order_fixtures()
    fake_db = _FakeMealPlanDatabase([meal, plan, menu_item])
    monkeypatch.setattr(meal_planner_commands, "get_session_factory", lambda: lambda: fake_db)
    monkeypatch.setattr(
        meal_planner_commands,
        "get_session",
        lambda *args: asyncio.sleep(0, result={
            "ordering_platform": "NourishBot",
            "cart": [{"item_name": "Existing item"}],
        }),
    )

    reply = await meal_planner_commands._start_planned_meal_order(
        meal_id=17,
        chat_id="chat-1",
        channel_type="WHATSAPP",
        channel=SimpleNamespace(customer_id=7, restaurant_id=1),
    )

    assert "existing cart is not empty" in reply
    assert "No order was placed" in reply


@pytest.mark.asyncio
async def test_plan_meal_order_requires_reconfirmation_after_menu_metadata_changes(monkeypatch):
    meal, plan, menu_item = _planned_order_fixtures(tags=["allergen-peanut"])
    fake_db = _FakeMealPlanDatabase([meal, plan, menu_item])
    monkeypatch.setattr(meal_planner_commands, "get_session_factory", lambda: lambda: fake_db)

    reply = await meal_planner_commands._start_planned_meal_order(
        meal_id=17,
        chat_id="chat-1",
        channel_type="WHATSAPP",
        channel=SimpleNamespace(customer_id=7, restaurant_id=1),
    )

    assert "metadata changed" in reply
    assert "reconfirm" in reply


@pytest.mark.asyncio
async def test_nutrition_target_plan_order_refuses_changed_verified_facts(monkeypatch):
    meal, plan, menu_item = _planned_order_fixtures()
    plan.constraints_json = {"nutrition_targets": TARGETS}
    meal.selection_json = {
        "menu_type": "VEG",
        "nutrition": {
            "status": "VERIFIED",
            "source": "Restaurant nutrition sheet",
            "serving_g": 300,
            "per_serving": {
                "energy_kcal": 450,
                "protein_g": 25,
                "carbohydrates_g": 50,
                "fat_g": 15,
                "fiber_g": 8,
            },
        },
    }
    menu_item.nutrition_status = "VERIFIED"
    menu_item.nutrition_source = "Restaurant nutrition sheet"
    menu_item.nutrition_serving_g = 300
    menu_item.energy_kcal = 450
    menu_item.protein_g = 26
    menu_item.carbohydrates_g = 50
    menu_item.fat_g = 15
    menu_item.fiber_g = 8
    fake_db = _FakeMealPlanDatabase([meal, plan, menu_item])
    monkeypatch.setattr(meal_planner_commands, "get_session_factory", lambda: lambda: fake_db)

    reply = await meal_planner_commands._start_planned_meal_order(
        meal_id=17,
        chat_id="chat-1",
        channel_type="WHATSAPP",
        channel=SimpleNamespace(customer_id=7, restaurant_id=1),
    )

    assert "Verified nutrition facts changed" in reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("meal_state", "plan_status", "desired_for", "menu_item", "expected"),
    [
        ("DRAFT", "DRAFT", datetime.now(timezone.utc) + timedelta(days=1), object(), "Confirm and review"),
        ("CONFIRMED", "CONFIRMED", datetime.now(timezone.utc) - timedelta(minutes=1), object(), "time has passed"),
        ("CONFIRMED", "CONFIRMED", datetime.now(timezone.utc) + timedelta(days=1), None, "no longer available"),
    ],
)
async def test_plan_meal_order_fails_closed_for_unconfirmed_stale_or_unavailable_data(
    monkeypatch,
    meal_state,
    plan_status,
    desired_for,
    menu_item,
    expected,
):
    meal, plan, _ = _planned_order_fixtures()
    meal.state = meal_state
    meal.desired_for = desired_for
    plan.status = plan_status
    values = [meal, plan]
    if meal_state == "CONFIRMED" and desired_for > datetime.now(timezone.utc):
        values.append(menu_item)
    fake_db = _FakeMealPlanDatabase(values)
    monkeypatch.setattr(meal_planner_commands, "get_session_factory", lambda: lambda: fake_db)

    reply = await meal_planner_commands._start_planned_meal_order(
        meal_id=17,
        chat_id="chat-1",
        channel_type="WHATSAPP",
        channel=SimpleNamespace(customer_id=7, restaurant_id=1),
    )

    assert expected.lower() in reply.lower()


def test_daily_nutrition_api_reports_known_values_without_guessing_unknown_meals():
    day = SimpleNamespace(
        desired_for=datetime(2026, 10, 19, 3, 0, tzinfo=timezone.utc),
        selection_json={
            "nutrition": {
                "status": "VERIFIED",
                "per_serving": verified_candidates()[0].nutrition_per_serving,
            }
        },
    )
    unknown = SimpleNamespace(
        desired_for=day.desired_for,
        selection_json={"nutrition": {"status": "UNKNOWN", "per_serving": None}},
    )
    plan = SimpleNamespace(
        timezone="Asia/Kolkata",
        constraints_json={"nutrition_targets": TARGETS},
    )

    summary = _daily_nutrition_payload(plan, [day, unknown])

    assert summary[0]["status"] == "INCOMPLETE"
    assert summary[0]["totals"]["protein_g"] == 25
    assert "difference_from_targets" not in summary[0]


def test_daily_nutrition_api_only_marks_a_complete_four_meal_day_verified():
    meals = [
        SimpleNamespace(
            desired_for=datetime(2026, 10, 19, hour, 0, tzinfo=timezone.utc),
            meal_period=period,
            selection_json={
                "nutrition": {
                    "status": "VERIFIED",
                    "per_serving": candidate.nutrition_per_serving,
                }
            },
        )
        for hour, period, candidate in zip((3, 8, 12, 15), ("BREAKFAST", "LUNCH", "SNACKS", "DINNER"), verified_candidates())
    ]
    plan = SimpleNamespace(
        timezone="Asia/Kolkata",
        constraints_json={"nutrition_targets": TARGETS},
    )

    summary = _daily_nutrition_payload(plan, meals)

    assert summary[0]["status"] == "VERIFIED"
    assert summary[0]["totals"]["protein_g"] == TARGETS["protein_g"]
    assert summary[0]["difference_from_targets"]["protein_g"] == 0
