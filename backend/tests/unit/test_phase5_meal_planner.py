from datetime import date, datetime, time, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.routes.meal_plans import router as meal_plans_router
from app.db.models import MealPlan, MealPlanningEvent, MealRoutineException, MealSchedule
from app.domain.meal_planning import (
    MEAL_PERIODS,
    MealPlanningError,
    MenuCandidate,
    PlanConstraints,
    assert_allergy_safe,
    assert_diet_compatible,
    canonical_snapshot,
    meal_dates,
    parse_recurrence_rule,
    recommend_complete_plan,
    recurrence_dates,
    routine_occurrence_key,
)
from app.domain.scheduling import JOB_TYPES, PLANNED_MEAL_REMINDER, PLAN_CONFIRMATION_REMINDER


def candidates():
    return [
        MenuCandidate(1, "Protein Breakfast Bowl", 120, "VEG", ("breakfast", "high-protein", "allergen-verified")),
        MenuCandidate(2, "Balanced Lunch", 140, "VEG", ("lunch", "allergen-verified")),
        MenuCandidate(3, "Fruit Snacks", 80, "VEGAN", ("snacks", "allergen-verified")),
        MenuCandidate(4, "Light Dinner", 150, "VEG", ("dinner", "less-oil", "allergen-verified")),
    ]


def test_phase5_models_extend_plans_schedules_and_append_only_history():
    assert MealRoutineException.__tablename__ == "meal_routine_exceptions"
    assert MealPlanningEvent.__tablename__ == "meal_planning_events"
    assert hasattr(MealPlan, "constraints_json")
    assert hasattr(MealPlan, "confirmation_snapshot_hash")
    assert hasattr(MealSchedule, "occurrence_key")
    assert hasattr(MealSchedule, "allergen_tags")


def test_daily_and_weekly_plans_have_complete_date_ranges():
    start = date(2026, 10, 19)
    assert meal_dates("DAILY", start) == [start]
    week = meal_dates("WEEKLY", start)
    assert len(week) == 7
    assert week[-1] == date(2026, 10, 25)


def test_weekly_recommendation_has_all_four_slots_and_visible_constraints():
    constraints = PlanConstraints.from_values(
        dietary_flags=["vegetarian"],
        health_goals=["high-protein"],
        budget_cents=5600,
        currency="INR",
    )
    plan = recommend_complete_plan(
        candidates(),
        scope="WEEKLY",
        starts_on=date(2026, 10, 19),
        timezone_name="Asia/Kolkata",
        constraints=constraints,
    )
    assert len(plan) == 28
    assert {item.meal_period for item in plan} == set(MEAL_PERIODS)
    assert sum(item.candidate.price_cents for item in plan) <= constraints.budget_cents
    assert constraints.public_dict()["dietary_flags"] == ["vegetarian"]
    assert constraints.public_dict()["budget_cents"] == 5600


def test_allergy_conflicts_and_unverified_metadata_are_blocked_by_code():
    with pytest.raises(MealPlanningError, match="Allergy conflict"):
        assert_allergy_safe(["allergen:peanut", "allergen-verified"], ["peanuts"])
    with pytest.raises(MealPlanningError, match="not verified"):
        assert_allergy_safe(["lunch"], ["dairy"])
    assert_allergy_safe(["free-from:dairy"], ["milk"])


def test_diet_constraints_are_deterministic_not_prompt_only():
    with pytest.raises(MealPlanningError, match="vegetarian"):
        assert_diet_compatible("NON_VEG", [], ["vegetarian"])
    with pytest.raises(MealPlanningError, match="vegan"):
        assert_diet_compatible("VEG", [], ["vegan"])
    assert_diet_compatible("VEGAN", ["high-protein"], ["vegan"])


def test_budget_filter_refuses_an_impossible_plan():
    with pytest.raises(MealPlanningError, match="per-meal budget"):
        recommend_complete_plan(
            candidates(),
            scope="DAILY",
            starts_on=date(2026, 10, 19),
            timezone_name="Asia/Kolkata",
            constraints=PlanConstraints.from_values(budget_cents=100),
        )


def test_recurrence_supports_daily_weekdays_weekends_and_custom_days():
    assert parse_recurrence_rule("DAILY") == frozenset(range(7))
    assert parse_recurrence_rule("WEEKDAYS") == frozenset({0, 1, 2, 3, 4})
    assert parse_recurrence_rule("WEEKENDS") == frozenset({5, 6})
    assert parse_recurrence_rule("CUSTOM:MON,WED,FRI") == frozenset({0, 2, 4})
    with pytest.raises(MealPlanningError, match="MON through SUN"):
        parse_recurrence_rule("CUSTOM:MON,FUNDAY")


def test_recurrence_dates_stay_inside_controlled_range():
    values = recurrence_dates("WEEKDAYS", date(2026, 10, 19), date(2026, 10, 25))
    assert values == [
        date(2026, 10, 19),
        date(2026, 10, 20),
        date(2026, 10, 21),
        date(2026, 10, 22),
        date(2026, 10, 23),
    ]


def test_routine_occurrence_key_prevents_duplicates_and_versions_edits():
    first = routine_occurrence_key(9, 1, date(2026, 10, 19))
    assert first == routine_occurrence_key(9, 1, date(2026, 10, 19))
    assert first != routine_occurrence_key(9, 2, date(2026, 10, 19))


def test_confirmation_snapshot_is_stable_and_changes_with_a_meal_revision():
    plan = SimpleNamespace(
        id=7,
        revision=1,
        scope="DAILY",
        starts_on=date(2026, 10, 19),
        ends_on=date(2026, 10, 19),
        timezone="Asia/Kolkata",
        constraints_json={"dietary_flags": ["vegetarian"]},
    )
    meal = SimpleNamespace(
        id=1,
        revision=1,
        meal_period="BREAKFAST",
        desired_for=datetime(2026, 10, 19, 3, 0, tzinfo=timezone.utc),
        menu_item_id=2,
        title="Breakfast Bowl",
        estimated_cost_cents=120,
        diet_tags=["veg"],
        allergen_tags=[],
    )
    first_snapshot, first_hash = canonical_snapshot(plan, [meal])
    second_snapshot, second_hash = canonical_snapshot(plan, [meal])
    assert first_snapshot == second_snapshot
    assert first_hash == second_hash
    meal.revision = 2
    assert canonical_snapshot(plan, [meal])[1] != first_hash


def test_planner_routes_cover_plans_templates_routines_and_actions():
    paths = {route.path for route in meal_plans_router.routes}
    assert {
        "/api/v1/customer/meal-plans",
        "/api/v1/customer/meal-plans/{plan_id}",
        "/api/v1/customer/meal-plans/{plan_id}/meals/{meal_id}",
        "/api/v1/customer/meal-plans/{plan_id}/confirm",
        "/api/v1/customer/planned-meals/{meal_id}/confirm",
        "/api/v1/customer/meal-templates",
        "/api/v1/customer/meal-routines",
        "/api/v1/customer/meal-routines/{routine_id}/actions",
    }.issubset(paths)


def test_durable_scheduler_knows_draft_and_confirmed_plan_reminders():
    assert PLAN_CONFIRMATION_REMINDER in JOB_TYPES
    assert PLANNED_MEAL_REMINDER in JOB_TYPES
    worker = (
        Path(__file__).resolve().parents[2] / "app" / "workers" / "scheduled_order_worker.py"
    ).read_text()
    assert "_send_plan_notification" in worker
    assert "no order has been placed" in worker.lower()
    assert "REMINDERS_NOT_GRANTED" in worker


def test_phase5_migration_is_chained_bounded_and_append_only():
    migration = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0014_phase5_meal_planner.py"
    ).read_text()
    assert 'down_revision: Union[str, Sequence[str], None] = "0013"' in migration
    assert "meal_routine_exceptions" in migration
    assert "meal_planning_events" in migration
    assert "trg_meal_planning_events_append_only" in migration
    assert "materialization_horizon_days BETWEEN 1 AND 31" in migration
    assert "occurrence_key" in migration
