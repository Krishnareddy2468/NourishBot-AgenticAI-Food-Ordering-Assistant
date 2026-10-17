"""Deterministic meal-plan, recommendation, recurrence, and allergy policy."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MEAL_PERIODS = ("BREAKFAST", "LUNCH", "SNACKS", "DINNER")
DEFAULT_MEAL_TIMES = {
    "BREAKFAST": time(8, 30),
    "LUNCH": time(13, 0),
    "SNACKS": time(17, 0),
    "DINNER": time(20, 0),
}
WEEKDAYS = {0, 1, 2, 3, 4}
WEEKENDS = {5, 6}
DAY_NAMES = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
NUTRITION_TARGET_KEYS = (
    "energy_kcal",
    "protein_g",
    "carbohydrates_g",
    "fat_g",
    "fiber_g",
)


class MealPlanningError(ValueError):
    """A user-correctable deterministic planning validation failure."""


def _terms(values: Iterable[str] | None) -> tuple[str, ...]:
    normalized = {
        re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
        for value in (values or [])
        if str(value).strip()
    }
    aliases = {
        "groundnut": "peanut",
        "groundnuts": "peanut",
        "peanuts": "peanut",
        "milk": "dairy",
        "tree-nuts": "tree-nut",
        "nuts": "tree-nut",
    }
    return tuple(sorted(aliases.get(value, value) for value in normalized))


@dataclass(frozen=True)
class PlanConstraints:
    dietary_flags: tuple[str, ...] = ()
    allergies: tuple[str, ...] = ()
    health_goals: tuple[str, ...] = ()
    budget_cents: int | None = None
    currency: str = "INR"
    nutrition_targets: dict[str, float] | None = None

    @classmethod
    def from_values(
        cls,
        *,
        dietary_flags: Iterable[str] | None = None,
        allergies: Iterable[str] | None = None,
        health_goals: Iterable[str] | None = None,
        budget_cents: int | None = None,
        currency: str = "INR",
        nutrition_targets: dict[str, Any] | None = None,
    ) -> "PlanConstraints":
        if budget_cents is not None and int(budget_cents) < 0:
            raise MealPlanningError("Meal-plan budget cannot be negative")
        targets = normalize_nutrition_targets(nutrition_targets)
        return cls(
            dietary_flags=_terms(dietary_flags),
            allergies=_terms(allergies),
            health_goals=_terms(health_goals),
            budget_cents=int(budget_cents) if budget_cents is not None else None,
            currency=(currency or "INR").strip().upper()[:3],
            nutrition_targets=targets,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "dietary_flags": list(self.dietary_flags),
            "allergies": list(self.allergies),
            "health_goals": list(self.health_goals),
            "budget_cents": self.budget_cents,
            "currency": self.currency,
            "nutrition_targets": self.nutrition_targets,
        }


def normalize_nutrition_targets(value: dict[str, Any] | None) -> dict[str, float] | None:
    if value is None:
        return None
    if set(value) != set(NUTRITION_TARGET_KEYS):
        raise MealPlanningError(
            "Nutrition targets must include energy_kcal, protein_g, carbohydrates_g, fat_g, and fiber_g"
        )
    targets: dict[str, float] = {}
    for key in NUTRITION_TARGET_KEYS:
        if isinstance(value[key], bool):
            raise MealPlanningError(f"Nutrition target {key} must be a positive number")
        try:
            amount = float(value[key])
        except (TypeError, ValueError) as exc:
            raise MealPlanningError(f"Nutrition target {key} must be a positive number") from exc
        if not math.isfinite(amount) or amount <= 0:
            raise MealPlanningError(f"Nutrition target {key} must be a positive number")
        targets[key] = amount
    return targets


@dataclass(frozen=True)
class MenuCandidate:
    id: int
    name: str
    price_cents: int
    menu_type: str
    tags: tuple[str, ...] = ()
    category: str = ""
    history_count: int = 0
    nutrition_per_serving: dict[str, float] | None = None
    nutrition_status: str = "UNKNOWN"
    nutrition_source: str | None = None
    nutrition_serving_g: float | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MenuCandidate":
        return cls(
            id=int(value["id"]),
            name=str(value["name"]),
            price_cents=int(value.get("price_cents") or 0),
            menu_type=str(value.get("menu_type") or value.get("type") or "").upper(),
            tags=_terms(value.get("tags") or []),
            category=str(value.get("category") or ""),
            history_count=int(value.get("history_count") or 0),
            nutrition_per_serving={
                key: float(value[key])
                for key in NUTRITION_TARGET_KEYS
                if value.get(key) is not None
            } or None,
            nutrition_status=str(value.get("nutrition_status") or "UNKNOWN").upper(),
            nutrition_source=str(value.get("nutrition_source") or "").strip() or None,
            nutrition_serving_g=(
                float(value["nutrition_serving_g"])
                if value.get("nutrition_serving_g") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PlannedMeal:
    day: date
    meal_period: str
    desired_for: datetime
    candidate: MenuCandidate
    reason: tuple[str, ...] = field(default_factory=tuple)


def _nutrition_score(
    candidate: MenuCandidate,
    totals: dict[str, float],
    targets: dict[str, float],
    meals_selected: int,
) -> float:
    facts = candidate.nutrition_per_serving or {}
    fraction = (meals_selected + 1) / len(MEAL_PERIODS)
    return sum(
        abs(totals[key] + facts[key] - targets[key] * fraction) / targets[key]
        for key in NUTRITION_TARGET_KEYS
    )


def meal_dates(scope: str, starts_on: date) -> list[date]:
    normalized = scope.strip().upper()
    if normalized == "DAILY":
        return [starts_on]
    if normalized == "WEEKLY":
        return [starts_on + timedelta(days=offset) for offset in range(7)]
    raise MealPlanningError("Plan scope must be DAILY or WEEKLY")


def parse_timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise MealPlanningError("Unknown meal-plan timezone") from exc


def declared_allergens(tags: Iterable[str]) -> set[str]:
    allergens: set[str] = set()
    for tag in _terms(tags):
        for prefix in ("allergen-", "contains-"):
            if tag.startswith(prefix):
                allergens.add(tag.removeprefix(prefix))
    return allergens


def assert_allergy_safe(tags: Iterable[str], allergies: Iterable[str]) -> None:
    normalized_tags = set(_terms(tags))
    normalized_allergies = set(_terms(allergies))
    if not normalized_allergies:
        return
    conflicts = declared_allergens(normalized_tags) & normalized_allergies
    if conflicts:
        raise MealPlanningError(
            "Allergy conflict: " + ", ".join(sorted(conflicts))
        )
    verified_free = {
        tag.removeprefix("free-from-")
        for tag in normalized_tags
        if tag.startswith("free-from-")
    }
    if "allergen-verified" not in normalized_tags and not normalized_allergies.issubset(verified_free):
        raise MealPlanningError(
            "Allergen metadata is not verified for: " + ", ".join(sorted(normalized_allergies))
        )


def assert_diet_compatible(menu_type: str, tags: Iterable[str], dietary_flags: Iterable[str]) -> None:
    flags = set(_terms(dietary_flags))
    normalized_tags = set(_terms(tags))
    item_type = (menu_type or "").upper()
    if "vegan" in flags and item_type != "VEGAN" and "vegan" not in normalized_tags:
        raise MealPlanningError("Meal does not satisfy the vegan constraint")
    if "vegetarian" in flags and item_type not in {"VEG", "VEGAN"}:
        raise MealPlanningError("Meal does not satisfy the vegetarian constraint")
    if "no-pork" in flags and ("pork" in normalized_tags or "contains-pork" in normalized_tags):
        raise MealPlanningError("Meal conflicts with the no-pork constraint")
    if "no-beef" in flags and ("beef" in normalized_tags or "contains-beef" in normalized_tags):
        raise MealPlanningError("Meal conflicts with the no-beef constraint")


def validate_candidate(candidate: MenuCandidate, constraints: PlanConstraints) -> None:
    if candidate.price_cents < 0:
        raise MealPlanningError("Meal price cannot be negative")
    assert_diet_compatible(candidate.menu_type, candidate.tags, constraints.dietary_flags)
    assert_allergy_safe(candidate.tags, constraints.allergies)


def has_complete_verified_nutrition(candidate: MenuCandidate) -> bool:
    values = candidate.nutrition_per_serving or {}
    return (
        candidate.nutrition_status == "VERIFIED"
        and bool(candidate.nutrition_source)
        and candidate.nutrition_serving_g is not None
        and candidate.nutrition_serving_g > 0
        and set(values) == set(NUTRITION_TARGET_KEYS)
        and all(math.isfinite(values[key]) and values[key] >= 0 for key in NUTRITION_TARGET_KEYS)
    )


def assert_nutrition_verified(candidate: MenuCandidate) -> None:
    if not has_complete_verified_nutrition(candidate):
        raise MealPlanningError(
            "Menu item needs complete verified nutrition values and serving size for a nutrition-targeted plan"
        )


def _period_score(candidate: MenuCandidate, period: str, goals: tuple[str, ...]) -> tuple[int, list[str]]:
    text = " ".join((candidate.name, candidate.category, *candidate.tags)).lower()
    score = min(candidate.history_count, 5)
    reasons: list[str] = []
    if period.lower() in text:
        score += 8
        reasons.append(f"tagged for {period.lower()}")
    matched_goals = [goal for goal in goals if goal in text]
    if matched_goals:
        score += 4 * len(matched_goals)
        reasons.append("matches " + ", ".join(matched_goals))
    if candidate.history_count:
        reasons.append("based on meal history")
    if not reasons:
        reasons.append("fits the active constraints")
    return score, reasons


def recommend_complete_plan(
    candidates: Iterable[MenuCandidate | dict[str, Any]],
    *,
    scope: str,
    starts_on: date,
    timezone_name: str,
    constraints: PlanConstraints,
    meal_times: dict[str, time] | None = None,
) -> list[PlannedMeal]:
    """Build four safe, budget-visible draft meals per plan day."""
    parsed = [item if isinstance(item, MenuCandidate) else MenuCandidate.from_mapping(item) for item in candidates]
    safe: list[MenuCandidate] = []
    for candidate in parsed:
        try:
            validate_candidate(candidate, constraints)
        except MealPlanningError:
            continue
        safe.append(candidate)
    if constraints.nutrition_targets is not None:
        safe = [candidate for candidate in safe if has_complete_verified_nutrition(candidate)]
    if not safe:
        if constraints.nutrition_targets is not None:
            raise MealPlanningError(
                "No menu items have complete verified nutrition data for these daily targets"
            )
        raise MealPlanningError("No menu item satisfies the diet and verified-allergy constraints")

    dates = meal_dates(scope, starts_on)
    slot_count = len(dates) * len(MEAL_PERIODS)
    per_slot_budget = constraints.budget_cents // slot_count if constraints.budget_cents is not None else None
    timezone_info = parse_timezone(timezone_name)
    times = {**DEFAULT_MEAL_TIMES, **(meal_times or {})}
    recommendations: list[PlannedMeal] = []

    for day_index, plan_day in enumerate(dates):
        day_totals = {key: 0.0 for key in NUTRITION_TARGET_KEYS}
        selected_ids: set[int] = set()
        for period_index, period in enumerate(MEAL_PERIODS):
            eligible = [
                item for item in safe
                if per_slot_budget is None or item.price_cents <= per_slot_budget
            ]
            if not eligible:
                raise MealPlanningError(
                    f"No safe {period.lower()} option fits the per-meal budget of "
                    f"{per_slot_budget} {constraints.currency} cents"
                )
            targets = constraints.nutrition_targets
            if targets is not None:
                nutrition_targets = dict(targets)
                distinct = [item for item in eligible if item.id not in selected_ids]
                nutrition_options = distinct or eligible
                candidate = min(
                    nutrition_options,
                    key=lambda item: (
                        _nutrition_score(
                            item,
                            day_totals,
                            nutrition_targets,
                            period_index,
                        ),
                        -_period_score(item, period, constraints.health_goals)[0],
                        item.price_cents,
                        item.id,
                    ),
                )
                for key in NUTRITION_TARGET_KEYS:
                    day_totals[key] += (candidate.nutrition_per_serving or {})[key]
                selected_ids.add(candidate.id)
            else:
                ranked = sorted(
                    eligible,
                    key=lambda item: (
                        -_period_score(item, period, constraints.health_goals)[0],
                        item.price_cents,
                        item.id,
                    ),
                )
                candidate = ranked[(day_index + period_index) % min(len(ranked), 4)]
            _, reasons = _period_score(candidate, period, constraints.health_goals)
            if per_slot_budget is not None:
                reasons.append(f"within {per_slot_budget} {constraints.currency} cents per meal")
            if targets is not None:
                reasons.append("selected using verified menu nutrition values")
            local_dt = datetime.combine(plan_day, times[period], tzinfo=timezone_info)
            recommendations.append(PlannedMeal(
                day=plan_day,
                meal_period=period,
                desired_for=local_dt.astimezone(timezone.utc),
                candidate=candidate,
                reason=tuple(reasons),
            ))

    estimated_total = sum(item.candidate.price_cents for item in recommendations)
    if constraints.budget_cents is not None and estimated_total > constraints.budget_cents:
        raise MealPlanningError("Generated meal plan exceeds the requested budget")
    return recommendations


def parse_recurrence_rule(value: str) -> frozenset[int]:
    normalized = (value or "").strip().upper().replace(" ", "")
    if normalized == "DAILY":
        return frozenset(range(7))
    if normalized == "WEEKDAYS":
        return frozenset(WEEKDAYS)
    if normalized == "WEEKENDS":
        return frozenset(WEEKENDS)
    if normalized.startswith("CUSTOM:"):
        names = [part for part in normalized.removeprefix("CUSTOM:").split(",") if part]
        if not names or any(name not in DAY_NAMES for name in names):
            raise MealPlanningError("Custom recurrence must use MON through SUN")
        return frozenset(DAY_NAMES[name] for name in names)
    raise MealPlanningError("Recurrence must be DAILY, WEEKDAYS, WEEKENDS, or CUSTOM:MON,WED")


def recurrence_dates(rule: str, starts_on: date, ends_on: date) -> list[date]:
    if ends_on < starts_on:
        raise MealPlanningError("Recurrence end must not be before its start")
    weekdays = parse_recurrence_rule(rule)
    count = (ends_on - starts_on).days + 1
    return [
        starts_on + timedelta(days=offset)
        for offset in range(count)
        if (starts_on + timedelta(days=offset)).weekday() in weekdays
    ]


def routine_occurrence_key(routine_id: int, version: int, occurrence_date: date) -> str:
    return f"routine:{int(routine_id)}:v{int(version)}:{occurrence_date.isoformat()}"


def canonical_snapshot(plan: Any, meals: Iterable[Any]) -> tuple[dict[str, Any], str]:
    snapshot = {
        "plan_id": int(plan.id),
        "revision": int(plan.revision),
        "scope": plan.scope,
        "starts_on": plan.starts_on.isoformat(),
        "ends_on": plan.ends_on.isoformat(),
        "timezone": plan.timezone,
        "constraints": plan.constraints_json or {},
        "meals": [
            {
                "id": int(meal.id),
                "revision": int(meal.revision),
                "meal_period": meal.meal_period,
                "desired_for": meal.desired_for.isoformat(),
                "menu_item_id": meal.menu_item_id,
                "title": meal.title,
                "estimated_cost_cents": meal.estimated_cost_cents,
                "diet_tags": meal.diet_tags or [],
                "allergen_tags": meal.allergen_tags or [],
                "nutrition": (getattr(meal, "selection_json", None) or {}).get("nutrition", {}),
            }
            for meal in sorted(meals, key=lambda item: (item.desired_for, item.meal_period, item.id))
        ],
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return snapshot, hashlib.sha256(encoded).hexdigest()
