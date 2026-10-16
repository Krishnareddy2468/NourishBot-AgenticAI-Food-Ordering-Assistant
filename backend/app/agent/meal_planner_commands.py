"""Deterministic Telegram/WhatsApp commands for meal plans and routines."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.core.config import settings
from app.db.models import Channel, MealPlan, MealRoutine, MealSchedule
from app.db.session import get_session_factory
from app.domain.meal_planning import MealPlanningError
from app.services.meal_planner import (
    change_routine,
    confirm_plan,
    confirm_planned_meal,
    create_recommended_plan,
)


_PLAN_CREATE_RE = re.compile(r"\b(?:plan my|create (?:a )?)(day|week|daily|weekly)(?: meal plan)?\b", re.I)
_PLAN_CONFIRM_RE = re.compile(r"^\s*confirm\s+meal\s+plan\s+#?(\d+)\s*$", re.I)
_MEAL_CONFIRM_RE = re.compile(r"^\s*confirm\s+planned\s+meal\s+#?(\d+)\s*$", re.I)
_ROUTINE_STATE_RE = re.compile(r"^\s*(pause|resume)\s+(?:meal\s+)?routine\s+#?(\d+)\s*$", re.I)
_ROUTINE_SKIP_RE = re.compile(r"^\s*skip\s+(?:meal\s+)?routine\s+#?(\d+)\s+(?:on\s+)?(\d{4}-\d{2}-\d{2})\s*$", re.I)
_BUDGET_RE = re.compile(r"(?:₹|\brs\.?|\binr)\s*([0-9]+(?:\.[0-9]{1,2})?)", re.I)


async def _channel(chat_id: str | int, channel_type: str) -> Channel | None:
    async with get_session_factory()() as db:
        return (await db.execute(select(Channel).where(
            Channel.restaurant_id == settings.DEFAULT_RESTAURANT_ID,
            Channel.type == channel_type.upper(),
            Channel.external_id == str(chat_id),
            Channel.customer_id.is_not(None),
        ))).scalar_one_or_none()


def _request_constraints(message: str) -> tuple[list[str], list[str], list[str], int | None]:
    lowered = message.lower()
    diets = [
        value for value in ("vegan", "vegetarian", "no-pork", "no-beef")
        if value in lowered
    ]
    goals = [
        value for value in ("high-protein", "weight-loss", "lower-sugar", "less-oil")
        if value in lowered.replace(" ", "-")
    ]
    allergies: list[str] = []
    allergy_match = re.search(r"allerg(?:ic|y)\s+(?:to\s+)?([a-z,\s-]+?)(?:\s+(?:under|within|for)\b|$)", lowered)
    if allergy_match:
        allergies = [part.strip() for part in re.split(r",|\band\b", allergy_match.group(1)) if part.strip()]
    budget_match = _BUDGET_RE.search(message)
    budget_cents = int(float(budget_match.group(1)) * 100) if budget_match else None
    return diets, allergies, goals, budget_cents


def _plan_message(plan: MealPlan, meals: list[MealSchedule]) -> str:
    constraints = plan.constraints_json or {}
    constraint_parts = []
    if constraints.get("dietary_flags"):
        constraint_parts.append("diet: " + ", ".join(constraints["dietary_flags"]))
    if constraints.get("health_goals"):
        constraint_parts.append("goals: " + ", ".join(constraints["health_goals"]))
    if constraints.get("allergies"):
        constraint_parts.append("allergies blocked: " + ", ".join(constraints["allergies"]))
    if constraints.get("budget_cents") is not None:
        constraint_parts.append(f"budget: {constraints['currency']} {constraints['budget_cents'] / 100:.2f}")
    by_day: dict[str, list[str]] = {}
    zone = ZoneInfo(plan.timezone)
    for meal in meals:
        local = meal.desired_for.astimezone(zone)
        by_day.setdefault(local.date().isoformat(), []).append(
            f"{meal.meal_period.title()}: {meal.title} ({meal.currency} {meal.estimated_cost_cents / 100:.2f})"
        )
    lines = [
        f"Meal plan #{plan.id}: {plan.name}",
        f"Status: {plan.status} — every meal is a draft until you confirm the plan.",
        "Constraints: " + ("; ".join(constraint_parts) if constraint_parts else "none specified"),
        f"Estimated total: {plan.currency} {plan.estimated_total_cents / 100:.2f}",
    ]
    for day, entries in by_day.items():
        lines.append(f"\n{day}")
        lines.extend(f"• {entry}" for entry in entries)
    lines.append(f"\nSend “confirm meal plan {plan.id}” when it is correct. Confirmation does not place an order.")
    return "\n".join(lines)[:4000]


async def handle_meal_planner_command(
    message: str,
    *,
    chat_id: str | int,
    channel_type: str,
) -> str | None:
    """Return a deterministic planning reply, or None when the text is not a planning command."""
    text = (message or "").strip()
    create_match = _PLAN_CREATE_RE.search(text)
    show_intent = text.casefold() in {
        "my meal plan", "show my meal plan", "show my meal plans", "list my meal plans", "meal plans",
    }
    confirm_match = _PLAN_CONFIRM_RE.match(text)
    meal_confirm_match = _MEAL_CONFIRM_RE.match(text)
    routine_state = _ROUTINE_STATE_RE.match(text)
    routine_skip = _ROUTINE_SKIP_RE.match(text)
    if not any((create_match, show_intent, confirm_match, meal_confirm_match, routine_state, routine_skip)):
        return None

    channel = await _channel(chat_id, channel_type)
    if not channel:
        return (
            "Meal planning needs a linked customer profile. Place one identified order or link this "
            "channel to your existing profile, then try again."
        )
    customer_id = int(channel.customer_id)
    restaurant_id = channel.restaurant_id

    async with get_session_factory()() as db:
        try:
            if create_match:
                requested = create_match.group(1).lower()
                scope = "DAILY" if requested in {"day", "daily"} else "WEEKLY"
                diets, allergies, goals, budget_cents = _request_constraints(text)
                start = datetime.now(ZoneInfo("Asia/Kolkata")).date() + timedelta(days=1)
                plan, meals = await create_recommended_plan(
                    db,
                    restaurant_id=restaurant_id,
                    customer_id=customer_id,
                    scope=scope,
                    starts_on=start,
                    timezone_name="Asia/Kolkata",
                    dietary_flags=diets,
                    allergies=allergies,
                    health_goals=goals,
                    budget_cents=budget_cents,
                    currency="INR",
                    source=channel_type,
                    channel_id=channel.id,
                    actor_id=str(channel.id),
                )
                await db.commit()
                return _plan_message(plan, meals)

            if show_intent:
                plan = (await db.execute(
                    select(MealPlan)
                    .where(
                        MealPlan.restaurant_id == restaurant_id,
                        MealPlan.customer_id == customer_id,
                    )
                    .order_by(MealPlan.id.desc())
                    .limit(1)
                )).scalar_one_or_none()
                if not plan:
                    return "You do not have a meal plan yet. Send “plan my week” to create a draft."
                meals = (await db.execute(
                    select(MealSchedule)
                    .where(MealSchedule.meal_plan_id == plan.id)
                    .order_by(MealSchedule.desired_for, MealSchedule.meal_period)
                )).scalars().all()
                return _plan_message(plan, meals)

            if confirm_match:
                plan_id = int(confirm_match.group(1))
                plan = (await db.execute(select(MealPlan).where(
                    MealPlan.id == plan_id,
                    MealPlan.restaurant_id == restaurant_id,
                    MealPlan.customer_id == customer_id,
                ).with_for_update())).scalar_one_or_none()
                if not plan:
                    return "That meal plan was not found on your linked profile."
                _, snapshot_hash = await confirm_plan(db, plan, actor_id=str(channel.id))
                await db.commit()
                return (
                    f"Meal plan #{plan.id} is confirmed (receipt {snapshot_hash[:12]}). "
                    "This confirms your planning schedule only; no food order has been placed."
                )

            if meal_confirm_match:
                meal_id = int(meal_confirm_match.group(1))
                meal = (await db.execute(select(MealSchedule).where(
                    MealSchedule.id == meal_id,
                    MealSchedule.restaurant_id == restaurant_id,
                    MealSchedule.customer_id == customer_id,
                ).with_for_update())).scalar_one_or_none()
                if not meal:
                    return "That planned meal was not found on your linked profile."
                snapshot_hash = await confirm_planned_meal(db, meal, actor_id=str(channel.id))
                await db.commit()
                return (
                    f"Planned meal #{meal.id} is confirmed (receipt {snapshot_hash[:12]}). "
                    "No food order has been placed."
                )

            match = routine_state or routine_skip
            routine_id = int(match.group(2) if routine_state else match.group(1))
            routine = (await db.execute(select(MealRoutine).where(
                MealRoutine.id == routine_id,
                MealRoutine.restaurant_id == restaurant_id,
                MealRoutine.customer_id == customer_id,
            ).with_for_update())).scalar_one_or_none()
            if not routine:
                return "That meal routine was not found on your linked profile."
            if routine_state:
                action = routine_state.group(1).upper()
                await change_routine(db, routine, action=action, actor_id=str(channel.id))
                await db.commit()
                return f"Routine #{routine.id} is now {routine.state.lower()}. Existing occurrences were updated without duplicates."
            occurrence = date.fromisoformat(routine_skip.group(2))
            await change_routine(
                db,
                routine,
                action="SKIP",
                occurrence_date=occurrence,
                actor_id=str(channel.id),
            )
            await db.commit()
            return f"Skipped routine #{routine.id} for {occurrence.isoformat()}. No duplicate reminder will be created."
        except MealPlanningError as exc:
            await db.rollback()
            return f"I couldn't safely update the meal plan: {exc}."
