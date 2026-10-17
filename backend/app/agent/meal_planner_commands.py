"""Deterministic Telegram/WhatsApp commands for meal plans and routines."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import or_, select

from app.core.config import settings
from app.db.crud import get_session
from app.db.models import Channel, MealPlan, MealRoutine, MealSchedule, MenuItem
from app.db.session import get_session_factory
from app.domain.meal_planning import (
    MealPlanningError,
    NUTRITION_TARGET_KEYS,
    normalize_nutrition_targets,
)
from app.services.planned_meal_order import (
    PlannedMealOrderError,
    load_orderable_planned_meal,
)
from app.services.meal_planner import (
    change_routine,
    confirm_plan,
    confirm_planned_meal,
    create_recommended_plan,
    update_planned_meal,
)


_PLAN_CREATE_RE = re.compile(r"\b(?:plan my|create (?:a )?)(day|week|daily|weekly)(?: meal plan)?\b", re.I)
_PLAN_CONFIRM_RE = re.compile(r"^\s*confirm\s+meal\s+plan\s+#?(\d+)\s*$", re.I)
_MEAL_CONFIRM_RE = re.compile(r"^\s*confirm\s+planned\s+meal\s+#?(\d+)\s*$", re.I)
_MEAL_EDIT_RE = re.compile(
    r"^\s*change\s+planned\s+meal\s+#?(\d+)\s+to\s+menu\s+item\s+#?(\d+)"
    r"(?:\s+at\s+(\S+))?\s*$",
    re.I,
)
_ORDER_PLANNED_MEAL_RE = re.compile(r"^\s*order\s+planned\s+meal\s+#?(\d+)\s*$", re.I)
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


async def _start_planned_meal_order(
    *,
    meal_id: int,
    chat_id: str | int,
    channel_type: str,
    channel: Channel,
) -> str:
    customer_id = int(channel.customer_id)
    restaurant_id = channel.restaurant_id
    async with get_session_factory()() as db:
        try:
            planned = await load_orderable_planned_meal(
                db,
                meal_id=meal_id,
                restaurant_id=restaurant_id,
                customer_id=customer_id,
            )
        except PlannedMealOrderError as exc:
            return str(exc)
        scheduled_text = planned.scheduled_text
        plan_context = planned.planning_context

    session = await get_session(chat_id, channel_type)
    provider = str(session.get("ordering_platform") or "NourishBot").strip()
    order_request = f"Order planned meal #{meal_id} for {scheduled_text}."
    if provider.upper() in {"", "NOURISHBOT"}:
        if session.get("cart"):
            return (
                "Your existing cart is not empty. Finish or clear it first so this planned meal "
                "cannot accidentally be combined with other items. No order was placed."
            )
        from app.agent.pipeline import process_message

        return await process_message(
            order_request,
            chat_id=chat_id,
            user_name=str(session.get("user_name") or ""),
            channel_type=channel_type,
            planning_context=plan_context,
        )

    if provider.upper() not in {"SWIGGY", "ZOMATO"}:
        return "The selected ordering provider is not supported for plan orders. No order was placed."
    from app.agent.mcp_agent import get_mcp_response

    reply = await get_mcp_response(
        user_message=order_request,
        chat_id=chat_id,
        user_name=str(session.get("user_name") or ""),
        platform=provider,
        channel_type=channel_type,
        planning_context=plan_context,
    )
    return reply or (
        f"{provider} plan ordering is unavailable right now. No order was placed; "
        "choose NourishBot or try again later."
    )


def _nutrition_targets(message: str) -> dict[str, float] | None:
    patterns = {
        "energy_kcal": r"\b(?:calories|kcal|energy)\s*[:=]?\s*(\d+(?:\.\d+)?)",
        "protein_g": r"\bprotein\s*[:=]?\s*(\d+(?:\.\d+)?)\s*g?\b",
        "carbohydrates_g": r"\b(?:carbs?|carbohydrates?)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*g?\b",
        "fat_g": r"\bfat\s*[:=]?\s*(\d+(?:\.\d+)?)\s*g?\b",
        "fiber_g": r"\bfib(?:er|re)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*g?\b",
    }
    found = {
        key: float(match.group(1))
        for key, pattern in patterns.items()
        if (match := re.search(pattern, message, re.I))
    }
    if not found:
        return None
    if set(found) != set(NUTRITION_TARGET_KEYS):
        raise MealPlanningError(
            "For a nutrition-aware day plan, provide all five daily targets: "
            "calories, protein g, carbs g, fat g, and fibre g. Use targets from a qualified professional."
        )
    return normalize_nutrition_targets(found)


def _request_constraints(
    message: str,
) -> tuple[list[str], list[str], list[str], int | None, dict[str, float] | None]:
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
    return diets, allergies, goals, budget_cents, _nutrition_targets(message)


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
    by_day: dict[str, list[MealSchedule]] = {}
    zone = ZoneInfo(plan.timezone)
    for meal in meals:
        local = meal.desired_for.astimezone(zone)
        by_day.setdefault(local.date().isoformat(), []).append(meal)
    lines = [
        f"Meal plan #{plan.id}: {plan.name}",
        (
            f"Status: {plan.status} — review this draft before confirming."
            if plan.status == "DRAFT"
            else f"Status: {plan.status} — this confirms the planning schedule only."
        ),
        "Constraints: " + ("; ".join(constraint_parts) if constraint_parts else "none specified"),
        f"Estimated total: {plan.currency} {plan.estimated_total_cents / 100:.2f}",
    ]
    targets = constraints.get("nutrition_targets")
    if targets:
        lines.append("Nutrition targets supplied by you; not medical advice.")
    for day, day_meals in by_day.items():
        lines.append(f"\n{day}")
        lines.extend(
            f"• Planned meal #{meal.id} (menu item #{meal.menu_item_id}): "
            f"{meal.meal_period.title()} — {meal.title} "
            f"({meal.currency} {meal.estimated_cost_cents / 100:.2f})"
            for meal in day_meals
        )
        verified_values: list[dict[str, float]] = []
        for meal in day_meals:
            nutrition = (meal.selection_json or {}).get("nutrition") or {}
            values = nutrition.get("per_serving") or {}
            if nutrition.get("status") != "VERIFIED" or any(key not in values for key in NUTRITION_TARGET_KEYS):
                verified_values = []
                break
            verified_values.append({key: float(values[key]) for key in NUTRITION_TARGET_KEYS})
        if verified_values and len(verified_values) == len(day_meals):
            totals = {
                key: round(sum(value[key] for value in verified_values), 1)
                for key in NUTRITION_TARGET_KEYS
            }
            shown = (
                f"Energy {totals['energy_kcal']:.0f} kcal; protein {totals['protein_g']:.1f} g; "
                f"carbs {totals['carbohydrates_g']:.1f} g; fat {totals['fat_g']:.1f} g; "
                f"fibre {totals['fiber_g']:.1f} g."
            )
            if targets:
                shown += " Daily targets: " + ", ".join(
                    f"{key.replace('_', ' ')} {targets[key]}"
                    for key in NUTRITION_TARGET_KEYS
                ) + "."
            lines.append("Verified menu nutrition totals: " + shown)
        elif targets:
            lines.append("Nutrition target coverage is incomplete; no nutrition values were guessed.")
    if plan.status == "DRAFT":
        lines.append(
            f"\nSend “confirm meal plan {plan.id}” to confirm the planning draft. "
            "This does not place an order. To order a meal, request it separately with its "
            "delivery date and time; the regular order flow will verify the menu and ask for "
            "explicit confirmation before placing that individual order."
        )
    else:
        lines.append(
            "\nThis plan confirmation did not place any orders. Each order must be requested "
            "and explicitly confirmed through the normal ordering flow."
        )
    lines.append(
        "To adjust a draft meal, send: change planned meal #<meal id> to menu item "
        "#<menu item id> [at <ISO-8601 time with timezone>]."
    )
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
    meal_edit_match = _MEAL_EDIT_RE.match(text)
    routine_state = _ROUTINE_STATE_RE.match(text)
    routine_skip = _ROUTINE_SKIP_RE.match(text)
    planned_meal_order = _ORDER_PLANNED_MEAL_RE.match(text)
    if not any((
        create_match, show_intent, confirm_match, meal_confirm_match,
        meal_edit_match, planned_meal_order, routine_state, routine_skip,
    )):
        return None

    channel = await _channel(chat_id, channel_type)
    if not channel:
        return (
            "Meal planning needs a linked customer profile. Place one identified order or link this "
            "channel to your existing profile, then try again."
        )
    customer_id = int(channel.customer_id)
    restaurant_id = channel.restaurant_id

    if planned_meal_order:
        return await _start_planned_meal_order(
            meal_id=int(planned_meal_order.group(1)),
            chat_id=chat_id,
            channel_type=channel_type,
            channel=channel,
        )

    async with get_session_factory()() as db:
        try:
            if create_match:
                requested = create_match.group(1).lower()
                scope = "DAILY" if requested in {"day", "daily"} else "WEEKLY"
                diets, allergies, goals, budget_cents, nutrition_targets = _request_constraints(text)
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
                    nutrition_targets=nutrition_targets,
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

            if meal_edit_match:
                meal_id = int(meal_edit_match.group(1))
                menu_item_id = int(meal_edit_match.group(2))
                desired_for = None
                if meal_edit_match.group(3):
                    try:
                        desired_for = datetime.fromisoformat(
                            meal_edit_match.group(3).replace("Z", "+00:00")
                        )
                    except ValueError as exc:
                        raise MealPlanningError(
                            "Meal time must be an ISO-8601 datetime with a timezone."
                        ) from exc
                    if desired_for.tzinfo is None:
                        raise MealPlanningError(
                            "Meal time must include a timezone, for example +05:30."
                        )
                meal = (await db.execute(select(MealSchedule).where(
                    MealSchedule.id == meal_id,
                    MealSchedule.restaurant_id == restaurant_id,
                    MealSchedule.customer_id == customer_id,
                ).with_for_update())).scalar_one_or_none()
                if not meal:
                    return "That planned meal was not found on your linked profile."
                plan = (await db.execute(select(MealPlan).where(
                    MealPlan.id == meal.meal_plan_id,
                    MealPlan.restaurant_id == restaurant_id,
                    MealPlan.customer_id == customer_id,
                ).with_for_update())).scalar_one_or_none()
                if not plan:
                    return "That meal plan was not found on your linked profile."
                menu_item = (await db.execute(select(MenuItem).where(
                    MenuItem.id == menu_item_id,
                    MenuItem.restaurant_id == restaurant_id,
                    MenuItem.available.is_(True),
                    or_(MenuItem.stock_qty.is_(None), MenuItem.stock_qty > 0),
                ))).scalar_one_or_none()
                if not menu_item:
                    return "That menu item is unavailable. Choose an available item from the current menu."
                await update_planned_meal(
                    db,
                    plan=plan,
                    schedule=meal,
                    menu_item=menu_item,
                    desired_for=desired_for,
                    actor_id=str(channel.id),
                )
                await db.commit()
                return (
                    f"Planned meal #{meal.id} is now {meal.title} for "
                    f"{meal.desired_for.astimezone(ZoneInfo(plan.timezone)).isoformat()}. "
                    f"Plan #{plan.id} returned to draft and needs review again. "
                    "No order has been placed."
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
