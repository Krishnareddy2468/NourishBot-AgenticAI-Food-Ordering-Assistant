"""Meal-plan creation, editing, confirmation, templates, and routine materialization."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from app.core.config import settings
from app.db.models import (
    CustomerConsent,
    MealBudget,
    MealPlan,
    MealPlanningEvent,
    MealRoutine,
    MealRoutineException,
    MealSchedule,
    MealTemplate,
    MenuCategory,
    MenuItem,
    Order,
    OrderItem,
    ScheduledJob,
    UserPreferences,
)
from app.db.session import get_session_factory
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
    parse_timezone,
    recommend_complete_plan,
    recurrence_dates,
    routine_occurrence_key,
)
from app.domain.scheduling import PLAN_CONFIRMATION_REMINDER, PLANNED_MEAL_REMINDER


def planning_event(
    db,
    *,
    restaurant_id: int,
    customer_id: int,
    event_type: str,
    actor_type: str,
    actor_id: str | None = None,
    plan_id: int | None = None,
    schedule_id: int | None = None,
    routine_id: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> MealPlanningEvent:
    event = MealPlanningEvent(
        restaurant_id=restaurant_id,
        customer_id=customer_id,
        meal_plan_id=plan_id,
        meal_schedule_id=schedule_id,
        routine_id=routine_id,
        event_type=event_type,
        actor_type=actor_type,
        actor_id=actor_id,
        metadata_json=metadata or {},
    )
    db.add(event)
    return event


async def _latest_consent(db, restaurant_id: int, customer_id: int, purpose: str) -> bool:
    value = (await db.execute(
        select(CustomerConsent.granted)
        .where(
            CustomerConsent.restaurant_id == restaurant_id,
            CustomerConsent.customer_id == customer_id,
            CustomerConsent.purpose == purpose,
        )
        .order_by(CustomerConsent.id.desc())
        .limit(1)
    )).scalar_one_or_none()
    return value is True


async def effective_constraints(
    db,
    *,
    restaurant_id: int,
    customer_id: int,
    dietary_flags: list[str] | None = None,
    allergies: list[str] | None = None,
    health_goals: list[str] | None = None,
    budget_cents: int | None = None,
    currency: str = "INR",
    starts_on: date | None = None,
    ends_on: date | None = None,
) -> tuple[PlanConstraints, bool]:
    """Merge explicit safety constraints with consent-gated preferences."""
    use_memory = await _latest_consent(db, restaurant_id, customer_id, "PERSONALIZATION")
    preference = None
    if use_memory:
        preference = (await db.execute(
            select(UserPreferences).where(UserPreferences.customer_id == customer_id)
        )).scalar_one_or_none()

    diets = list(dietary_flags or [])
    known_allergies = list(allergies or [])
    goals = list(health_goals or [])
    if preference:
        diets.extend(preference.dietary_flags or [])
        # Stored allergies remain safety data once the customer explicitly
        # supplies them; they are never weakened by request overrides.
        known_allergies.extend(preference.allergies or [])
        goals.extend(preference.health_goals or [])

    if budget_cents is None and starts_on is not None:
        budget = (await db.execute(
            select(MealBudget)
            .where(
                MealBudget.restaurant_id == restaurant_id,
                MealBudget.customer_id == customer_id,
                MealBudget.active.is_(True),
                MealBudget.starts_on <= starts_on,
                MealBudget.ends_on >= (ends_on or starts_on),
            )
            .order_by(MealBudget.starts_on.desc())
            .limit(1)
        )).scalar_one_or_none()
        if budget:
            budget_cents = budget.limit_cents
            currency = budget.currency

    return PlanConstraints.from_values(
        dietary_flags=diets,
        allergies=known_allergies,
        health_goals=goals,
        budget_cents=budget_cents,
        currency=currency,
    ), use_memory


async def _menu_candidates(db, restaurant_id: int, customer_id: int, *, use_history: bool) -> list[MenuCandidate]:
    history: dict[int, int] = {}
    if use_history:
        history = dict((await db.execute(
            select(OrderItem.item_id, func.sum(OrderItem.qty))
            .join(Order, Order.id == OrderItem.order_id)
            .where(
                Order.restaurant_id == restaurant_id,
                Order.customer_id == customer_id,
                Order.status.in_(["DELIVERED", "COMPLETED"]),
            )
            .group_by(OrderItem.item_id)
        )).all())

    rows = (await db.execute(
        select(MenuItem, MenuCategory.name)
        .outerjoin(MenuCategory, MenuCategory.id == MenuItem.category_id)
        .where(
            MenuItem.restaurant_id == restaurant_id,
            MenuItem.available.is_(True),
            (MenuItem.stock_qty.is_(None) | (MenuItem.stock_qty > 0)),
        )
        .order_by(MenuItem.id)
    )).all()
    return [
        MenuCandidate.from_mapping({
            "id": item.id,
            "name": item.name,
            "price_cents": item.special_price_cents if item.special_price_cents is not None else item.price_cents,
            "menu_type": (item.type or "").upper(),
            "tags": item.tags or [],
            "category": category or "",
            "history_count": int(history.get(item.id, 0)),
        })
        for item, category in rows
    ]


def _add_schedule_job(
    db,
    schedule: MealSchedule,
    *,
    job_type: str,
    run_at: datetime,
    deadline_at: datetime,
) -> ScheduledJob:
    job = ScheduledJob(
        restaurant_id=schedule.restaurant_id,
        meal_schedule_id=schedule.id,
        order_id=None,
        job_type=job_type,
        dedupe_key=f"meal-schedule:{schedule.id}:{job_type.lower()}:v{schedule.revision}",
        run_at=run_at,
        deadline_at=deadline_at,
        max_attempts=settings.SCHEDULE_JOB_MAX_ATTEMPTS,
        payload_json={
            "customer_id": schedule.customer_id,
            "meal_plan_id": schedule.meal_plan_id,
            "routine_id": schedule.routine_id,
            "meal_period": schedule.meal_period,
            "desired_for": schedule.desired_for.isoformat(),
        },
    )
    db.add(job)
    return job


async def _cancel_schedule_jobs(db, schedule_ids: list[int], *, reason: str) -> None:
    if not schedule_ids:
        return
    jobs = (await db.execute(
        select(ScheduledJob).where(
            ScheduledJob.meal_schedule_id.in_(schedule_ids),
            ScheduledJob.state.in_(["PENDING", "RETRY_WAIT"]),
        )
    )).scalars().all()
    now = datetime.now(timezone.utc)
    for job in jobs:
        job.state = "CANCELLED"
        job.completed_at = now
        job.next_action = reason


async def create_recommended_plan(
    db,
    *,
    restaurant_id: int,
    customer_id: int,
    scope: str,
    starts_on: date,
    name: str | None = None,
    timezone_name: str = "Asia/Kolkata",
    dietary_flags: list[str] | None = None,
    allergies: list[str] | None = None,
    health_goals: list[str] | None = None,
    budget_cents: int | None = None,
    currency: str = "INR",
    source: str = "WEB",
    channel_id: int | None = None,
    actor_type: str = "CUSTOMER",
    actor_id: str | None = None,
) -> tuple[MealPlan, list[MealSchedule]]:
    scope = scope.strip().upper()
    dates = meal_dates(scope, starts_on)
    if len(dates) > settings.MEAL_PLAN_MAX_HORIZON_DAYS:
        raise MealPlanningError("Requested plan exceeds the configured planning horizon")
    zone = parse_timezone(timezone_name)
    if starts_on < datetime.now(zone).date():
        raise MealPlanningError("Meal plans cannot start in the past")
    constraints, use_history = await effective_constraints(
        db,
        restaurant_id=restaurant_id,
        customer_id=customer_id,
        dietary_flags=dietary_flags,
        allergies=allergies,
        health_goals=health_goals,
        budget_cents=budget_cents,
        currency=currency,
        starts_on=starts_on,
        ends_on=dates[-1],
    )
    candidates = await _menu_candidates(db, restaurant_id, customer_id, use_history=use_history)
    recommendations = recommend_complete_plan(
        candidates,
        scope=scope,
        starts_on=starts_on,
        timezone_name=timezone_name,
        constraints=constraints,
    )
    plan = MealPlan(
        restaurant_id=restaurant_id,
        customer_id=customer_id,
        name=(name or f"{scope.title()} meal plan from {starts_on.isoformat()}")[:160],
        status="DRAFT",
        scope=scope,
        starts_on=dates[0],
        ends_on=dates[-1],
        timezone=timezone_name,
        constraints_json=constraints.public_dict(),
        estimated_total_cents=sum(item.candidate.price_cents for item in recommendations),
        currency=constraints.currency,
        revision=1,
        created_by_channel_id=channel_id,
    )
    db.add(plan)
    await db.flush()

    schedules: list[MealSchedule] = []
    now = datetime.now(timezone.utc)
    for item in recommendations:
        candidate = item.candidate
        schedule = MealSchedule(
            restaurant_id=restaurant_id,
            customer_id=customer_id,
            meal_plan_id=plan.id,
            menu_item_id=candidate.id,
            title=candidate.name,
            meal_period=item.meal_period,
            state="DRAFT",
            desired_for=item.desired_for,
            timezone=timezone_name,
            source=source.upper(),
            selection_json={
                "recommendation_reasons": list(item.reason),
                "history_used": use_history,
                "menu_type": candidate.menu_type,
            },
            constraints_json=constraints.public_dict(),
            estimated_cost_cents=candidate.price_cents,
            currency=constraints.currency,
            diet_tags=sorted({candidate.menu_type.lower(), *candidate.tags}),
            allergen_tags=sorted(tag for tag in candidate.tags if tag.startswith(("allergen-", "contains-"))),
            revision=1,
        )
        db.add(schedule)
        schedules.append(schedule)
    await db.flush()
    for schedule in schedules:
        reminder_at = schedule.desired_for - timedelta(hours=settings.PLAN_CONFIRMATION_REMINDER_HOURS)
        if reminder_at > now:
            _add_schedule_job(
                db,
                schedule,
                job_type=PLAN_CONFIRMATION_REMINDER,
                run_at=reminder_at,
                deadline_at=schedule.desired_for,
            )
    planning_event(
        db,
        restaurant_id=restaurant_id,
        customer_id=customer_id,
        plan_id=plan.id,
        event_type="PLAN_DRAFT_CREATED",
        actor_type=actor_type,
        actor_id=actor_id,
        metadata={
            "scope": scope,
            "meal_count": len(schedules),
            "constraints": constraints.public_dict(),
            "estimated_total_cents": plan.estimated_total_cents,
        },
    )
    return plan, schedules


async def update_planned_meal(
    db,
    *,
    plan: MealPlan,
    schedule: MealSchedule,
    title: str | None = None,
    menu_item: MenuItem | None = None,
    desired_for: datetime | None = None,
    meal_period: str | None = None,
    actor_id: str | None = None,
) -> MealSchedule:
    if schedule.meal_plan_id != plan.id or schedule.customer_id != plan.customer_id:
        raise MealPlanningError("Planned meal does not belong to this plan")
    period = (meal_period or schedule.meal_period).upper()
    if period not in MEAL_PERIODS:
        raise MealPlanningError("Meal period must be breakfast, lunch, snacks, or dinner")
    constraints = PlanConstraints.from_values(**{
        "dietary_flags": (plan.constraints_json or {}).get("dietary_flags"),
        "allergies": (plan.constraints_json or {}).get("allergies"),
        "health_goals": (plan.constraints_json or {}).get("health_goals"),
        "budget_cents": (plan.constraints_json or {}).get("budget_cents"),
        "currency": (plan.constraints_json or {}).get("currency", "INR"),
    })
    if menu_item is not None:
        tags = tuple(menu_item.tags or [])
        candidate = MenuCandidate(
            id=menu_item.id,
            name=menu_item.name,
            price_cents=menu_item.special_price_cents if menu_item.special_price_cents is not None else menu_item.price_cents,
            menu_type=(menu_item.type or "").upper(),
            tags=tags,
        )
        assert_diet_compatible(candidate.menu_type, tags, constraints.dietary_flags)
        assert_allergy_safe(tags, constraints.allergies)
        schedule.menu_item_id = candidate.id
        schedule.title = candidate.name
        schedule.estimated_cost_cents = candidate.price_cents
        schedule.diet_tags = sorted({candidate.menu_type.lower(), *candidate.tags})
        schedule.allergen_tags = sorted(tag for tag in candidate.tags if tag.startswith(("allergen-", "contains-")))
        schedule.selection_json = {
            **(schedule.selection_json or {}),
            "menu_type": candidate.menu_type,
            "recommendation_reasons": ["selected by the customer"],
        }
    elif title is not None:
        schedule.title = title.strip()[:160]
    if not schedule.title:
        raise MealPlanningError("Planned meal title is required")
    if desired_for is not None:
        if desired_for.tzinfo is None:
            raise MealPlanningError("Planned meal time must include a timezone")
        schedule.rescheduled_from = schedule.desired_for
        schedule.desired_for = desired_for.astimezone(timezone.utc)
    schedule.meal_period = period
    schedule.state = "DRAFT"
    schedule.confirmed_at = None
    schedule.revision = int(schedule.revision or 1) + 1
    plan.status = "DRAFT"
    plan.confirmed_at = None
    plan.confirmation_snapshot_hash = None
    plan.revision = int(plan.revision or 1) + 1
    meals = (await db.execute(
        select(MealSchedule).where(MealSchedule.meal_plan_id == plan.id)
    )).scalars().all()
    await _cancel_schedule_jobs(db, [item.id for item in meals], reason="PLAN_SNAPSHOT_INVALIDATED")
    now = datetime.now(timezone.utc)
    for item in meals:
        if item.id != schedule.id:
            item.revision = int(item.revision or 1) + 1
        item.state = "DRAFT"
        item.confirmed_at = None
        reminder_at = item.desired_for - timedelta(hours=settings.PLAN_CONFIRMATION_REMINDER_HOURS)
        if reminder_at > now:
            _add_schedule_job(
                db,
                item,
                job_type=PLAN_CONFIRMATION_REMINDER,
                run_at=reminder_at,
                deadline_at=item.desired_for,
            )
    planning_event(
        db,
        restaurant_id=plan.restaurant_id,
        customer_id=plan.customer_id,
        plan_id=plan.id,
        schedule_id=schedule.id,
        event_type="PLANNED_MEAL_EDITED",
        actor_type="CUSTOMER",
        actor_id=actor_id,
        metadata={"revision": schedule.revision, "meal_period": period},
    )
    plan.estimated_total_cents = sum(item.estimated_cost_cents for item in meals)
    if constraints.budget_cents is not None and plan.estimated_total_cents > constraints.budget_cents:
        raise MealPlanningError("This edit would exceed the plan budget")
    return schedule


async def confirm_plan(db, plan: MealPlan, *, actor_id: str | None = None) -> tuple[dict[str, Any], str]:
    if plan.status not in {"DRAFT", "CONFIRMED"}:
        raise MealPlanningError("Only a draft meal plan can be confirmed")
    meals = (await db.execute(
        select(MealSchedule)
        .where(MealSchedule.meal_plan_id == plan.id)
        .order_by(MealSchedule.desired_for, MealSchedule.meal_period)
        .with_for_update()
    )).scalars().all()
    expected = len(meal_dates(plan.scope, plan.starts_on)) * len(MEAL_PERIODS)
    active_meals = [meal for meal in meals if meal.state not in {"SKIPPED", "SUPERSEDED", "CANCELLED"}]
    if len(active_meals) != expected:
        raise MealPlanningError(f"A complete {plan.scope.lower()} plan requires {expected} meal slots")
    if any(meal.state not in {"DRAFT", "CONFIRMED"} for meal in active_meals):
        raise MealPlanningError("Every active planned meal must be a draft before confirmation")
    snapshot, snapshot_hash = canonical_snapshot(plan, active_meals)
    if plan.status == "CONFIRMED" and plan.confirmation_snapshot_hash == snapshot_hash:
        return snapshot, snapshot_hash
    constraints = plan.constraints_json or {}
    for meal in active_meals:
        assert_diet_compatible(
            str((meal.selection_json or {}).get("menu_type") or ""),
            meal.diet_tags or [],
            constraints.get("dietary_flags") or [],
        )
        assert_allergy_safe(
            [*(meal.diet_tags or []), *(meal.allergen_tags or [])],
            constraints.get("allergies") or [],
        )
        if meal.desired_for <= datetime.now(timezone.utc):
            raise MealPlanningError("Past planned meals must be rescheduled before confirmation")

    now = datetime.now(timezone.utc)
    plan.status = "CONFIRMED"
    plan.confirmed_at = now
    plan.confirmation_snapshot_hash = snapshot_hash
    await _cancel_schedule_jobs(db, [meal.id for meal in active_meals], reason="PLAN_CONFIRMED")
    for meal in active_meals:
        meal.state = "CONFIRMED"
        meal.confirmed_at = now
        reminder_at = meal.desired_for - timedelta(minutes=settings.PLANNED_MEAL_REMINDER_MINUTES)
        if reminder_at > now:
            _add_schedule_job(
                db,
                meal,
                job_type=PLANNED_MEAL_REMINDER,
                run_at=reminder_at,
                deadline_at=meal.desired_for,
            )
    planning_event(
        db,
        restaurant_id=plan.restaurant_id,
        customer_id=plan.customer_id,
        plan_id=plan.id,
        event_type="PLAN_CONFIRMED",
        actor_type="CUSTOMER",
        actor_id=actor_id,
        metadata={"snapshot_hash": snapshot_hash, "meal_count": len(active_meals)},
    )
    return snapshot, snapshot_hash


async def confirm_planned_meal(
    db,
    schedule: MealSchedule,
    *,
    actor_id: str | None = None,
) -> str:
    """Confirm one routine/template meal without creating an order."""
    constraints = schedule.constraints_json or {}
    assert_diet_compatible(
        str((schedule.selection_json or {}).get("menu_type") or ""),
        schedule.diet_tags or [],
        constraints.get("dietary_flags") or [],
    )
    assert_allergy_safe(
        [*(schedule.diet_tags or []), *(schedule.allergen_tags or [])],
        constraints.get("allergies") or [],
    )
    if schedule.desired_for <= datetime.now(timezone.utc):
        raise MealPlanningError("Past planned meals must be rescheduled before confirmation")
    receipt_source = {
        "meal_schedule_id": schedule.id,
        "revision": schedule.revision,
        "title": schedule.title,
        "meal_period": schedule.meal_period,
        "desired_for": schedule.desired_for.isoformat(),
        "menu_item_id": schedule.menu_item_id,
        "estimated_cost_cents": schedule.estimated_cost_cents,
        "constraints": constraints,
    }
    receipt_hash = hashlib.sha256(
        json.dumps(receipt_source, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    if schedule.state == "CONFIRMED":
        stored = (schedule.selection_json or {}).get("confirmation_snapshot_hash")
        if stored == receipt_hash:
            return receipt_hash
        raise MealPlanningError("Confirmed meal snapshot no longer matches its receipt")
    if schedule.state != "DRAFT":
        raise MealPlanningError("Only a draft planned meal can be confirmed")
    now = datetime.now(timezone.utc)
    await _cancel_schedule_jobs(db, [schedule.id], reason="PLANNED_MEAL_CONFIRMED")
    schedule.state = "CONFIRMED"
    schedule.confirmed_at = now
    schedule.selection_json = {
        **(schedule.selection_json or {}),
        "confirmation_snapshot_hash": receipt_hash,
    }
    reminder_at = schedule.desired_for - timedelta(minutes=settings.PLANNED_MEAL_REMINDER_MINUTES)
    if reminder_at > now:
        _add_schedule_job(
            db,
            schedule,
            job_type=PLANNED_MEAL_REMINDER,
            run_at=reminder_at,
            deadline_at=schedule.desired_for,
        )
    planning_event(
        db,
        restaurant_id=schedule.restaurant_id,
        customer_id=schedule.customer_id,
        plan_id=schedule.meal_plan_id,
        schedule_id=schedule.id,
        routine_id=schedule.routine_id,
        event_type="PLANNED_MEAL_CONFIRMED",
        actor_type="CUSTOMER",
        actor_id=actor_id,
        metadata={"snapshot_hash": receipt_hash},
    )
    return receipt_hash


async def create_template_from_plan(
    db,
    *,
    plan: MealPlan,
    name: str,
    scope: str,
    schedule: MealSchedule | None = None,
    actor_id: str | None = None,
) -> MealTemplate:
    normalized_scope = scope.strip().upper()
    if normalized_scope not in {"MEAL", "DAY"}:
        raise MealPlanningError("Template scope must be MEAL or DAY")
    if normalized_scope == "MEAL":
        if schedule is None or schedule.meal_plan_id != plan.id:
            raise MealPlanningError("A meal template requires a planned meal")
        definitions = [_schedule_definition(schedule)]
    else:
        schedules = (await db.execute(
            select(MealSchedule)
            .where(MealSchedule.meal_plan_id == plan.id)
            .order_by(MealSchedule.desired_for, MealSchedule.meal_period)
        )).scalars().all()
        if not schedules:
            raise MealPlanningError("The plan has no meals to save")
        first_day = schedules[0].desired_for.astimezone(parse_timezone(plan.timezone)).date()
        definitions = [
            _schedule_definition(item)
            for item in schedules
            if item.desired_for.astimezone(parse_timezone(plan.timezone)).date() == first_day
        ]
    template = MealTemplate(
        restaurant_id=plan.restaurant_id,
        customer_id=plan.customer_id,
        name=name.strip()[:160],
        scope=normalized_scope,
        definition_json={"meals": definitions, "constraints": plan.constraints_json or {}},
        active=True,
    )
    if not template.name:
        raise MealPlanningError("Template name is required")
    db.add(template)
    await db.flush()
    planning_event(
        db,
        restaurant_id=plan.restaurant_id,
        customer_id=plan.customer_id,
        plan_id=plan.id,
        event_type="TEMPLATE_CREATED",
        actor_type="CUSTOMER",
        actor_id=actor_id,
        metadata={"template_id": template.id, "scope": normalized_scope},
    )
    return template


def _schedule_definition(schedule: MealSchedule) -> dict[str, Any]:
    return {
        "title": schedule.title,
        "menu_item_id": schedule.menu_item_id,
        "meal_period": schedule.meal_period,
        "estimated_cost_cents": schedule.estimated_cost_cents,
        "currency": schedule.currency,
        "diet_tags": schedule.diet_tags or [],
        "allergen_tags": schedule.allergen_tags or [],
        "menu_type": (schedule.selection_json or {}).get("menu_type") or "",
        "constraints": schedule.constraints_json or {},
    }


async def create_routine(
    db,
    *,
    template: MealTemplate,
    name: str,
    recurrence_rule: str,
    meal_period: str,
    time_of_day: time,
    timezone_name: str,
    horizon_days: int | None = None,
    actor_id: str | None = None,
) -> tuple[MealRoutine, list[MealSchedule]]:
    if template.scope != "MEAL":
        raise MealPlanningError("Recurring routines require a single-meal template")
    parse_recurrence_rule(recurrence_rule)
    parse_timezone(timezone_name)
    period = meal_period.strip().upper()
    if period not in MEAL_PERIODS:
        raise MealPlanningError("Routine meal period is invalid")
    horizon = int(horizon_days or settings.ROUTINE_MATERIALIZATION_HORIZON_DAYS)
    if horizon < 1 or horizon > settings.MEAL_PLAN_MAX_HORIZON_DAYS:
        raise MealPlanningError("Routine materialization horizon is outside the configured range")
    routine = MealRoutine(
        restaurant_id=template.restaurant_id,
        customer_id=template.customer_id,
        template_id=template.id,
        name=name.strip()[:160],
        state="ACTIVE",
        meal_period=period,
        timezone=timezone_name,
        recurrence_rule=recurrence_rule.strip().upper().replace(" ", ""),
        time_of_day=time_of_day,
        materialization_horizon_days=horizon,
        version=1,
        next_materialization_at=datetime.now(timezone.utc),
    )
    if not routine.name:
        raise MealPlanningError("Routine name is required")
    db.add(routine)
    await db.flush()
    created = await materialize_routine(db, routine, template=template)
    planning_event(
        db,
        restaurant_id=routine.restaurant_id,
        customer_id=routine.customer_id,
        routine_id=routine.id,
        event_type="ROUTINE_CREATED",
        actor_type="CUSTOMER",
        actor_id=actor_id,
        metadata={"recurrence_rule": routine.recurrence_rule, "materialized": len(created)},
    )
    return routine, created


async def materialize_routine(
    db,
    routine: MealRoutine,
    *,
    template: MealTemplate | None = None,
    today: date | None = None,
) -> list[MealSchedule]:
    if routine.state != "ACTIVE":
        return []
    template = template or (await db.execute(
        select(MealTemplate).where(MealTemplate.id == routine.template_id)
    )).scalar_one()
    definitions = list((template.definition_json or {}).get("meals") or [])
    if len(definitions) != 1:
        raise MealPlanningError("Routine template must contain exactly one meal")
    definition = dict(definitions[0])
    constraints = (template.definition_json or {}).get("constraints") or definition.get("constraints") or {}
    menu_item_id = definition.get("menu_item_id")
    if menu_item_id:
        menu_item = (await db.execute(select(MenuItem).where(
            MenuItem.id == menu_item_id,
            MenuItem.restaurant_id == routine.restaurant_id,
        ))).scalar_one_or_none()
        if not menu_item or not menu_item.available or (
            menu_item.stock_qty is not None and menu_item.stock_qty <= 0
        ):
            routine.state = "PAUSED"
            routine.paused_at = datetime.now(timezone.utc)
            planning_event(
                db,
                restaurant_id=routine.restaurant_id,
                customer_id=routine.customer_id,
                routine_id=routine.id,
                event_type="ROUTINE_PAUSED_MENU_UNAVAILABLE",
                actor_type="SYSTEM",
                metadata={"menu_item_id": menu_item_id},
            )
            return []
        live_candidate = MenuCandidate.from_mapping({
            "id": menu_item.id,
            "name": menu_item.name,
            "price_cents": menu_item.special_price_cents if menu_item.special_price_cents is not None else menu_item.price_cents,
            "menu_type": menu_item.type or "",
            "tags": menu_item.tags or [],
        })
        definition.update({
            "title": live_candidate.name,
            "estimated_cost_cents": live_candidate.price_cents,
            "menu_type": live_candidate.menu_type,
            "diet_tags": sorted({live_candidate.menu_type.lower(), *live_candidate.tags}),
            "allergen_tags": sorted(
                tag for tag in live_candidate.tags if tag.startswith(("allergen-", "contains-"))
            ),
        })
    try:
        assert_diet_compatible(
            str(definition.get("menu_type") or ""),
            definition.get("diet_tags") or [],
            constraints.get("dietary_flags") or [],
        )
        assert_allergy_safe(
            [*(definition.get("diet_tags") or []), *(definition.get("allergen_tags") or [])],
            constraints.get("allergies") or [],
        )
    except MealPlanningError as exc:
        routine.state = "PAUSED"
        routine.paused_at = datetime.now(timezone.utc)
        planning_event(
            db,
            restaurant_id=routine.restaurant_id,
            customer_id=routine.customer_id,
            routine_id=routine.id,
            event_type="ROUTINE_PAUSED_SAFETY_CONFLICT",
            actor_type="SYSTEM",
            metadata={"reason": str(exc)},
        )
        return []
    start = today or datetime.now(parse_timezone(routine.timezone)).date()
    horizon = min(routine.materialization_horizon_days, settings.MEAL_PLAN_MAX_HORIZON_DAYS)
    end = start + timedelta(days=horizon - 1)
    dates = recurrence_dates(routine.recurrence_rule, start, end)
    exceptions = {
        item.occurrence_date: item
        for item in (await db.execute(
            select(MealRoutineException).where(
                MealRoutineException.routine_id == routine.id,
                MealRoutineException.occurrence_date >= start,
                MealRoutineException.occurrence_date <= end,
            )
        )).scalars().all()
    }
    existing_rows = (await db.execute(
        select(MealSchedule.occurrence_key, MealSchedule.state).where(
            MealSchedule.routine_id == routine.id,
            MealSchedule.occurrence_key.is_not(None),
        )
    )).all()
    existing_dates = {
        key.rsplit(":", 1)[-1]
        for key, state in existing_rows
        if state not in {"SUPERSEDED", "CANCELLED", "SKIPPED"}
    }
    zone = parse_timezone(routine.timezone)
    created: list[MealSchedule] = []
    for occurrence_date in dates:
        key = routine_occurrence_key(routine.id, routine.version, occurrence_date)
        exception = exceptions.get(occurrence_date)
        if occurrence_date.isoformat() in existing_dates or (exception and exception.action == "SKIP"):
            continue
        desired_for = (
            exception.replacement_for
            if exception and exception.action == "RESCHEDULE"
            else datetime.combine(occurrence_date, routine.time_of_day, tzinfo=zone).astimezone(timezone.utc)
        )
        if desired_for <= datetime.now(timezone.utc):
            continue
        schedule = MealSchedule(
            restaurant_id=routine.restaurant_id,
            customer_id=routine.customer_id,
            template_id=template.id,
            routine_id=routine.id,
            menu_item_id=definition.get("menu_item_id"),
            title=str(definition.get("title") or routine.name)[:160],
            meal_period=routine.meal_period,
            state="DRAFT",
            desired_for=desired_for,
            timezone=routine.timezone,
            source="ROUTINE",
            selection_json={
                "routine_version": routine.version,
                "menu_type": definition.get("menu_type") or "",
            },
            constraints_json=constraints,
            estimated_cost_cents=int(definition.get("estimated_cost_cents") or 0),
            currency=str(definition.get("currency") or "INR"),
            diet_tags=definition.get("diet_tags") or [],
            allergen_tags=definition.get("allergen_tags") or [],
            occurrence_key=key,
            revision=1,
            rescheduled_from=(
                datetime.combine(occurrence_date, routine.time_of_day, tzinfo=zone).astimezone(timezone.utc)
                if exception and exception.action == "RESCHEDULE"
                else None
            ),
        )
        db.add(schedule)
        created.append(schedule)
    await db.flush()
    now = datetime.now(timezone.utc)
    for schedule in created:
        reminder_at = schedule.desired_for - timedelta(hours=settings.PLAN_CONFIRMATION_REMINDER_HOURS)
        if reminder_at > now:
            _add_schedule_job(
                db,
                schedule,
                job_type=PLAN_CONFIRMATION_REMINDER,
                run_at=reminder_at,
                deadline_at=schedule.desired_for,
            )
        planning_event(
            db,
            restaurant_id=routine.restaurant_id,
            customer_id=routine.customer_id,
            schedule_id=schedule.id,
            routine_id=routine.id,
            event_type="ROUTINE_MEAL_MATERIALIZED",
            actor_type="SYSTEM",
            metadata={"occurrence_key": schedule.occurrence_key},
        )
    routine.materialized_through = end
    routine.next_materialization_at = now + timedelta(days=1)
    template.times_used = int(template.times_used or 0) + len(created)
    return created


async def change_routine(
    db,
    routine: MealRoutine,
    *,
    action: str,
    occurrence_date: date | None = None,
    replacement_for: datetime | None = None,
    recurrence_rule: str | None = None,
    time_of_day: time | None = None,
    reason: str | None = None,
    actor_id: str | None = None,
) -> list[MealSchedule]:
    normalized = action.strip().upper()
    now = datetime.now(timezone.utc)
    future = (await db.execute(
        select(MealSchedule).where(
            MealSchedule.routine_id == routine.id,
            MealSchedule.desired_for >= now,
            MealSchedule.state.in_(["DRAFT", "PAUSED"]),
        )
    )).scalars().all()
    if normalized == "PAUSE":
        if routine.state != "ACTIVE":
            raise MealPlanningError("Only an active routine can be paused")
        routine.state = "PAUSED"
        routine.paused_at = now
        for schedule in future:
            schedule.state = "PAUSED"
        await _cancel_schedule_jobs(db, [item.id for item in future], reason="ROUTINE_PAUSED")
        created: list[MealSchedule] = []
    elif normalized == "RESUME":
        if routine.state != "PAUSED":
            raise MealPlanningError("Only a paused routine can be resumed")
        routine.state = "ACTIVE"
        routine.paused_at = None
        routine.next_materialization_at = now
        for schedule in future:
            schedule.state = "DRAFT"
            reminder_at = schedule.desired_for - timedelta(hours=settings.PLAN_CONFIRMATION_REMINDER_HOURS)
            if reminder_at > now:
                schedule.revision = int(schedule.revision or 1) + 1
                _add_schedule_job(
                    db,
                    schedule,
                    job_type=PLAN_CONFIRMATION_REMINDER,
                    run_at=reminder_at,
                    deadline_at=schedule.desired_for,
                )
        created = await materialize_routine(db, routine)
    elif normalized in {"SKIP", "RESCHEDULE"}:
        if routine.state not in {"ACTIVE", "PAUSED"}:
            raise MealPlanningError("Only an active or paused routine can be changed")
        if occurrence_date is None:
            raise MealPlanningError("Routine skip or reschedule requires an occurrence date")
        if normalized == "RESCHEDULE" and (replacement_for is None or replacement_for.tzinfo is None):
            raise MealPlanningError("Rescheduling requires a timezone-aware replacement time")
        if normalized == "RESCHEDULE" and replacement_for.astimezone(timezone.utc) <= now:
            raise MealPlanningError("A routine occurrence cannot be rescheduled into the past")
        exception = (await db.execute(
            select(MealRoutineException).where(
                MealRoutineException.routine_id == routine.id,
                MealRoutineException.occurrence_date == occurrence_date,
            )
        )).scalar_one_or_none()
        if exception is None:
            exception = MealRoutineException(
                restaurant_id=routine.restaurant_id,
                routine_id=routine.id,
                occurrence_date=occurrence_date,
                action=normalized,
            )
            db.add(exception)
        exception.action = normalized
        exception.replacement_for = replacement_for.astimezone(timezone.utc) if replacement_for else None
        exception.reason = (reason or "")[:500] or None
        schedule = (await db.execute(
            select(MealSchedule)
            .where(
                MealSchedule.routine_id == routine.id,
                MealSchedule.occurrence_key.like(f"routine:{routine.id}:v%:{occurrence_date.isoformat()}"),
                MealSchedule.state.notin_(["SUPERSEDED", "CANCELLED"]),
            )
            .order_by(MealSchedule.id.desc())
            .limit(1)
        )).scalar_one_or_none()
        created = []
        if schedule:
            await _cancel_schedule_jobs(db, [schedule.id], reason=f"ROUTINE_{normalized}")
            schedule.revision = int(schedule.revision or 1) + 1
            if normalized == "SKIP":
                schedule.state = "SKIPPED"
                schedule.skipped_at = now
            else:
                schedule.rescheduled_from = schedule.desired_for
                schedule.desired_for = replacement_for.astimezone(timezone.utc)
                schedule.state = "DRAFT"
                reminder_at = schedule.desired_for - timedelta(hours=settings.PLAN_CONFIRMATION_REMINDER_HOURS)
                if reminder_at > now:
                    _add_schedule_job(
                        db,
                        schedule,
                        job_type=PLAN_CONFIRMATION_REMINDER,
                        run_at=reminder_at,
                        deadline_at=schedule.desired_for,
                    )
        elif normalized == "RESCHEDULE":
            created = await materialize_routine(db, routine, today=occurrence_date)
    elif normalized == "EDIT":
        if recurrence_rule is None and time_of_day is None:
            raise MealPlanningError("Routine edit requires a recurrence rule or time")
        if recurrence_rule is not None:
            parse_recurrence_rule(recurrence_rule)
            routine.recurrence_rule = recurrence_rule.strip().upper().replace(" ", "")
        if time_of_day is not None:
            routine.time_of_day = time_of_day
        routine.version = int(routine.version or 1) + 1
        routine.next_materialization_at = now
        for schedule in future:
            schedule.state = "SUPERSEDED"
        await _cancel_schedule_jobs(db, [item.id for item in future], reason="ROUTINE_EDITED")
        created = await materialize_routine(db, routine)
    else:
        raise MealPlanningError("Routine action must be pause, resume, skip, reschedule, or edit")

    planning_event(
        db,
        restaurant_id=routine.restaurant_id,
        customer_id=routine.customer_id,
        routine_id=routine.id,
        event_type=f"ROUTINE_{normalized}",
        actor_type="CUSTOMER",
        actor_id=actor_id,
        metadata={
            "occurrence_date": occurrence_date.isoformat() if occurrence_date else None,
            "replacement_for": replacement_for.isoformat() if replacement_for else None,
            "reason": reason,
            "version": routine.version,
        },
    )
    return created


async def materialize_due_routines(*, now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(timezone.utc)
    async with get_session_factory()() as db:
        routines = (await db.execute(
            select(MealRoutine)
            .where(
                MealRoutine.state == "ACTIVE",
                (MealRoutine.next_materialization_at.is_(None) | (MealRoutine.next_materialization_at <= now)),
            )
            .order_by(MealRoutine.id)
            .limit(100)
            .with_for_update(skip_locked=True)
        )).scalars().all()
        created = 0
        for routine in routines:
            created += len(await materialize_routine(db, routine))
        await db.commit()
    return {"routines": len(routines), "meals_created": created}
