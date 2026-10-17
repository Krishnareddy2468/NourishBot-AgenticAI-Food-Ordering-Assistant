"""Linked-customer API for meal plans, templates, and recurring routines."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import or_, select

from app.api.routes.customer_privacy import linked_customer_channel
from app.db.models import MealPlan, MealRoutine, MealSchedule, MealTemplate, MenuItem
from app.db.session import AsyncSessionLocal
from app.domain.meal_planning import MEAL_PERIODS, MealPlanningError
from app.services.planned_meal_order import (
    PlannedMealOrderError,
    place_planned_meal_web_order,
    preview_planned_meal_web_order,
)
from app.services.meal_planner import (
    change_routine,
    confirm_planned_meal,
    confirm_plan,
    create_recommended_plan,
    create_routine,
    create_template_from_plan,
    update_planned_meal,
)


async def _prevent_planner_caching(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


router = APIRouter(
    prefix="/api/v1/customer",
    tags=["meal-planner"],
    dependencies=[Depends(_prevent_planner_caching)],
)


class NutritionTargets(BaseModel):
    energy_kcal: float = Field(strict=True, gt=0, allow_inf_nan=False)
    protein_g: float = Field(strict=True, gt=0, allow_inf_nan=False)
    carbohydrates_g: float = Field(strict=True, gt=0, allow_inf_nan=False)
    fat_g: float = Field(strict=True, gt=0, allow_inf_nan=False)
    fiber_g: float = Field(strict=True, gt=0, allow_inf_nan=False)


class PlanCreate(BaseModel):
    scope: str = Field(pattern="^(?i:daily|weekly)$")
    starts_on: date
    name: str | None = Field(default=None, max_length=160)
    timezone: str = Field(default="Asia/Kolkata", min_length=1, max_length=64)
    dietary_flags: list[str] = Field(default_factory=list, max_length=32)
    allergies: list[str] = Field(default_factory=list, max_length=32)
    health_goals: list[str] = Field(default_factory=list, max_length=32)
    nutrition_targets: NutritionTargets | None = None
    budget_cents: int | None = Field(default=None, ge=0)
    currency: str = Field(default="INR", min_length=3, max_length=3)


class PlannedMealUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    menu_item_id: int | None = Field(default=None, ge=1)
    desired_for: datetime | None = None
    meal_period: str | None = Field(default=None, pattern="^(?i:breakfast|lunch|snacks|dinner)$")


class PlannedMealOrderPreview(BaseModel):
    order_type: str = Field(default="DELIVERY", pattern="^(?i:delivery|pickup)$")
    delivery_address: str | None = Field(default=None, max_length=500)
    contact_phone: str | None = Field(default=None, max_length=20)


class PlannedMealOrderConfirm(BaseModel):
    confirmation_id: str = Field(min_length=8, max_length=64)
    order_type: str = Field(default="DELIVERY", pattern="^(?i:delivery|pickup)$")
    delivery_address: str | None = Field(default=None, max_length=500)


class TemplateCreate(BaseModel):
    plan_id: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=160)
    scope: str = Field(pattern="^(?i:meal|day)$")
    meal_schedule_id: int | None = Field(default=None, ge=1)


class RoutineCreate(BaseModel):
    template_id: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=160)
    recurrence_rule: str = Field(min_length=1, max_length=80)
    meal_period: str = Field(pattern="^(?i:breakfast|lunch|snacks|dinner)$")
    time_of_day: time
    timezone: str = Field(default="Asia/Kolkata", min_length=1, max_length=64)
    materialization_horizon_days: int | None = Field(default=None, ge=1, le=31)


class RoutineAction(BaseModel):
    action: str = Field(pattern="^(?i:pause|resume|skip|reschedule|edit)$")
    occurrence_date: date | None = None
    replacement_for: datetime | None = None
    recurrence_rule: str | None = Field(default=None, max_length=80)
    time_of_day: time | None = None
    reason: str | None = Field(default=None, max_length=500)


def _meal_payload(meal: MealSchedule) -> dict[str, Any]:
    return {
        "id": meal.id,
        "plan_id": meal.meal_plan_id,
        "routine_id": meal.routine_id,
        "title": meal.title,
        "meal_period": meal.meal_period,
        "state": meal.state,
        "desired_for": meal.desired_for.isoformat(),
        "timezone": meal.timezone,
        "menu_item_id": meal.menu_item_id,
        "estimated_cost_cents": meal.estimated_cost_cents,
        "currency": meal.currency,
        "diet_tags": meal.diet_tags or [],
        "allergen_tags": meal.allergen_tags or [],
        "constraints": meal.constraints_json or {},
        "recommendation": meal.selection_json or {},
        "nutrition": (meal.selection_json or {}).get("nutrition", {
            "status": "UNKNOWN",
            "source": None,
            "serving_g": None,
            "per_serving": None,
        }),
        "revision": meal.revision,
        "confirmed_at": meal.confirmed_at.isoformat() if meal.confirmed_at else None,
        "occurrence_key": meal.occurrence_key,
    }


def _plan_payload(plan: MealPlan, meals: list[MealSchedule] | None = None) -> dict[str, Any]:
    value = {
        "id": plan.id,
        "name": plan.name,
        "status": plan.status,
        "scope": plan.scope,
        "starts_on": plan.starts_on.isoformat(),
        "ends_on": plan.ends_on.isoformat(),
        "timezone": plan.timezone,
        "constraints": plan.constraints_json or {},
        "estimated_total_cents": plan.estimated_total_cents,
        "currency": plan.currency,
        "revision": plan.revision,
        "confirmation_snapshot_hash": plan.confirmation_snapshot_hash,
        "confirmed_at": plan.confirmed_at.isoformat() if plan.confirmed_at else None,
        "nutrition_targets": (plan.constraints_json or {}).get("nutrition_targets"),
    }
    if meals is not None:
        value["meals"] = [_meal_payload(meal) for meal in meals]
        value["daily_nutrition"] = _daily_nutrition_payload(plan, meals)
    return value


def _daily_nutrition_payload(plan: MealPlan, meals: list[MealSchedule]) -> list[dict[str, Any]]:
    grouped: dict[str, list[MealSchedule]] = {}
    zone = ZoneInfo(plan.timezone)
    for meal in meals:
        day = meal.desired_for.astimezone(zone).date().isoformat()
        grouped.setdefault(day, []).append(meal)

    targets = (plan.constraints_json or {}).get("nutrition_targets")
    summaries: list[dict[str, Any]] = []
    for day, day_meals in sorted(grouped.items()):
        totals = {
            "energy_kcal": 0.0,
            "protein_g": 0.0,
            "carbohydrates_g": 0.0,
            "fat_g": 0.0,
            "fiber_g": 0.0,
        }
        complete = (
            len(day_meals) == len(MEAL_PERIODS)
            and {meal.meal_period for meal in day_meals} == set(MEAL_PERIODS)
        )
        for meal in day_meals:
            nutrition = (meal.selection_json or {}).get("nutrition") or {}
            per_serving = nutrition.get("per_serving") or {}
            if nutrition.get("status") != "VERIFIED" or any(key not in per_serving for key in totals):
                complete = False
                continue
            for key in totals:
                totals[key] += float(per_serving[key])
        summary: dict[str, Any] = {
            "date": day,
            "status": "VERIFIED" if complete else "INCOMPLETE",
            "meal_count": len(day_meals),
            "totals": totals,
            "targets": targets,
        }
        if complete and targets:
            summary["difference_from_targets"] = {
                key: round(totals[key] - float(targets[key]), 2)
                for key in totals
            }
        summaries.append(summary)
    return summaries


def _template_payload(template: MealTemplate) -> dict[str, Any]:
    return {
        "id": template.id,
        "name": template.name,
        "scope": template.scope,
        "definition": template.definition_json,
        "active": template.active,
        "times_used": template.times_used,
    }


def _routine_payload(routine: MealRoutine, created: list[MealSchedule] | None = None) -> dict[str, Any]:
    value = {
        "id": routine.id,
        "template_id": routine.template_id,
        "name": routine.name,
        "state": routine.state,
        "meal_period": routine.meal_period,
        "timezone": routine.timezone,
        "recurrence_rule": routine.recurrence_rule,
        "time_of_day": routine.time_of_day.isoformat(),
        "materialization_horizon_days": routine.materialization_horizon_days,
        "materialized_through": routine.materialized_through.isoformat() if routine.materialized_through else None,
        "version": routine.version,
    }
    if created is not None:
        value["materialized_meals"] = [_meal_payload(meal) for meal in created]
    return value


async def _scope(request: Request) -> tuple[int, int, int]:
    channel = await linked_customer_channel(request)
    return channel.restaurant_id, int(channel.customer_id), channel.id


async def _owned_plan(db, plan_id: int, restaurant_id: int, customer_id: int, *, lock: bool = False) -> MealPlan:
    query = select(MealPlan).where(
        MealPlan.id == plan_id,
        MealPlan.restaurant_id == restaurant_id,
        MealPlan.customer_id == customer_id,
    )
    if lock:
        query = query.with_for_update()
    plan = (await db.execute(query)).scalar_one_or_none()
    if not plan:
        raise HTTPException(404, "Meal plan not found")
    return plan


@router.get("/meal-plans")
async def list_meal_plans(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    restaurant_id, customer_id, _ = await _scope(request)
    async with AsyncSessionLocal() as db:
        plans = (await db.execute(
            select(MealPlan)
            .where(MealPlan.restaurant_id == restaurant_id, MealPlan.customer_id == customer_id)
            .order_by(MealPlan.starts_on.desc(), MealPlan.id.desc())
            .limit(limit)
        )).scalars().all()
    return [_plan_payload(plan) for plan in plans]


@router.post("/meal-plans", status_code=201)
async def create_meal_plan(body: PlanCreate, request: Request):
    restaurant_id, customer_id, channel_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        try:
            plan, meals = await create_recommended_plan(
                db,
                restaurant_id=restaurant_id,
                customer_id=customer_id,
                scope=body.scope,
                starts_on=body.starts_on,
                name=body.name,
                timezone_name=body.timezone,
                dietary_flags=body.dietary_flags,
                allergies=body.allergies,
                health_goals=body.health_goals,
                nutrition_targets=(
                    body.nutrition_targets.model_dump()
                    if body.nutrition_targets is not None
                    else None
                ),
                budget_cents=body.budget_cents,
                currency=body.currency,
                source="WEB",
                channel_id=channel_id,
                actor_id=str(channel_id),
            )
            await db.commit()
        except MealPlanningError as exc:
            await db.rollback()
            raise HTTPException(422, str(exc)) from exc
        await db.refresh(plan)
    return _plan_payload(plan, meals)


@router.get("/menu-items")
async def list_customer_menu_items(request: Request):
    restaurant_id, _, _ = await _scope(request)
    async with AsyncSessionLocal() as db:
        items = (await db.execute(
            select(MenuItem)
            .where(
                MenuItem.restaurant_id == restaurant_id,
                MenuItem.available.is_(True),
                or_(MenuItem.stock_qty.is_(None), MenuItem.stock_qty > 0),
            )
            .order_by(MenuItem.name, MenuItem.id)
        )).scalars().all()
    return [
        {
            "id": item.id,
            "name": item.name,
            "type": item.type,
            "price_cents": (
                item.special_price_cents
                if item.special_price_cents is not None
                else item.price_cents
            ),
            "nutrition_status": item.nutrition_status,
        }
        for item in items
    ]


@router.get("/meal-plans/{plan_id}")
async def get_meal_plan(plan_id: int, request: Request):
    restaurant_id, customer_id, _ = await _scope(request)
    async with AsyncSessionLocal() as db:
        plan = await _owned_plan(db, plan_id, restaurant_id, customer_id)
        meals = (await db.execute(
            select(MealSchedule)
            .where(MealSchedule.meal_plan_id == plan.id)
            .order_by(MealSchedule.desired_for, MealSchedule.meal_period)
        )).scalars().all()
    return _plan_payload(plan, meals)


@router.patch("/meal-plans/{plan_id}/meals/{meal_id}")
async def edit_planned_meal(plan_id: int, meal_id: int, body: PlannedMealUpdate, request: Request):
    restaurant_id, customer_id, channel_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        plan = await _owned_plan(db, plan_id, restaurant_id, customer_id, lock=True)
        meal = (await db.execute(
            select(MealSchedule).where(
                MealSchedule.id == meal_id,
                MealSchedule.meal_plan_id == plan.id,
                MealSchedule.customer_id == customer_id,
            ).with_for_update()
        )).scalar_one_or_none()
        if not meal:
            raise HTTPException(404, "Planned meal not found")
        menu_item = None
        if body.menu_item_id:
            menu_item = (await db.execute(select(MenuItem).where(
                MenuItem.id == body.menu_item_id,
                MenuItem.restaurant_id == restaurant_id,
                MenuItem.available.is_(True),
                or_(MenuItem.stock_qty.is_(None), MenuItem.stock_qty > 0),
            ))).scalar_one_or_none()
            if not menu_item:
                raise HTTPException(404, "Available menu item not found")
        try:
            await update_planned_meal(
                db,
                plan=plan,
                schedule=meal,
                title=body.title,
                menu_item=menu_item,
                desired_for=body.desired_for,
                meal_period=body.meal_period,
                actor_id=str(channel_id),
            )
            await db.commit()
        except MealPlanningError as exc:
            await db.rollback()
            raise HTTPException(422, str(exc)) from exc
    return _meal_payload(meal)


@router.post("/meal-plans/{plan_id}/confirm")
async def confirm_meal_plan(plan_id: int, request: Request):
    restaurant_id, customer_id, channel_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        plan = await _owned_plan(db, plan_id, restaurant_id, customer_id, lock=True)
        try:
            _snapshot, snapshot_hash = await confirm_plan(db, plan, actor_id=str(channel_id))
            await db.commit()
        except MealPlanningError as exc:
            await db.rollback()
            raise HTTPException(409, str(exc)) from exc
    return {
        "id": plan.id,
        "status": plan.status,
        "snapshot_hash": snapshot_hash,
        "message": "Meal plan confirmed. This confirms the plan only; no food order has been placed.",
    }


@router.post("/planned-meals/{meal_id}/confirm")
async def confirm_one_planned_meal(meal_id: int, request: Request):
    restaurant_id, customer_id, channel_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        meal = (await db.execute(select(MealSchedule).where(
            MealSchedule.id == meal_id,
            MealSchedule.restaurant_id == restaurant_id,
            MealSchedule.customer_id == customer_id,
        ).with_for_update())).scalar_one_or_none()
        if not meal:
            raise HTTPException(404, "Planned meal not found")
        try:
            snapshot_hash = await confirm_planned_meal(db, meal, actor_id=str(channel_id))
            await db.commit()
        except MealPlanningError as exc:
            await db.rollback()
            raise HTTPException(409, str(exc)) from exc
    return {
        "id": meal.id,
        "status": meal.state,
        "snapshot_hash": snapshot_hash,
        "message": "Planned meal confirmed. No food order has been placed.",
    }


@router.post("/planned-meals/{meal_id}/order/preview")
async def preview_planned_meal_order(meal_id: int, body: PlannedMealOrderPreview, request: Request):
    """Return the exact order a later explicit confirmation may place. Places nothing."""
    channel = await linked_customer_channel(request)
    try:
        return await preview_planned_meal_web_order(
            channel=channel,
            meal_id=meal_id,
            order_type=body.order_type.upper(),
            delivery_address=body.delivery_address or "",
            contact_phone=body.contact_phone or "",
        )
    except PlannedMealOrderError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/planned-meals/{meal_id}/order/confirm")
async def confirm_planned_meal_order(meal_id: int, body: PlannedMealOrderConfirm, request: Request):
    """Place the reviewed planned-meal order against its unused confirmation."""
    channel = await linked_customer_channel(request)
    try:
        return await place_planned_meal_web_order(
            channel=channel,
            meal_id=meal_id,
            confirmation_id=body.confirmation_id,
            order_type=body.order_type.upper(),
            delivery_address=body.delivery_address or "",
        )
    except PlannedMealOrderError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/meal-templates")
async def list_meal_templates(request: Request):
    restaurant_id, customer_id, _ = await _scope(request)
    async with AsyncSessionLocal() as db:
        templates = (await db.execute(
            select(MealTemplate)
            .where(
                MealTemplate.restaurant_id == restaurant_id,
                MealTemplate.customer_id == customer_id,
                MealTemplate.active.is_(True),
            )
            .order_by(MealTemplate.id.desc())
        )).scalars().all()
    return [_template_payload(template) for template in templates]


@router.post("/meal-templates", status_code=201)
async def save_meal_template(body: TemplateCreate, request: Request):
    restaurant_id, customer_id, channel_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        plan = await _owned_plan(db, body.plan_id, restaurant_id, customer_id)
        schedule = None
        if body.meal_schedule_id:
            schedule = (await db.execute(select(MealSchedule).where(
                MealSchedule.id == body.meal_schedule_id,
                MealSchedule.meal_plan_id == plan.id,
            ))).scalar_one_or_none()
        try:
            template = await create_template_from_plan(
                db,
                plan=plan,
                name=body.name,
                scope=body.scope,
                schedule=schedule,
                actor_id=str(channel_id),
            )
            await db.commit()
        except MealPlanningError as exc:
            await db.rollback()
            raise HTTPException(422, str(exc)) from exc
    return _template_payload(template)


@router.get("/meal-routines")
async def list_meal_routines(request: Request):
    restaurant_id, customer_id, _ = await _scope(request)
    async with AsyncSessionLocal() as db:
        routines = (await db.execute(
            select(MealRoutine)
            .where(MealRoutine.restaurant_id == restaurant_id, MealRoutine.customer_id == customer_id)
            .order_by(MealRoutine.id.desc())
        )).scalars().all()
    return [_routine_payload(routine) for routine in routines]


@router.post("/meal-routines", status_code=201)
async def save_meal_routine(body: RoutineCreate, request: Request):
    restaurant_id, customer_id, channel_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        template = (await db.execute(select(MealTemplate).where(
            MealTemplate.id == body.template_id,
            MealTemplate.restaurant_id == restaurant_id,
            MealTemplate.customer_id == customer_id,
            MealTemplate.active.is_(True),
        ))).scalar_one_or_none()
        if not template:
            raise HTTPException(404, "Meal template not found")
        try:
            routine, created = await create_routine(
                db,
                template=template,
                name=body.name,
                recurrence_rule=body.recurrence_rule,
                meal_period=body.meal_period,
                time_of_day=body.time_of_day,
                timezone_name=body.timezone,
                horizon_days=body.materialization_horizon_days,
                actor_id=str(channel_id),
            )
            await db.commit()
        except MealPlanningError as exc:
            await db.rollback()
            raise HTTPException(422, str(exc)) from exc
    return _routine_payload(routine, created)


@router.post("/meal-routines/{routine_id}/actions")
async def apply_routine_action(routine_id: int, body: RoutineAction, request: Request):
    restaurant_id, customer_id, channel_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        routine = (await db.execute(select(MealRoutine).where(
            MealRoutine.id == routine_id,
            MealRoutine.restaurant_id == restaurant_id,
            MealRoutine.customer_id == customer_id,
        ).with_for_update())).scalar_one_or_none()
        if not routine:
            raise HTTPException(404, "Meal routine not found")
        try:
            created = await change_routine(
                db,
                routine,
                action=body.action,
                occurrence_date=body.occurrence_date,
                replacement_for=body.replacement_for,
                recurrence_rule=body.recurrence_rule,
                time_of_day=body.time_of_day,
                reason=body.reason,
                actor_id=str(channel_id),
            )
            await db.commit()
        except MealPlanningError as exc:
            await db.rollback()
            raise HTTPException(409, str(exc)) from exc
    return _routine_payload(routine, created)
