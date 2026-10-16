"""Authenticated customer dashboard, preferences, budgets, and order controls."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.api.routes.customer_privacy import linked_customer_channel
from app.db.models import (
    CustomerConsent,
    MealBudget,
    MealPlan,
    MealRoutine,
    MealSchedule,
    MealTemplate,
    Order,
    ProviderAttempt,
    SpendingLedgerEntry,
    UserPreferences,
)
from app.db.session import AsyncSessionLocal
from app.domain.order_state import InvalidStateTransition, sync_terminal_schedule, transition_order
from app.providers.registry import current_policy, public_policy
from app.providers.service import cancel_attempt


router = APIRouter(prefix="/api/v1/customer", tags=["customer-dashboard"])
PROVIDERS = ("NOURISHBOT", "SWIGGY", "ZOMATO")


class PreferencesUpdate(BaseModel):
    dietary_flags: list[str] = Field(default_factory=list, max_length=30)
    allergies: list[str] = Field(default_factory=list, max_length=30)
    health_goals: list[str] = Field(default_factory=list, max_length=30)
    price_band: str = Field(default="mid", pattern="^(budget|mid|premium)$")


class BudgetUpdate(BaseModel):
    period: str = Field(default="WEEKLY", pattern="^(?i:weekly|monthly)$")
    limit_cents: int = Field(gt=0, le=100_000_000)
    currency: str = Field(default="INR", min_length=3, max_length=3)
    starts_on: date
    alert_thresholds: list[int] = Field(default_factory=lambda: [50, 80, 100], max_length=10)


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


async def _scope(request: Request) -> tuple[int, int, int]:
    channel = await linked_customer_channel(request)
    return int(channel.restaurant_id), int(channel.customer_id), int(channel.id)


def _terms(values: list[str]) -> list[str]:
    return sorted({
        "-".join(str(value).strip().lower().split())[:64]
        for value in values
        if str(value).strip()
    })


def _schedule_payload(meal: MealSchedule, zone: ZoneInfo) -> dict[str, Any]:
    desired = meal.desired_for.astimezone(zone)
    return {
        "id": meal.id,
        "plan_id": meal.meal_plan_id,
        "routine_id": meal.routine_id,
        "order_id": meal.order_id,
        "title": meal.title,
        "meal_period": meal.meal_period,
        "state": meal.state,
        "desired_for": meal.desired_for.isoformat(),
        "local_date": desired.date().isoformat(),
        "local_time": desired.strftime("%H:%M"),
        "estimated_cost_cents": meal.estimated_cost_cents,
        "currency": meal.currency,
        "can_confirm": meal.state == "DRAFT",
        "can_skip": bool(meal.routine_id and meal.state in {"DRAFT", "PAUSED"}),
        "can_reschedule": bool(meal.routine_id and meal.state in {"DRAFT", "PAUSED"}),
    }


def _order_payload(order: Order) -> dict[str, Any]:
    return {
        "id": order.id,
        "order_ref": order.order_ref,
        "provider": order.provider,
        "order_type": order.order_type,
        "status": order.status,
        "schedule_state": order.schedule_state,
        "provider_state": order.provider_state,
        "fulfillment_state": order.fulfillment_state,
        "desired_for": order.desired_for.isoformat() if order.desired_for else None,
        "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
        "completed_at": order.completed_at.isoformat() if order.completed_at else None,
        "total_cents": order.total_cents,
        "currency": "INR",
        "terminal_reason": order.terminal_reason,
        "can_cancel": order.status in {"CREATED", "CONFIRMED", "ACCEPTED", "PREPARING", "READY"},
    }


@router.get("/dashboard")
async def customer_dashboard(
    request: Request,
    response: Response,
    week_start: date | None = Query(default=None),
):
    _no_store(response)
    restaurant_id, customer_id, _channel_id = await _scope(request)
    zone = ZoneInfo("Asia/Kolkata")
    today = datetime.now(zone).date()
    start = week_start or (today - timedelta(days=today.weekday()))
    end = start + timedelta(days=6)
    start_at = datetime.combine(start, datetime.min.time(), tzinfo=zone).astimezone(timezone.utc)
    end_at = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=zone).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        meals = (await db.execute(select(MealSchedule).where(
            MealSchedule.restaurant_id == restaurant_id,
            MealSchedule.customer_id == customer_id,
            MealSchedule.desired_for >= start_at,
            MealSchedule.desired_for < end_at,
            MealSchedule.state.notin_(["SUPERSEDED", "CANCELLED"]),
        ).order_by(MealSchedule.desired_for))).scalars().all()
        upcoming = (await db.execute(select(MealSchedule).where(
            MealSchedule.restaurant_id == restaurant_id,
            MealSchedule.customer_id == customer_id,
            MealSchedule.desired_for >= now,
            MealSchedule.state.in_(["CONFIRMED", "SCHEDULED", "ACTIVATED"]),
        ).order_by(MealSchedule.desired_for).limit(12))).scalars().all()
        plans = (await db.execute(select(MealPlan).where(
            MealPlan.restaurant_id == restaurant_id,
            MealPlan.customer_id == customer_id,
        ).order_by(MealPlan.starts_on.desc()).limit(20))).scalars().all()
        templates = (await db.execute(select(MealTemplate).where(
            MealTemplate.restaurant_id == restaurant_id,
            MealTemplate.customer_id == customer_id,
            MealTemplate.active.is_(True),
        ).order_by(MealTemplate.id.desc()).limit(20))).scalars().all()
        routines = (await db.execute(select(MealRoutine).where(
            MealRoutine.restaurant_id == restaurant_id,
            MealRoutine.customer_id == customer_id,
        ).order_by(MealRoutine.id.desc()).limit(20))).scalars().all()
        orders = (await db.execute(select(Order).where(
            Order.restaurant_id == restaurant_id,
            Order.customer_id == customer_id,
        ).order_by(Order.created_at.desc()).limit(50))).scalars().all()
        order_ids = [order.id for order in orders]
        attempts = []
        if order_ids:
            attempts = (await db.execute(select(ProviderAttempt).where(
                ProviderAttempt.restaurant_id == restaurant_id,
                ProviderAttempt.order_id.in_(order_ids),
            ).order_by(ProviderAttempt.created_at.desc()).limit(50))).scalars().all()
        preferences = (await db.execute(select(UserPreferences).where(
            UserPreferences.customer_id == customer_id,
        ))).scalar_one_or_none()
        consent_rows = (await db.execute(select(CustomerConsent).where(
            CustomerConsent.restaurant_id == restaurant_id,
            CustomerConsent.customer_id == customer_id,
        ).order_by(CustomerConsent.id.desc()))).scalars().all()
        consents: dict[str, bool] = {}
        for consent in consent_rows:
            consents.setdefault(consent.purpose, bool(consent.granted))
        budgets = (await db.execute(select(MealBudget).where(
            MealBudget.restaurant_id == restaurant_id,
            MealBudget.customer_id == customer_id,
            MealBudget.active.is_(True),
        ).order_by(MealBudget.starts_on.desc()).limit(12))).scalars().all()
        ledger_total = int((await db.execute(select(func.coalesce(func.sum(SpendingLedgerEntry.amount_cents), 0)).where(
            SpendingLedgerEntry.restaurant_id == restaurant_id,
            SpendingLedgerEntry.customer_id == customer_id,
        ))).scalar_one())
        week_spend = int((await db.execute(select(func.coalesce(func.sum(SpendingLedgerEntry.amount_cents), 0)).where(
            SpendingLedgerEntry.restaurant_id == restaurant_id,
            SpendingLedgerEntry.customer_id == customer_id,
            SpendingLedgerEntry.occurred_at >= start_at,
            SpendingLedgerEntry.occurred_at < end_at,
        ))).scalar_one())
        policies = [public_policy(await current_policy(db, provider)) for provider in PROVIDERS]

    return {
        "week": {
            "starts_on": start.isoformat(),
            "ends_on": end.isoformat(),
            "days": [{
                "date": (start + timedelta(days=offset)).isoformat(),
                "meals": [
                    _schedule_payload(meal, zone)
                    for meal in meals
                    if meal.desired_for.astimezone(zone).date() == start + timedelta(days=offset)
                ],
            } for offset in range(7)],
        },
        "upcoming_confirmed": [_schedule_payload(meal, zone) for meal in upcoming],
        "draft_plans": [{
            "id": plan.id,
            "name": plan.name,
            "scope": plan.scope,
            "status": plan.status,
            "starts_on": plan.starts_on.isoformat(),
            "ends_on": plan.ends_on.isoformat(),
            "estimated_total_cents": plan.estimated_total_cents,
            "currency": plan.currency,
            "constraints": plan.constraints_json,
        } for plan in plans if plan.status == "DRAFT"],
        "templates": [{
            "id": item.id, "name": item.name, "scope": item.scope,
            "times_used": item.times_used,
        } for item in templates],
        "routines": [{
            "id": item.id, "name": item.name, "state": item.state,
            "meal_period": item.meal_period, "recurrence_rule": item.recurrence_rule,
            "time_of_day": item.time_of_day.isoformat(), "timezone": item.timezone,
        } for item in routines],
        "orders": [_order_payload(order) for order in orders],
        "provider_attempts": [{
            "id": item.id,
            "order_id": item.order_id,
            "provider": item.provider,
            "mode": item.mode,
            "operation": item.operation,
            "status": item.status,
            "submitted_at": item.submitted_at.isoformat() if item.submitted_at else None,
            "completed_at": item.completed_at.isoformat() if item.completed_at else None,
            "next_action": item.next_action,
        } for item in attempts],
        "preferences": {
            "dietary_flags": list(preferences.dietary_flags or []) if preferences else [],
            "allergies": list(preferences.allergies or []) if preferences else [],
            "health_goals": list(preferences.health_goals or []) if preferences else [],
            "price_band": preferences.price_band if preferences else "mid",
        },
        "consents": {purpose: consents.get(purpose, False) for purpose in ("PERSONALIZATION", "LOCATION", "REMINDERS", "ANALYTICS")},
        "budgets": [{
            "id": item.id, "period": item.period, "limit_cents": item.limit_cents,
            "currency": item.currency, "starts_on": item.starts_on.isoformat(),
            "ends_on": item.ends_on.isoformat(), "alert_thresholds": item.alert_thresholds_json,
        } for item in budgets],
        "spending": {
            "week_cents": week_spend,
            "lifetime_cents": ledger_total,
            "currency": "INR",
            "source": "IMMUTABLE_SPENDING_LEDGER",
        },
        "provider_capabilities": policies,
    }


@router.patch("/dashboard/preferences")
async def update_dashboard_preferences(body: PreferencesUpdate, request: Request, response: Response):
    _no_store(response)
    _restaurant_id, customer_id, _channel_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        value = (await db.execute(select(UserPreferences).where(
            UserPreferences.customer_id == customer_id,
        ).with_for_update())).scalar_one_or_none()
        if value is None:
            value = UserPreferences(customer_id=customer_id)
            db.add(value)
        value.dietary_flags = _terms(body.dietary_flags)
        value.allergies = _terms(body.allergies)
        value.health_goals = _terms(body.health_goals)
        value.price_band = body.price_band
        value.health_onboarding_done = 2
        await db.commit()
    return {
        "dietary_flags": value.dietary_flags,
        "allergies": value.allergies,
        "health_goals": value.health_goals,
        "price_band": value.price_band,
    }


@router.put("/dashboard/budget")
async def update_dashboard_budget(body: BudgetUpdate, request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id, _channel_id = await _scope(request)
    period = body.period.upper()
    if period == "WEEKLY":
        ends_on = body.starts_on + timedelta(days=6)
    else:
        next_month = (body.starts_on.replace(day=28) + timedelta(days=4)).replace(day=1)
        ends_on = next_month - timedelta(days=1)
    thresholds = sorted({int(value) for value in body.alert_thresholds if 1 <= int(value) <= 100})
    async with AsyncSessionLocal() as db:
        value = (await db.execute(select(MealBudget).where(
            MealBudget.restaurant_id == restaurant_id,
            MealBudget.customer_id == customer_id,
            MealBudget.period == period,
            MealBudget.starts_on == body.starts_on,
        ).with_for_update())).scalar_one_or_none()
        if value is None:
            value = MealBudget(
                restaurant_id=restaurant_id,
                customer_id=customer_id,
                period=period,
                starts_on=body.starts_on,
            )
            db.add(value)
        value.ends_on = ends_on
        value.limit_cents = body.limit_cents
        value.currency = body.currency.upper()
        value.alert_thresholds_json = thresholds
        value.active = True
        await db.commit()
        await db.refresh(value)
    return {
        "id": value.id, "period": value.period, "limit_cents": value.limit_cents,
        "currency": value.currency, "starts_on": value.starts_on.isoformat(),
        "ends_on": value.ends_on.isoformat(), "alert_thresholds": value.alert_thresholds_json,
    }


@router.post("/orders/{order_id}/cancel")
async def cancel_customer_order(order_id: int, request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id, channel_id = await _scope(request)
    attempt_id = None
    async with AsyncSessionLocal() as db:
        order = (await db.execute(select(Order).where(
            Order.id == order_id,
            Order.restaurant_id == restaurant_id,
            Order.customer_id == customer_id,
        ))).scalar_one_or_none()
        if not order:
            raise HTTPException(404, "Order not found")
        if order.status not in {"CREATED", "CONFIRMED", "ACCEPTED", "PREPARING", "READY"}:
            raise HTTPException(409, f"Order cannot be cancelled from {order.status}")
        if order.provider != "NOURISHBOT" and order.provider_state in {"CONFIRMED", "UNKNOWN"}:
            attempt_id = (await db.execute(select(ProviderAttempt.id).where(
                ProviderAttempt.restaurant_id == restaurant_id,
                ProviderAttempt.order_id == order.id,
            ).order_by(ProviderAttempt.created_at.desc()).limit(1))).scalar_one_or_none()
            if not attempt_id:
                raise HTTPException(409, "Provider cancellation needs manual review")

    if attempt_id:
        try:
            result = await cancel_attempt(attempt_id)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        if result.outcome != "CANCELLED":
            raise HTTPException(409, "Provider cancellation is not confirmed; manual review is required")

    async with AsyncSessionLocal() as db:
        order = (await db.execute(select(Order).where(
            Order.id == order_id,
            Order.restaurant_id == restaurant_id,
            Order.customer_id == customer_id,
        ).with_for_update())).scalar_one()
        try:
            transition_order(
                db, order, "CANCELLED",
                actor_type="CUSTOMER", actor_id=channel_id,
                source="CUSTOMER_DASHBOARD", reason_code="CUSTOMER_CANCELLED",
            )
            schedule = (await db.execute(select(MealSchedule).where(
                MealSchedule.order_id == order.id,
                MealSchedule.customer_id == customer_id,
            ))).scalar_one_or_none()
            sync_terminal_schedule(
                db, order, meal_schedule=schedule,
                actor_type="CUSTOMER", actor_id=channel_id,
                source="CUSTOMER_DASHBOARD", reason_code="CUSTOMER_CANCELLED",
            )
            await db.commit()
        except InvalidStateTransition as exc:
            await db.rollback()
            raise HTTPException(409, str(exc)) from exc
    return {"id": order.id, "order_ref": order.order_ref, "status": order.status}
