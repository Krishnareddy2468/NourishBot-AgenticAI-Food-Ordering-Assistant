"""Tenant-scoped operational analytics based on lifecycle timestamps."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select

from app.auth.deps import require_manager, tenant_id
from app.db.models import (
    MealPlanningEvent,
    Order,
    OrderEvent,
    ProviderAttempt,
    ScheduledJob,
    ScheduledJobAttempt,
    SpendingLedgerEntry,
)
from app.db.session import AsyncSessionLocal


router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


def safe_rate(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


def _window(date_from: date | None, date_to: date | None) -> tuple[datetime, datetime]:
    today = datetime.now(timezone.utc).date()
    start_date = date_from or (today - timedelta(days=29))
    end_date = date_to or today
    if end_date < start_date:
        raise HTTPException(422, "date_to must be on or after date_from")
    if (end_date - start_date).days > 366:
        raise HTTPException(422, "Analytics range cannot exceed 366 days")
    return (
        datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc),
    )


@router.get("/operations")
async def operations_analytics(
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    user=Depends(require_manager),
):
    rid = tenant_id(user)
    start, end = _window(date_from, date_to)
    async with AsyncSessionLocal() as db:
        scheduled_rows = (await db.execute(
            select(func.date_trunc("hour", Order.desired_for).label("bucket"), func.count(Order.id))
            .where(
                Order.restaurant_id == rid,
                Order.desired_for >= start,
                Order.desired_for < end,
            )
            .group_by("bucket")
            .order_by("bucket")
        )).all()

        lifecycle = (await db.execute(select(
            func.count(Order.id),
            func.count(Order.submitted_at),
            func.count(Order.completed_at),
            func.sum(case((Order.status.in_(["FAILED", "CANCELLED", "EXPIRED"]), 1), else_=0)),
        ).where(
            Order.restaurant_id == rid,
            Order.desired_for >= start,
            Order.desired_for < end,
        ))).one()
        scheduled_count = int(lifecycle[0] or 0)
        submitted_count = int(lifecycle[1] or 0)
        completed_count = int(lifecycle[2] or 0)
        terminal_failure_count = int(lifecycle[3] or 0)

        scheduler = (await db.execute(select(
            func.count(ScheduledJobAttempt.id),
            func.coalesce(func.avg(func.extract(
                "epoch", ScheduledJobAttempt.started_at - ScheduledJob.run_at
            )), 0),
            func.count(func.distinct(case((ScheduledJob.state == "DEAD_LETTER", ScheduledJob.id)))),
            func.sum(case((ScheduledJobAttempt.started_at > ScheduledJob.deadline_at, 1), else_=0)),
        ).select_from(ScheduledJobAttempt).join(
            ScheduledJob, ScheduledJob.id == ScheduledJobAttempt.scheduled_job_id
        ).where(
            ScheduledJobAttempt.restaurant_id == rid,
            ScheduledJobAttempt.started_at >= start,
            ScheduledJobAttempt.started_at < end,
        ))).one()

        provider_rows = (await db.execute(
            select(ProviderAttempt.status, func.count(ProviderAttempt.id))
            .where(
                ProviderAttempt.restaurant_id == rid,
                func.coalesce(ProviderAttempt.submitted_at, ProviderAttempt.created_at) >= start,
                func.coalesce(ProviderAttempt.submitted_at, ProviderAttempt.created_at) < end,
            )
            .group_by(ProviderAttempt.status)
        )).all()
        provider_counts = {str(status): int(count) for status, count in provider_rows}
        provider_total = sum(provider_counts.values())

        revenue_rows = (await db.execute(
            select(
                func.date_trunc("day", SpendingLedgerEntry.occurred_at).label("bucket"),
                func.sum(SpendingLedgerEntry.amount_cents),
            )
            .where(
                SpendingLedgerEntry.restaurant_id == rid,
                SpendingLedgerEntry.occurred_at >= start,
                SpendingLedgerEntry.occurred_at < end,
            )
            .group_by("bucket")
            .order_by("bucket")
        )).all()

        fulfillment_rows = (await db.execute(select(
            Order.order_type,
            func.count(Order.id),
            func.count(Order.completed_at),
            func.sum(case((Order.status.in_(["FAILED", "CANCELLED", "EXPIRED"]), 1), else_=0)),
            func.coalesce(func.avg(func.extract("epoch", Order.completed_at - Order.submitted_at)), 0),
        ).where(
            Order.restaurant_id == rid,
            Order.desired_for >= start,
            Order.desired_for < end,
        ).group_by(Order.order_type))).all()

        cancellation_rows = (await db.execute(select(
            func.coalesce(OrderEvent.reason_code, "UNSPECIFIED"),
            func.count(OrderEvent.id),
        ).where(
            OrderEvent.restaurant_id == rid,
            OrderEvent.to_state == "CANCELLED",
            OrderEvent.occurred_at >= start,
            OrderEvent.occurred_at < end,
        ).group_by(func.coalesce(OrderEvent.reason_code, "UNSPECIFIED")))).all()

        reason_expr = func.coalesce(MealPlanningEvent.metadata_json["reason"].astext, "UNSPECIFIED")
        reschedule_rows = (await db.execute(select(
            reason_expr,
            func.count(MealPlanningEvent.id),
        ).where(
            MealPlanningEvent.restaurant_id == rid,
            MealPlanningEvent.event_type == "ROUTINE_RESCHEDULE",
            MealPlanningEvent.occurred_at >= start,
            MealPlanningEvent.occurred_at < end,
        ).group_by(reason_expr))).all()

    return {
        "range": {"from": start.date().isoformat(), "to": (end.date() - timedelta(days=1)).isoformat()},
        "scheduled_volume": [{
            "desired_hour": bucket.isoformat(), "count": int(count),
        } for bucket, count in scheduled_rows],
        "lifecycle": {
            "scheduled": scheduled_count,
            "submitted": submitted_count,
            "completed": completed_count,
            "terminal_failures": terminal_failure_count,
            "submission_success_rate": safe_rate(submitted_count, scheduled_count),
            "fulfillment_success_rate": safe_rate(completed_count, submitted_count),
        },
        "scheduler": {
            "attempts": int(scheduler[0] or 0),
            "average_lag_seconds": round(float(scheduler[1] or 0), 2),
            "dead_letter_count": int(scheduler[2] or 0),
            "missed_window_count": int(scheduler[3] or 0),
        },
        "providers": {
            "by_status": provider_counts,
            "failure_count": provider_counts.get("FAILED", 0),
            "unknown_count": provider_counts.get("UNKNOWN", 0),
            "failure_rate": safe_rate(provider_counts.get("FAILED", 0), provider_total),
            "unknown_rate": safe_rate(provider_counts.get("UNKNOWN", 0), provider_total),
        },
        "completed_revenue": [{
            "fulfillment_date": bucket.date().isoformat(),
            "amount_cents": int(amount or 0),
            "currency": "INR",
        } for bucket, amount in revenue_rows],
        "fulfillment_performance": [{
            "type": order_type,
            "scheduled": int(total or 0),
            "completed": int(completed or 0),
            "failed_or_cancelled": int(failed or 0),
            "success_rate": safe_rate(int(completed or 0), int(total or 0)),
            "average_seconds_from_submission": round(float(avg_seconds or 0), 2),
        } for order_type, total, completed, failed, avg_seconds in fulfillment_rows],
        "reasons": {
            "cancellations": [{"reason": reason, "count": int(count)} for reason, count in cancellation_rows],
            "reschedules": [{"reason": reason, "count": int(count)} for reason, count in reschedule_rows],
        },
        "revenue_source": "IMMUTABLE_SPENDING_LEDGER",
        "timestamp_contract": {
            "scheduled_volume": "orders.desired_for",
            "submission": "orders.submitted_at",
            "fulfillment": "orders.completed_at",
            "provider": "provider_attempts.submitted_at/completed_at",
            "revenue": "spending_ledger_entries.occurred_at",
        },
    }
