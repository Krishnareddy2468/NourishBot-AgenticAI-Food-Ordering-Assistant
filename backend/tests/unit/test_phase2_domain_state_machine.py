from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.entrypoint import (
    CANONICAL_AGENT_ENTRYPOINT,
    QUARANTINED_AGENT_ENTRYPOINTS,
    reject_legacy_agent,
)
from app.db.models import (
    MealBudget,
    MealPlan,
    MealRoutine,
    MealSchedule,
    MealTemplate,
    OrderEvent,
    ScheduleEvent,
)
from app.domain.order_state import (
    InvalidStateTransition,
    sync_terminal_schedule,
    transition_order,
    transition_provider,
    transition_schedule,
)


class FakeSession:
    def __init__(self):
        self.added = []

    def add(self, value):
        self.added.append(value)


def make_order(**overrides):
    values = {
        "id": 17,
        "restaurant_id": 3,
        "status": "CONFIRMED",
        "schedule_state": "SCHEDULED",
        "provider_state": "PENDING",
        "fulfillment_state": "NOT_STARTED",
        "cancelled_at": None,
        "failed_at": None,
        "expired_at": None,
        "terminal_reason": None,
        "activated_at": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_phase2_domain_tables_are_registered():
    assert {
        MealPlan.__tablename__,
        MealTemplate.__tablename__,
        MealRoutine.__tablename__,
        MealBudget.__tablename__,
        MealSchedule.__tablename__,
        OrderEvent.__tablename__,
        ScheduleEvent.__tablename__,
    } == {
        "meal_plans",
        "meal_templates",
        "meal_routines",
        "meal_budgets",
        "meal_schedules",
        "order_events",
        "schedule_events",
    }


def test_valid_order_transition_updates_state_and_appends_audit_event():
    session = FakeSession()
    order = make_order()

    event = transition_order(
        session,
        order,
        "ACCEPTED",
        actor_type="SYSTEM",
        source="TEST",
        reason_code="SCHEDULE_ACTIVATED",
    )

    assert order.status == "ACCEPTED"
    assert order.fulfillment_state == "ACCEPTED"
    assert event.from_state == "CONFIRMED"
    assert event.to_state == "ACCEPTED"
    assert session.added == [event]


def test_invalid_order_transition_fails_without_audit_event():
    session = FakeSession()
    order = make_order(status="DELIVERED", fulfillment_state="DELIVERED")

    with pytest.raises(InvalidStateTransition, match="DELIVERED to PREPARING"):
        transition_order(
            session,
            order,
            "PREPARING",
            actor_type="STAFF",
            source="TEST",
        )

    assert order.status == "DELIVERED"
    assert session.added == []


def test_delivery_failure_is_not_recorded_as_cancellation():
    session = FakeSession()
    order = make_order(
        status="OUT_FOR_DELIVERY",
        schedule_state="ACTIVATED",
        fulfillment_state="OUT_FOR_DELIVERY",
    )
    schedule = SimpleNamespace(
        id=9,
        restaurant_id=order.restaurant_id,
        order_id=order.id,
        state="ACTIVATED",
    )

    transition_order(
        session,
        order,
        "FAILED",
        actor_type="DRIVER",
        source="TEST",
        reason_code="DELIVERY_FAILED",
    )
    sync_terminal_schedule(
        session,
        order,
        meal_schedule=schedule,
        actor_type="DRIVER",
        source="TEST",
        reason_code="DELIVERY_FAILED",
    )

    assert order.status == "FAILED"
    assert order.fulfillment_state == "FAILED"
    assert order.terminal_reason == "DELIVERY_FAILED"
    assert order.cancelled_at is None
    assert schedule.state == "FAILED"


def test_schedule_and_provider_states_transition_independently():
    session = FakeSession()
    order = make_order()

    transition_schedule(
        session,
        order,
        "ACTIVATED",
        actor_type="SYSTEM",
        source="TEST",
    )
    transition_provider(
        session,
        order,
        "SUBMITTING",
        actor_type="SYSTEM",
        source="TEST",
    )

    assert order.status == "CONFIRMED"
    assert order.schedule_state == "ACTIVATED"
    assert order.provider_state == "SUBMITTING"


def test_provider_rejection_has_a_distinct_terminal_reason():
    session = FakeSession()
    order = make_order(provider_state="SUBMITTING")

    transition_provider(
        session,
        order,
        "REJECTED",
        actor_type="PROVIDER",
        source="TEST",
        reason_code="PROVIDER_REJECTED",
    )

    assert order.provider_state == "REJECTED"
    assert order.status == "FAILED"
    assert order.terminal_reason == "PROVIDER_REJECTED"
    assert [event.event_type for event in session.added] == [
        "PROVIDER_STATE_CHANGED",
        "ORDER_STATE_CHANGED",
    ]


def test_phase2_migration_follows_auth_and_guards_event_history():
    migration = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0011_phase2_domain_state_machine.py"
    ).read_text()

    assert 'down_revision: Union[str, Sequence[str], None] = "0010"' in migration
    assert "trg_order_events_append_only" in migration
    assert "trg_schedule_events_append_only" in migration
    assert "LEGACY_STATUS_MIGRATION" in migration


def test_only_pipeline_is_a_supported_agent_entrypoint():
    assert CANONICAL_AGENT_ENTRYPOINT == "app.agent.pipeline.process_message"
    assert QUARANTINED_AGENT_ENTRYPOINTS == {
        "app.agent.nourishbot_agent.get_nourishbot_response",
        "app.agent.orchestrator.get_bot_response",
    }
    with pytest.raises(RuntimeError, match="quarantined"):
        reject_legacy_agent("app.agent.orchestrator.get_bot_response")
