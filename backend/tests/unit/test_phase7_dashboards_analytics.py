from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.routes.analytics import router as analytics_router, safe_rate
from app.api.routes.dashboard import router as dashboard_router
from app.db.models import Order, SpendingLedgerEntry, User
from app.db.models.analytics import _reject_spending_mutation
from app.domain.order_state import transition_order


class FakeSession:
    def __init__(self):
        self.added = []

    def add(self, value):
        self.added.append(value)


def make_ready_order():
    return SimpleNamespace(
        id=17,
        restaurant_id=3,
        customer_id=9,
        total_cents=725,
        status="READY",
        fulfillment_state="READY",
        submitted_at=datetime(2026, 10, 16, 6, 0, tzinfo=timezone.utc),
        completed_at=None,
        cancelled_at=None,
        failed_at=None,
        expired_at=None,
        terminal_reason=None,
    )


def test_phase7_models_register_customer_link_lifecycle_timestamps_and_ledger():
    assert hasattr(User, "customer_id")
    assert hasattr(Order, "submitted_at")
    assert hasattr(Order, "completed_at")
    assert SpendingLedgerEntry.__tablename__ == "spending_ledger_entries"
    assert SpendingLedgerEntry.amount_cents.nullable is False


def test_completed_order_records_timestamp_and_one_immutable_spending_entry():
    session = FakeSession()
    order = make_ready_order()
    completed_at = datetime(2026, 10, 16, 7, 0, tzinfo=timezone.utc)

    transition_order(
        session,
        order,
        "DELIVERED",
        actor_type="STAFF",
        actor_id=4,
        source="TEST",
        occurred_at=completed_at,
    )

    ledger_entries = [item for item in session.added if isinstance(item, SpendingLedgerEntry)]
    assert order.completed_at == completed_at
    assert len(ledger_entries) == 1
    assert ledger_entries[0].amount_cents == 725
    assert ledger_entries[0].restaurant_id == 3
    assert ledger_entries[0].customer_id == 9
    assert ledger_entries[0].order_id == 17

    second_session = FakeSession()
    transition_order(
        second_session,
        order,
        "COMPLETED",
        actor_type="SYSTEM",
        source="TEST",
    )
    assert not any(isinstance(item, SpendingLedgerEntry) for item in second_session.added)


def test_spending_ledger_rejects_updates_and_deletes_in_application_code():
    entry = SpendingLedgerEntry()
    with pytest.raises(ValueError, match="append-only"):
        _reject_spending_mutation(None, None, entry)


def test_analytics_rates_are_stable_for_empty_and_populated_ranges():
    assert safe_rate(0, 0) == 0.0
    assert safe_rate(8, 10) == 80.0
    assert safe_rate(1, 3) == 33.33


def test_phase7_routes_cover_customer_controls_and_operational_analytics():
    customer_paths = {route.path for route in dashboard_router.routes}
    staff_paths = {route.path for route in analytics_router.routes}
    assert {
        "/api/v1/customer/dashboard",
        "/api/v1/customer/dashboard/preferences",
        "/api/v1/customer/dashboard/budget",
        "/api/v1/customer/orders/{order_id}/cancel",
    }.issubset(customer_paths)
    assert "/api/v1/analytics/operations" in staff_paths


def test_dashboard_and_analytics_queries_enforce_customer_and_tenant_scope():
    root = Path(__file__).resolve().parents[2] / "app" / "api" / "routes"
    dashboard = (root / "dashboard.py").read_text()
    analytics = (root / "analytics.py").read_text()
    assert "MealSchedule.restaurant_id == restaurant_id" in dashboard
    assert "MealSchedule.customer_id == customer_id" in dashboard
    assert "Order.restaurant_id == restaurant_id" in dashboard
    assert "Order.customer_id == customer_id" in dashboard
    assert "rid = tenant_id(user)" in analytics
    assert "Order.restaurant_id == rid" in analytics
    assert "SpendingLedgerEntry.restaurant_id == rid" in analytics


def test_authenticated_account_identity_precedes_a_stale_browser_link():
    source = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "api"
        / "routes"
        / "customer_privacy.py"
    ).read_text()
    function = source[source.index("async def linked_customer_channel"):source.index("@router.post(\"/identity/link\")")]
    assert function.index("AUTH_COOKIE_NAME") < function.index("CUSTOMER_IDENTITY_COOKIE_NAME")


def test_phase7_migration_is_chained_backfilled_append_only_and_reversible():
    migration = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0016_phase7_dashboards_analytics.py"
    ).read_text()
    assert 'down_revision: Union[str, Sequence[str], None] = "0015"' in migration
    assert "submitted_at" in migration
    assert "completed_at" in migration
    assert "PHASE7_BACKFILL" in migration
    assert "trg_spending_ledger_append_only" in migration
    assert "ck_customer_user_has_profile" in migration
    assert 'op.drop_table("spending_ledger_entries")' in migration
