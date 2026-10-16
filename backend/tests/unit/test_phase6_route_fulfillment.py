from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.api.routes.fulfillment import router as fulfillment_router
from app.db.models import RouteDiscovery, SavedLocation, SavedRoute
from app.domain.route_fulfillment import (
    FulfillmentType,
    OpeningState,
    RouteFulfillmentError,
    decide_provider_fulfillment,
    evaluate_future_opening,
    normalize_fulfillment_type,
    recovery_reason,
    score_restaurant_candidates,
)
from app.providers.base import (
    CapabilityState,
    ProviderCapabilities,
    ProviderMode,
    ProviderPolicy,
)


IST = timezone(timedelta(hours=5, minutes=30))


def provider_policy(*, transactional=False, **capabilities) -> ProviderPolicy:
    values = {"discovery": True, **capabilities}
    if transactional:
        values.update({
            "scheduled_ordering": True,
            "order_status_lookup": True,
            "stable_idempotency": True,
        })
    return ProviderPolicy(
        provider="ZOMATO",
        version_id=2,
        version=2,
        state=CapabilityState.VERIFIED,
        mode=ProviderMode.TRANSACTIONAL if transactional else ProviderMode.DISCOVERY,
        capabilities=ProviderCapabilities(**values),
        tool_schema_hash="verified-schema",
    )


def test_phase6_models_register_consent_scoped_location_route_and_history():
    assert SavedLocation.__tablename__ == "saved_locations"
    assert SavedRoute.__tablename__ == "saved_routes"
    assert RouteDiscovery.__tablename__ == "route_discoveries"
    assert SavedLocation.consent_id.nullable is False
    assert SavedRoute.consent_id.nullable is False
    assert RouteDiscovery.consent_id.nullable is False


def test_future_opening_is_evaluated_separately_from_open_now():
    periods = [{
        "open": {"day": 1, "hour": 9, "minute": 0},
        "close": {"day": 1, "hour": 17, "minute": 0},
    }]
    monday_lunch = datetime(2026, 10, 19, 13, 0, tzinfo=IST)
    monday_dinner = datetime(2026, 10, 19, 20, 0, tzinfo=IST)
    assert evaluate_future_opening(periods, monday_lunch, 330) == OpeningState.OPEN
    assert evaluate_future_opening(periods, monday_dinner, 330) == OpeningState.CLOSED
    assert evaluate_future_opening([], monday_lunch, 330) == OpeningState.UNKNOWN


def test_future_time_requires_timezone_and_never_uses_current_open_state():
    with pytest.raises(RouteFulfillmentError, match="timezone"):
        evaluate_future_opening(
            [{"open": {"day": 1}, "close": {"day": 2}}],
            datetime(2026, 10, 19, 13, 0),
            330,
        )


def test_route_score_explains_detour_rating_diet_budget_and_future_status():
    requested = datetime(2026, 10, 19, 13, 0, tzinfo=IST)
    periods = [{
        "open": {"day": 1, "hour": 9},
        "close": {"day": 1, "hour": 17},
    }]
    ranked = score_restaurant_candidates([
        {
            "place_id": "near",
            "name": "Green Bowl",
            "business_status": "OPERATIONAL",
            "open_now": False,
            "regular_opening_periods": periods,
            "utc_offset_minutes": 330,
            "route_duration_minutes": 4,
            "rating": 4.5,
            "rating_count": 500,
            "diet_tags": ["vegetarian"],
            "estimated_cost_cents": 350,
        },
        {
            "place_id": "closed",
            "name": "Currently Open But Closed Later",
            "business_status": "OPERATIONAL",
            "open_now": True,
            "regular_opening_periods": [{
                "open": {"day": 1, "hour": 8},
                "close": {"day": 1, "hour": 12},
            }],
            "utc_offset_minutes": 330,
            "route_duration_minutes": 1,
            "rating": 5,
            "rating_count": 1000,
            "diet_tags": ["vegetarian"],
            "estimated_cost_cents": 300,
        },
    ], desired_for=requested, required_diet_tags=["vegetarian"], budget_cents=400)
    assert ranked[0]["place_id"] == "near"
    assert ranked[0]["future_open_state"] == "OPEN"
    assert ranked[0]["maps_checkout_eligible"] is False
    assert ranked[0]["requires_provider_verification"] is True
    assert set(ranked[0]["score_components"]) == {
        "business_status", "current_open_state", "future_open_state",
        "route_detour", "rating_confidence", "diet", "budget",
    }


def test_drive_through_requires_explicit_provider_evidence():
    policy = provider_policy(transactional=True, pickup=True, curbside_drive_through=True)
    verified = decide_provider_fulfillment(
        policy,
        {
            "restaurant": {"google_place_id": "place-7", "name": "Highway Kitchen"},
            "fulfillment_types": ["PICKUP", "DRIVE_THROUGH"],
            "available_at_requested_time": True,
        },
        requested_type="DRIVE_THROUGH",
        place_id="place-7",
        restaurant_name="Highway Kitchen",
        transactional_release_enabled=True,
    )
    assert verified.provider_verified is True
    assert verified.checkout_eligible is True
    assert verified.resolved_type == FulfillmentType.DRIVE_THROUGH.value


def test_verified_provider_still_cannot_checkout_while_public_release_gate_is_closed():
    decision = decide_provider_fulfillment(
        provider_policy(transactional=True, delivery=True),
        {
            "restaurant": {"google_place_id": "place-7"},
            "fulfillment_types": ["DELIVERY"],
            "available_at_requested_time": True,
        },
        requested_type="DELIVERY",
        place_id="place-7",
        restaurant_name="Highway Kitchen",
    )
    assert decision.provider_verified is True
    assert decision.checkout_eligible is False
    assert decision.manual_selection_required is True


def test_unverified_drive_through_falls_back_only_to_verified_pickup():
    policy = provider_policy(transactional=True, pickup=True, curbside_drive_through=False)
    fallback = decide_provider_fulfillment(
        policy,
        {
            "restaurant": {"google_place_id": "place-7"},
            "fulfillment_types": ["PICKUP"],
            "available_at_requested_time": True,
        },
        requested_type="drive-through",
        place_id="place-7",
        restaurant_name="Highway Kitchen",
    )
    assert fallback.provider_verified is True
    assert fallback.resolved_type == FulfillmentType.PICKUP.value
    assert "ordinary scheduled pickup" in fallback.reason.lower()

    manual = decide_provider_fulfillment(
        provider_policy(pickup=False, curbside_drive_through=False),
        {"restaurant": {"google_place_id": "place-7"}},
        requested_type="DRIVE_THROUGH",
        place_id="place-7",
        restaurant_name="Highway Kitchen",
    )
    assert manual.checkout_eligible is False
    assert manual.manual_selection_required is True


def test_verified_discovery_does_not_enable_checkout_without_transactional_mode():
    decision = decide_provider_fulfillment(
        provider_policy(delivery=True),
        {
            "restaurant": {"google_place_id": "place-7"},
            "fulfillment_type": "DELIVERY",
            "available_at_requested_time": True,
        },
        requested_type="DELIVERY",
        place_id="place-7",
        restaurant_name="Highway Kitchen",
    )
    assert decision.provider_verified is True
    assert decision.checkout_eligible is False
    assert decision.manual_selection_required is True


def test_provider_restaurant_must_match_maps_selection():
    decision = decide_provider_fulfillment(
        provider_policy(delivery=True),
        {
            "restaurant": {"google_place_id": "different-place"},
            "fulfillment_types": ["DELIVERY"],
            "available_at_requested_time": True,
        },
        requested_type="DELIVERY",
        place_id="selected-place",
        restaurant_name="Selected Restaurant",
    )
    assert decision.provider_verified is False
    assert decision.manual_selection_required is True


def test_route_change_and_future_closure_trigger_recovery():
    assert recovery_reason(
        previous_route_fingerprint="old",
        current_route_fingerprint="new",
        selected_future_open_state="OPEN",
    ) == "ROUTE_CHANGED"
    assert recovery_reason(
        previous_route_fingerprint="same",
        current_route_fingerprint="same",
        selected_future_open_state="CLOSED",
    ) == "SELECTED_RESTAURANT_CLOSED"


def test_fulfillment_type_normalization_is_strict():
    assert normalize_fulfillment_type("drive-thru") == FulfillmentType.DRIVE_THROUGH
    with pytest.raises(RouteFulfillmentError):
        normalize_fulfillment_type("dine-in")


def test_phase6_routes_cover_storage_discovery_verification_and_recovery():
    paths = {route.path for route in fulfillment_router.routes}
    assert {
        "/api/v1/customer/locations",
        "/api/v1/customer/locations/{location_id}",
        "/api/v1/customer/routes",
        "/api/v1/customer/routes/{route_id}",
        "/api/v1/customer/fulfillment/discover",
        "/api/v1/customer/fulfillment/verify",
        "/api/v1/customer/fulfillment/recover",
        "/api/v1/customer/route-history",
    }.issubset(paths)


def test_phase6_migration_is_chained_consent_bound_and_reversible():
    migration = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0015_phase6_route_fulfillment.py"
    ).read_text()
    assert 'down_revision: Union[str, Sequence[str], None] = "0014"' in migration
    assert "saved_locations" in migration
    assert "saved_routes" in migration
    assert "route_discoveries" in migration
    assert "customer_consents.id" in migration
    assert "retention_expires_at" in migration
    assert "op.drop_table(\"saved_locations\")" in migration
