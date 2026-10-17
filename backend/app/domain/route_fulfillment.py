"""Deterministic route scoring and provider-verification policy.

Google Maps is a discovery input only.  Nothing in this module converts a Maps
result into checkout eligibility; that decision requires explicit evidence from
the selected ordering provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Iterable

from app.providers.base import CapabilityState, ProviderPolicy


class OpeningState(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class FulfillmentType(StrEnum):
    DELIVERY = "DELIVERY"
    PICKUP = "PICKUP"
    DRIVE_THROUGH = "DRIVE_THROUGH"


class RouteFulfillmentError(ValueError):
    """A user-correctable route or fulfillment validation error."""


def normalize_fulfillment_type(value: str) -> FulfillmentType:
    normalized = (value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if normalized == "DRIVE_THRU":
        normalized = "DRIVE_THROUGH"
    try:
        return FulfillmentType(normalized)
    except ValueError as exc:
        raise RouteFulfillmentError(
            "Fulfillment type must be DELIVERY, PICKUP, or DRIVE_THROUGH"
        ) from exc


def estimate_route_arrival(
    departure_at: datetime,
    travel_minutes: int | float | None,
) -> datetime | None:
    """Estimate arrival from route travel time only; this is not a food-ready estimate."""
    if departure_at.tzinfo is None:
        raise RouteFulfillmentError("Route departure time must include a timezone")
    if travel_minutes is None:
        return None
    duration = float(travel_minutes)
    if not math.isfinite(duration) or duration < 0:
        raise RouteFulfillmentError("Route travel time must be a finite non-negative number")
    return departure_at.astimezone(timezone.utc) + timedelta(minutes=duration)


def evaluate_future_opening(
    periods: Iterable[dict[str, Any]] | None,
    desired_for: datetime | None,
    utc_offset_minutes: int | None,
) -> OpeningState:
    """Evaluate Google regular-opening periods at the requested future instant.

    Google day numbers use Sunday=0.  Missing or malformed hours fail to
    UNKNOWN rather than borrowing the current ``openNow`` value.
    """
    if desired_for is None or utc_offset_minutes is None or not periods:
        return OpeningState.UNKNOWN
    if desired_for.tzinfo is None:
        raise RouteFulfillmentError("Requested fulfillment time must include a timezone")

    local = desired_for.astimezone(timezone(timedelta(minutes=int(utc_offset_minutes))))
    google_day = (local.weekday() + 1) % 7
    requested_minute = google_day * 24 * 60 + local.hour * 60 + local.minute
    week_minutes = 7 * 24 * 60

    valid_period_seen = False
    for period in periods:
        if not isinstance(period, dict):
            continue
        opened = period.get("open") or {}
        closed = period.get("close") or {}
        try:
            start = int(opened["day"]) * 1440 + int(opened.get("hour", 0)) * 60 + int(opened.get("minute", 0))
        except (KeyError, TypeError, ValueError):
            continue
        # A missing close in the Places response represents continuously open.
        if not closed:
            return OpeningState.OPEN
        try:
            end = int(closed["day"]) * 1440 + int(closed.get("hour", 0)) * 60 + int(closed.get("minute", 0))
        except (KeyError, TypeError, ValueError):
            continue
        valid_period_seen = True
        if end <= start:
            end += week_minutes
        if start <= requested_minute < end or start <= requested_minute + week_minutes < end:
            return OpeningState.OPEN
    return OpeningState.CLOSED if valid_period_seen else OpeningState.UNKNOWN


def _terms(values: Iterable[str] | None) -> set[str]:
    return {
        str(value).strip().lower().replace("_", "-")
        for value in (values or [])
        if str(value).strip()
    }


def score_restaurant_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    desired_for: datetime | None,
    required_diet_tags: Iterable[str] | None = None,
    budget_cents: int | None = None,
) -> list[dict[str, Any]]:
    """Return explainable route-aware rankings without granting eligibility."""
    if budget_cents is not None and int(budget_cents) < 0:
        raise RouteFulfillmentError("Budget cannot be negative")
    required_tags = _terms(required_diet_tags)
    ranked: list[dict[str, Any]] = []

    for source in candidates:
        candidate = dict(source)
        future = evaluate_future_opening(
            candidate.get("regular_opening_periods"),
            desired_for,
            candidate.get("utc_offset_minutes"),
        )
        operational = candidate.get("business_status") == "OPERATIONAL"
        open_now = candidate.get("open_now")
        detour_minutes = max(0.0, float(candidate.get("route_duration_minutes") or 0))
        rating = min(5.0, max(0.0, float(candidate.get("rating") or 0)))
        rating_count = max(0, int(candidate.get("rating_count") or 0))
        confidence = min(1.0, math.log1p(rating_count) / math.log1p(500))
        rating_points = round((rating / 5.0) * confidence * 25.0, 2)

        tags = _terms(candidate.get("diet_tags") or candidate.get("tags") or [])
        if required_tags and required_tags.issubset(tags):
            diet_points = 15.0
            diet_reason = "matches requested diet tags"
        elif required_tags:
            diet_points = 0.0
            diet_reason = "diet match is not verified"
        else:
            diet_points = 5.0
            diet_reason = "no diet filter requested"

        estimated_cost = candidate.get("estimated_cost_cents")
        if budget_cents is None:
            budget_points, budget_reason = 5.0, "no budget filter requested"
        elif estimated_cost is None:
            budget_points, budget_reason = 0.0, "price is not verified"
        elif int(estimated_cost) <= int(budget_cents):
            budget_points, budget_reason = 10.0, "estimated price fits budget"
        else:
            budget_points, budget_reason = -25.0, "estimated price exceeds budget"

        status_points = 20.0 if operational else -100.0
        current_points = 3.0 if open_now is True else (-3.0 if open_now is False else 0.0)
        future_points = {
            OpeningState.OPEN: 25.0,
            OpeningState.CLOSED: -100.0,
            OpeningState.UNKNOWN: -5.0,
        }[future]
        detour_points = -min(40.0, round(detour_minutes * 1.5, 2))
        score = round(
            status_points + current_points + future_points + detour_points
            + rating_points + diet_points + budget_points,
            2,
        )
        candidate.update({
            "future_open_state": future.value,
            "route_score": score,
            "score_components": {
                "business_status": status_points,
                "current_open_state": current_points,
                "future_open_state": future_points,
                "route_detour": detour_points,
                "rating_confidence": rating_points,
                "diet": diet_points,
                "budget": budget_points,
            },
            "score_reasons": [diet_reason, budget_reason],
            "maps_checkout_eligible": False,
            "requires_provider_verification": True,
        })
        ranked.append(candidate)
    return sorted(ranked, key=lambda item: (-item["route_score"], item.get("name") or ""))


def _walk_values(value: Any, key_names: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in key_names:
                found.append(nested)
            found.extend(_walk_values(nested, key_names))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_walk_values(nested, key_names))
    return found


def _provider_fulfillment_types(response: dict[str, Any]) -> set[FulfillmentType]:
    values = _walk_values(
        response,
        {
            "fulfillment_type", "fulfilment_type", "fulfillment_types", "fulfilment_types",
            "supported_fulfillment", "supported_fulfilment", "service_modes",
        },
    )
    flattened: list[Any] = []
    for value in values:
        flattened.extend(value if isinstance(value, list) else [value])
    result: set[FulfillmentType] = set()
    for value in flattened:
        try:
            result.add(normalize_fulfillment_type(str(value)))
        except RouteFulfillmentError:
            continue
    return result


def _provider_restaurant_matches(response: dict[str, Any], place_id: str, name: str) -> bool:
    identifiers = {
        str(value).strip().lower()
        for value in _walk_values(response, {"place_id", "google_place_id", "restaurant_id", "id"})
        if not isinstance(value, (dict, list)) and str(value).strip()
    }
    names = {
        str(value).strip().lower()
        for value in _walk_values(response, {"restaurant_name", "name"})
        if not isinstance(value, (dict, list)) and str(value).strip()
    }
    return bool((place_id and place_id.strip().lower() in identifiers) or (name and name.strip().lower() in names))


def _provider_future_available(response: dict[str, Any]) -> bool:
    values = _walk_values(
        response,
        {
            "available",
            "is_available",
            "available_at_requested_time",
            "scheduled_available",
            "supports_requested_time",
        },
    )
    for value in values:
        if value is True:
            return True
        if isinstance(value, str) and value.strip().upper() in {"TRUE", "YES", "AVAILABLE", "OPEN"}:
            return True
    return False


@dataclass(frozen=True)
class FulfillmentDecision:
    requested_type: str
    resolved_type: str | None
    provider_verified: bool
    checkout_eligible: bool
    manual_selection_required: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_type": self.requested_type,
            "resolved_type": self.resolved_type,
            "provider_verified": self.provider_verified,
            "checkout_eligible": self.checkout_eligible,
            "manual_selection_required": self.manual_selection_required,
            "reason": self.reason,
        }


def decide_provider_fulfillment(
    policy: ProviderPolicy,
    response: dict[str, Any],
    *,
    requested_type: str,
    place_id: str,
    restaurant_name: str,
    transactional_release_enabled: bool = False,
) -> FulfillmentDecision:
    """Require matching, verified provider evidence; safely downgrade drive-through."""
    requested = normalize_fulfillment_type(requested_type)
    supported = _provider_fulfillment_types(response)
    restaurant_matches = _provider_restaurant_matches(response, place_id, restaurant_name)
    future_available = _provider_future_available(response)
    verified_policy = policy.state == CapabilityState.VERIFIED
    caps = policy.capabilities
    capability = {
        FulfillmentType.DELIVERY: caps.delivery,
        FulfillmentType.PICKUP: caps.pickup,
        FulfillmentType.DRIVE_THROUGH: caps.curbside_drive_through,
    }[requested]

    if verified_policy and restaurant_matches and future_available and capability and requested in supported:
        checkout_eligible = transactional_release_enabled and policy.allows_transactional_mode
        return FulfillmentDecision(
            requested.value,
            requested.value,
            True,
            checkout_eligible,
            not checkout_eligible,
            (
                "Selected restaurant, future time, and fulfillment type were verified by the ordering provider."
                if checkout_eligible
                else "Fulfillment was verified, but provider checkout is not in verified transactional mode; continue in the provider experience."
            ),
        )

    if requested == FulfillmentType.DRIVE_THROUGH:
        pickup_verified = (
            verified_policy
            and restaurant_matches
            and future_available
            and caps.pickup
            and FulfillmentType.PICKUP in supported
        )
        if pickup_verified:
            checkout_eligible = transactional_release_enabled and policy.allows_transactional_mode
            return FulfillmentDecision(
                requested.value,
                FulfillmentType.PICKUP.value,
                True,
                checkout_eligible,
                not checkout_eligible,
                (
                    "Drive-through could not be verified; using ordinary scheduled pickup instead."
                    if checkout_eligible
                    else "Drive-through could not be verified. Ordinary scheduled pickup is verified, but checkout must continue in the provider experience."
                ),
            )

    return FulfillmentDecision(
        requested.value, None, False, False, True,
        "Provider verification was unavailable or inconclusive. Choose a restaurant manually and verify it in the provider experience.",
    )


def recovery_reason(
    *,
    previous_route_fingerprint: str | None,
    current_route_fingerprint: str | None,
    selected_future_open_state: str | None,
) -> str | None:
    if previous_route_fingerprint and current_route_fingerprint and previous_route_fingerprint != current_route_fingerprint:
        return "ROUTE_CHANGED"
    if (selected_future_open_state or "").upper() == OpeningState.CLOSED.value:
        return "SELECTED_RESTAURANT_CLOSED"
    return None
