"""Google Maps discovery tools used by both NourishBot agent paths.

Maps is deliberately read-only in this product. It discovers restaurants and
route-side drive-through candidates; Swiggy, Zomato, or the in-house ordering
pipeline remains responsible for menu verification and scheduled checkout.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import settings
from app.domain.route_fulfillment import estimate_route_arrival


PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
ROUTES_COMPUTE_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
PLACE_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,"
    "places.rating,places.userRatingCount,places.priceLevel,places.businessStatus,"
    "places.currentOpeningHours,places.regularOpeningHours,places.utcOffsetMinutes,"
    "places.types,places.googleMapsUri"
)


class MapsServiceError(RuntimeError):
    """Safe, user-facing Maps configuration or request failure."""


def _require_api_key() -> str:
    key = settings.GOOGLE_MAPS_API_KEY.strip()
    if not key:
        raise MapsServiceError(
            "Route-aware restaurant search is not configured yet. Set GOOGLE_MAPS_API_KEY."
        )
    return key


async def _post_json(url: str, payload: dict, field_mask: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _require_api_key(),
        "X-Goog-FieldMask": field_mask,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.MAPS_REQUEST_TIMEOUT_S) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise MapsServiceError("Maps search timed out. Please try again.") from exc
    except httpx.HTTPStatusError as exc:
        raise MapsServiceError(
            f"Maps search failed with status {exc.response.status_code}."
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise MapsServiceError("Maps search is temporarily unavailable.") from exc
    return data if isinstance(data, dict) else {}


def _duration_minutes(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text.endswith("s"):
        return None
    try:
        return max(0, round(float(text[:-1]) / 60))
    except ValueError:
        return None


def _compact_place(place: dict, routing_summary: dict | None = None) -> dict:
    display_name = place.get("displayName") or {}
    opening = place.get("currentOpeningHours") or {}
    location = place.get("location") or {}
    route = routing_summary or {}
    legs = route.get("legs") or []
    leg = legs[0] if legs and isinstance(legs[0], dict) else {}
    return {
        "place_id": str(place.get("id") or ""),
        "name": str(display_name.get("text") or ""),
        "address": str(place.get("formattedAddress") or ""),
        "rating": place.get("rating"),
        "rating_count": place.get("userRatingCount"),
        "price_level": place.get("priceLevel"),
        "business_status": place.get("businessStatus"),
        "open_now": opening.get("openNow"),
        "regular_opening_periods": (place.get("regularOpeningHours") or {}).get("periods") or [],
        "utc_offset_minutes": place.get("utcOffsetMinutes"),
        "place_types": list(place.get("types") or []),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "maps_url": str(place.get("googleMapsUri") or ""),
        "route_distance_meters": leg.get("distanceMeters"),
        "route_duration_minutes": _duration_minutes(leg.get("duration")),
    }


def _rank_places(places: list[dict]) -> list[dict]:
    return sorted(
        places,
        key=lambda item: (
            item.get("business_status") != "OPERATIONAL",
            item.get("open_now") is False,
            -(float(item.get("rating") or 0)),
            -(int(item.get("rating_count") or 0)),
        ),
    )


async def search_best_restaurants(
    location: str,
    query: str = "restaurants",
    max_results: int = 5,
) -> dict:
    """Find highly rated restaurants around a named place or coordinates."""
    location = (location or "").strip()
    query = (query or "restaurants").strip()
    if len(location) < 3:
        raise MapsServiceError("Share an area, address, or latitude/longitude to search Maps.")

    count = max(1, min(int(max_results or 5), 10))
    payload = {
        "textQuery": f"{query} near {location}",
        "pageSize": count,
        "rankPreference": "RELEVANCE",
    }
    data = await _post_json(PLACES_TEXT_SEARCH_URL, payload, PLACE_FIELD_MASK)
    results = _rank_places([_compact_place(place) for place in data.get("places") or []])
    return {
        "source": "Google Maps",
        "search_location": location,
        "query": query,
        "restaurants": results[:count],
        "availability_note": (
            "Maps shows current business status only. Verify the future meal slot, menu, "
            "price, and fulfillment option on Swiggy or Zomato before confirmation."
        ),
    }


async def find_drive_through_on_route(
    origin: str,
    destination: str,
    cuisine: str = "",
    scheduled_for: str = "",
    max_results: int = 5,
    depart_at: str | None = None,
) -> dict:
    """Compatibility wrapper for route-side drive-through discovery."""
    return await find_restaurants_on_route(
        origin,
        destination,
        cuisine=cuisine,
        scheduled_for=scheduled_for,
        fulfillment_type="DRIVE_THROUGH",
        max_results=max_results,
        depart_at=depart_at,
    )


async def find_restaurants_on_route(
    origin: str,
    destination: str,
    cuisine: str = "",
    scheduled_for: str = "",
    fulfillment_type: str = "PICKUP",
    max_results: int = 5,
    depart_at: str | None = None,
) -> dict:
    """Compute a driving route and discover low-detour restaurant candidates."""
    origin = (origin or "").strip()
    destination = (destination or "").strip()
    if len(origin) < 3 or len(destination) < 3:
        raise MapsServiceError("Share both your starting point and destination for route search.")

    route_payload: dict[str, Any] = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "polylineQuality": "OVERVIEW",
        "polylineEncoding": "ENCODED_POLYLINE",
    }
    departure_text = (depart_at or scheduled_for or "").strip()
    departure_at = datetime.now(timezone.utc)
    if departure_text:
        try:
            departure_at = datetime.fromisoformat(departure_text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MapsServiceError("Route departure time must be an ISO-8601 datetime with a timezone.") from exc
        if departure_at.tzinfo is None:
            raise MapsServiceError("Route departure time must include a timezone.")
    route_payload["departureTime"] = departure_at.isoformat()

    route_data = await _post_json(
        ROUTES_COMPUTE_URL,
        route_payload,
        "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline",
    )
    routes = route_data.get("routes") or []
    if not routes:
        return {
            "source": "Google Maps",
            "origin": origin,
            "destination": destination,
            "restaurants": [],
            "availability_note": "No driving route was found for those locations.",
        }

    route = routes[0]
    encoded_polyline = ((route.get("polyline") or {}).get("encodedPolyline") or "").strip()
    if not encoded_polyline:
        raise MapsServiceError("Maps returned a route without a searchable route line.")

    count = max(1, min(int(max_results or 5), 10))
    normalized_type = fulfillment_type.strip().upper().replace("-", "_")
    qualifier = "drive-through restaurants" if normalized_type in {"DRIVE_THROUGH", "DRIVE_THRU"} else "restaurants with pickup"
    text_query = qualifier
    if cuisine.strip():
        text_query = f"{cuisine.strip()} {qualifier}"
    places_payload = {
        "textQuery": text_query,
        "pageSize": count,
        "searchAlongRouteParameters": {
            "polyline": {"encodedPolyline": encoded_polyline},
        },
    }
    field_mask = f"{PLACE_FIELD_MASK},routingSummaries"
    places_data = await _post_json(PLACES_TEXT_SEARCH_URL, places_payload, field_mask)
    places = places_data.get("places") or []
    summaries = places_data.get("routingSummaries") or []
    candidates = [
        _compact_place(place, summaries[index] if index < len(summaries) else None)
        for index, place in enumerate(places)
    ]
    for candidate in candidates:
        arrival = estimate_route_arrival(
            departure_at,
            candidate.get("route_duration_minutes"),
        )
        candidate["pickup_timing"] = {
            "departure_at": departure_at.isoformat(),
            "estimated_arrival_at": arrival.isoformat() if arrival else None,
            "estimate_source": "GOOGLE_MAPS_ROUTE_SUMMARY" if arrival else None,
            "preparation_time_minutes": None,
            "estimated_ready_at": None,
            "note": (
                "Route travel-time estimate only. Restaurant preparation time is not verified, "
                "so food readiness at arrival cannot be guaranteed."
            ),
        }
    return {
        "source": "Google Maps",
        "origin": origin,
        "destination": destination,
        "route_distance_meters": route.get("distanceMeters"),
        "route_duration_minutes": _duration_minutes(route.get("duration")),
        "depart_at": departure_at.isoformat(),
        "restaurants": _rank_places(candidates)[:count],
        "availability_note": (
            "These are route-side discovery results, not checkout or fulfillment proof. "
            "Verify the selected restaurant, requested future slot, fulfillment mode, menu, "
            "and price through the ordering provider before scheduling."
        ),
    }


def build_maps_tools() -> list:
    """Return LangChain tools for the external-provider agent."""
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            coroutine=search_best_restaurants,
            name="search_best_restaurants_on_maps",
            description=(
                "Search Google Maps for highly rated restaurants near an area or coordinates. "
                "Use for discovery only, then verify the chosen restaurant on Swiggy or Zomato."
            ),
        ),
        StructuredTool.from_function(
            coroutine=find_drive_through_on_route,
            name="find_drive_through_restaurants_on_route",
            description=(
                "Find route-side drive-through restaurant candidates using a driving origin and "
                "destination. Use for discovery only; provider tools must verify pickup support."
            ),
        ),
    ]
