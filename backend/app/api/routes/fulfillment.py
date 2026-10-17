"""Consent-gated locations and route-aware fulfillment discovery APIs."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from app.agent.maps_client import (
    MapsServiceError,
    find_restaurants_on_route,
    search_best_restaurants,
)
from app.api.routes.customer_privacy import linked_customer_channel
from app.core.config import settings
from app.db.models import CustomerConsent, RouteDiscovery, SavedLocation, SavedRoute
from app.db.session import AsyncSessionLocal
from app.domain.route_fulfillment import (
    OpeningState,
    RouteFulfillmentError,
    normalize_fulfillment_type,
    recovery_reason,
    estimate_route_arrival,
    score_restaurant_candidates,
)
from app.providers.service import verify_selected_fulfillment


router = APIRouter(prefix="/api/v1/customer", tags=["route-fulfillment"])


class LocationCreate(BaseModel):
    label: str = Field(min_length=1, max_length=64)
    address: str = Field(min_length=3, max_length=500)
    place_id: str | None = Field(default=None, max_length=255)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class RouteCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    origin_location_id: int
    destination_location_id: int


class DiscoveryRequest(BaseModel):
    origin: str | None = Field(default=None, max_length=500)
    destination: str | None = Field(default=None, max_length=500)
    saved_route_id: int | None = None
    cuisine: str = Field(default="", max_length=100)
    desired_for: datetime
    depart_at: datetime | None = None
    fulfillment_type: str
    diet_tags: list[str] = Field(default_factory=list, max_length=20)
    budget_cents: int | None = Field(default=None, ge=0)
    max_results: int = Field(default=5, ge=1, le=10)
    remember_history: bool = False


class RecoveryRequest(DiscoveryRequest):
    previous_route_fingerprint: str | None = Field(default=None, max_length=128)
    selected_future_open_state: str | None = Field(default=None, max_length=16)


class VerificationRequest(BaseModel):
    history_id: str | None = Field(default=None, max_length=64)
    provider: str = Field(min_length=1, max_length=32)
    place_id: str = Field(min_length=1, max_length=255)
    restaurant_name: str = Field(min_length=1, max_length=255)
    requested_type: str
    desired_for: datetime
    future_open_state: str = "UNKNOWN"


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


def _empty_no_store() -> Response:
    return Response(
        status_code=204,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise HTTPException(422, "desired_for must include a timezone offset")
    return value.astimezone(timezone.utc)


def _fulfillment(value: str):
    try:
        return normalize_fulfillment_type(value)
    except RouteFulfillmentError as exc:
        raise HTTPException(422, str(exc)) from exc


def _fingerprint(*values: object) -> str:
    canonical = json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _retention_deadline() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=settings.LOCATION_RETENTION_DAYS)


async def _scope(request: Request) -> tuple[int, int]:
    channel = await linked_customer_channel(request)
    return int(channel.restaurant_id), int(channel.customer_id)


async def _location_consent(db, restaurant_id: int, customer_id: int) -> CustomerConsent | None:
    consent = (await db.execute(
        select(CustomerConsent)
        .where(
            CustomerConsent.restaurant_id == restaurant_id,
            CustomerConsent.customer_id == customer_id,
            CustomerConsent.purpose == "LOCATION",
        )
        .order_by(CustomerConsent.id.desc())
        .limit(1)
    )).scalar_one_or_none()
    return consent if consent and consent.granted else None


async def _require_location_consent(db, restaurant_id: int, customer_id: int) -> CustomerConsent:
    consent = await _location_consent(db, restaurant_id, customer_id)
    if not consent:
        raise HTTPException(403, "Location consent is required before saving locations, routes, or route history")
    return consent


def _location_payload(value: SavedLocation) -> dict[str, Any]:
    return {
        "id": value.id,
        "label": value.label,
        "address": value.address,
        "place_id": value.place_id,
        "latitude": value.latitude,
        "longitude": value.longitude,
        "retention_expires_at": value.retention_expires_at.isoformat(),
    }


@router.get("/locations")
async def list_saved_locations(request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        values = (await db.execute(
            select(SavedLocation).where(
                SavedLocation.restaurant_id == restaurant_id,
                SavedLocation.customer_id == customer_id,
                SavedLocation.retention_expires_at > datetime.now(timezone.utc),
            ).order_by(SavedLocation.label)
        )).scalars().all()
    return [_location_payload(value) for value in values]


@router.post("/locations", status_code=201)
async def save_location(body: LocationCreate, request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        consent = await _require_location_consent(db, restaurant_id, customer_id)
        value = SavedLocation(
            restaurant_id=restaurant_id,
            customer_id=customer_id,
            label=body.label.strip(),
            address=body.address.strip(),
            place_id=(body.place_id or "").strip() or None,
            latitude=body.latitude,
            longitude=body.longitude,
            consent_id=consent.id,
            retention_expires_at=_retention_deadline(),
        )
        db.add(value)
        try:
            await db.commit()
        except IntegrityError as exc:
            await db.rollback()
            raise HTTPException(409, "A saved location already uses that label") from exc
        await db.refresh(value)
    return _location_payload(value)


@router.delete("/locations/{location_id}", status_code=204)
async def delete_location(location_id: int, request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        result = await db.execute(delete(SavedLocation).where(
            SavedLocation.id == location_id,
            SavedLocation.restaurant_id == restaurant_id,
            SavedLocation.customer_id == customer_id,
        ))
        if not result.rowcount:
            raise HTTPException(404, "Saved location not found")
        await db.commit()
    return _empty_no_store()


@router.get("/routes")
async def list_saved_routes(request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        values = (await db.execute(select(SavedRoute).where(
            SavedRoute.restaurant_id == restaurant_id,
            SavedRoute.customer_id == customer_id,
            SavedRoute.retention_expires_at > datetime.now(timezone.utc),
        ).order_by(SavedRoute.name))).scalars().all()
    return [{
        "id": value.id,
        "name": value.name,
        "origin_location_id": value.origin_location_id,
        "destination_location_id": value.destination_location_id,
        "route_fingerprint": value.route_fingerprint,
        "retention_expires_at": value.retention_expires_at.isoformat(),
    } for value in values]


@router.post("/routes", status_code=201)
async def save_route(body: RouteCreate, request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id = await _scope(request)
    if body.origin_location_id == body.destination_location_id:
        raise HTTPException(422, "Route origin and destination must be different")
    async with AsyncSessionLocal() as db:
        consent = await _require_location_consent(db, restaurant_id, customer_id)
        locations = (await db.execute(select(SavedLocation).where(
            SavedLocation.id.in_([body.origin_location_id, body.destination_location_id]),
            SavedLocation.restaurant_id == restaurant_id,
            SavedLocation.customer_id == customer_id,
            SavedLocation.retention_expires_at > datetime.now(timezone.utc),
        ))).scalars().all()
        if len(locations) != 2:
            raise HTTPException(404, "One or both saved locations were not found")
        value = SavedRoute(
            restaurant_id=restaurant_id,
            customer_id=customer_id,
            name=body.name.strip(),
            origin_location_id=body.origin_location_id,
            destination_location_id=body.destination_location_id,
            route_fingerprint=_fingerprint(body.origin_location_id, body.destination_location_id),
            consent_id=consent.id,
            retention_expires_at=min(
                _retention_deadline(),
                *(location.retention_expires_at for location in locations),
            ),
        )
        db.add(value)
        try:
            await db.commit()
        except IntegrityError as exc:
            await db.rollback()
            raise HTTPException(409, "A saved route already uses that name") from exc
        await db.refresh(value)
    return {"id": value.id, "name": value.name, "route_fingerprint": value.route_fingerprint}


@router.delete("/routes/{route_id}", status_code=204)
async def delete_route(route_id: int, request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        result = await db.execute(delete(SavedRoute).where(
            SavedRoute.id == route_id,
            SavedRoute.restaurant_id == restaurant_id,
            SavedRoute.customer_id == customer_id,
        ))
        if not result.rowcount:
            raise HTTPException(404, "Saved route not found")
        await db.commit()
    return _empty_no_store()


async def _resolve_route(body: DiscoveryRequest, restaurant_id: int, customer_id: int) -> tuple[str, str, str, int | None]:
    if body.saved_route_id is None:
        origin, destination = (body.origin or "").strip(), (body.destination or "").strip()
        if _fulfillment(body.fulfillment_type).value == "DELIVERY":
            if len(destination) < 3:
                raise HTTPException(422, "Share a delivery destination or choose a saved route")
            return origin, destination, _fingerprint("DELIVERY", destination), None
        if len(origin) < 3 or len(destination) < 3:
            raise HTTPException(422, "Share both route origin and destination, or choose a saved route")
        return origin, destination, _fingerprint(origin, destination), None

    async with AsyncSessionLocal() as db:
        await _require_location_consent(db, restaurant_id, customer_id)
        route = (await db.execute(select(SavedRoute).where(
            SavedRoute.id == body.saved_route_id,
            SavedRoute.restaurant_id == restaurant_id,
            SavedRoute.customer_id == customer_id,
            SavedRoute.retention_expires_at > datetime.now(timezone.utc),
        ))).scalar_one_or_none()
        if not route:
            raise HTTPException(404, "Saved route not found")
        locations = (await db.execute(select(SavedLocation).where(
            SavedLocation.id.in_([route.origin_location_id, route.destination_location_id]),
            SavedLocation.restaurant_id == restaurant_id,
            SavedLocation.customer_id == customer_id,
            SavedLocation.retention_expires_at > datetime.now(timezone.utc),
        ))).scalars().all()
        by_id = {value.id: value for value in locations}
        if route.origin_location_id not in by_id or route.destination_location_id not in by_id:
            raise HTTPException(409, "Saved route locations are unavailable; choose locations manually")
        return (
            by_id[route.origin_location_id].address,
            by_id[route.destination_location_id].address,
            route.route_fingerprint,
            route.id,
        )


async def _remember_discovery(
    *,
    restaurant_id: int,
    customer_id: int,
    saved_route_id: int | None,
    query_fingerprint: str,
    body: DiscoveryRequest,
    status: str,
    score_summary: dict[str, Any],
    reason: str | None = None,
) -> str:
    async with AsyncSessionLocal() as db:
        consent = await _require_location_consent(db, restaurant_id, customer_id)
        record = RouteDiscovery(
            id=str(uuid.uuid4()),
            restaurant_id=restaurant_id,
            customer_id=customer_id,
            saved_route_id=saved_route_id,
            consent_id=consent.id,
            query_fingerprint=query_fingerprint,
            requested_for=_aware(body.desired_for),
            requested_fulfillment_type=_fulfillment(body.fulfillment_type).value,
            discovery_status=status,
            recovery_reason=reason,
            score_summary_json=score_summary,
            retention_expires_at=_retention_deadline(),
        )
        db.add(record)
        await db.commit()
        return record.id


async def _update_verification_history(
    history_id: str | None,
    *,
    restaurant_id: int,
    customer_id: int,
    body: VerificationRequest,
    decision: dict[str, Any],
) -> None:
    if not history_id:
        return
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            update(RouteDiscovery)
            .where(
                RouteDiscovery.id == history_id,
                RouteDiscovery.restaurant_id == restaurant_id,
                RouteDiscovery.customer_id == customer_id,
            )
            .values(
                selected_place_id=body.place_id,
                selected_restaurant_name=body.restaurant_name,
                provider=body.provider.strip().upper(),
                resolved_fulfillment_type=decision.get("resolved_type"),
                provider_verified=bool(decision.get("provider_verified")),
                discovery_status=(
                    "PROVIDER_VERIFIED"
                    if decision.get("checkout_eligible")
                    else "MANUAL_SELECTION_REQUIRED"
                ),
            )
        )
        if not result.rowcount:
            raise HTTPException(404, "Route discovery history record not found")
        await db.commit()


async def _discover(body: DiscoveryRequest, request: Request, response: Response, reason: str | None = None):
    _no_store(response)
    desired_for = _aware(body.desired_for)
    depart_at = _aware(body.depart_at) if body.depart_at else datetime.now(timezone.utc)
    fulfillment = _fulfillment(body.fulfillment_type)
    restaurant_id, customer_id = await _scope(request)
    origin, destination, route_fingerprint, saved_route_id = await _resolve_route(body, restaurant_id, customer_id)
    search_terms = " ".join((*body.diet_tags, body.cuisine)).strip()

    try:
        if fulfillment.value == "DELIVERY":
            maps_result = await search_best_restaurants(
                destination,
                query=(search_terms + " restaurants").strip(),
                max_results=body.max_results,
            )
        else:
            maps_result = await find_restaurants_on_route(
                origin,
                destination,
                cuisine=search_terms,
                scheduled_for=depart_at.isoformat(),
                fulfillment_type=fulfillment.value,
                max_results=body.max_results,
            )
        candidates = score_restaurant_candidates(
            maps_result.get("restaurants") or [],
            desired_for=desired_for,
            required_diet_tags=body.diet_tags,
            budget_cents=body.budget_cents,
        )
        if fulfillment.value != "DELIVERY":
            for candidate in candidates:
                arrival = estimate_route_arrival(
                    depart_at,
                    candidate.get("route_duration_minutes"),
                )
                candidate["pickup_timing"] = {
                    "departure_at": depart_at.isoformat(),
                    "estimated_arrival_at": arrival.isoformat() if arrival else None,
                    "estimate_source": "GOOGLE_MAPS_ROUTE_SUMMARY" if arrival else None,
                    "preparation_time_minutes": None,
                    "estimated_ready_at": None,
                    "note": (
                        "Route travel-time estimate only. Restaurant preparation time is not "
                        "verified, so food readiness at arrival cannot be guaranteed."
                    ),
                }
        status = "CANDIDATES_FOUND" if candidates else "NO_CANDIDATES"
    except (MapsServiceError, RouteFulfillmentError) as exc:
        candidates, status = [], "MAPS_UNAVAILABLE"
        maps_result = {"availability_note": str(exc)}

    history_id = None
    if body.remember_history:
        history_id = await _remember_discovery(
            restaurant_id=restaurant_id,
            customer_id=customer_id,
            saved_route_id=saved_route_id,
            query_fingerprint=_fingerprint(
                route_fingerprint,
                body.cuisine,
                desired_for,
                depart_at,
                fulfillment.value,
            ),
            body=body,
            status=status,
            reason=reason,
            score_summary={
                "candidate_count": len(candidates),
                "top_place_id": candidates[0].get("place_id") if candidates else None,
                "top_score": candidates[0].get("route_score") if candidates else None,
            },
        )

    return {
        "discovery_status": status,
        "route_fingerprint": route_fingerprint,
        "recovery_reason": reason,
        "history_id": history_id,
        "requested_for": desired_for.isoformat(),
        "depart_at": depart_at.isoformat() if fulfillment.value != "DELIVERY" else None,
        "requested_fulfillment_type": fulfillment.value,
        "restaurants": candidates,
        "maps_checkout_eligible": False,
        "manual_selection": {
            "required": not candidates,
            "message": "Choose a restaurant manually in the provider experience and verify the future slot and fulfillment mode." if not candidates else None,
        },
        "availability_note": maps_result.get("availability_note"),
        "pickup_timing_note": (
            "Candidate arrival times are route estimates only. No restaurant preparation "
            "time has been verified; confirm pickup timing with the provider."
            if fulfillment.value != "DELIVERY"
            else None
        ),
    }


@router.post("/fulfillment/discover")
async def discover_fulfillment(body: DiscoveryRequest, request: Request, response: Response):
    return await _discover(body, request, response)


@router.post("/fulfillment/recover")
async def recover_fulfillment(body: RecoveryRequest, request: Request, response: Response):
    restaurant_id, customer_id = await _scope(request)
    _origin, _destination, current, _saved_route_id = await _resolve_route(body, restaurant_id, customer_id)
    reason = recovery_reason(
        previous_route_fingerprint=body.previous_route_fingerprint,
        current_route_fingerprint=current,
        selected_future_open_state=body.selected_future_open_state,
    ) or "DISCOVERY_RETRY"
    return await _discover(body, request, response, reason=reason)


@router.post("/fulfillment/verify")
async def verify_fulfillment(body: VerificationRequest, request: Request, response: Response):
    _no_store(response)
    desired_for = _aware(body.desired_for)
    requested_type = _fulfillment(body.requested_type)
    restaurant_id, customer_id = await _scope(request)
    if body.future_open_state.upper() == OpeningState.CLOSED.value:
        result = {
            "requested_type": requested_type.value,
            "resolved_type": None,
            "provider_verified": False,
            "checkout_eligible": False,
            "manual_selection_required": True,
            "reason": "The restaurant is closed at the requested future time. Run route recovery or choose another restaurant.",
        }
        await _update_verification_history(
            body.history_id,
            restaurant_id=restaurant_id,
            customer_id=customer_id,
            body=body,
            decision=result,
        )
        return result
    decision = await verify_selected_fulfillment(
        provider=body.provider.strip().upper(),
        restaurant_id=restaurant_id,
        place_id=body.place_id,
        restaurant_name=body.restaurant_name,
        requested_type=requested_type.value,
        desired_for=desired_for,
    )
    result = decision.as_dict()
    await _update_verification_history(
        body.history_id,
        restaurant_id=restaurant_id,
        customer_id=customer_id,
        body=body,
        decision=result,
    )
    return result


@router.get("/route-history")
async def list_route_history(
    request: Request,
    response: Response,
    limit: int = Query(default=20, ge=1, le=100),
):
    _no_store(response)
    restaurant_id, customer_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        values = (await db.execute(select(RouteDiscovery).where(
            RouteDiscovery.restaurant_id == restaurant_id,
            RouteDiscovery.customer_id == customer_id,
            RouteDiscovery.retention_expires_at > datetime.now(timezone.utc),
        ).order_by(RouteDiscovery.created_at.desc()).limit(limit))).scalars().all()
    return [{
        "id": value.id,
        "requested_for": value.requested_for.isoformat(),
        "requested_fulfillment_type": value.requested_fulfillment_type,
        "discovery_status": value.discovery_status,
        "recovery_reason": value.recovery_reason,
        "score_summary": value.score_summary_json,
        "retention_expires_at": value.retention_expires_at.isoformat(),
    } for value in values]


@router.delete("/route-history", status_code=204)
async def delete_route_history(request: Request, response: Response):
    _no_store(response)
    restaurant_id, customer_id = await _scope(request)
    async with AsyncSessionLocal() as db:
        await db.execute(delete(RouteDiscovery).where(
            RouteDiscovery.restaurant_id == restaurant_id,
            RouteDiscovery.customer_id == customer_id,
        ))
        await db.commit()
    return _empty_no_store()
