import pytest

from app.agent import maps_client


@pytest.mark.asyncio
async def test_search_best_restaurants_ranks_operational_rating(monkeypatch):
    async def fake_post(url, payload, field_mask):
        assert url == maps_client.PLACES_TEXT_SEARCH_URL
        assert payload["textQuery"] == "healthy food near Hitech City"
        assert "places.rating" in field_mask
        return {
            "places": [
                {
                    "id": "closed",
                    "displayName": {"text": "Closed Star"},
                    "rating": 5,
                    "businessStatus": "CLOSED_TEMPORARILY",
                },
                {
                    "id": "open",
                    "displayName": {"text": "Green Bowl"},
                    "rating": 4.6,
                    "userRatingCount": 420,
                    "businessStatus": "OPERATIONAL",
                    "currentOpeningHours": {"openNow": True},
                },
            ]
        }

    monkeypatch.setattr(maps_client, "_post_json", fake_post)
    result = await maps_client.search_best_restaurants("Hitech City", "healthy food")

    assert result["restaurants"][0]["name"] == "Green Bowl"
    assert "Swiggy or Zomato" in result["availability_note"]


@pytest.mark.asyncio
async def test_drive_through_search_passes_computed_route_to_places(monkeypatch):
    calls = []

    async def fake_post(url, payload, field_mask):
        calls.append((url, payload, field_mask))
        if url == maps_client.ROUTES_COMPUTE_URL:
            return {
                "routes": [{
                    "duration": "1800s",
                    "distanceMeters": 21000,
                    "polyline": {"encodedPolyline": "route-polyline"},
                }]
            }
        return {
            "places": [{
                "id": "route-stop",
                "displayName": {"text": "Highway Kitchen"},
                "businessStatus": "OPERATIONAL",
                "rating": 4.5,
            }],
            "routingSummaries": [{"legs": [{"duration": "240s", "distanceMeters": 1800}]}],
        }

    monkeypatch.setattr(maps_client, "_post_json", fake_post)
    result = await maps_client.find_drive_through_on_route(
        "Madhapur", "Hyderabad Airport", cuisine="vegetarian"
    )

    places_payload = calls[1][1]
    assert places_payload["searchAlongRouteParameters"]["polyline"]["encodedPolyline"] == "route-polyline"
    assert places_payload["textQuery"] == "vegetarian drive-through restaurants"
    assert result["route_duration_minutes"] == 30
    assert result["restaurants"][0]["route_duration_minutes"] == 4
    assert result["restaurants"][0]["pickup_timing"]["preparation_time_minutes"] is None
    assert "cannot be guaranteed" in result["restaurants"][0]["pickup_timing"]["note"]


@pytest.mark.asyncio
async def test_route_search_uses_explicit_departure_and_returns_arrival_estimate(monkeypatch):
    calls = []

    async def fake_post(url, payload, field_mask):
        calls.append((url, payload))
        if url == maps_client.ROUTES_COMPUTE_URL:
            return {"routes": [{
                "duration": "1800s",
                "distanceMeters": 21000,
                "polyline": {"encodedPolyline": "route-polyline"},
            }]}
        return {
            "places": [{"id": "stop", "displayName": {"text": "Route Stop"}}],
            "routingSummaries": [{"legs": [{"duration": "240s"}]}],
        }

    monkeypatch.setattr(maps_client, "_post_json", fake_post)
    result = await maps_client.find_drive_through_on_route(
        "Madhapur",
        "Hyderabad Airport",
        depart_at="2026-10-19T08:00:00+05:30",
    )

    assert calls[0][1]["departureTime"] == "2026-10-19T08:00:00+05:30"
    assert result["restaurants"][0]["pickup_timing"]["estimated_arrival_at"] == "2026-10-19T02:34:00+00:00"


@pytest.mark.asyncio
async def test_route_search_rejects_naive_departure_time():
    with pytest.raises(maps_client.MapsServiceError, match="include a timezone"):
        await maps_client.find_drive_through_on_route(
            "Madhapur",
            "Hyderabad Airport",
            depart_at="2026-10-19T08:00:00",
        )


@pytest.mark.asyncio
async def test_route_search_requires_both_endpoints():
    with pytest.raises(maps_client.MapsServiceError, match="starting point and destination"):
        await maps_client.find_drive_through_on_route("", "Hyderabad Airport")
