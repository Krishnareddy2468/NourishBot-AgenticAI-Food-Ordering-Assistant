# Route-Aware Fulfillment

Phase 6 adds consented saved locations, route-aware restaurant ranking, future-time opening
evaluation, and provider-side fulfillment verification. Google Maps remains a read-only
discovery source and never grants checkout eligibility.

## Drive-through / on-route pickup flow

The product goal is travel without wasted time: the customer orders ahead from a restaurant on
the way, collects the food in person (no delivery), and the meal is ready when they arrive.

1. The user shares the route origin and destination, optionally by sending a current location.
2. Google Routes computes the driving route; Places searches for drive-through candidates near
   that route.
3. Results are ranked using operational status, rating, rating volume, and routing summary
   where available.
4. The user chooses a candidate and a future pickup time matching their expected arrival;
   NourishBot searches that restaurant on Swiggy or Zomato and verifies pickup support.
5. Only a provider tool that supports the chosen fulfillment and a future time may complete
   checkout, after explicit confirmation.

The Maps route tool can now return a route-summary-based estimated arrival for each candidate
when a timezone-aware departure time is supplied. The estimate is travel time only, not a
restaurant preparation or ready-time estimate. The customer must still choose and confirm a
future pickup time; NourishBot must never promise that food will be ready on arrival based on
the route estimate.

## Trust boundary

```text
Consented location or one-time route input
                    │
                    ▼
       Google Routes + Places discovery
                    │
                    ▼
 Deterministic ranking for the requested future time
  detour · status · rating confidence · diet · budget
                    │
                    ▼
       User selects one restaurant candidate
                    │
                    ▼
 Verified provider adapter checks exact restaurant,
 future availability, and requested fulfillment type
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 verified and eligible   manual provider selection
          │
          ▼
 explicit order confirmation remains required
```

Maps fields are discovery evidence only. Every Maps candidate includes
`maps_checkout_eligible: false` and `requires_provider_verification: true`. A restaurant can
be currently open while closed at the requested time; `open_now` and `future_open_state` are
therefore retained as separate facts.

## Consent and retention

- Inline locations can be used for one request without being persisted.
- Saving a location, route, or discovery-history record requires the latest `LOCATION`
  consent decision to be granted.
- Every retained record references the consent decision that authorized it and receives a
  `retention_expires_at` deadline based on `LOCATION_RETENTION_DAYS`.
- The daily privacy worker physically deletes expired history, routes, and locations.
- Withdrawing location consent physically deletes saved route/location data immediately.
- Customer endpoints support deleting one saved location, one saved route, or all route
  history. Queries are always tenant- and customer-scoped.
- Raw Maps responses and route polylines are never written to route history. History keeps a
  one-way query fingerprint, the selected place identity, decision status, and compact score
  summary only.

## Ranking contract

`score_restaurant_candidates` is deterministic and exposes every score component:

- operational business status;
- current opening state, used only as a small informational signal;
- opening state at the requested future time;
- route-detour duration;
- rating weighted by review-count confidence;
- verified diet-tag match; and
- estimated price against the request budget.

Missing future hours, diet metadata, or prices are represented as unknown. The ranking does
not invent these facts and cannot authorize an order.

## Provider verification and fallback

The selected restaurant is checked through the provider adapter using its place ID/name,
requested time, and fulfillment type. Eligibility requires all of the following:

1. the provider capability version is `VERIFIED`;
2. discovery is enabled in that capability version;
3. the response matches the selected restaurant;
4. the response explicitly confirms availability at the requested time;
5. the capability registry and response both support the requested fulfillment type.

This verifies the selection, not necessarily checkout. `checkout_eligible` becomes true only
when the same provider is also in verified transactional Mode C with scheduled ordering,
idempotency, and status reconciliation. Discovery or handoff modes keep checkout disabled.

Drive-through is never inferred from Maps wording or place types. If drive-through cannot be
verified but ordinary pickup is explicitly verified, the decision safely resolves to scheduled
`PICKUP`. Otherwise the response requires manual selection in the provider experience. Every
provider verification response is stored as a redacted provider receipt.

## Recovery

The recovery endpoint re-runs discovery when the route fingerprint changes, the selected
restaurant is closed at the requested time, or discovery is retried. If Maps is unavailable or
returns no candidates, the API returns a successful, explicit manual-selection path rather than
claiming that fulfillment is unavailable or attempting checkout.

## Customer API

| Endpoint | Purpose |
|---|---|
| `GET/POST /api/v1/customer/locations` | List or save consented locations |
| `DELETE /api/v1/customer/locations/{location_id}` | Delete one saved location and dependent routes |
| `GET/POST /api/v1/customer/routes` | List or save consented routes |
| `DELETE /api/v1/customer/routes/{route_id}` | Delete one route and dependent history |
| `POST /api/v1/customer/fulfillment/discover` | Rank nearby or low-detour candidates |
| `POST /api/v1/customer/fulfillment/verify` | Verify the exact selection with the provider |
| `POST /api/v1/customer/fulfillment/recover` | Re-discover after route or availability change |
| `GET/DELETE /api/v1/customer/route-history` | Inspect or delete retained decision history |

Migration `0015` creates `saved_locations`, `saved_routes`, and `route_discoveries`. Rollback
drops history first, followed by routes and locations, preserving foreign-key order.
