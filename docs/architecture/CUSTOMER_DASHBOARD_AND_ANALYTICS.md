# Customer Dashboard and Operational Analytics

Phase 7 adds two read models over the existing transactional system: a private customer control center and tenant-scoped staff analytics. Neither surface bypasses the order, planning, provider, consent, or scheduling domain services.

## Identity and scope

Every website customer account has a one-to-one `customer_id` and a verified `WEB` channel named `account:{user_id}`. Signup and OAuth linking provision these records in the same transaction. Existing customer users are backfilled by migration `0016`.

Customer dashboard requests resolve either the existing opaque linked-browser identity or an authenticated customer session. The API derives `restaurant_id`, `customer_id`, and `channel_id` from that verified channel. It never accepts these identifiers from the request body.

Staff analytics require the existing manager authorization dependency. The restaurant tenant is derived from the authenticated user, and each analytics query includes that tenant predicate.

## Customer dashboard

`GET /api/v1/customer/dashboard` returns a bounded aggregate containing:

- The selected seven-day calendar and upcoming confirmed meals.
- Draft plans, saved templates, and recurring routines.
- Customer-owned orders and their provider attempts.
- Diet, allergy, health, consent, budget, and provider-capability state.
- Weekly and lifetime spending totals sourced only from the spending ledger.

The `/account` interface exposes plan confirmation, routine pause/resume, occurrence skip/reschedule, reminder consent, preference changes, budget limits, and eligible order cancellation. Existing domain services still validate each mutation. Confirming a meal plan does not place an order.

External-order cancellation remains conservative. NourishBot first requires a confirmed provider cancellation when the external provider has accepted the order; an unknown or failed cancellation is sent to manual review instead of being represented as cancelled.

## Lifecycle timestamps

Analytics deliberately separates three meanings of time:

| Timestamp | Meaning | Used for |
|---|---|---|
| `orders.desired_for` | Customer's requested meal time | Scheduled demand volume and fulfillment cohorts |
| `orders.submitted_at` | First accepted/preparing lifecycle transition | Submission success and fulfillment duration |
| `orders.completed_at` | First delivered/completed lifecycle transition | Completion success and revenue recognition |

Migration `0016` backfills legacy timestamps from the best available lifecycle data. New writes are maintained centrally by `transition_order`.

## Immutable spending ledger

`spending_ledger_entries` is the accounting source for customer spending summaries and completed-revenue analytics. A completed customer order appends one positive `COMPLETED_SPEND` entry. The `(order_id, entry_type)` uniqueness constraint prevents duplicate recognition.

Ledger rows cannot be updated or deleted:

- SQLAlchemy listeners reject application-level mutation.
- A PostgreSQL trigger rejects database-level update and delete statements.

Corrections must be represented as new `REFUND_CREDIT` rows with negative values. Dashboard totals sum signed entries and therefore remain reconcilable.

## Staff analytics

`GET /api/v1/analytics/operations` accepts a bounded date range of at most 366 days and reports:

- Scheduled volume grouped by desired hour.
- Submission and completion rates.
- Scheduler attempt lag, missed deadlines, and dead-letter count.
- Provider failures and unknown outcomes.
- Completed revenue grouped by fulfillment date.
- Delivery, pickup, and drive-through completion performance.
- Cancellation and routine-reschedule reasons.

The response includes a `timestamp_contract` and `revenue_source` so clients cannot silently reinterpret the measures.

## Deployment

Apply migration `0016` after `0015` before starting the updated API. The migration adds the account-to-customer link, lifecycle timestamps, ledger backfill, indexes, constraints, and append-only trigger. Its downgrade removes the Phase 7 schema additions but intentionally does not delete account channels created during migration, because they may have acquired dependent audit history.
