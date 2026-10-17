# NourishBot V2.0 — Product Upgrade and Implementation Plan

**Product:** NourishBot — Your AI Meal Planning & Scheduling Companion
**Plan status:** Approved for phased development
**Created:** 2026-10-15
**Release posture:** Not ready for public transactional use until the Phase 0 release gate passes

## 1. Product Direction

NourishBot is an AI meal planning, scheduling, and ordering companion. Users plan meals for breakfast, lunch, snacks, and dinner through Telegram, WhatsApp, and a customer web dashboard. Orders must be scheduled for a future time; instant ordering is outside the product scope.

NourishBot supports:

- In-house NourishBot ordering and fulfillment.
- Home delivery, with a partially implemented nutrition-aware full-day meal-planning experience.
- Scheduled pickup and drive-through collection.
- Google Maps restaurant discovery and route-aware pickup discovery.
- Swiggy and Zomato integrations only within verified provider capabilities.
- Explicit user confirmation before any transactional action.
- Diet, allergy, health-goal, budget, and meal-time preferences.
- Daily and weekly meal plans, saved templates, reminders, and recurring routines.

Google Maps is a discovery service only. It does not prove menu availability, price, provider availability, future opening hours, or drive-through eligibility. Swiggy/Zomato capabilities must be verified independently before NourishBot relies on them.

### Product direction to develop

The customer product has two complementary experiences:

1. **Nutrition-aware home meals:** help a customer plan a complete day's meals around their
   dietary preferences and nutrition goals, then schedule suitable individual orders for home
   delivery at selected times. The customer reviews and confirms each order; confirming a meal
   plan alone must never create orders.
2. **On-route pickup:** find suitable restaurants along a customer's route, verify provider
   pickup support, and enable order-ahead collection when the provider supports the selected
   fulfillment and time.

The prototype planner can total source-labelled menu nutrition facts and requires all five
verified values when nutrition targets are used. `VERIFIED` currently means a manager-entered
assertion with a recorded source, not independent scientific review. Nutrition planning must not
invent nutrition values: any unverified values must be identified as estimates. Do not present
targets as medical advice or guarantee health outcomes.

Meal planning and recurring reminders are distinct from recurring food delivery. The product
does not include a multi-day home-delivery schedule for situations such as a household cook
being away.

## 2. Non-Negotiable Product Boundaries

1. No instant or ASAP ordering.
2. Every meal order must include a future time and timezone.
3. No checkout, payment, or provider mutation without explicit user confirmation.
4. The confirmation must refer to an unchanged order snapshot.
5. Maps data must never be presented as provider-confirmed availability.
6. NourishBot must not claim a provider order succeeded without a verified receipt.
7. Unknown provider outcomes must be reconciled before retrying.
8. Personalization and long-term memory require user consent.
9. Every tenant-scoped resource must be authorized using the authenticated tenant.
10. Sensitive customer, location, payment, and provider data must be minimized and redacted.

## 3. Current Architecture Audit

```text
Telegram / WhatsApp / Staff Web
              |
              v
     Identity + Chat Session
              |
              v
 Context -> LLM Planner -> Safety Guard
              |
              v
 Deterministic Executor -> Verifier
              |
       +------+---------+
       |                |
       v                v
NourishBot DB      Swiggy/Zomato MCP
       |                |
       |- Orders        `- Provider tools
       |- Payments
       |- Delivery
       `- Scheduled worker
              |
              v
     Staff/Kitchen Dashboard
```

The project already follows a valuable safety principle: the LLM proposes actions, while deterministic application code performs database mutations. This architecture should be retained and strengthened.

### 3.1 Existing capabilities

| Capability                | Status                    | Notes                                                       |
| ------------------------- | ------------------------- | ----------------------------------------------------------- |
| Scheduled in-house orders | Partial                   | Orders persist as `SCHEDULED` and activate later          |
| Telegram ordering         | Implemented               | Main conversational channel                                 |
| WhatsApp ordering         | Partial                   | Signed webhook and duplicate-message protection exist       |
| Meal periods              | Partial                   | Conversational concept, not a structured meal-plan domain   |
| Dietary preferences       | Partial                   | Preference fields exist but lack complete management APIs   |
| Home delivery             | Partial                   | Address and delivery objects exist                          |
| Pickup and drive-through  | Partial                   | Order types and route discovery exist                       |
| Google Maps discovery     | Implemented for discovery | Correctly documented as read-only discovery                 |
| Swiggy/Zomato integration | Experimental              | MCP connections exist; provider capabilities are unverified |
| Explicit confirmation     | Unsafe                    | Prompt instruction exists, deterministic proof does not     |
| Staff dashboard           | Implemented               | Orders, kitchen, menu, team, deliveries, basic analytics    |
| Customer dashboard        | Missing                   | No planner/calendar/templates/budgets/routines              |
| Recurring meal routines   | Missing                   | No recurrence engine or persisted routine model             |
| Budget controls           | Missing                   | Accumulated spend is not a budget system                    |
| Consent-based memory      | Missing                   | Preferences are learned automatically                       |
| Durable provider retries  | Missing                   | No provider attempt or reconciliation ledger                |
| Production observability  | Partial                   | Logs/readiness exist; operational metrics and alerts do not |

### 3.2 Release-blocking findings

#### P0: Cross-tenant data access

Several order, delivery, menu, and payment operations do not consistently filter by the authenticated user's `restaurant_id`. A staff user may be able to read or mutate another restaurant's data.

#### P0: Public tracking and GPS exposure

- Driver GPS updates are not authenticated.
- Numeric delivery tracking can expose driver and customer details.
- Order references have low entropy for a public tracking credential.
- WebSocket subscriptions trust a caller-supplied restaurant ID.

#### P0: Confirmation is not deterministically enforced

The planner prompt requests confirmation, but the execution guard does not prove that the current user confirmed the exact cart, amount, address, provider, fulfillment type, and scheduled time.

#### P0: Logical order idempotency is ineffective

The order idempotency key includes a newly generated order reference. A retry therefore receives a different key and can create a duplicate logical order.

#### P0: External provider uncertainty is represented as success

An MCP abort/connection-cleanup error can produce a message saying the request was sent even when NourishBot cannot prove whether a provider transaction succeeded.

#### P0: No durable external transaction record

External provider calls lack a durable attempt record with request fingerprint, stable idempotency key, provider request/order ID, retry metadata, last error, and reconciliation state.

#### P0: Scheduler is not enabled in every required deployment profile

Celery Beat is included in the cloud profile but not the MCP profile, so scheduled activation can be missing from an MCP deployment.

### 3.3 High-priority findings

#### P1: Memory is not consent-based

Preference learning has no consent, retention, export, or deletion controls. Learning begins when an order is placed instead of after successful fulfillment. The memory update path also crosses database sessions and can operate on a detached model instance.

#### P1: State models are inconsistent

Conversation, order, provider, payment, and delivery lifecycles use different and sometimes conflicting states. A failed delivery can be converted into `CANCELLED`, losing the true failure reason.

#### P1: Provider capability detection is too weak

Searching tool schemas for scheduling-related field names does not prove that scheduling, cancellation, tracking, delivery, pickup, or drive-through work correctly.

#### P1: Outbox processing is not multi-replica safe

The API-hosted outbox worker has no durable lease or row locking, so multiple replicas may process the same event.

#### P1: Payment webhook processing is incomplete

Webhook events do not have a replay/idempotency ledger, and all events other than `payment.captured` are treated as failures.

#### P1: Authentication and role coverage need hardening

- Staff tokens are stored in browser `localStorage`.
- CSP and related frontend security headers are missing.
- Backend roles and available frontend workspaces do not match.

### 3.4 Lower-priority quality findings

- Staff analytics use creation time where scheduled or fulfilled time is expected.
- Revenue labeled as "today" is not restricted to today's completed orders.
- Illustrative ratings and prices on the landing page should be clearly labeled or removed.
- Legacy and current agent implementations coexist and can drift.
- Frontend unit, accessibility, and end-to-end test infrastructure is missing.
- Live Swiggy/Zomato capability tests have not been completed.

## 4. Gap Analysis

| V2 requirement                              | Current status    | Required result                                             |
| ------------------------------------------- | ----------------- | ----------------------------------------------------------- |
| Identity across Telegram, WhatsApp, and web | Partial           | Verified identity linking and account ownership             |
| Consent-based memory                        | Missing           | Consent, enable/disable, export/delete, retention           |
| Daily and weekly planner                    | Complete          | Four-slot daily and seven-day draft plans with constraints  |
| Saved meal templates                        | Complete          | Single-meal and full-day template persistence               |
| Meal modification                           | Complete          | Persisted edits invalidate confirmation snapshots           |
| Reminders                                   | Complete          | Consent-gated durable Telegram and WhatsApp reminders       |
| Provider Mode A: discovery                  | Complete          | Registry-guarded read-only tools and visible mode messaging  |
| Provider Mode B: handoff                    | Complete          | User-controlled provider handoff without platform checkout   |
| Provider Mode C: transaction                | Gated             | Available only after capability and exact-schema verification|
| Provider capability registry                | Complete          | Immutable versions, schema hashes, receipts, and health state|
| Scheduled delivery                          | Partial           | Desired time, provider slot, submission time, ready-by time |
| Route-aware pickup                          | Partial           | Eligibility verification and consented route persistence    |
| Restaurant search                           | Partial           | Dietary, budget, route, and availability scoring            |
| Explicit confirmation                       | Unsafe            | Snapshot-bound deterministic confirmation                   |
| Cancellation/rescheduling                   | Partial           | Provider cancellation is capability-gated; rescheduling remains planned |
| Reliable retry behavior                     | Missing           | Leases, backoff, reconciliation, and dead-letter state      |
| Uncertain outcomes                          | Complete          | Durable `UNKNOWN` state and status-before-retry reconciliation |
| Customer dashboard                          | Missing           | Calendar, history, templates, preferences, spending         |
| Budget controls                             | Partial           | Plan limits and estimates implemented; spending ledger remains |
| Recurring routines                          | Complete          | Versioned rules, lifecycle actions, and bounded materialization |
| Analytics                                   | Partial           | Correct time semantics and operational/business metrics     |
| Security and privacy                        | Incomplete        | Tenant isolation, tracking privacy, consent, audit trail    |
| Comprehensive tests                         | Missing           | Unit, integration, E2E, security, provider contract tests   |

## 5. Target Architecture

```text
Telegram ----+
WhatsApp ----+--> Identity Resolver --> Conversation Session
Customer Web+                              |
                                             v
                                  Planner / Recommendation
                                             |
                                  Deterministic Policy Gate
                                             |
                     +-----------------------+-----------------------+
                     |                       |                       |
                     v                       v                       v
                Meal Planner            Order Service         Route Discovery
                     |                       |                       |
                     v                       v                       v
           Plans/Templates/Routines    Scheduler/Jobs           Google Maps
                                             |
                                  Provider Capability Router
                               +-------------+-------------+
                               |             |             |
                               v             v             v
                          NourishBot       Mode A/B       Mode C
                          in-house     discovery/handoff transaction
                                             |
                                             v
                                Attempts/Receipts/Reconciliation
```

### 5.1 Domain entities

The V2 domain should separate planning, scheduling, ordering, and provider execution:

- `user_identities`
- `user_consents`
- `saved_locations`
- `saved_routes`
- `meal_plans`
- `meal_plan_items`
- `meal_templates`
- `meal_routines`
- `scheduled_meals`
- `orders`
- `order_events`
- `confirmation_challenges`
- `provider_capabilities`
- `provider_accounts`
- `provider_attempts`
- `reminders`
- `budgets`
- `spending_ledger`
- `audit_events`

A meal plan does not become a transactional order until the user confirms an exact immutable snapshot.

### 5.2 Time model

Every scheduled meal should distinguish:

- `desired_meal_at`
- `user_timezone`
- `desired_ready_at`
- `provider_slot_start`
- `provider_slot_end`
- `submit_not_before`
- `submit_deadline`
- `next_retry_at`
- `confirmed_at`
- `provider_confirmed_at`

All timestamps are stored in UTC with the user's IANA timezone stored separately for display and recurrence calculations.

### 5.3 Order lifecycle

```text
DRAFT
  -> PENDING_CONFIRMATION
  -> CONFIRMED_BY_USER
  -> SCHEDULED
  -> SUBMISSION_QUEUED
  -> SUBMITTING
  -> PROVIDER_CONFIRMED
  -> PREPARING
  -> READY
  -> OUT_FOR_DELIVERY / READY_FOR_PICKUP
  -> COMPLETED
```

Recovery and terminal states:

```text
CANCELLED
FAILED
UNKNOWN
RESCHEDULE_REQUIRED
```

State changes must be validated by one domain transition service and recorded as immutable events.

### 5.4 Provider operating modes

#### Mode A — Discovery

Search restaurants, menus, ratings, and provider availability. No order mutation.

#### Mode B — Handoff

Prepare a verified cart or secure provider handoff. The user completes checkout on the provider surface.

#### Mode C — Transactional

NourishBot submits the order only when the provider has verified support for:

- The requested future scheduling window.
- The requested fulfillment type.
- Explicit confirmation and stable idempotency.
- Order status lookup and uncertain-outcome reconciliation.
- A provider receipt or order ID.

Swiggy and Zomato default to Mode A until higher modes pass live capability verification.

## 6. Implementation Phases

## Phase 0 — Public Release Safety Gate

**Goal:** Close vulnerabilities that can expose data, mutate another tenant's records, or create unconfirmed/duplicate orders.

**Implementation status:** Complete in the working tree on 2026-10-15. Migration `0008` must be applied during deployment before the updated application starts.

### Work

- [X] Enforce tenant filters on all staff reads and writes.
- [X] Centralize authenticated tenant extraction and reject missing tenant scope.
- [X] Allowlist customer-chat tools so prompt injection cannot reach staff mutations.
- [X] Authenticate WebSocket connections and derive tenant from the HttpOnly staff session.
- [X] Authenticate driver GPS/location updates.
- [X] Restrict delivery status/proof actions to the assigned driver or authorized manager.
- [X] Replace public numeric tracking with high-entropy, revocable tracking tokens.
- [X] Redact driver phone, exact location history, address, and proof data by audience.
- [X] Add an immutable confirmation snapshot and one-time confirmation challenge.
- [X] Derive idempotency keys from the stable confirmation/request identifier.
- [X] Replace false-success MCP responses with durable `UNKNOWN` outcome handling.
- [X] Default Swiggy/Zomato to non-transactional discovery until capability verification exists.
- [X] Enable scheduler services in all relevant deployment profiles.
- [X] Add security regression tests before changing public behavior.
- [X] Move staff JWTs out of browser storage into HttpOnly cookies with origin/CSRF checks.

### Primary files

- `backend/app/api/routes/orders.py`
- `backend/app/api/routes/deliveries.py`
- `backend/app/api/routes/payments.py`
- `backend/app/api/main.py`
- `backend/app/auth/`
- `backend/app/agent/pipeline.py`
- `backend/app/agent/executor.py`
- `backend/app/agent/mcp_agent.py`
- `backend/docker-compose.yml`
- `frontend/context/AuthContext.jsx`
- `frontend/hooks/useWebSocket.js`

### Dependencies and risks

- Requires a database migration for confirmation and tracking tokens.
- Tracking links already shared during testing may need temporary compatibility handling.
- Tenant filtering must not break legitimate admin access.

### Acceptance criteria

- [X] Cross-tenant reads and writes return 403 or 404.
- [X] Anonymous location mutation is impossible.
- [X] A client cannot subscribe to another tenant's WebSocket stream.
- [X] A changed cart/address/schedule invalidates prior confirmation.
- [X] Replayed confirmation creates no duplicate logical order.
- [X] Unknown provider outcomes never produce a success message.
- [X] Scheduled activation runs in every supported deployment profile.

### Phase 0 validation evidence

- Backend tests: 36 passed.
- Python compilation: passed.
- Frontend production build (including Next.js compile/type validation): passed.
- Alembic migration chain: `0008` is the single head.
- Alembic PostgreSQL offline SQL generation for `0007 -> 0008`: passed.
- Docker Compose cloud profile: API, worker, Beat, and Redis present.
- Docker Compose MCP profile: MCP API, worker, Beat, and Redis present without an API port collision.

## Phase 1 — Identity, Confirmation, and Consent

**Goal:** Establish a verified customer identity and privacy-safe personalization foundation.

**Implementation status:** Complete in the working tree on 2026-10-15. Migration `0009` must be applied during deployment before the updated application starts.

### Work

- [X] Add unified identity records for Telegram, WhatsApp, and web.
- [X] Add channel identity verification/linking flows.
- [X] Extend confirmation audit records with actor/channel attribution and invalidation reasons.
- [X] Add consent purposes for personalization, locations, reminders, and analytics.
- [X] Add memory enable/disable, export, and deletion operations.
- [X] Add retention rules for messages, locations, and provider responses.
- [X] Fix preference updates to use one database transaction/session.
- [X] Learn implicit preferences only after fulfillment or explicit feedback.

### Primary files/modules

- New identity, consent, and confirmation models/migrations.
- `backend/app/agent/context_builder.py`
- `backend/app/agent/memory_agent.py`
- `backend/app/agent/pipeline.py`
- `backend/app/db/crud.py`
- Telegram and WhatsApp channel handlers.

### Acceptance criteria

- [X] No long-term memory is created without consent.
- [X] Users can view, disable, export, and delete stored personalization.
- [X] One customer can securely link multiple channels.
- [X] Confirmation audit history identifies the actor, channel, and invalidation reason.
- [X] Cancelled/unfulfilled meals do not count as completed taste signals.

### Phase 1 validation evidence

- Backend tests: 49 passed, including identity linking, web identity, privacy-command, retention, and learning-signal coverage.
- Python compilation: passed with the bytecode cache redirected to `/tmp`.
- Frontend production build (including Next.js compile/type validation): passed.
- Alembic migration chain: `0009` is the single head.
- Alembic PostgreSQL offline SQL generation for both `0008 -> 0009` and `0009 -> 0008`: passed.
- Isolated PostgreSQL 16 migration upgrade, rollback to `0008`, and re-upgrade to `0009`: passed.
- Live PostgreSQL service checks passed for tenant-scoped identity uniqueness, append-only consent resolution, one-time link replay rejection, consent-gated fulfillment learning, cancelled-order exclusion, learning idempotency, export, disable, and deletion.
- Existing-customer profiles cannot be claimed by supplying a matching phone number; cross-channel joins require a one-time link code.
- Link tokens are stored only as hashes, expire, are target-channel-bound, and cannot be reused.

## Phase 2 — Domain Model and State Machine

**Goal:** Create one consistent model for plans, schedules, orders, fulfillment, and provider execution.

**Implementation status:** Complete in the working tree on 2026-10-15. Migration `0011` must be applied after website-auth migration `0010` before the updated application starts.

### Work

- [X] Add meal plan, template, routine, budget, provider attempt, and event tables.
- [X] Introduce a central order transition service.
- [X] Separate meal scheduling state from provider order state.
- [X] Add immutable order and schedule event histories.
- [X] Add desired time, ready-by time, provider slot, and submission window fields.
- [X] Migrate current order statuses safely.
- [X] Distinguish cancellation, delivery failure, provider rejection, and expiry.
- [X] Remove or quarantine legacy duplicate agent implementations after parity tests.

### Acceptance criteria

- [X] Every transition is validated and audited.
- [X] Invalid transitions fail consistently.
- [X] Existing orders migrate without data loss.
- [X] Delivery failures remain distinct from cancellations.
- [X] The current agent entry point is unambiguous.

### Phase 2 validation evidence

- Backend tests: 61 passed, including eight dedicated domain/state-machine tests.
- Python compilation: passed with the bytecode cache redirected outside the source tree.
- Alembic migration chain: `0011` is the single head.
- Alembic PostgreSQL offline upgrade and downgrade SQL generation for `0010 <-> 0011`: passed.
- Legacy `scheduled_for` values are retained and copied into `desired_for`; legacy state normalization is recorded in the order event history.
- Order and schedule event histories are protected by ORM guards and PostgreSQL update/delete rejection triggers.
- Telegram and WhatsApp use the canonical pipeline; legacy agent entry points fail closed with an explicit migration error.

## Phase 3 — Reliable Scheduling Engine

**Goal:** Ensure scheduled meals survive restarts and execute exactly once logically.

**Implementation status:** Complete in the working tree on 2026-10-15. Migration `0012` must be applied after Phase 2 migration `0011` before updated API, worker, or Beat processes start.

### Work

- [X] Create durable scheduled-job records.
- [X] Add worker leases/locks and lease expiry recovery.
- [X] Add attempt counts, bounded exponential backoff, and `next_retry_at`.
- [X] Add submission deadlines and late-order policy.
- [X] Add reminders before confirmation and fulfillment.
- [X] Add failed-schedule and reschedule-required notifications.
- [X] Add dead-letter and manual-review queues.
- [X] Make outbox processing safe across multiple replicas.
- [X] Add scheduler health and lag metrics.

### Acceptance criteria

- [X] Worker restarts lose no schedules.
- [X] Concurrent workers do not duplicate logical execution.
- [X] Every failure records a structured reason and next action.
- [X] Jobs beyond retry limits enter manual review.
- [X] Reminder delivery is auditable.

### Phase 3 validation evidence

- Backend tests: 71 passed, including ten dedicated scheduler reliability tests.
- Python compilation: passed with the bytecode cache redirected outside the source tree.
- Alembic migration chain: `0012` is the single head.
- Alembic PostgreSQL offline upgrade and downgrade SQL generation for `0011 <-> 0012`: passed.
- Activation and reminder jobs are created atomically with a scheduled order and use unique logical keys.
- Scheduler and outbox claims use leased `FOR UPDATE SKIP LOCKED` rows for multi-worker safety.
- Every scheduler outcome appends an immutable attempt record; exhausted work enters the tenant-scoped manual-review queue.
- Manager health reporting includes heartbeat freshness, scheduling lag, outbox lag, and dead-letter counts.

## Phase 4 — Provider Capability Registry and Modes

**Goal:** Make provider behavior explicit, testable, and safe.

**Implementation status:** Complete in the working tree on 2026-10-16. Migration `0013` must be applied during deployment before the updated application starts. External providers remain conservatively in Mode A until their live capabilities and exact tool schemas are verified.

### Work

- [X] Define a provider adapter interface.
- [X] Add a versioned provider capability registry.
- [X] Record tool schema/version hashes and verification timestamps.
- [X] Add capability states: `UNVERIFIED`, `VERIFIED`, `DEGRADED`, `DISABLED`.
- [X] Implement Mode A discovery.
- [X] Implement Mode B user-controlled handoff.
- [X] Enable Mode C only for verified capabilities.
- [X] Persist every provider attempt and response receipt.
- [X] Add status-before-retry reconciliation.
- [X] Add provider cancellation and tracking adapters where supported.
- [X] Redact tokens, addresses, phone numbers, and payment fields from logs.

### Capability fields

- Discovery/search
- Menu lookup
- Live price verification
- Scheduled ordering
- Scheduling horizon
- Delivery
- Pickup
- Curbside/drive-through
- Cancellation
- Order status lookup
- User-owned authentication
- Stable idempotency support

### Acceptance criteria

- [X] Unsupported capabilities cannot reach transactional tools.
- [X] Users can see whether a provider is discovery, handoff, or transactional.
- [X] Mode C always returns a verified receipt or enters `UNKNOWN`.
- [X] An `UNKNOWN` attempt is reconciled before any retry.

### Phase 4 validation evidence

- Provider adapters, capability policy, schema inventory, redaction, Mode C receipt handling, API exposure, and migration invariants have 12 dedicated unit tests.
- Backend tests: 83 passed.
- Python source compilation passes with the bytecode cache redirected outside the source tree.
- Alembic migration chain: `0013` is the single head.
- Alembic PostgreSQL offline upgrade and downgrade SQL generation for `0012 <-> 0013` passes.
- Tool-schema changes automatically create a new `UNVERIFIED` discovery version instead of inheriting transactional authority.
- Transactional tools require verified Mode C plus schedule and idempotency fields on the live tool schema.
- Missing order IDs and ambiguous provider results enter durable `UNKNOWN` state and are reconciled through status lookup before retry.
- Provider payloads and errors are redacted before persistence or application logging.

## Phase 5 — Meal Planner and Recurring Routines

**Goal:** Deliver the primary V2 customer value: structured meal planning rather than one-off ordering.

**Implementation status:** Complete in the working tree on 2026-10-16. Migration `0014` must be applied before the updated API and workers start.

### Work

- [X] Add daily and weekly meal plans.
- [X] Add breakfast, lunch, snacks, and dinner meal slots.
- [X] Generate recommendations using diet, allergies, health goals, budget, time, and history.
- [X] Add editable meal-plan drafts.
- [X] Add saved meal and full-day templates.
- [X] Add recurrence rules for weekdays, weekends, and custom schedules.
- [X] Add pause, skip, resume, edit, and reschedule actions.
- [X] Materialize recurrence instances within a controlled horizon.
- [X] Add confirmation and reminder messages over Telegram and WhatsApp.
- [X] Prevent allergy conflicts with deterministic validation.

### Acceptance criteria

- [X] A user can create and modify a complete weekly plan.
- [X] A routine can be paused or skipped without creating duplicate jobs.
- [X] Allergy conflicts are blocked by code.
- [X] Budget and diet constraints are visible in the generated plan.
- [X] Every planned meal remains a draft until confirmed.

### Phase 5 validation evidence

- Backend tests: 96 passed, including 13 dedicated Phase 5 tests.
- Python source compilation passes with the bytecode cache redirected outside the source tree.
- Alembic migration chain: `0014` is the single head.
- Alembic PostgreSQL offline upgrade and downgrade SQL generation for `0013 <-> 0014` passes.
- Complete-plan generation creates four standard slots per day and twenty-eight slots per week.
- Diet and allergy checks are deterministic; allergy-constrained recommendations fail closed without verified metadata.
- Plan constraints, recommendation reasons, estimated meal costs, and total budget remain visible in stored and API responses.
- Routine occurrence keys, row locking, exceptions, and versioned reminder keys prevent duplicate materialization and jobs.
- Planning confirmation receipts are stable hashes and do not create orders or call provider tools.
- Reminder jobs require current consent and use the durable scheduler's retry and manual-review behavior.

## Phase 6 — Maps and Route-Aware Fulfillment

**Goal:** Provide useful discovery for home delivery and low-detour pickup without overstating provider support.

**Implementation status:** Complete in the working tree on 2026-10-16. Migration `0015` must be applied before the updated API starts.

### Work

- [X] Add consented saved locations and routes.
- [X] Add retention and delete controls for route/location history.
- [X] Score restaurants by route detour, current status, rating confidence, diet, and budget.
- [X] Evaluate the requested future time separately from current opening state.
- [X] Verify the selected restaurant and fulfillment type through the provider adapter.
- [X] Add route-change and closed-restaurant recovery.
- [X] Fall back to ordinary scheduled pickup when drive-through cannot be verified.

### Acceptance criteria

- [X] Maps is never treated as proof of checkout eligibility.
- [X] Drive-through is offered only after provider-side verification.
- [X] Location and route data are stored only with consent.
- [X] Failed route discovery provides a safe manual selection path.

### Phase 6 validation evidence

- Backend tests: 108 passed, including 12 dedicated Phase 6 tests.
- Python source compilation passes with the bytecode cache redirected outside the source tree.
- Alembic migration chain: `0015` is the single head.
- Alembic PostgreSQL offline upgrade and downgrade SQL generation for `0014 <-> 0015` passes.
- Dedicated deterministic tests cover future opening periods, explainable route scoring,
  provider restaurant matching, drive-through verification, pickup fallback, recovery reasons,
  API surface, consent-linked models, and migration rollback structure.
- Maps responses expose current and future opening inputs separately; every candidate is marked
  ineligible until provider verification succeeds.
- Saved locations, routes, and retained discovery history reference the granting consent row,
  expire automatically, and are physically removed when location consent is withdrawn.
- Provider verification requires a verified capability version, exact restaurant evidence,
  explicit future availability, and the requested fulfillment type. Inconclusive results return
  a manual-selection path.
- Migration `0015` follows `0014` and supports ordered upgrade/downgrade.

## Phase 7 — Customer Dashboard and Analytics

**Goal:** Give customers control over plans, routines, consent, and spending while improving staff visibility.

**Implementation status:** Complete in the working tree on 2026-10-16. Migration `0016` must be applied before the updated API starts.

### Customer dashboard

- [X] Weekly meal calendar.
- [X] Upcoming confirmed meals.
- [X] Draft meal plans.
- [X] Templates and recurring routines.
- [X] Order and provider-attempt history.
- [X] Diet, allergy, and health preferences.
- [X] Consent and privacy controls.
- [X] Budget limits and spending summaries.
- [X] Provider mode/capability status.
- [X] Reminder, cancellation, skip, and reschedule controls.

### Staff analytics

- [X] Scheduled volume by desired meal time.
- [X] Submission and fulfillment success rates.
- [X] Scheduler lag and missed-window metrics.
- [X] Provider failure and unknown-outcome rates.
- [X] Completed revenue by fulfillment date.
- [X] Delivery, pickup, and drive-through performance.
- [X] Cancellation and rescheduling reasons.

### Acceptance criteria

- [X] Customer data is scoped to the authenticated customer.
- [X] Staff data is scoped to the authenticated tenant.
- [X] Analytics use scheduled, submitted, and completed timestamps correctly.
- [X] Budget totals reconcile with the immutable spending ledger.

### Phase 7 validation evidence

- Backend tests: 116 passed, including 8 dedicated Phase 7 tests.
- Production Next.js build passes with all customer and staff routes prerendered or compiled successfully.
- Python source compilation passes; migration `0016` is the single Alembic head.
- Alembic PostgreSQL offline upgrade and downgrade SQL generation for `0015 <-> 0016` passes.
- The authenticated web customer is linked to one tenant-scoped customer profile and verified account channel.
- Customer dashboard queries constrain every plan, schedule, routine, order, attempt, budget, consent, and ledger read to the resolved customer scope.
- Staff operational analytics require a manager session and constrain every query to the authenticated restaurant tenant.
- Orders record distinct submission and completion timestamps; completed customer orders append one immutable ledger entry.
- PostgreSQL and SQLAlchemy both reject spending-ledger updates and deletes.
- Customer controls cover preferences, budget, plan confirmation, reminder consent, routine pause/resume, occurrence skip/reschedule, and eligible order cancellation.
- Staff analytics report desired-time demand, scheduler lag, missed windows, provider outcomes, fulfillment-mode performance, completion revenue, and lifecycle reasons.

## Phase 8 — Production Validation and Public Release

**Goal:** Prove that the product is safe, recoverable, observable, and supportable.

**Engineering status (17 October 2026):** Complete. The fail-closed release gate,
operational readiness API, structured logging, production idempotency behavior,
timezone hardening, privacy notice, validation tooling, disposable dependency stack,
and operator runbooks are implemented. Public transactional release remains in safe
mode until production-specific evidence is collected and approved; unchecked items
below are deliberately operational attestations, not unfinished application code.

### Automated validation

- [X] Explicit confirmation and stale-confirmation tests.
- [X] Duplicate message, order, payment, and webhook tests.
- [X] Scheduler restart and concurrent-worker tests.
- [X] Timezone, DST, naive timestamp, and invalid-time tests.
- [X] Provider timeout and unknown-outcome reconciliation tests.
- [X] Cancellation and rescheduling tests.
- [X] Stale menu, unavailable item, and price-change tests.
- [X] Pickup and drive-through eligibility tests.
- [X] Tenant-isolation and authorization tests.
- [X] Memory consent, export, retention, and deletion tests.
- [X] Budget and recurring-routine tests.
- [X] API integration tests with PostgreSQL and Redis (local validation).
- [X] Frontend unit and accessibility tests (local validation).
- [ ] Frontend end-to-end tests against a candidate deployment.
- [ ] Provider contract and live sandbox verification where available.

### Operational readiness

- [X] Database backup and restore drill (local validation).
- [X] Migration forward/rollback rehearsal (local validation).
- [X] Secret rotation procedure.
- [ ] Error monitoring and structured alerts.
- [ ] Scheduler lag, queue depth, and provider-health dashboards.
- [X] Customer-support and manual-review workflow.
- [X] Incident response and provider-disable runbook.
- [X] Privacy policy and data deletion workflow.
- [X] Bounded load and rate-limit testing (local validation).
- [X] Deployment smoke test (local validation).
- [ ] Candidate deployment rollback rehearsal.

### Public release gate

Public transactional release is allowed only when:

- [ ] Every P0 and P1 finding is closed or explicitly risk-accepted.
- [X] All migrations and rollback procedures pass local rehearsal.
- [X] Confirmation, idempotency, and reconciliation tests pass locally.
- [X] Tenant isolation and public tracking security tests pass locally.
- [X] Backup and restore are proven locally.
- [ ] Monitoring and alerts are active.
- [ ] Provider claims match verified capabilities.
- [X] A provider defaults safely to Mode A when its status is uncertain.

## Phase 9 — Nutrition-Aware Home Meals and On-Route Pickup

**Status:** Partially implemented and locally validated. The prototype supports provenance-aware
menu nutrition fields, a verified-facts-only daily planner, nutrition summaries, web and chat
plan review/editing, handoff of confirmed planned meals into existing individual order flows from
both chat and the customer web dashboard, and route-summary arrival estimates. It does not yet
have independently reviewed nutrition data, provider preparation-time estimates, or complete
route chat journeys.

**Goal:** Help customers plan a complete day of meals around their dietary and nutrition goals,
order chosen meals for home delivery at suitable times, and make travel pickup practical by
timing collection around the route.

### Nutrition-aware home meals

- [x] Define the five daily targets, validation rules, and informational-only user-facing
  limitations.
- [ ] Review nutrition guidance and supported claims with qualified nutrition expertise.
- [x] Define menu-level nutrition fields for energy, protein, carbohydrates, fat, and fibre,
  including source, verification status, serving size, and last-updated time.
- [x] Add manager entry and shape validation for source-labelled menu nutrition; do not silently
  generate or treat model estimates as verified facts.
- [ ] Add an independently reviewed nutrition-ingestion process and restaurant dataset. Current
  `VERIFIED` status is a manager assertion, not independent scientific verification.
- [x] Extend the daily planner to recommend a complete day's meals against the customer's
  selected nutrition goals, while preserving diet, allergy, availability, and budget constraints.
- [x] Show daily totals, target gaps, serving assumptions, and missing or estimated values
  clearly; never claim the plan guarantees a health outcome.
- [x] Let the customer review and adjust menu choices and future meal times in the customer web
  dashboard and WhatsApp/Telegram planning commands.
- [x] Let a customer request a confirmed planned meal in WhatsApp or Telegram and hand it to the
  existing individual order flow with its planned future time; require the normal exact-order
  review and explicit confirmation, and reject stale or changed menu data.
- [x] Add direct individual-order handoff and confirmation from the customer web dashboard using
  the same confirmation, provider-capability, idempotency, and reconciliation gates. Review and
  explicit confirmation are separate requests that reuse the chat confirmation challenge,
  snapshot comparison, and idempotency key; dashboard checkout covers the in-house kitchen and
  fails closed to the chat conversation for any external provider.
- [x] Support plan viewing/editing in WhatsApp, Telegram, and the customer web experience.
- [x] Keep the chat order-confirmation conversation on the existing explicit-confirmation and
  provider-capability safety gates.
- [x] Add unit coverage for target validation, verified-facts-only planning, nutrition summaries,
  manager entry validation, chat plan editing, and the planned-meal order handoff guards.
- [x] End-to-end tests against PostgreSQL for the web order entry point: changed menu data,
  stale and reused confirmations, duplicate requests, unavailable items, and the schedule link.
- [ ] Extend those end-to-end tests to estimated/unknown nutrition values, allergy constraints,
  and provider failures, and to the chat order entry point.

### Route pickup experience

- [x] Return a timezone-aware candidate arrival estimate from Google route-summary duration when
  available. This is advisory travel time, not a food-ready time.
- [ ] Add a dependable arrival uncertainty window and present it in the route conversation when
  route and detour confidence data support one.
- [ ] Obtain a restaurant/provider preparation-time estimate where supported and use it to
  suggest when to place the order; do not promise readiness when timing data is unavailable.
- [x] Keep the customer's explicit future pickup time as the checkout requirement; a route ETA
  never substitutes for the confirmed scheduled time or a provider readiness estimate.
- [ ] Support the full route-pickup conversation in WhatsApp and Telegram: origin/destination
  or shared current location, candidate selection, meal selection, timing, and explicit order
  confirmation.
- [x] Add unit coverage for route ETA conversion and route-summary handling.
- [ ] Complete end-to-end tests for route changes, preparation-time uncertainty, pickup
  verification, fallback behavior, and WhatsApp/Telegram journeys.

### Phase 9 acceptance criteria

- [x] The planner never invents nutrition facts or labels unverified values as verified.
- [x] A meal plan alone never creates a delivery or pickup order.
- [x] The web order entry point has end-to-end evidence for the immutable confirmation,
  idempotency, and provider-capability controls (`tests/integration/test_phase9_web_order_end_to_end.py`,
  5 passed against local PostgreSQL on 17 October 2026).
- [ ] The chat order entry point has the same end-to-end evidence, and both have reconciliation
  evidence for an external provider once one is verified.
- [ ] No public transactional provider behavior is enabled before the Phase 8 release gate
  passes for the candidate deployment and provider.
- [x] The weekly cook-away / multi-day home-delivery scheduling use case remains out of scope.

### Phase 9 validation evidence (17 October 2026)

- Backend unit tests: 182 passed, including 19 dedicated to the dashboard order handoff.
- Dashboard order end-to-end tests against local PostgreSQL 16: 5 passed. They cover the stored
  confirmation, the placed order and its linked planned meal, durable job creation, a resubmitted
  confirmation returning the first order, a price change after review being rejected as
  `SNAPSHOT_CHANGED`, an unavailable item blocking review, and an ordered meal being refused a
  second time. They require a disposable database because the order and schedule audit trails are
  append-only by database trigger.
- Frontend unit and accessibility suite: 18 passed. Production Next.js build passed.
- This work found and fixed a latent defect: `app/agent/executor.py` referenced `settings` in
  `place_order` without importing it at module scope, so every in-house order placement raised
  `NameError` at runtime. No earlier test exercised that path.

## 7. Test Strategy

### Unit tests

- State transition matrix.
- Scheduling policy and time conversion.
- Confirmation snapshot hashing and expiry.
- Capability routing.
- Budget calculations.
- Recurrence generation.
- Dietary and allergy policy checks.

### Integration tests

- PostgreSQL transactions and tenant scoping.
- Redis queues, locks, rate limits, and duplicate suppression.
- Celery scheduling, retry, and recovery.
- Outbox concurrency.
- Payment webhook replay.
- Identity linking and consent enforcement.

### Contract tests

- Google Maps request/response mapping.
- Swiggy/Zomato MCP schema discovery.
- Provider capability snapshots.
- Provider receipt and status normalization.
- Handoff URL/session behavior.

### End-to-end tests

- Telegram scheduled home-delivery flow.
- WhatsApp scheduled home-delivery flow.
- Route-aware pickup/drive-through flow.
- Weekly plan creation and confirmation.
- Routine pause, skip, and resume.
- Provider timeout and reconciliation.
- Customer dashboard privacy boundaries.
- Staff tenant boundaries.

## 8. Development Order and Gates

Development proceeds in this order:

1. Phase 0: public release safety.
2. Phase 1: identity, confirmation, and consent.
3. Phase 2: domain model and state machine.
4. Security and migration review gate.
5. Phase 3: durable scheduling.
6. Phase 4: provider capability modes.
7. Provider verification gate.
8. Phase 5: meal planning and routines.
9. Phase 6: route-aware fulfillment.
10. Phase 7: dashboards and analytics.
11. Phase 8: full production validation.
12. Phase 9: nutrition-aware daily home meals and arrival-aware route pickup, behind the
    existing safety and provider-verification gates.

No later phase may bypass an unresolved safety or data-integrity dependency in an earlier phase.

## 9. Documentation Deliverables

Documentation must be updated alongside implementation:

- README with verified capabilities only.
- Architecture and data-flow diagrams.
- Provider capability and limitation matrix.
- Scheduling and state-transition specification.
- API and webhook documentation.
- Privacy, consent, retention, and deletion behavior.
- Deployment and rollback guide.
- Testing guide and release checklist.
- Customer and staff operating guides.

The product-vision PDF may describe future capabilities, but each feature must be labeled as current, experimental, planned, or provider-dependent.

## 10. Definition of Done

A phase is complete only when:

- Its schema changes include tested migrations and rollback considerations.
- Its authorization and privacy behavior are tested.
- Its unit and integration tests pass.
- Its failure and recovery behavior are defined.
- Logs exclude secrets and unnecessary personal data.
- User-facing behavior accurately describes capability limitations.
- Relevant documentation is updated.
- No unresolved P0 defect remains in the phase scope.

## 11. Decision Log

| Date       | Decision                                                                               | Reason                                                                                                                              |
| ---------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-15 | Keep the NourishBot name and food-ordering niche                                       | Preserves the project identity while separating it from NDA-sensitive startup implementation                                        |
| 2026-10-15 | Position the product around planning and scheduling                                    | Scheduled meals are the core differentiator; instant ordering is excluded                                                           |
| 2026-10-15 | Support Telegram, WhatsApp, and a customer dashboard                                   | Gives users conversational and visual planning interfaces                                                                           |
| 2026-10-15 | Keep home delivery and route-aware pickup/drive-through                                | Both fulfillment paths remain central to the product                                                                                |
| 2026-10-15 | Limit providers to NourishBot, Swiggy, and Zomato                                      | Avoids unsupported or invented MCP provider claims                                                                                  |
| 2026-10-15 | Treat Google Maps as discovery only                                                    | Maps cannot prove provider menu, price, scheduling, or checkout support                                                             |
| 2026-10-15 | Default external providers to Mode A until verified                                    | Prevents unsupported transactional claims and unsafe checkout                                                                       |
| 2026-10-15 | Complete the safety foundation before feature expansion                                | Public ordering requires tenant isolation, confirmation, idempotency, and recovery first                                            |
| 2026-10-17 | Plan nutrition-aware full-day home meals and route-timed pickup as the next product phase | Differentiates the customer experience; requires verified nutrition data and reliable provider timing before transactional claims |
| 2026-10-15 | Deploy the Next.js frontend on Vercel, not Azure                                       | Vercel is the selected public web platform; the project root is `frontend`                                                        |
| 2026-10-15 | Keep API, worker, beat, and polling processes on an always-on runtime until refactored | The current long-running processes do not fit request-scoped Vercel Functions without webhook, cron, locking, and lifecycle changes |
