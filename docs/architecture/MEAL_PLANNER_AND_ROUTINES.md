# Meal Planner and Recurring Routines

Phase 5 adds a deterministic planning layer above ordering. A meal plan is not an order, and confirming a plan never calls a provider or creates an order. Transactional ordering still requires the separate immutable order-confirmation workflow.

## Plan model

A daily plan contains one day and a weekly plan contains seven consecutive days. Each day has exactly four standard slots:

- Breakfast at 08:30 local time.
- Lunch at 13:00 local time.
- Snacks at 17:00 local time.
- Dinner at 20:00 local time.

The times and selected meals are editable before confirmation. Each `MealSchedule` stores the visible constraints, recommendation reasons, estimated cost, diet and allergen evidence, timezone, revision, and state. A complete daily plan has four active meals; a complete weekly plan has twenty-eight.

New plans and routine occurrences are always `DRAFT`. Confirmation creates a stable SHA-256 snapshot receipt and changes the relevant meals to `CONFIRMED`. Editing any meal invalidates the plan receipt and returns every meal in that plan to `DRAFT`. No order is created by either plan or individual-meal confirmation.

## Deterministic recommendations

The recommendation service combines:

- Explicit diet, allergy, health-goal, budget, and timezone input.
- Stored preferences and fulfilled-order history only when personalization consent is currently granted.
- Live in-house menu availability, stock, prices, types, categories, and tags.
- Meal-period tags and health-goal tags for deterministic scoring.

Budget is stored on the plan and shown with every response. The generator applies a per-slot cap derived from the total plan budget, and rejects a plan when it cannot produce every required slot without exceeding that budget.

## Allergy boundary

Allergy safety is enforced in application code, not in an LLM prompt. Tags are normalized, including common aliases such as groundnut/peanut and milk/dairy.

A candidate is rejected when:

- Its `allergen:*` or `contains:*` metadata conflicts with a customer allergy.
- The customer has an allergy and the item lacks either `allergen-verified` metadata or a matching `free-from:*` declaration.

This intentionally fails closed. An item without verified metadata cannot be recommended or confirmed for an allergy-constrained plan.

## Templates

A customer can save:

- A single-meal template from one planned meal.
- A full-day template containing that day's four slots.

Templates retain cost, diet, allergen, and constraint evidence. A recurring routine currently requires a single-meal template; full-day templates are available for reuse by plan clients without silently creating four independent routines.

## Routine rules and lifecycle

Supported rules are:

- `DAILY`
- `WEEKDAYS`
- `WEEKENDS`
- `CUSTOM:MON,WED,FRI` using any valid day combination

Each routine has a timezone, local time, version, state, and a bounded materialization horizon. The default horizon is fourteen days and the hard maximum is thirty-one days.

Materialized occurrences use a unique key:

```text
routine:{routine_id}:v{version}:{occurrence_date}
```

The hourly materializer holds row locks with `SKIP LOCKED`, checks existing occurrences across versions, and creates only missing drafts. This makes repeated and concurrent runs idempotent.

Lifecycle actions behave as follows:

- Pause: stops materialization, pauses future drafts, and cancels their pending reminders.
- Resume: restores paused drafts, creates new versioned reminders, and fills only missing horizon dates.
- Skip: records a unique occurrence exception, marks an existing occurrence skipped, and cancels its reminders.
- Reschedule: records a replacement timestamp and updates or materializes the original occurrence without duplicating it.
- Edit: versions the routine, supersedes unconfirmed future drafts, cancels old jobs, and materializes the revised schedule. Previously confirmed meals remain authoritative and are not duplicated.

Every plan, template, routine, occurrence, edit, confirmation, pause, resume, skip, and reschedule action appends a `meal_planning_events` audit record. PostgreSQL rejects updates or deletes to this history.

## Reminders and channels

Draft meals can receive a confirmation reminder twenty-four hours before the planned time. Confirmed meals can receive a planning reminder sixty minutes before the planned time. Both messages explicitly state that no food order has been placed.

Reminders:

- Require current `REMINDERS` consent.
- Use the durable leased scheduler and unique job keys.
- Deliver through the customer's linked Telegram or WhatsApp channel.
- Retry transient failures and enter the existing dead-letter/manual-review path after exhaustion.

Telegram and WhatsApp also support deterministic commands:

```text
plan my day
plan my week vegetarian under ₹2000
show my meal plan
confirm meal plan 12
confirm planned meal 45
pause routine 3
resume routine 3
skip routine 3 2026-10-21
```

## Customer API

These routes require the existing opaque linked-browser customer identity. Tenant and customer ownership are derived from that cookie; clients cannot choose another customer ID.

| Route | Purpose |
|---|---|
| `GET/POST /api/v1/customer/meal-plans` | List plans or generate a complete draft. |
| `GET /api/v1/customer/meal-plans/{id}` | Read a plan with all meals and visible constraints. |
| `PATCH /api/v1/customer/meal-plans/{id}/meals/{meal_id}` | Change a meal, slot, or scheduled time. |
| `POST /api/v1/customer/meal-plans/{id}/confirm` | Confirm one immutable complete-plan snapshot. |
| `POST /api/v1/customer/planned-meals/{id}/confirm` | Confirm one routine-generated draft meal. |
| `GET/POST /api/v1/customer/meal-templates` | List or save meal/full-day templates. |
| `GET/POST /api/v1/customer/meal-routines` | List or create recurring routines. |
| `POST /api/v1/customer/meal-routines/{id}/actions` | Pause, resume, skip, reschedule, or edit a routine. |

## Deployment

Apply Alembic migration `0014` before starting the updated API or workers. Run both Celery worker and beat processes so routine materialization and reminders execute.

The relevant environment settings are:

```text
MEAL_PLAN_MAX_HORIZON_DAYS=31
ROUTINE_MATERIALIZATION_HORIZON_DAYS=14
PLAN_CONFIRMATION_REMINDER_HOURS=24
PLANNED_MEAL_REMINDER_MINUTES=60
```
