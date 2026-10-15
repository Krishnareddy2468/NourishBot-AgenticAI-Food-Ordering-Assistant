# Reliable Scheduler

Phase 3 replaces the scheduled-order table scan with a durable job engine. Scheduler state is stored in PostgreSQL, so API, Celery, or host restarts do not lose future work.

## Durable jobs

Every scheduled order creates three uniquely keyed jobs in the same database transaction as the order:

- `ACTIVATE_ORDER` opens the live fulfillment workflow.
- `CONFIRMATION_REMINDER` asks the customer to review the future meal schedule.
- `FULFILLMENT_REMINDER` notifies the customer shortly before fulfillment.

If safe activation cannot complete, the scheduler creates a `RESCHEDULE_REQUIRED` notification job and exposes the failed job in the manual-review queue.

Migration `0012` backfills these jobs for existing orders whose `schedule_state` is `SCHEDULED`. `dedupe_key` is unique, so migration replay and application retries cannot create duplicate logical jobs.

## Leasing and recovery

Each scheduler tick:

1. Recovers expired `LEASED` jobs.
2. Selects due `PENDING` or `RETRY_WAIT` jobs with `FOR UPDATE SKIP LOCKED`.
3. Commits a time-bounded worker lease before execution.
4. Executes and finalizes each job only when its lease owner still matches.

This allows multiple Celery workers to poll safely. If a worker dies, the lease expires and another tick records `LEASE_EXPIRED` before retrying. Activation is also state-idempotent: a recovered job sees an already activated schedule and finishes without repeating its order transition.

## Retries, deadlines, and manual review

Failures store a bounded error message plus a structured `last_error_code` and `next_action`. Retry delays use bounded exponential backoff:

```text
delay = min(maximum, base × 2^(attempt - 1))
```

Each attempt appends a `ScheduledJobAttempt`. PostgreSQL and SQLAlchemy both reject updates or deletes to attempt history.

When the attempt limit or submission deadline is reached:

- The job moves to `DEAD_LETTER`.
- `next_action` becomes `MANUAL_REVIEW`.
- A `schedule.manual_review_required` outbox event is created.
- A still-waiting activation is expired with a structured reason.
- A durable `RESCHEDULE_REQUIRED` customer notification is queued.

The supported late policy is intentionally conservative: `MANUAL_REVIEW`. NourishBot does not silently submit a meal after its safe submission window.

## Reminder delivery

Routine confirmation and fulfillment reminders require the customer's latest `REMINDERS` consent. Consent-denied reminders finish as `SKIPPED_NO_CONSENT`; this is recorded in the attempt history. Reschedule-required messages are transactional service notices and are not suppressed by reminder preference.

Successful channel delivery is recorded as the attempt outcome `DELIVERED`. Channel unavailability produces `NOTIFICATION_UNAVAILABLE` and follows the normal retry/dead-letter policy.

## Replica-safe outbox

Outbox events now have their own owner, lease expiry, attempt count, bounded backoff, and dead-letter timestamp. API replicas claim rows with `FOR UPDATE SKIP LOCKED`, commit the claim, and finalize only their own leases. Realtime clients receive a stable `event_id` for deduplication.

## Operations API

Managers and administrators can use:

- `GET /api/v1/scheduler/health` for heartbeat age, job counts, scheduler lag, outbox lag, and dead-letter counts.
- `GET /api/v1/scheduler/jobs` for the tenant's recent jobs.
- `GET /api/v1/scheduler/jobs/{job_id}/attempts` for immutable execution and reminder-delivery history.
- `GET /api/v1/scheduler/manual-review` for dead-letter work.
- `POST /api/v1/scheduler/jobs/{job_id}/retry` for an eligible manual retry.

An expired activation must first be rescheduled; the retry endpoint will not bypass the terminal schedule state.

## Deployment

Apply the migration before starting updated API, worker, or Beat processes:

```bash
cd backend
alembic upgrade head
```

Run both a Celery worker and Celery Beat. Beat invokes `run_scheduler_tick` every minute. The API process continues to drain realtime outbox events, and multiple API replicas are safe.

The key environment controls are documented in `backend/.env.example`, including lease durations, attempt limits, reminder lead times, late grace, backoff bounds, and heartbeat staleness.
