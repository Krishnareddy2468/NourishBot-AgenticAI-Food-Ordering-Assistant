# Incident, Provider Disable, and Customer Support Runbook

## Severity

- P0: cross-tenant exposure, unauthorized transaction, confirmed duplicate charge/order, or unrecoverable data loss.
- P1: missed scheduled fulfillment at scale, payment/webhook integrity failure, provider outcomes stuck unknown, backup failure, or monitoring outage.
- P2: isolated fulfillment failure with a safe manual workaround.

For P0/P1, stop external mutations by setting `PUBLIC_TRANSACTIONAL_RELEASE=false`. If provider-specific, append a `DISABLED` capability version for that provider. Do not delete capability history or provider receipts.

## First response

1. Assign an incident commander and start an UTC timeline.
2. Preserve request IDs, order references, provider attempt IDs, scheduler job IDs, and deployment SHA. Do not copy customer content into the incident channel.
3. Check `/api/ready`, `/api/v1/operations/readiness`, scheduler health, provider attempts in `UNKNOWN`, and the manual-review queue.
4. Stop the unsafe action while keeping status lookup and cancellation recovery available.
5. Reconcile uncertain provider outcomes before any retry.
6. Notify affected customers with verified facts only. Never state that an order was cancelled, paid, or fulfilled until the corresponding system confirms it.

## Manual-review workflow

- Scheduler: inspect `/api/v1/scheduler/manual-review` and attempt history. Activation jobs whose meal is no longer scheduled must be rescheduled, not blindly retried.
- Provider: inspect redacted receipts, use the reconcile action, and escalate unknown results to the provider. A retry is prohibited while `reconciliation_required=true`.
- Payment: match provider payment and order IDs. Webhook replays are idempotent and should return success without repeating side effects.
- Fulfillment: confirm whether food preparation started, record the distinct cancellation/failure reason, and offer a safe reschedule or refund path.

Every manual action needs an operator identity, timestamp, reason, previous state, resulting state, and customer communication record.

## Provider disable sequence

1. Close `PUBLIC_TRANSACTIONAL_RELEASE` globally if scope is uncertain.
2. Add a provider capability version with `DISABLED` state or downgrade to discovery/handoff.
3. Reconcile all `SUBMITTING` and `UNKNOWN` attempts.
4. Keep read-only discovery disabled too if response integrity is uncertain.
5. Verify the customer UI reports safe mode and makes no placement claim.
6. Re-enable only after a new schema hash, sandbox contract run, monitoring verification, and release approval.

## Recovery and review

After containment, verify tenant boundaries, immutable histories, ledger reconciliation, queue recovery, and customer outcomes. Record root cause, contributing controls, detection gap, corrective owner, deadline, and regression test. A P0/P1 cannot be closed by deleting evidence.
