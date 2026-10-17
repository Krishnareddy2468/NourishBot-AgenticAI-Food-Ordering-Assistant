# Phase 8 Validation Status

Status date: 17 October 2026

## Outcome

Phase 8 engineering is complete. NourishBot now has a fail-closed public release
gate, operational readiness checks, production-safe provider behavior, structured
request telemetry, dependency-backed integration tooling, recovery scripts, and
release/incident/privacy runbooks.

Public transactional release is **not approved by this repository change**. It
correctly remains in safe discovery or user-controlled handoff mode until evidence
from the actual production environment passes the gate.

## Verified locally

- Backend test suite: 137 passed; the 3 opt-in database integration tests were
  skipped in the default run and passed separately against local PostgreSQL and Redis.
- Frontend unit and accessibility/security suite: 8 passed.
- Production Next.js build: passed, including `/privacy`.
- Smoke test: 6/6 checks passed against the local API and frontend.
- Empty evidence template: correctly returns `KEEP_SAFE_MODE` with a non-zero exit.

## Implemented controls

- External mutations require the provider capability gate, the MCP transaction
  switch, the independent public-release switch, and fresh live-sandbox evidence
  for that exact provider; one provider can never authorize another.
- Production Redis failure closes webhook duplicate claims and request rate
  protection instead of silently degrading.
- Razorpay and WhatsApp deliveries require stable duplicate-suppression keys.
- Naive timestamps require an explicit IANA timezone; invalid zones and DST folds
  or gaps are rejected.
- Logs carry request IDs and route templates without recording identifiers from
  URL paths; provider debug logging is disabled in production.
- The manager readiness endpoint checks migration head, Redis, scheduler heartbeat,
  job/outbox lag, dead letters, unknown provider outcomes, alerts, and evidence.
- Backup/restore tooling checks SHA-256 and exact table row counts, refuses the
  source database as a target, and keeps passwords out of process arguments.

## Local rehearsal (17 October 2026, local PostgreSQL 16 and Redis, not production)

- PostgreSQL/Redis integration suite: 3 passed.
- Migration rehearsal: `upgrade head` → `downgrade base` → `upgrade head` passed.
  This found and fixed a rollback defect in migration 0001 (`restaurants` was dropped
  before its dependents, and `sessions`/`carts` reference each other); it now drops with
  `CASCADE` because every table in that migration is being removed.
- Backup/restore drill: passed (51 tables; checksum and exact row counts matched).
- Smoke test: 6/6 passed after fixing a case-sensitive header lookup in
  `scripts/smoke_test.py`. Bounded load probe: 200 requests at concurrency 10,
  0% errors, p95 10.8 ms.
- Public tracking security tests cover short-token rejection before database access,
  indistinguishable not-found responses for unknown tokens, and exclusion of
  customer contact/address fields; tenant-scoped staff tracking is also tested.
- End-to-end browser testing against a candidate deployment has not been run.

These runs prove the tooling and migrations work; they do not replace evidence from the
candidate deployment listed below.

## Evidence still required from the deployment environment

1. Repeat smoke/load checks against the approved candidate deployment and rehearse
   its rollback procedure.
2. Run frontend browser end-to-end tests against the candidate deployment.
3. Activate production monitoring, deliver and acknowledge a test alert, and verify
   scheduler, queue, and provider-health dashboards.
4. Verify live provider sandbox contracts if transactional mode is requested;
   otherwise record discovery/handoff with mutations disabled.
5. Record the full candidate Git SHA, close or formally accept every P0/P1 finding,
   complete public-tracking security validation, and obtain release approval.

Use `docs/release/phase8-evidence.template.json` for those attestations and run:

```bash
python scripts/check_release_gate.py /secure/evidence/phase8.json
```

Do not enable `PUBLIC_TRANSACTIONAL_RELEASE` unless that command exits successfully.
