# Production Release Runbook

This runbook is the operational contract for Phase 8. A successful code build is not permission to enable external transactions. `PUBLIC_TRANSACTIONAL_RELEASE` stays `false` until the evidence gate passes in production.

## Roles and separation of duties

- Release lead: owns the deployment decision and evidence file.
- Database operator: performs backup, restore, and migration rehearsals.
- Incident commander: owns active incidents and rollback decisions.
- Provider owner: verifies exact provider schemas and sandbox behavior.
- Privacy owner: handles export/deletion requests and retention evidence.

No person should approve their own unresolved P0/P1 risk acceptance.

## Pre-deployment gate

1. Freeze the candidate commit and record its full Git SHA.
2. Run backend unit tests and frontend tests/build.
3. Start disposable PostgreSQL and Redis with `make phase8-up`.
4. Run `make phase8-migrate`, `make phase8-integration`, `make phase8-migration-rehearsal`, and `make phase8-backup-restore`.
5. Run the deployed smoke and bounded load probes.
6. Verify scheduler, outbox, provider, error-rate, latency, database, and Redis alerts in the production monitoring system. Trigger and acknowledge a test alert.
7. Record evidence using `docs/release/phase8-evidence.template.json` and evaluate it with `python scripts/check_release_gate.py <evidence.json>`.
8. Keep both `MCP_TRANSACTIONAL_ENABLED=false` and `PUBLIC_TRANSACTIONAL_RELEASE=false` if the command exits non-zero.

`ALLOW_SAFE_MODE_RELEASE` approves only discovery/handoff operation. A provider may
expose mutation tools only when the result is `ALLOW_VERIFIED_TRANSACTIONAL_RELEASE`
and that exact provider appears in `transactional_providers`.

## Database backup and restore drill

The backup script uses PostgreSQL custom format, writes a SHA-256 sidecar, and never prints credentials. The restore command refuses to target the source database and requires the exact disposable target name.

```bash
python scripts/postgres_backup_restore.py backup \
  --source "$DATABASE_URL_SYNC" \
  --output /secure/backups/nourishbot-$(date +%Y%m%d%H%M%S).dump

python scripts/postgres_backup_restore.py drill \
  --source "$DATABASE_URL_SYNC" \
  --target "$RESTORE_DRILL_DATABASE_URL_SYNC" \
  --backup /secure/backups/<backup>.dump \
  --confirm-target-db <disposable_database_name> \
  --report /secure/evidence/restore-report.json
```

Store backups in encrypted object storage with restricted service-account access and a lifecycle policy. A successful command on the production source and isolated restore target is required every 30 days. Verify exact per-table row counts and retain the generated report.

## Migration rehearsal and deployment

1. Restore the most recent production backup to a staging database.
2. Run `alembic current` and record the starting revision.
3. Run `alembic upgrade head` and the integration suite.
4. Run `alembic downgrade <previous_revision>`, verify service compatibility, then upgrade to head again.
5. Take a fresh production backup.
6. Put writes into maintenance mode if the migration notes require it.
7. Apply migrations once, then deploy API, workers, Beat, and frontend from the same candidate SHA.
8. Run `scripts/smoke_test.py` and observe alerts, queue lag, and error rate for at least one full scheduling interval.

Migration `0016` is the expected head for the current release. Change `EXPECTED_ALEMBIC_HEAD` only with the next reviewed migration.

## Secret rotation

Rotate one credential class at a time. Prepare the new value, deploy dual-key verification where supported, switch producers, verify traffic, revoke the old value, then record completion without storing the secret.

Rotation order:

1. Database and Redis credentials.
2. JWT signing key. Existing sessions expire when the old key is removed; announce this impact.
3. Razorpay webhook and API secrets.
4. WhatsApp app secret, verify token, and access token.
5. Google OAuth and Maps credentials.
6. Gemini and Telegram credentials.
7. MCP OAuth cache and provider credentials.

Immediately rotate any value that appears in logs, source control, support messages, or an unapproved device. Confirm that Redis endpoint logging remains credential-redacted.

## Monitoring and alert thresholds

Scrape or poll:

- `/api/ready` for PostgreSQL and Redis availability.
- `/api/v1/operations/readiness` for the authenticated tenant release view.
- `/api/v1/scheduler/health` for worker heartbeat, scheduler lag, outbox lag, and dead letters.
- `/api/v1/analytics/operations` for provider unknown/failure rates and fulfillment health.

Critical alerts: readiness failure, stale scheduler heartbeat, scheduler lag above five minutes, any provider unknown outcome, any dead letter, migration mismatch, sustained 5xx errors, or external transaction execution while the release gate is closed. Warning alerts: outbox lag above five minutes, rising cancellation rate, or provider degradation.

## Smoke, load, and rollback

```bash
python scripts/smoke_test.py --api https://api.example --frontend https://app.example --report smoke.json
python scripts/load_probe.py --url https://api.example/api/health --requests 500 --concurrency 20 --report load.json
```

The load probe is intentionally read-only and bounded. Use an approved load-testing environment for transactional workflows.

Rollback triggers include migration errors, confirmation/idempotency regression, incorrect tenant scope, sustained readiness failure, unknown provider outcomes without reconciliation, or missed schedule activation. Close `PUBLIC_TRANSACTIONAL_RELEASE`, disable the affected provider capability, stop new write traffic, preserve logs/evidence, roll back application code, and use the rehearsed database procedure. Never retry an unknown provider submission before reconciliation.

## Evidence retention

Retain the candidate SHA, test logs, migration transcripts, restore report, smoke/load reports, alert screenshots or event IDs, provider schema hashes, approvals, and rollback result. Evidence must not contain secrets, full provider payloads, addresses, phone numbers, or chat content.
