# Vercel Deployment Guide

## Deployment decision

Vercel is the production web target for NourishBot. The deployable Vercel project is the
Next.js application in `frontend/`. Azure is not a deployment target.

The current backend remains a separate always-on runtime because one FastAPI process starts a
continuous outbox worker and Telegram polling, while Celery worker and beat processes activate
scheduled meals and enforce retention. Moving only the HTTP entrypoint to a request-scoped
function would leave these responsibilities without a reliable owner.

## Current production topology

```text
Browser
  |
  v
Vercel: Next.js frontend (`frontend/`)
  | HTTPS + secure WebSocket
  v
Always-on backend runtime
  |-- FastAPI API + WhatsApp webhook + Telegram polling + outbox loop
  |-- Celery worker
  `-- Celery beat
       |-- managed PostgreSQL
       `-- managed Redis
```

This is a platform boundary, not an Azure requirement. Choose a backend runtime that supports
always-on containers, multiple process types, health checks, secure environment variables, and
private or TLS-protected PostgreSQL and Redis connections.

## Create the Vercel project

1. Import this Git repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Keep the detected **Next.js** framework preset.
4. Keep the build command as `npm run build`; Vercel installs from `frontend/package-lock.json`.
5. Do not override the output directory. Next.js and Vercel manage it automatically.
6. Add the variables below separately for Preview and Production.
7. Deploy a preview, verify it against a staging API, then promote or deploy Production.

### Frontend environment variables

| Variable | Preview example | Production example |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `https://api-staging.example.com` | `https://api.example.com` |
| `NEXT_PUBLIC_WS_URL` | `wss://api-staging.example.com/api/v1/ws` | `wss://api.example.com/api/v1/ws` |

These are public browser configuration values. Never put database URLs, bot tokens, API
secrets, signing keys, or provider credentials in a `NEXT_PUBLIC_*` variable.

On the backend, set `ALLOWED_ORIGINS` to exact trusted frontend origins. Production should
include the stable Vercel/custom domain. If arbitrary preview URLs need API access, do not use a
wildcard with credentials; validate preview origins deliberately or use a stable staging domain.

## Release verification

Before moving traffic to a Vercel deployment:

1. Run `npm ci && npm run build` in `frontend/`.
2. Confirm the Preview deployment loads the landing page, login, and authenticated workspaces.
3. Confirm browser API calls use HTTPS and no mixed-content errors occur.
4. Confirm the browser opens the configured `wss://` endpoint and receives authorized events.
5. Confirm login cookies work with the configured domain and `SameSite` policy.
6. Confirm the backend `/api/ready` endpoint reports PostgreSQL and Redis healthy.
7. Confirm Celery worker and beat are running and a staged scheduled meal activates exactly once.
8. Confirm Telegram and WhatsApp callbacks reach the backend, not the frontend deployment.
9. Exercise privacy export/deletion, tracking-token access, and staff tenant isolation.

## Rollback

Use Vercel's deployment history to promote the last known-good frontend deployment. A frontend
rollback does not roll back the database or backend containers. Keep API changes backward
compatible across at least the active and previous frontend versions.

For a backend rollback:

1. Stop new releases, but keep the scheduler and workers observable.
2. Restore the prior container image only if its database contract is compatible.
3. Use the tested Alembic downgrade only when the migration runbook explicitly permits it.
4. Re-run `/api/ready`, one scheduled-order activation, and the affected channel flow.

## Requirements before a full-Vercel backend migration

Vercel supports FastAPI applications and WebSocket connections, but the present backend cannot
be moved unchanged. Complete all of the following first:

- Add a request-scoped FastAPI entrypoint that does not launch continuous work during lifespan.
- Replace Telegram polling with a verified webhook.
- Move outbox delivery to an idempotent job endpoint or an external worker.
- Replace Celery beat triggers with authenticated cron endpoints or an external scheduler.
- Protect every scheduled job with database/Redis locking, deduplication, and retry-safe state.
- Keep PostgreSQL and Redis external and regionally close to the Vercel functions.
- Treat WebSockets as connection-scoped functions and use Redis/pub-sub for cross-instance state.
- Re-test MCP subprocess/network behavior inside the Vercel runtime and function limits.
- Prove that activation, retries, reconciliation, and retention continue when a function is
  retried, overlaps, times out, or receives duplicate cron delivery.

The existing activation cadence is once per minute. Vercel Hobby cron jobs cannot provide that
frequency; use a Vercel plan that supports the required schedule or retain an external scheduler.

## Official references

- [Vercel monorepo root directories](https://vercel.com/docs/monorepos)
- [FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi)
- [Vercel Cron Jobs](https://vercel.com/docs/cron-jobs)
- [Cron usage and pricing](https://vercel.com/docs/cron-jobs/usage-and-pricing)
- [Managing Cron Jobs](https://vercel.com/docs/cron-jobs/manage-cron-jobs)
- [WebSockets on Vercel Functions](https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections)
