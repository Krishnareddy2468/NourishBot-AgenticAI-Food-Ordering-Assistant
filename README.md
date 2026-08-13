# Dzukku — Agentic AI Food Ordering Assistant

A restaurant operating system built around a conversational agent. Guests order through
Telegram; staff run the floor, kitchen, and admin desk through a Next.js web app. Both
sides talk to one FastAPI backend over REST and WebSockets.

The agent can take orders against the in-house menu, or hand off to Zomato/Swiggy via
Model Context Protocol (MCP) when live third-party ordering is enabled.

---

## Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running with Docker](#running-with-docker)
- [API surface](#api-surface)
- [Database and migrations](#database-and-migrations)
- [Testing and CI](#testing-and-ci)
- [Project status](#project-status)
- [Further documentation](#further-documentation)

---

## Architecture

A single FastAPI process boots the REST API, the Telegram bot (polling, on the same event
loop so asyncpg sessions stay bound to one loop), the WebSocket hub, and the outbox worker.
Celery worker and beat run alongside for notifications and scheduled jobs.

```
Telegram ──┐
           ├──▶  FastAPI (app.api.main:api)  ──▶  PostgreSQL  (SQLAlchemy async + Alembic)
Next.js  ──┘         │                        ──▶  Redis       (session cache + Celery broker)
   REST + WS         │
                     ├──▶  Agent pipeline  ──▶  OpenAI
                     ├──▶  MCP agent       ──▶  Zomato / Swiggy (LangGraph + mcp-remote)
                     ├──▶  Outbox worker   ──▶  WebSocket broadcast
                     └──▶  Celery worker / beat
```

### The agent pipeline

`backend/app/agent/pipeline.py` implements five stages per incoming message. The governing
rule is that **the LLM never writes to the database** — it proposes, deterministic code commits.

| Stage | Module | Responsibility |
|-------|--------|----------------|
| 1. Context | [context_builder.py](backend/app/agent/context_builder.py) | Read a full DB snapshot for the chat |
| 2. Plan | [planner.py](backend/app/agent/planner.py) | LLM returns a JSON plan: goal, slots, proposed actions |
| 3. Execute | [executor.py](backend/app/agent/executor.py) | Deterministic tool runner, applies policy checks, commits |
| 4. Verify | [verifier.py](backend/app/agent/verifier.py) | Re-reads the DB, validates totals and availability |
| 5. Respond | [responder.py](backend/app/agent/responder.py) | Turns verified facts into a friendly reply |

Tool-call loops are bounded by `AGENT_MAX_ITERATIONS` (default 6). A conversation state
machine ([state_machine.py](backend/app/agent/state_machine.py)) tracks progress through
ordering and reservation flows. LLM calls fall back through a three-model chain
(`OPENAI_PRIMARY_MODEL` → `OPENAI_FALLBACK_MODEL` → `OPENAI_FALLBACK_2_MODEL`).

### Third-party ordering (MCP)

When `MCP_ENABLED=true`, a LangGraph ReAct agent binds MCP tools from Zomato and Swiggy,
bridged from HTTP to stdio by `npx mcp-remote`. This is a **single-tenant** mode: orders go
to the bot owner's Zomato/Swiggy account using a shared OAuth token cache. With MCP disabled
(the cloud default), guests who pick Zomato or Swiggy get redirect links to those apps instead.

### Staff frontend

Next.js 14 App Router, one route per role:

| Route | Purpose |
|-------|---------|
| [/](frontend/app/page.jsx) | Role selector |
| [/login](frontend/app/login/page.jsx) | JWT sign-in |
| [/admin](frontend/app/admin/page.jsx) | Menu, staff, reports |
| [/waiter](frontend/app/waiter/page.jsx) | Table and order management |
| [/kitchen](frontend/app/kitchen/page.jsx) | Kitchen display system |
| [/track/[orderRef]](frontend/app/track/[orderRef]/page.jsx) | Guest-facing order tracking |

Authorization is role-based: `ADMIN`, `MANAGER`, `WAITER`, `KITCHEN`, `CASHIER`, `DRIVER`
(see [deps.py](backend/app/auth/deps.py)). Live updates arrive over `/api/v1/ws`;
[offlineOrderQueue.js](frontend/services/offlineOrderQueue.js) buffers writes when the
connection drops.

---

## Repository layout

```
backend/            FastAPI + Telegram bot + agent (Python 3.11)
  app/agent/          5-stage pipeline, MCP agent, policies, persona
  app/api/routes/     REST routers, all under /api/v1
  app/db/             SQLAlchemy models, CRUD, session
  app/realtime/       WebSocket manager, events, notifications
  app/workers/        Celery app, outbox worker, notification worker
  alembic/versions/   Migrations 0001–0006
frontend/           Next.js 14 staff + tracking app
packages/           Shared TS workspaces — auth, contracts, logger, ui, utils (scaffolds)
sdk/                Python and TypeScript client SDKs (scaffolds)
infra/              Dockerfiles, Terraform (dev env)
deploy/             Azure App Service / static hosting config
docs/               Architecture, guides, plans, brochures
.azuredevops/       CI and CD pipelines
```

---

## Prerequisites

| Requirement | Version | Needed for |
|-------------|---------|------------|
| Node.js | ≥ 20 | Frontend, Turborepo, `npx mcp-remote` |
| pnpm | ≥ 9 | Workspace package manager |
| Python | 3.11 | Backend |
| PostgreSQL | 14+ with `pgvector` | Primary datastore (migration `0005` enables the extension) |
| Redis | 7 | Session cache, Celery broker |

You will also need an [OpenAI API key](https://platform.openai.com/) and a Telegram bot
token from [@BotFather](https://t.me/BotFather).

---

## Quick start

```bash
# 1. Install both toolchains
make install          # pnpm install + pip install -r backend/requirements.txt

# 2. Configure
cp .env.example .env  # then fill in DATABASE_URL, OPENAI_API_KEY, TELEGRAM_TOKEN

# 3. Migrate the database
cd backend && alembic upgrade head && cd ..

# 4. Seed the menu from the bundled workbook (optional)
python backend/scripts/seed_from_excel.py

# 5. Run
python backend/main.py   # API + Telegram bot + outbox worker on :8000
pnpm dev                 # Next.js frontend on :3000
```

Health check: `curl http://localhost:8000/api/health`.
Interactive API docs: <http://localhost:8000/docs>.

To run the API without the Telegram bot (useful when two developers share one token), set
`TELEGRAM_BOT_ENABLED=false`.

### Root scripts

| Command | Effect |
|---------|--------|
| `make install` | Install Node and Python dependencies |
| `pnpm dev` | `turbo run dev` across workspaces |
| `pnpm build` | `turbo run build` |
| `pnpm lint` | `turbo run lint` |
| `make test` | `pnpm test` + `pytest backend/tests` (see [Testing and CI](#testing-and-ci)) |

---

## Configuration

Copy [.env.example](.env.example) to `.env`. The variables that must be set before the app
will start:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Async PostgreSQL DSN (`postgresql+asyncpg://…`) |
| `DATABASE_URL_SYNC` | Sync DSN used by Alembic (`postgresql+psycopg2://…`) |
| `OPENAI_API_KEY` | LLM access |
| `TELEGRAM_TOKEN` | Bot token from @BotFather |

Commonly adjusted:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_PRIMARY_MODEL` | `gpt-4o` | Primary model |
| `OPENAI_FALLBACK_MODEL` | `gpt-4o-mini` | First fallback |
| `OPENAI_FALLBACK_2_MODEL` | `gpt-3.5-turbo` | Second fallback |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache and Celery broker |
| `PORT` | `8000` | API port |
| `JWT_SECRET` | dev placeholder | **Must be changed in production** |
| `JWT_EXPIRY_MINUTES` | `480` | Staff session lifetime |
| `MCP_ENABLED` | `false` | Enable live Zomato/Swiggy ordering |
| `MCP_AUTH_DIR` | — | Path to the `mcp-remote` token cache in containers |
| `STORAGE_PROVIDER` | `local` | `local`, `s3`, `gcs`, or `azure` |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Frontend → API |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/api/v1/ws` | Frontend → WebSocket |

Razorpay (`RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`) and Google
Sheets (`GOOGLE_SHEET_ID`, `GOOGLE_CREDENTIALS`) are optional; both degrade gracefully when
unset. Cloud object-storage clients are commented out in
[requirements.txt](backend/requirements.txt) — uncomment the one you need.

---

## Running with Docker

[backend/docker-compose.yml](backend/docker-compose.yml) defines two profiles.

**Cloud** — the default. MCP off, Redis + API + Celery worker + beat:

```bash
cd backend
docker compose --profile cloud up -d
```

**MCP** — single-tenant live ordering. Requires a one-time OAuth dance on the host, which
populates `~/.mcp-auth`; the container mounts that cache read-only:

```bash
npx -y mcp-remote https://mcp-server.zomato.com/mcp
npx -y mcp-remote https://mcp.swiggy.com/food
docker compose --profile mcp up -d
```

Use [backend/scripts/export_mcp_auth.sh](backend/scripts/export_mcp_auth.sh) to move that
token cache to another host as a secret.

---

## API surface

All routers are mounted under `/api/v1` ([main.py](backend/app/api/main.py)):

| Prefix | Module |
|--------|--------|
| `/api/v1/auth` | [auth.py](backend/app/api/routes/auth.py) — login, JWT issue |
| `/api/v1/menu` | [menu.py](backend/app/api/routes/menu.py) |
| `/api/v1/orders` | [orders.py](backend/app/api/routes/orders.py) |
| `/api/v1/tables` | [tables.py](backend/app/api/routes/tables.py) |
| `/api/v1/kitchen` | [kitchen.py](backend/app/api/routes/kitchen.py) |
| `/api/v1/payments` | [payments.py](backend/app/api/routes/payments.py) — Razorpay |
| `/api/v1/deliveries` | [deliveries.py](backend/app/api/routes/deliveries.py) |
| `/api/v1/reservations` | [reservations.py](backend/app/api/routes/reservations.py) |
| `/api/v1/staff` | [staff.py](backend/app/api/routes/staff.py) |
| `/api/v1/invoices` | [invoices.py](backend/app/api/routes/invoices.py) |

Plus `GET /api/health` and `WS /api/v1/ws?restaurant_id=<id>`.

---

## Database and migrations

PostgreSQL via SQLAlchemy 2 async, with Alembic migrations in
[backend/alembic/versions/](backend/alembic/versions/):

| Revision | Change |
|----------|--------|
| `0001` | vNext initial schema |
| `0002` | Fix boolean server defaults |
| `0003` | Add session ordering platform |
| `0004` | Add delivery proof columns |
| `0005` | Enable `pgvector` |
| `0006` | Add user preferences and rating |

```bash
cd backend
alembic upgrade head                          # apply
alembic revision --autogenerate -m "message"  # create
alembic downgrade -1                          # roll back one
```

Models live in [backend/app/db/models/](backend/app/db/models/), split by domain: menu,
cart/orders, dine-in, delivery, payments, invoices, reservations, outbox, user preferences.

Order and reservation writes go through a transactional outbox
([outbox.py](backend/app/db/models/outbox.py)); [outbox_worker.py](backend/app/workers/outbox_worker.py)
drains it and fans events out over WebSockets, so a broadcast failure never loses a write.

---

## Testing and CI

[.azuredevops/pipelines/ci.yml](.azuredevops/pipelines/ci.yml) runs on PRs into `main`
touching `frontend/`, `backend/`, `packages/`, or `sdk/`. It validates the frontend
(`pnpm install --frozen-lockfile` → `pnpm build` → `pnpm lint`) and the backend
(`pip install` → `pytest backend/tests`). `cd-backend.yml` and `cd-frontend.yml` handle
deployment.

> **Note:** the test suite is not yet written. `backend/tests/` does not exist, so both
> `make test` and the CI backend job fail at the pytest step. The
> [tests/](tests/) directory (`unit/`, `smoke/`, `evals/`) holds `.gitkeep` placeholders.
> Adding tests is the highest-value open task in this repo.

---

## Project status

Working: the agent pipeline, Telegram channel, REST API, staff frontend, PostgreSQL schema
and migrations, WebSocket realtime, outbox worker, Celery workers, JWT auth with role gates,
Razorpay integration, MCP ordering, and Docker/Azure deployment config.

Scaffolded but empty: `packages/*` (auth, contracts, logger, ui, utils), `sdk/python`,
`sdk/typescript`, `data/migrations`, `data/seed`, `docs/adr`, and `.azuredevops/templates`.

---

## Further documentation

- [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) — flow diagrams, routing matrix, session state
- [docs/architecture/PRODUCTION_ARCHITECTURE_DZUKKU.md](docs/architecture/PRODUCTION_ARCHITECTURE_DZUKKU.md)
- [docs/architecture/CTO_DEEP_TECH_ARCHITECTURE_DZUKKU.md](docs/architecture/CTO_DEEP_TECH_ARCHITECTURE_DZUKKU.md)
- [docs/guides/CODEBASE_GUIDE.md](docs/guides/CODEBASE_GUIDE.md) — file-by-file walkthrough
- [docs/testing-docs/FEATURES_AND_TESTING_GUIDE.md](docs/testing-docs/FEATURES_AND_TESTING_GUIDE.md)
- [docs/plans/FOOD_OS_MASTER_BLUEPRINT.md](docs/plans/FOOD_OS_MASTER_BLUEPRINT.md)

> Parts of `docs/` predate two migrations — the move from SQLite to PostgreSQL and from
> Google Gemini to OpenAI (commit `ee54600`). Where a document and the code disagree, the
> code is current.
