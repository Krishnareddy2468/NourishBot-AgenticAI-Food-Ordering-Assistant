# NourishBot — AI Scheduled Meal Ordering Assistant

NourishBot helps people plan breakfast, lunch, snacks, and dinner through Telegram or
WhatsApp. It can discover restaurants with Google Maps, schedule an order through NourishBot,
Swiggy, or Zomato, and coordinate either home delivery or a drive-through pickup on the
user's route.

**Instant ordering is intentionally disabled. Every order requires a valid future date,
time, and timezone.**

The project keeps the food-ordering niche while using an original scheduled-meal workflow.
It is designed as a portfolio and product prototype and uses synthetic data by default.

## Product flow

```text
Telegram / WhatsApp
        │
        ▼
Describe meal + diet preference
        │
        ▼
Choose NourishBot / Swiggy / Zomato
        │
        ▼
Choose home delivery or drive-through pickup
        │
        ▼
Home delivery: confirm destination and serviceability
Drive-through: share route origin + destination
        │
        ▼
Google Maps discovery → nearby or low-detour restaurant candidates
        │
        ▼
Provider verification → future date/time → explicit confirmation
        │
        ├── NourishBot: store as SCHEDULED, activate, prepare, and coordinate fulfillment
        │
        └── Swiggy / Zomato: transact only when scheduling and chosen fulfillment are supported
```

If an external provider exposes only immediate checkout, NourishBot blocks that transaction
instead of silently placing an instant order.

## Supported meal discovery

- Breakfast, lunch, snacks, and dinner
- Vegetarian, vegan, high-protein, lighter, balanced, and lower-sugar preferences when the
  menu metadata supports the claim
- Budget, quantity, and menu-availability constraints
- Best-restaurant discovery near an area using Google Maps
- Route-aware drive-through discovery using a Google driving route and low-detour Places results
- Scheduled home delivery coordinated by NourishBot, plus scheduled drive-through pickup

Maps is a discovery layer, not an ordering provider. A Maps result is never treated as proof of
future opening hours, menu availability, price, or drive-through support. NourishBot verifies the
selected restaurant and fulfillment option through Swiggy, Zomato, or its in-house menu before
asking for confirmation.

Dietary suggestions are preferences, not medical advice. The assistant does not claim that
food is allergy-safe without verified ingredient and kitchen controls.

## Channels and providers

| Component | Current behavior |
|---|---|
| Telegram | Polling bot with provider selection and natural-language scheduling |
| WhatsApp | Signed Cloud API webhook, duplicate-message suppression, text/interactive replies, and shared-location parsing |
| NourishBot | In-house scheduled pickup, drive-through, or home-delivery flow backed by PostgreSQL |
| Swiggy / Zomato | Discovery-only by default; transactions remain disabled until scheduling, fulfillment, reconciliation, and receipt capabilities are verified |
| Google Maps | Read-only nearby restaurant and drive-through-along-route discovery |

Provider names do not imply an official partnership. Endpoint availability, OAuth policy,
tool schemas, account permissions, and scheduled-order support must be verified with each
provider before enabling it in a public environment.

## Safety architecture

```text
Context → Planner → Deterministic Guard → Executor → Verifier → Response
                         │                  │
                         │                  └── committed PostgreSQL state
                         └── schedule, cart, identity, and provider checks

Celery Beat → due scheduled orders → live kitchen queue → WebSocket operations views
```

The model can propose actions, but application code decides whether they may execute.
Release-critical controls include:

- Minimum lead time and maximum scheduling horizon
- Operating-hours validation in `Asia/Kolkata`
- Mandatory explicit confirmation
- Database price and availability revalidation
- Discovery-only MCP access by default; transactional tools stay disabled until provider capability verification exists
- One-time confirmation snapshots, stable idempotency keys, and durable unknown-outcome records
- A strict customer-chat tool allowlist that blocks staff mutations before database access
- Tenant-scoped staff APIs and authenticated WebSocket subscriptions
- HttpOnly staff sessions with origin checks for cookie-authenticated mutations
- High-entropy, revocable public tracking tokens with audience-safe responses
- Maps discovery separated from provider-side menu, availability, and checkout verification
- WhatsApp HMAC signature verification and Redis idempotency keys
- Redis-backed login, webhook, and API rate limits
- Explicit CORS origins in production
- Validated staff order-state transitions
- Readiness checks for PostgreSQL and Redis

## Application surfaces

| Route or channel | Purpose |
|---|---|
| `/` | Public landing page for the scheduled-meal product |
| Telegram bot | Customer meal planning and provider selection |
| WhatsApp webhook | Customer meal planning through WhatsApp Cloud API |
| `/login` | Staff authentication |
| `/workspace` | Authenticated role selection |
| `/admin` | Upcoming schedules, live operations, menu, and staff overview |
| `/kitchen` | Activated order and item preparation queue |
| `/waiter` | In-house pickup and drive-through collection coordination |
| `/track/[trackingToken]` | Scheduled time and privacy-safe customer order status |
| `/api/health` | Process liveness |
| `/api/ready` | PostgreSQL and Redis readiness |

The frontend uses a shared warm cream, NourishBot orange, and deep green visual system across
the public site, authentication, operations, kitchen, fulfillment, dispatch, and tracking experiences.

## Technology

- Python 3.11, FastAPI, SQLAlchemy, and Alembic
- PostgreSQL for orders, sessions, menus, customers, and outbox events
- Redis for session caching, webhook idempotency, and Celery
- Celery worker and beat for scheduled-order activation
- OpenAI-compatible agent pipeline and LangGraph MCP adapter
- Google Maps Places API (New) and Routes API for restaurant and route discovery
- Telegram Bot API and WhatsApp Cloud API
- Next.js 14, React, and WebSockets
- Docker Compose profiles for in-house and MCP-enabled runtime modes

## Repository layout

```text
backend/
  app/agent/             planning, policy, execution, verification, MCP guard
  app/api/routes/        REST API and WhatsApp webhook
  app/bot/               Telegram interface
  app/db/                models and persistence
  app/realtime/          WebSocket events and channel notifications
  app/workers/           Celery and scheduled-order activation
  alembic/               database migrations
  tests/                 backend unit tests
frontend/                landing page, staff workspaces, and customer tracking
infra/                   deployment scaffolding
```

## Local setup

Requirements: Node.js 20+, pnpm 9+, Python 3.11, PostgreSQL 14+, and Redis 7+.

```bash
make install
cp backend/.env.example backend/.env
cd backend
alembic upgrade head
cd ..
```

Run the API and chat services:

```bash
cd backend
python main.py
```

Run the frontend in another terminal:

```bash
cd frontend
pnpm dev
```

Run the scheduler in separate terminals:

```bash
cd backend
celery -A app.workers.celery_app worker --loglevel=info --concurrency=2
celery -A app.workers.celery_app beat --loglevel=info
```

The landing page is at `http://localhost:3000`; API documentation is at
`http://localhost:8000/docs`.

## Configuration

Copy [backend/.env.example](backend/.env.example) and configure:

| Variable | Purpose |
|---|---|
| `ENVIRONMENT` | Use `production` to enable strict startup validation |
| `DATABASE_URL` / `DATABASE_URL_SYNC` | Async application and sync worker/migration connections |
| `REDIS_URL` | Cache, idempotency, Celery broker, and result backend |
| `JWT_SECRET` | Staff authentication signing secret |
| `AUTH_COOKIE_NAME` / `AUTH_COOKIE_SAMESITE` | HttpOnly staff-session cookie settings |
| `ALLOWED_ORIGINS` | Comma-separated frontend origins; no wildcard in production |
| `CONFIRMATION_TTL_MINUTES` | Lifetime of a one-time order confirmation challenge |
| `MCP_TRANSACTIONAL_ENABLED` | Development-only transaction gate; production rejects it until provider verification is implemented |
| `*_RATE_LIMIT_PER_MINUTE` | Per-client API, login, and webhook request ceilings |
| `OPENAI_API_KEY` | Planner and response generation |
| `TELEGRAM_TOKEN` | Telegram bot token when polling is enabled |
| `WHATSAPP_*` | Cloud API token, phone ID, verification token, app secret, and API version |
| `MIN_SCHEDULE_LEAD_MINUTES` | Earliest allowed future order |
| `MAX_SCHEDULE_DAYS` | Furthest allowed future order |
| `SCHEDULE_ACTIVATION_LEAD_MINUTES` | When an in-house order enters the kitchen queue |
| `GOOGLE_MAPS_API_KEY` | Places API (New) and Routes API key for nearby and on-route discovery |
| `MAPS_REQUEST_TIMEOUT_S` | Timeout for Maps discovery requests |
| `NEXT_PUBLIC_API_BASE_URL` | Browser API URL |
| `NEXT_PUBLIC_WS_URL` | Browser WebSocket URL |

## Tests and release checks

```bash
PYTHONPYCACHEPREFIX=/tmp/nourishbot-pycache python -m compileall -q backend/app backend/alembic/versions
python -m pytest backend/tests -q
cd frontend && pnpm build
```

Before a public launch:

1. Use a managed PostgreSQL database and Redis instance, then run `alembic upgrade head`.
2. Set `ENVIRONMENT=production`, a strong `JWT_SECRET`, and exact `ALLOWED_ORIGINS`.
3. Create least-privilege staff accounts and remove demo credentials and synthetic fixtures
   that should not be exposed.
4. Configure HTTPS, the public frontend/API URLs, monitoring, backups, and log redaction.
5. Register and verify the WhatsApp webhook; test duplicate delivery, retries, token expiry,
   and the 24-hour messaging rules applicable to the account.
6. Verify the Swiggy and Zomato MCP endpoints, OAuth identity, transaction ownership, native
   scheduling fields, and supported fulfillment modes. Leave a provider disabled if any check fails.
7. Restrict the Google Maps key to the production workload, enable Places API (New) and Routes
   API, configure quotas, and test nearby and on-route searches in staging.
8. Run unit, migration, webhook, end-to-end chat, cancellation, scheduling-boundary,
   concurrency, security, and load tests in staging.
9. Confirm the Celery worker and beat processes are healthy; without them, scheduled in-house
   orders will not enter the live kitchen queue.
10. Confirm `/api/ready` returns 200 and perform a rollback drill before opening traffic.
11. Review the repository and rewritten Git history against the applicable NDA or IP terms.

## Current readiness

Implemented in this repository:

- Mandatory future scheduling in the in-house pipeline
- Scheduled order schema and migration
- Timed activation worker and operations outbox event
- Scheduled pickup, drive-through, and home-delivery fulfillment for in-house orders
- Telegram and WhatsApp channel routing
- Swiggy and Zomato discovery-only MCP connections; transactional access is fail-closed
- Google Maps nearby restaurant and route-aware drive-through discovery tools
- Responsive landing page and unified staff authentication/theme
- Tokenized, privacy-safe scheduled customer tracking and tenant-scoped staff views
- HttpOnly staff authentication, authenticated WebSockets, and origin/CSRF enforcement
- Immutable one-time order confirmations, stable idempotency, and durable provider-attempt outcomes
- Focused unit tests for scheduling policy, confirmation safety, tenant isolation, authentication, MCP tool safety, and WhatsApp parsing

Still environment-dependent before public traffic:

- Real Telegram, WhatsApp, OpenAI, database, Redis, Maps, and provider credentials
- WhatsApp business verification and webhook registration
- Confirmation that Swiggy and Zomato support the required scheduled transaction and fulfillment modes
- Staging end-to-end, security, concurrency, and load-test results
- Production hosting, DNS, TLS, alerting, backups, and rollback verification

Do not claim a provider is live merely because its button is visible. A provider is live only
after its authenticated tools pass the scheduling capability and end-to-end staging checks.

## Development rules

- Never let generated text directly mutate persistent state.
- Never place an instant order or emulate scheduling with delayed clicks.
- Validate menu availability, quantities, totals, identity, and schedule immediately before commit.
- Require confirmation for placement and cancellation.
- Keep demonstrations synthetic and do not commit credentials or customer data.
- Treat dietary requests as preferences unless verified nutrition and allergen controls exist.
- Do not push or deploy from this repository until the release checklist above passes.
