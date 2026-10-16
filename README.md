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
| Telegram | Polling bot with provider selection, natural-language scheduling, consent controls, and secure channel-link commands |
| WhatsApp | Signed Cloud API webhook, duplicate-message suppression, consent-gated shared locations, and secure channel linking |
| Customer web identity | HttpOnly opaque browser identity linked with a short-lived one-time code; privacy self-service API is available |
| NourishBot | In-house scheduled pickup, drive-through, or home-delivery flow backed by PostgreSQL |
| Swiggy / Zomato | Discovery-only by default; transactions remain disabled until scheduling, fulfillment, reconciliation, and receipt capabilities are verified |
| Google Maps | Read-only nearby restaurant and drive-through-along-route discovery |

Provider names do not imply an official partnership. Endpoint availability, OAuth policy,
tool schemas, account permissions, and scheduled-order support must be verified with each
provider before enabling it in a public environment.

### Privacy and channel-link commands

Telegram and WhatsApp handle these commands deterministically without sending them to the
language model:

- `privacy status` — view personalization, location, reminder, and analytics consent.
- `enable memory` / `disable memory` — opt in to or stop personalization.
- `export my memory` — receive the stored preference profile and masked linked-channel list.
- `delete memory`, followed by `delete my memory permanently` — permanently remove stored personalization.
- `enable location` / `disable location` — control whether shared coordinates may be used.
- `enable reminders`, `disable reminders`, `enable analytics`, or `disable analytics` — change those purpose-specific decisions.
- `link whatsapp`, `link telegram`, or `link web` — create an expiring code; send `link <code>` on the named target channel.

A phone number alone never merges customer profiles. Cross-channel linking requires control of
an existing verified channel and the target channel.

Linked Telegram and WhatsApp customers can also use deterministic planner commands such as
`plan my day`, `plan my week vegetarian under ₹2000`, `show my meal plan`,
`confirm meal plan 12`, `confirm planned meal 45`, `pause routine 3`,
`resume routine 3`, or `skip routine 3 2026-10-21`.

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
- Immutable provider capability versions with exact tool-schema hashes and `UNVERIFIED`, `VERIFIED`, `DEGRADED`, or `DISABLED` state
- Explicit Mode A discovery, Mode B user-controlled handoff, and verified Mode C transactional behavior shown to chat users
- Append-only redacted provider receipts plus status-before-retry reconciliation for every uncertain outcome
- Complete daily and weekly draft plans with breakfast, lunch, snacks, and dinner slots
- Deterministic diet, verified-allergy, health-goal, history, and budget-aware recommendations
- Saved meal/full-day templates and duplicate-safe recurring routines with pause, resume, skip, edit, and reschedule actions
- Consent-gated Telegram and WhatsApp reminders that never imply a plan has placed an order
- One-time confirmation snapshots, stable idempotency keys, and durable unknown-outcome records
- A strict customer-chat tool allowlist that blocks staff mutations before database access
- Tenant-scoped staff APIs and authenticated WebSocket subscriptions
- HttpOnly staff sessions with origin checks for cookie-authenticated mutations
- High-entropy, revocable public tracking tokens with audience-safe responses
- Maps discovery separated from provider-side menu, availability, and checkout verification
- WhatsApp HMAC signature verification and Redis idempotency keys
- Tenant-scoped Telegram, WhatsApp, and web identities linked only with hashed, expiring, one-time codes
- Append-only consent decisions for personalization, location, reminders, and analytics
- Consent-gated personalization learned only from fulfilled orders or explicit ratings
- Daily retention enforcement for chat turns, shared locations, and provider response/error excerpts
- Redis-backed login, webhook, and API rate limits
- Explicit CORS origins in production
- Central validated order, schedule, provider, and fulfillment transitions with append-only audit histories
- Distinct cancellation, fulfillment failure, provider rejection, and expiry outcomes
- Durable leased scheduling with restart recovery, bounded retries, reminders, and manual review
- Replica-safe leased outbox processing with retry and dead-letter visibility
- Readiness checks for PostgreSQL and Redis

## Application surfaces

| Route or channel | Purpose |
|---|---|
| `/` | Public landing page for the scheduled-meal product |
| Telegram bot | Customer meal planning and provider selection |
| WhatsApp webhook | Customer meal planning through WhatsApp Cloud API |
| `/api/v1/customer/identity/link` | Link an opaque web identity with a one-time code created on an existing channel |
| `/api/v1/customer/privacy*` | View consent, change consent, export memory, or permanently delete memory |
| `/api/v1/customer/meal-plans*` | Generate, inspect, edit, and confirm complete draft meal plans |
| `/api/v1/customer/meal-templates*` | Save and list single-meal or full-day templates |
| `/api/v1/customer/meal-routines*` | Create and manage bounded recurring meal routines |
| `/api/v1/scheduler/health` | Manager-only scheduler heartbeat, lag, queue, and outbox health |
| `/api/v1/scheduler/manual-review` | Manager-only dead-letter schedule queue |
| `/api/v1/providers/capabilities` | Staff-visible provider modes, verified capabilities, and schema versions |
| `/api/v1/providers/attempts*` | Manager-only provider receipts, tracking, cancellation, and reconciliation |
| `/login` | Customer signup, password sign-in, Google OAuth, and existing staff sign-in |
| `/account` | Non-privileged customer account home |
| `/workspace` | Authenticated staff role selection; customer accounts redirect to `/account` |
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
docs/architecture/       lifecycle and system design contracts
frontend/                landing page, staff workspaces, and customer tracking
deploy/                  Vercel deployment contract and runtime notes
infra/                   container and infrastructure scaffolding for backend services
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

### Google sign-in

Create a Google OAuth 2.0 **Web application** client and register the exact callback used by
your backend, for example `http://localhost:8000/api/v1/auth/oauth/google/callback`. Then set
these values in `backend/.env`:

```bash
FRONTEND_URL=http://localhost:3000
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/oauth/google/callback
```

Run `cd backend && alembic upgrade head` after pulling the authentication migration. Public
email and Google registrations receive the `CUSTOMER` role and cannot access staff APIs.

Run the scheduler in separate terminals:

```bash
cd backend
celery -A app.workers.celery_app worker --loglevel=info --concurrency=2
celery -A app.workers.celery_app beat --loglevel=info
```

The landing page is at `http://localhost:3000`; API documentation is at
`http://localhost:8000/docs`.

## Deployment target

The public Next.js application's deployment target is **Vercel**. In the Vercel project settings,
set the repository Root Directory to `frontend`; framework detection, builds, preview
deployments, and production deployments then run from that directory. The checked-in
[`frontend/vercel.json`](frontend/vercel.json) records the framework choice.

Set these Vercel environment variables for Preview and Production:

| Variable | Example | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `https://api.nourishbot.example` | Public HTTPS origin of the NourishBot API |
| `NEXT_PUBLIC_WS_URL` | `wss://api.nourishbot.example/api/v1/ws` | Public secure WebSocket endpoint |

The existing backend is not deployed as a Vercel Function yet. It starts Telegram polling and
an outbox loop inside the FastAPI lifespan and relies on separate Celery worker and beat
processes. Until those jobs are converted to webhooks, idempotent cron endpoints, and durable
external coordination, deploy the backend container, worker, and beat on an always-on runtime,
with managed PostgreSQL and Redis. This runtime must not be Azure; the specific backend host
can be selected independently of Vercel.

See [`docs/guides/VERCEL_DEPLOYMENT.md`](docs/guides/VERCEL_DEPLOYMENT.md) for setup,
verification, rollback, and the requirements for a future full-Vercel backend migration.

## Configuration

Copy [backend/.env.example](backend/.env.example) and configure:

| Variable | Purpose |
|---|---|
| `ENVIRONMENT` | Use `production` to enable strict startup validation |
| `DATABASE_URL` / `DATABASE_URL_SYNC` | Async application and sync worker/migration connections |
| `REDIS_URL` | Cache, idempotency, Celery broker, and result backend |
| `JWT_SECRET` | Website authentication signing secret |
| `AUTH_COOKIE_NAME` / `AUTH_COOKIE_SAMESITE` | HttpOnly website-session cookie settings |
| `FRONTEND_URL` | Frontend origin used after the Google OAuth callback |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Optional Google web OAuth credentials |
| `GOOGLE_REDIRECT_URI` | Exact Google callback URL registered in Google Cloud |
| `CUSTOMER_IDENTITY_COOKIE_NAME` | HttpOnly opaque cookie used by the linked customer privacy API |
| `ALLOWED_ORIGINS` | Comma-separated frontend origins; no wildcard in production |
| `CONFIRMATION_TTL_MINUTES` | Lifetime of a one-time order confirmation challenge |
| `IDENTITY_LINK_TTL_MINUTES` | Lifetime of a hashed, single-use channel-link code |
| `CONSENT_POLICY_VERSION` | Policy version recorded with each append-only consent decision |
| `MESSAGE_RETENTION_DAYS` | Maximum retained age for conversational turns |
| `LOCATION_RETENTION_DAYS` | Maximum retained age for shared route coordinates and transient session addresses |
| `PROVIDER_RESPONSE_RETENTION_DAYS` | Age after which provider response/error excerpts are redacted |
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

1. Connect the repository to Vercel with `frontend` as the project Root Directory and configure
   the Preview and Production public API/WebSocket variables.
2. Use a managed PostgreSQL database and Redis instance, then run `alembic upgrade head`.
3. Set `ENVIRONMENT=production`, a strong `JWT_SECRET`, and exact `ALLOWED_ORIGINS` that include
   the production Vercel/custom domain and approved preview origins.
4. Create least-privilege staff accounts and remove demo credentials and synthetic fixtures
   that should not be exposed.
5. Configure HTTPS, the public frontend/API URLs, monitoring, backups, and log redaction.
6. Register and verify the WhatsApp webhook; test duplicate delivery, retries, token expiry,
   and the 24-hour messaging rules applicable to the account.
7. Verify the Swiggy and Zomato MCP endpoints, OAuth identity, transaction ownership, native
   scheduling fields, and supported fulfillment modes. Leave a provider disabled if any check fails.
8. Restrict the Google Maps key to the production workload, enable Places API (New) and Routes
   API, configure quotas, and test nearby and on-route searches in staging.
9. Run unit, migration, webhook, end-to-end chat, cancellation, scheduling-boundary,
   concurrency, security, and load tests in staging.
10. Confirm the Celery worker and beat processes are healthy; without them, scheduled in-house
    orders will not enter the live kitchen queue.
11. Confirm `/api/ready` returns 200 from the deployed frontend's network path and perform a
    Vercel rollback drill before opening traffic.
12. Review the repository and rewritten Git history against the applicable NDA or IP terms.

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
- Unified customer identities for Telegram, WhatsApp, and web with one-time cross-channel linking
- Explicit consent controls for personalization, locations, reminders, and analytics
- Memory status, disable, export, and permanent deletion commands in Telegram and WhatsApp, plus customer web APIs
- Fulfillment/feedback-only personalization with idempotent learning records; cancelled and unfulfilled orders do not count as completed taste signals
- Automated message, location, and provider-response retention enforcement through Celery Beat
- Focused unit tests for scheduling policy, confirmation safety, identity/privacy, retention, tenant isolation, authentication, MCP tool safety, and WhatsApp parsing

Still environment-dependent before public traffic:

- Real Telegram, WhatsApp, OpenAI, database, Redis, Maps, and provider credentials
- WhatsApp business verification and webhook registration
- Confirmation that Swiggy and Zomato support the required scheduled transaction and fulfillment modes
- Staging end-to-end, security, concurrency, and load-test results
- Vercel project connection, production domain, environment variables, and rollback verification
- Always-on backend/worker/beat hosting, DNS, TLS, alerting, and backups

Do not claim a provider is live merely because its button is visible. A provider is live only
after its authenticated tools pass the scheduling capability and end-to-end staging checks.

## Development rules

- Never let generated text directly mutate persistent state.
- Never create or use long-term personalization unless the customer has explicitly enabled it.
- Never merge customer profiles from a caller-supplied phone number; use the one-time channel-link flow.
- Never place an instant order or emulate scheduling with delayed clicks.
- Validate menu availability, quantities, totals, identity, and schedule immediately before commit.
- Require confirmation for placement and cancellation.
- Keep demonstrations synthetic and do not commit credentials or customer data.
- Treat dietary requests as preferences unless verified nutrition and allergen controls exist.
- Do not push or deploy from this repository until the release checklist above passes.
