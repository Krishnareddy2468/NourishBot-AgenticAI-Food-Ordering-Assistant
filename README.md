# NourishBot — AI Canteen Pre-Order & Smart Pickup

NourishBot is an agentic AI food-ordering system for fictional campus and workplace
canteens. A customer can describe a meal in natural language, review the proposed cart,
place a pre-order, and follow it through preparation to pickup. Canteen staff receive the
same order in a live operations dashboard and kitchen display.

The project is intentionally scoped as a portfolio-grade ordering sandbox. It demonstrates
safe AI actions and real-time order coordination without acting as a delivery marketplace,
payment platform, or commercial restaurant operating system.

## Product idea

Busy canteens have short rush windows, long queues, and frequently changing availability.
NourishBot moves meal selection before the queue:

1. A customer asks for a meal using ordinary language.
2. The assistant checks the synthetic canteen menu and current availability.
3. It proposes an itemized order instead of writing directly to the database.
4. Deterministic code validates stock, quantities, and totals.
5. The confirmed order enters the kitchen queue.
6. The customer tracks preparation and collects the order when it is ready.

The next product milestone adds dietary filters, group lunch carts, and capacity-aware
pickup windows.

## Scope boundary

The active NourishBot concept includes:

- Conversational ordering against an in-house demo menu
- Deterministic cart and order validation
- Pickup-oriented order tracking
- Kitchen preparation workflow
- Staff operations dashboard
- Synthetic menu, customer, and operational data
- Optional Telegram interface for local demonstrations

It intentionally excludes:

- Ordering from third-party delivery marketplaces
- Delivery fleet management
- Live payments, settlements, or commissions
- Restaurant growth, advertising, or vendor-intelligence products
- Real customer, merchant, health, or transaction data
- WhatsApp production deployment or commercial go-to-market strategy

See [the refined product direction](docs/REFINED_PRODUCT_DIRECTION.md) for the design
boundary and [the public-release checklist](docs/PUBLIC_RELEASE_CHECKLIST.md) before
publishing or presenting the repository.

## Architecture

```text
Customer chat
     │
     ▼
Context → Planner → Policy checks → Executor → Verifier → Response
                         │              │
                         │              └── PostgreSQL order state
                         │
                         └── Only allow-listed actions can execute

Kitchen display ◀── WebSocket events ──▶ Pickup operations dashboard
```

The central safety rule is: **the language model may propose an action, but deterministic
application code validates and commits it.**

## Current application surfaces

| Surface | Purpose |
|---|---|
| Customer assistant | Browse the canteen menu and prepare an order conversationally |
| Operations dashboard | Monitor orders, availability, staff, and service health |
| Pickup desk | Manage active customer sessions and handoff |
| Kitchen display | Move line items from queued to cooking to ready |
| Order tracking | Show a customer the latest verified order state |

## Technology

- Python 3.11 and FastAPI
- PostgreSQL, SQLAlchemy, and Alembic
- Redis and Celery for background work
- OpenAI-compatible agent pipeline
- Next.js 14 and React
- WebSockets for live operational updates
- Docker and Terraform development scaffolding

## Repository layout

```text
backend/                 FastAPI API, agent pipeline, database, and workers
  app/agent/             Context, planning, policy, execution, and verification
  app/api/routes/        REST endpoints
  app/db/                Models and persistence
  app/realtime/          WebSocket events
frontend/                Next.js staff and tracking application
docs/                    Product boundary, architecture, and engineering guides
infra/                   Container and infrastructure configuration
tests/                   Unit, smoke, and evaluation suites
```

## Local setup

Requirements:

- Node.js 20+
- pnpm 9+
- Python 3.11
- PostgreSQL 14+
- Redis 7

Install dependencies:

```bash
make install
```

Configure the application:

```bash
cp .env.example .env
```

At minimum, configure the database URLs, an OpenAI API key, and a Telegram token if the
chat channel is enabled.

Run database migrations:

```bash
cd backend
alembic upgrade head
cd ..
```

Start the backend and frontend:

```bash
python backend/main.py
pnpm dev
```

- API: [http://localhost:8000](http://localhost:8000)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Staff application: [http://localhost:3000](http://localhost:3000)

## Configuration

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Async PostgreSQL connection |
| `DATABASE_URL_SYNC` | Migration connection |
| `REDIS_URL` | Cache and background-job broker |
| `OPENAI_API_KEY` | Assistant planning and response generation |
| `TELEGRAM_TOKEN` | Optional demo chat channel |
| `NEXT_PUBLIC_API_BASE_URL` | Browser-to-API base URL |
| `NEXT_PUBLIC_WS_URL` | Live event endpoint |

Use only fictional menu and user data in demonstrations.

## Project status

Working foundations:

- Five-stage agent pipeline
- Menu, order, kitchen, and tracking APIs
- PostgreSQL schema and migrations
- Role-based staff interface
- WebSocket order updates
- Background-worker scaffolding
- Offline queue support for the staff workflow

Current refinement work:

- Reframe staff workflows around pre-order and pickup
- Add explicit pickup-window capacity
- Add dietary and allergen metadata
- Add group-cart membership and confirmation
- Replace legacy demo assets with fully synthetic fixtures
- Build unit, API-contract, and agent-evaluation coverage

## Development rules

- Never let generated text directly mutate persistent state.
- Validate availability and totals immediately before commit.
- Require confirmation for order placement and cancellation.
- Keep demo data synthetic and free of real business information.
- Treat dietary information as a preference unless verified allergen controls exist.
- Do not claim that the prototype provides medical or allergy-safety guarantees.

## Important repository note

This repository retains earlier implementation history because the project will continue in
the same Git repository. Historical files may not represent the refined product direction.
Review the public-release checklist and the applicable NDA or IP-assignment agreement before
making the repository public.
