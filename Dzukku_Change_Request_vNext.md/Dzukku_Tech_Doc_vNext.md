# Project Dzukku — Technical Implementation Plan (vNext)

> **Owner:** Jacob V Kothapalli  
> **Audience:** Engineering / Tech Lead  
> **Last updated:** 2026-05-01

This doc is **pure tech**: what to change, where, and how to implement vNext.

---

## 1) Repo impact summary

### 1.1 Backend (DzukkuBot)

**vNext** requires:

- PostgreSQL + Alembic migrations
- Object storage client (S3/GCS/Azure Blob)
- WebSockets/SSE for real-time
- Razorpay payment intents + webhook verification
- Outbox worker for reliable events
- RBAC auth for portal roles
- Expanded agent toolset + policy engine

### 1.2 Frontend (restaurant portal)

- One portal, **role-based routing**:
  - `/admin/*` (POS/Admin)
  - `/waiter/*` (table service)
  - `/kitchen/*` (KDS)
- Login + role toggle
- WebSocket client for real-time

---

## 2) Backend changes — exact modules

### 2.1 Replace SQLite with PostgreSQL

- Add SQLAlchemy (recommended) + Alembic migrations.

**Modify / add files**

- `app/core/database.py` → Postgres engine/session
- `app/core/config.py` → `DATABASE_URL`
- `app/db/models/*.py` → ORM models
- `alembic/` + `alembic.ini`

### 2.2 Object storage for images

- Add `app/core/storage.py` abstraction:

  - `upload_image(file) -> url`
  - `delete_image(url)`

Env:

- `STORAGE_PROVIDER=s3|gcs|azure`
- `STORAGE_BUCKET=...`

### 2.3 Realtime layer (WS/SSE)

- Add `app/realtime/`:
  - `ws_manager.py` (connections)
  - `events.py` (publish/subscribe)

Expose:

- `WS /api/v1/ws` or `GET /api/v1/stream/*` (SSE)

### 2.4 Outbox pattern + worker

- On every state change transaction: write `outbox_events`.
- Worker drains and pushes to:
  - WS/SSE hub
  - notification engine

Add:

- `app/workers/outbox_worker.py`

### 2.5 Razorpay payments

Add:

- `POST /api/v1/payments/intents`
- `POST /api/v1/payments/webhooks/razorpay`

Implement modules:

- `app/payments/razorpay.py`
- `app/api/routes/payments.py`

Rules:

- webhook signature verification
- idempotent payment updates

### 2.6 RBAC auth

- Staff login + JWT
- Role checks on routes

Add:

- `app/auth/jwt.py`, `app/auth/deps.py`
- `app/api/routes/auth.py`

### 2.7 Agent v2

Update:

- `app/agent/orchestrator.py` → planner/executor/verifier
- Add `app/agent/policies.py` → operational constraints
- Expand tool implementations:
  - payments
  - delivery
  - table sessions
  - modifiers

### 2.8 Remove n8n + deprecate Sheets

- Remove `src/services/n8nService.js` + backend calls.
- Sheets: feature-flag or remove.

---

## 3) Frontend changes — one portal, three modes

### 3.1 Portal structure

- Admin: orders, menu CMS, staff, payments, drivers
- Waiter: table map, sessions, add items, fire-to-kitchen, bill
- Kitchen: KDS v2, item-level statuses, station views

### 3.2 Role toggle

- After login, show toggle (Admin / Waiter / Kitchen) based on assigned role(s).

### 3.3 Realtime

- WS client subscribes to:
  - `orders.*`
  - `order_items.*`
  - `table_sessions.*`
  - `deliveries.*`

---

## 4) Minimum API contracts

### Menu

- `POST /api/v1/menu/items`
- `PATCH /api/v1/menu/items/{{id}}`
- `POST /api/v1/menu/items/{{id}}/images`
- `PATCH /api/v1/menu/items/{{id}}/availability`

### Tables

- `POST /api/v1/tables/sessions`
- `POST /api/v1/tables/sessions/{{id}}/orders`
- `POST /api/v1/tables/sessions/{{id}}/fire`
- `POST /api/v1/tables/sessions/{{id}}/invoice`

### Kitchen

- `PATCH /api/v1/orders/{{id}}/items/{{item_id}}/status`

### Payments

- `POST /api/v1/payments/intents`
- `POST /api/v1/payments/webhooks/razorpay`

### Delivery

- `POST /api/v1/deliveries/assign`
- `POST /api/v1/deliveries/{{id}}/location`
- `GET /api/v1/deliveries/{{id}}/track`

---

## 5) Testing checklist

- Migration tests
- Payment webhook verification tests
- Idempotency tests (order + payment)
- E2E: chat order → payment → kitchen → delivery
- E2E: waiter order → kitchen → invoice

