# Project Dzukku — Change Request & Product Expansion Brief (vNext)

> **Owner:** Jacob V Kothapalli  
> **Audience:** Engineering / Tech Lead  
> **Last updated:** 2026-05-01

## 0) Product decisions (locked)

1. **Single-restaurant first**, design data model to become **multi-tenant later** with minimal refactor (add `restaurant_id` early, default to `1`).
2. Delivery: **restaurant-owned drivers first**; later add other dispatch models.
3. Payments: **Razorpay intents + webhook verification now** (no “later”).
4. Frontend: **one portal** with **login + role toggle** to switch between **Admin/POS**, **Waiter**, **Kitchen** screens.

---

## 1) Why this doc exists

The current build proves chat-first ordering + a lightweight POS dashboard. The next phase must make Dzukku **production-grade**, **cloud-hostable**, **database-first**, and **operationally complete** for:

- Image-rich menu, availability & stock
- Real-time orders (online + takeaway + dine-in)
- Payments (Razorpay)
- Delivery assignment & tracking
- Waiter/table service + invoice/billing
- Kitchen operations (KDS v2)
- A “true agentic” restaurant identity (policy-driven, memory-rich, proactive but non-spammy)

---

## 2) Non-negotiable architectural shifts

### 2.1 DB-first (Excel becomes import/export only)

- Replace SQLite + Excel-as-source-of-truth with **PostgreSQL** as the **system of record**.
- Keep Excel only for:
  - one-time import/seed
  - export/reporting

### 2.2 Images & media

- Store media in object storage (Azure Blob / GCS / S3).
- DB stores image metadata + CDN URL.

### 2.3 Real-time updates

- Replace polling with **WebSocket/SSE** for:
  - KDS updates
  - Waiter/table updates
  - Delivery tracking updates
  - Payment status updates

### 2.4 Remove n8n dependency

- Remove n8n integration; all workflows happen in-app.

---

## 3) Redesign: Agent decision model (v2)

### 3.1 The problem we’re solving

Today, the bot is a tool-using LLM with bounded loops and good guardrails. To make it **truly agentic** as the restaurant’s “face,” it must:

- Operate on **live business state** (menu, availability, stock, kitchen load)
- Follow **restaurant policies** (delivery radius, prep times, cancellation rules)
- Handle **payments**, **delivery**, and **dine-in flows** end-to-end
- Support **proactive communication** (order status, delays, new items) with strict non-spam controls

### 3.2 v2 Agent architecture (Planner → Executor → Verifier)

**Core rule:** LLM never writes to DB directly. It proposes actions; deterministic code validates and commits.

**Per message pipeline:**

1. **Context Build**
   - Fetch: customer profile, open cart/session, active order(s), reservation/table session, last N interactions
   - Inject time-of-day, language, restaurant policy summary, availability snapshot

2. **Planner (LLM)**
   - Output: an explicit JSON plan with:
     - `goal` (ORDER_ONLINE / DINE_IN / TAKEAWAY / RESERVATION / SUPPORT)
     - `missing_slots` (name/phone/address/payment/table/etc.)
     - `constraints` (budget, veg, spice, allergy)
     - `proposed_actions[]` (tool calls)

3. **Executor (Deterministic tools)**
   - Validate inputs (schema, ranges, policy)
   - Execute tool calls, commit to DB

4. **Verifier (Deterministic)**
   - Recompute totals from DB rows
   - Confirm item availability
   - Ensure confirmations captured for irreversible actions
   - Generate “safe summary” for response

5. **Responder (LLM)**
   - Converts verified summary into friendly, concise reply
   - Must end with a CTA/question

**Loop bounds:** max 6 tool-iterations per user turn.

### 3.3 v2 state machine (high-level)

- `IDLE`
- `BROWSING_MENU`
- `BUILDING_CART`
- `COLLECTING_DETAILS` (name/phone/address/table)
- `AWAITING_CONFIRMATION`
- `AWAITING_PAYMENT`
- `ORDER_PLACED`
- `ORDER_IN_PROGRESS` (kitchen)
- `OUT_FOR_DELIVERY`
- `DELIVERED` / `COMPLETED`
- `SUPPORT_CASE`

State is stored per channel session and linked to customer identity.

### 3.4 Toolset changes (v2)

Add/upgrade tools so agent can orchestrate full operations:

**Customer-facing tools**
- `search_menu(query, filters)` — semantic + filter search
- `get_menu(filters)`
- `get_item_details(item_id)` (includes images + modifiers)
- `add_to_cart(items[])` / `update_cart_item` / `remove_cart_item` / `view_cart` / `clear_cart`
- `set_order_type(DELIVERY|PICKUP|DINE_IN)`
- `set_delivery_address(address_struct)`
- `place_order(confirm=true)` (creates order + order_items)
- `create_payment_intent(order_id, provider=razorpay)`
- `check_payment_status(payment_id)`
- `cancel_order(order_id, reason)` (policy-gated)
- `create_group_order()` / `join_group_order(link)`

**Restaurant ops tools**
- `set_item_availability(item_id, available)`
- `update_stock(item_id, delta)`
- `update_order_status(order_id, status)`
- `assign_driver(order_id, driver_id)`
- `update_delivery_status(delivery_id, status)`

**Reservations / dine-in tools**
- `make_reservation(...)`
- `open_table_session(table_id, guests, waiter_id)`
- `add_table_order(table_session_id, items[])`
- `close_table_session(table_session_id)`
- `generate_invoice(entity_id, entity_type)`

### 3.5 “Restaurant identity” behaviors (v2)

- The agent represents the restaurant as a consistent persona:
  - warm, professional, concise
  - “food-first” boundary
  - multilingual mirroring
- Uses **policy and live state** to behave realistically:
  - if item out of stock → suggest alternatives
  - if kitchen load high → accurate ETA & apology
  - if delivery radius exceeded → offer pickup/dine-in

---

## 4) v2 DB schema (PostgreSQL) — single-restaurant now, multi-tenant ready

> Notes:
> - We include `restaurant_id` in all core tables, defaulting to `1`.
> - Use relational rows for commerce (no JSON blobs for order items).

### 4.1 Core

#### restaurants
- `id bigint primary key` (start with 1)
- `name text`, `phone text`, `address text`, `timezone text`
- `created_at timestamptz`

#### users (staff)
- `id bigserial primary key`
- `restaurant_id bigint not null references restaurants(id)`
- `name text not null`
- `phone text`
- `email text unique`
- `password_hash text not null`
- `role text not null`  -- ADMIN | MANAGER | CASHIER | WAITER | KITCHEN | DRIVER
- `active boolean default true`
- `created_at timestamptz`

#### customers
- `id bigserial primary key`
- `restaurant_id bigint not null references restaurants(id)`
- `name text`
- `phone text not null`
- `email text`
- `language_pref text`  -- en / hi / te / code-mix
- `marketing_opt_in boolean default false`
- `first_seen timestamptz`
- `last_seen timestamptz`
- unique(`restaurant_id`, `phone`)

#### channels
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `type text not null` -- TELEGRAM | WHATSAPP | WEB
- `external_id text not null` -- chat_id / wa_id
- `customer_id bigint references customers(id)`
- unique(`type`, `external_id`)

#### sessions
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `channel_id bigint not null references channels(id)`
- `state text not null`
- `cart_id bigint references carts(id)`
- `history_json jsonb` -- last N turns
- `updated_at timestamptz`

### 4.2 Menu + images + modifiers

#### menu_categories
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `name text not null`
- `sort_order int default 0`

#### menu_items
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `category_id bigint references menu_categories(id)`
- `name text not null`
- `description text`
- `type text`                    -- VEG | NON_VEG | EGG | VEGAN
- `price_cents int not null`     -- smallest unit
- `special_price_cents int`
- `available boolean default true`
- `stock_qty int`                -- null = unlimited
- `prep_time_sec int default 900`
- `tags text[]`
- `created_at timestamptz`
- `updated_at timestamptz`

#### menu_item_images
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `item_id bigint not null references menu_items(id)`
- `url text not null`
- `alt_text text`
- `sort_order int default 0`
- `checksum text`
- `created_at timestamptz`

#### modifier_groups
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `name text not null`
- `min_select int default 0`
- `max_select int default 1`

#### modifiers
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `group_id bigint not null references modifier_groups(id)`
- `name text not null`
- `price_cents int default 0`
- `available boolean default true`

#### menu_item_modifier_groups
- `restaurant_id bigint not null`
- `item_id bigint not null references menu_items(id)`
- `group_id bigint not null references modifier_groups(id)`
- primary key (`item_id`, `group_id`)

### 4.3 Cart + orders

#### carts
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `customer_id bigint references customers(id)`
- `status text default 'OPEN'`   -- OPEN|CONVERTED|ABANDONED
- `created_at timestamptz`
- `updated_at timestamptz`

#### cart_items
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `cart_id bigint not null references carts(id)`
- `item_id bigint not null references menu_items(id)`
- `qty int not null`
- `unit_price_cents int not null`
- `modifiers_json jsonb`

#### orders
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `order_ref text not null unique`  -- DZK-XXXXXX
- `customer_id bigint references customers(id)`
- `channel_id bigint references channels(id)`
- `order_type text not null`        -- DELIVERY|PICKUP|DINE_IN
- `status text not null`            -- CREATED|ACCEPTED|PREPARING|READY|OUT_FOR_DELIVERY|DELIVERED|CANCELLED
- `subtotal_cents int not null`
- `tax_cents int default 0`
- `packing_cents int default 0`
- `discount_cents int default 0`
- `total_cents int not null`
- `eta_ts timestamptz`
- `idempotency_key text unique`
- `notes text`
- `created_at timestamptz`
- `updated_at timestamptz`

#### order_items
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `order_id bigint not null references orders(id)`
- `item_id bigint not null references menu_items(id)`
- `item_name_snapshot text not null`
- `qty int not null`
- `unit_price_cents int not null`
- `modifiers_json jsonb`
- `status text default 'PENDING'`   -- PENDING|IN_PROGRESS|DONE|CANCELLED

### 4.4 Delivery

#### drivers
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `user_id bigint unique references users(id)`
- `vehicle_type text`              -- BIKE|CAR
- `vehicle_no text`
- `active boolean default true`

#### deliveries
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `order_id bigint not null references orders(id)`
- `driver_id bigint references drivers(id)`
- `status text not null`           -- ASSIGNED|PICKED_UP|EN_ROUTE|DELIVERED|FAILED
- `address_json jsonb not null`
- `customer_phone text`
- `assigned_at timestamptz`
- `picked_up_at timestamptz`
- `delivered_at timestamptz`

#### delivery_location_events
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `delivery_id bigint not null references deliveries(id)`
- `lat numeric(9,6) not null`
- `lng numeric(9,6) not null`
- `accuracy_m int`
- `recorded_at timestamptz not null`

### 4.5 Payments (Razorpay)

#### payments
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `order_id bigint not null references orders(id)`
- `provider text not null`          -- RAZORPAY
- `status text not null`            -- CREATED|AUTHORIZED|CAPTURED|FAILED|REFUNDED
- `amount_cents int not null`
- `currency text default 'INR'`
- `provider_order_id text`
- `provider_payment_id text`
- `provider_signature text`
- `created_at timestamptz`
- `updated_at timestamptz`

### 4.6 Tables + dine-in billing

#### dining_tables
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `name text not null`
- `capacity int not null`
- `active boolean default true`

#### table_sessions
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `table_id bigint not null references dining_tables(id)`
- `waiter_user_id bigint references users(id)`
- `guests int not null`
- `status text not null`           -- OPEN|CLOSED|CANCELLED
- `opened_at timestamptz`
- `closed_at timestamptz`
- `notes text`

#### table_session_orders
- `restaurant_id bigint not null`
- `table_session_id bigint not null references table_sessions(id)`
- `order_id bigint not null references orders(id)`
- primary key (`table_session_id`, `order_id`)

### 4.7 Reservations

#### reservations
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `reservation_ref text unique not null`  -- RSV-XXXXXX
- `customer_id bigint references customers(id)`
- `date date not null`
- `time time not null`
- `guests int not null`
- `special_request text`
- `status text not null`           -- CREATED|CONFIRMED|CANCELLED|NO_SHOW
- `created_at timestamptz`

### 4.8 Invoices

#### invoices
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `invoice_no text unique not null`
- `entity_type text not null`      -- ORDER|TABLE_SESSION
- `entity_id bigint not null`
- `subtotal_cents int not null`
- `tax_cents int default 0`
- `total_cents int not null`
- `pdf_url text`
- `created_at timestamptz`

### 4.9 Events / outbox

#### outbox_events
- `id bigserial primary key`
- `restaurant_id bigint not null`
- `event_type text not null`
- `payload jsonb not null`
- `created_at timestamptz`
- `processed_at timestamptz`

---

## 5) Exact flows: Waiter → Kitchen → Agent (and customer)

### 5.1 Dine-in flow (Waiter-driven)

1. **Waiter opens a table session**
   - Waiter → Table Map → select table → “Open Session”
   - DB: `table_sessions(OPEN)`

2. **Waiter adds items (with modifiers)**
   - Waiter → Menu → add to running ticket
   - DB: create `orders(order_type=DINE_IN, status=CREATED)` + `order_items`

3. **Waiter fires items to kitchen**
   - Action: “Send to Kitchen”
   - DB: order status → `ACCEPTED`
   - Event: `order.sent_to_kitchen`

4. **Kitchen cooks (KDS v2)**
   - Kitchen updates item status: `IN_PROGRESS` → `DONE`
   - DB: update `order_items.status`
   - Event: `order.item_status_updated`

5. **Waiter serves + closes loop**
   - Waiter sees “Ready to Serve” updates in real time

6. **Billing**
   - Waiter hits “Generate Bill”
   - System creates invoice and closes table session after payment

**Edge cases**
- Split bill by items/guests
- Move items between tables
- Void items (manager approval)

### 5.2 Online delivery flow (Agent-driven)

1. Customer chats → agent suggests items → builds cart
2. Agent collects details (name/phone/address)
3. Agent places order (DB commit)
4. Agent creates Razorpay intent
5. Payment webhook confirms capture
6. Kitchen receives order (KDS)
7. Restaurant assigns driver
8. Driver GPS pings
9. Customer tracking + updates
10. Proof of delivery

### 5.3 Takeaway/pickup flow

- Same as delivery but `order_type=PICKUP` and no driver.

---

## 6) UX copy & agent behavior rules (v2)

### 6.1 Tone
- Warm, professional, concise
- Mirrors the user’s language
- Emojis minimal and natural

### 6.2 Hard rules
- Never invent prices/availability
- Never place order without explicit confirmation
- Never ask for name/phone twice if known
- Always end with a question/CTA
- One upsell per session max

### 6.3 Copy templates (examples)

**Greeting**
- “Hey {{name}} 👋 Welcome to Dzukku! What are you craving today — quick bites or a full meal?”

**Add to cart**
- “Done ✅ Added **{{qty}}× {{item}}**. Total is **₹{{total}}**. Want to add anything else?”

**Confirm order**
- “Quick check: {{items_summary}} — Total **₹{{total}}**. Shall I place this order?”

**Payment**
- “Payment link ready: {{payment_link}}. Once paid, I’ll confirm the order instantly ✅”

**Tracking**
- “Out for delivery 🚴 Track here: {{tracking_link}}”

### 6.4 Notification policy
- Order-status notifications always allowed
- Broadcasts only for opt-in + capped frequency + quiet hours

---

## 7) Deliverables expected from engineering

- Alembic migrations for v2 schema
- Updated OpenAPI spec (`/api/v1`)
- Role-based portal routing + toggle
- WebSocket/SSE real-time implementation
- Razorpay payment intents + webhook verification
- Agent v2 orchestrator + expanded tools

