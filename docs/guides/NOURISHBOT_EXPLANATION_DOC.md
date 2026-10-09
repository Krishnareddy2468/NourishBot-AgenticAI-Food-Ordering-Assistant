# NourishBot — Full System Explanation
### From Frontend to Backend: Architecture, Ordering Flows, and Agentic Bot

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Frontend — React POS](#3-frontend--react-pos)
4. [Backend — FastAPI + PostgreSQL](#4-backend--fastapi--postgresql)
5. [Agentic Bot — 5-Stage Pipeline](#5-agentic-bot--5-stage-pipeline)
6. [Swiggy & Zomato MCP Integration](#6-swiggy--zomato-mcp-integration)
7. [Real-Time Layer — WebSocket](#7-real-time-layer--websocket)
8. [Full Ordering Flow: Scenario-Based Bot Conversations](#8-full-ordering-flow-scenario-based-bot-conversations)
   - Scenario A: Weather-Driven Comfort Food
   - Scenario B: Mood-Based Ordering
   - Scenario C: Budget / Cost Optimization
   - Scenario D: "What's Delicious Today?"
9. [Order Lifecycle: End-to-End](#9-order-lifecycle-end-to-end)
10. [Data & Security Design](#10-data--security-design)

---

## 1. Project Overview

**NourishBot** is a full-stack restaurant operations platform for a Hyderabad cloud kitchen. It has three interconnected systems running together:

| System | Technology | Purpose |
|---|---|---|
| POS Frontend | React 19 + Vite 8 | Staff portal for admin, waiter, kitchen |
| Backend API | FastAPI + PostgreSQL | All business logic, REST + WebSocket |
| Telegram Bot | Python + Gemini AI | Customer-facing AI ordering assistant |

The bot is **agentic** — it does not follow a script. It reasons, calls tools, reads live data from PostgreSQL, and responds conversationally. Customers order food by simply chatting on Telegram, just like texting a friend.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CUSTOMER (Telegram)                     │
│          "I want something warm, it's raining outside"      │
└─────────────────────┬───────────────────────────────────────┘
                      │ Telegram message
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               NOURISHBOT TELEGRAM BOT                           │
│   ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│   │ /start   │  │ Platform     │  │ Text Handler         │  │
│   │ /menu    │  │ Selection    │  │ (greeting detection) │  │
│   │ /order   │  │ (inline KB)  │  │                     │  │
│   └──────────┘  └──────────────┘  └─────────────────────┘  │
│         │               │                    │               │
│         ▼               ▼                    ▼               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              ROUTING MATRIX                         │   │
│   │  platform = NourishBot  → 5-Stage Pipeline (in-house)  │   │
│   │  platform = Zomato  → LangGraph MCP Agent          │   │
│   │  platform = Swiggy  → LangGraph MCP Agent          │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴──────────────┐
          ▼                          ▼
┌──────────────────┐      ┌──────────────────────┐
│  NOURISHBOT PIPELINE │      │  MCP AGENT (LangGraph)│
│  Stage 1: Context│      │  Gemini 2.5-flash LLM │
│  Stage 2: Planner│      │  Zomato MCP tools     │
│  Stage 3: Executor│     │  Swiggy MCP tools     │
│  Stage 4: Verifier│     │  Per-turn history     │
│  Stage 5: Responder│    └──────────────────────┘
└────────┬─────────┘
         │  DB reads/writes
         ▼
┌──────────────────────────────────────────────┐
│              PostgreSQL Database              │
│  Users · Customers · Menu · Cart · Orders    │
│  Payments · Reservations · Deliveries        │
│  Sessions · Channels · OutboxEvents          │
└──────────────────────────────────────────────┘
         │  WebSocket events
         ▼
┌──────────────────────────────────────────────┐
│         REACT POS FRONTEND (Staff)           │
│  Admin Dashboard · KDS · Waiter Portal       │
│  Tables · Deliveries · Invoices · Analytics  │
└──────────────────────────────────────────────┘
```

Everything runs in **one process**: FastAPI + Telegram bot polling + WebSocket server + Outbox worker — all sharing the same async event loop and PostgreSQL connection pool.

---

## 3. Frontend — React POS

### Stack
- **React 19** with Vite 8 build
- **React Router 7** — URL-based navigation with protected routes
- **Recharts** — analytics charts
- **Framer Motion** — UI animations
- **React Hot Toast** — notifications
- **Lucide React** — icons
- **WebSocket hook** — live real-time updates

### Role-Based Access

When a staff member logs in, they get a JWT token. The app shows a **Role Selector** page:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Admin / POS    │  │    Waiter        │  │    Kitchen       │
│  Orders, menu,  │  │  Table map,      │  │  KDS v2,         │
│  staff, drivers │  │  sessions, bill  │  │  item-level     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Pages and What They Do

#### Admin Portal (`/admin/*`)
The central control room. Left sidebar navigation with 11 sections:

| Tab | What it shows |
|---|---|
| Dashboard | Live order stats, revenue KPIs |
| Orders | All orders with status chips, accept/reject |
| Deliveries | Driver assignment, GPS tracking, proof of delivery |
| KDS (Kitchen Display) | Orders in Pending → Preparing → Ready columns |
| Tables | Floor map with live session status |
| Reservations | Upcoming bookings with confirm/cancel |
| Menu | Add/edit items, toggle availability, upload images |
| Employees | Staff list, create accounts, activate/deactivate |
| Invoices | Session invoices and order receipts |
| Settlements | Platform-wise fee breakdowns (Swiggy/Zomato/direct) |
| Analytics | Revenue by hour area chart |

The topbar shows **live connection status** (WebSocket), pending order count, items in kitchen, and a live clock.

#### Waiter Portal (`/waiter/*`)
The floor service tool. A waiter can:
1. See the table map — colour-coded AVAILABLE (green), OCCUPIED (amber), RESERVED (purple)
2. Click a table → open a session → set guest count
3. Browse the menu and build a ticket (cart)
4. Fire the ticket to kitchen with one tap
5. Watch item-level kitchen status live (Pending → Cooking → Done)
6. Generate bill only when all items are done

**Offline Mode**: If network drops mid-service, orders are saved to `localStorage` and synced automatically when connectivity returns. A banner shows "Offline — orders queued locally".

#### Kitchen KDS (`/kitchen/*`)
Item-level view. Three columns: **Queued → Cooking → Done**. Each card shows order ref, customer name, item count, and order type (Dine-In / Delivery / Pickup). Station filter tabs let the kitchen filter by order type.

At the bottom: **Expedite Queue** — only orders where every item is DONE appear here, ready to be marked as "Out for Delivery" or "Served".

#### Tracking Page (`/track/:orderRef`)
Customer-facing, no auth required. Shows a live progress stepper:
Order Placed → Confirmed → Preparing → Ready → Driver Assigned → Picked Up → On the Way → Delivered

Auto-refreshes every 15 seconds. Shows driver name, vehicle, and proof of delivery photo/signature.

### API Client (`platformApi.js`)

All API calls go through a central client that:
- Reads the JWT token from `localStorage`
- Adds `Authorization: Bearer <token>` to every request
- On 401 response → dispatches `nourishbot-auth-expired` event → auto-logout
- Generates idempotency keys for orders and payments (using `crypto.randomUUID()`)
- Normalises backend order shape into a consistent frontend model

### Real-Time WebSocket (`useWebSocket.js`)

```javascript
const WS_BASE = `ws://${hostname}:8000/api/v1/ws`

// Usage in any component
const { connected, on, off, send } = useWebSocket(restaurantId)

// Subscribe to any event type
on('order.status_changed', (evt) => refreshOrders())
on('*', (evt) => handleAnyEvent(evt))  // wildcard
```

Auto-reconnects every 3 seconds on disconnect. Components subscribe with `on(eventType, callback)` and the hook handles cleanup.

---

## 4. Backend — FastAPI + PostgreSQL

### Entry Point

`main.py` starts uvicorn pointing at `app.api.main:api`. The FastAPI `lifespan` context manager:

1. Verifies PostgreSQL connectivity on startup
2. Starts the outbox worker as a background task
3. Starts the Telegram bot polling (same event loop as FastAPI — avoids asyncpg "different loop" errors)
4. On shutdown: stops Telegram → cancels outbox → disposes engine

### REST API Routes (`/api/v1/*`)

| Prefix | Handles |
|---|---|
| `/auth/login` | Staff login → JWT token |
| `/orders` | List, get, state transitions, item status, mark-paid |
| `/menu/items` | CRUD, availability toggle, image upload |
| `/tables` | Floor tables + sessions (open, add order, fire, invoice, close) |
| `/kitchen/orders` | KDS feed — orders in ACCEPTED/PREPARING |
| `/payments/intents` | Create Razorpay payment order, webhook handler |
| `/deliveries` | Assign driver, status updates, GPS, proof of delivery |
| `/reservations` | List, confirm, cancel, status update |
| `/staff` | List, create, activate/deactivate |
| `/invoices` | List all invoices |
| `/api/health` | Health check |

### Authentication

Every protected route uses `Depends(extract_token)` which:
1. Reads `Authorization: Bearer <token>` header
2. Decodes the HS256 JWT using `PyJWT`
3. Attaches `user_id`, `restaurant_id`, `role` to the request

Role checks: `require_manager` (ADMIN/MANAGER), `require_kitchen` (ADMIN/KITCHEN).

### Database Models (PostgreSQL via SQLAlchemy async)

```
restaurants ─┬── users (staff)
              ├── customers (telegram users)
              │     └── channels (telegram chat_id binding)
              │           └── sessions (state machine + history)
              ├── menu_categories
              │     └── menu_items ── menu_item_images
              │                   └── modifier_groups ── modifiers
              ├── carts ── cart_items
              ├── orders ── order_items
              │     └── payments
              │     └── deliveries ── delivery_location_events
              │     └── table_session_orders
              ├── dining_tables ── table_sessions
              ├── reservations
              ├── invoices
              └── outbox_events
```

### Outbox Pattern

When an order is placed, an `OutboxEvent` row is written in the same transaction. The outbox worker polls this table and broadcasts WebSocket events to all connected clients — ensuring events are never lost even if a direct WS push fails.

### Real-Time WebSocket

`/api/v1/ws?restaurant_id=1` — persistent connection. The `ConnectionManager` maintains a `{restaurant_id: {client_id: WebSocket}}` dict. Any order/kitchen/delivery event calls `ws_manager.broadcast(restaurant_id, event_dict)` and all connected POS clients receive it instantly.

---

## 5. Agentic Bot — 5-Stage Pipeline

This is the core of the NourishBot in-house ordering experience. When a Telegram message arrives, it goes through five deterministic + LLM stages:

```
Customer message
      │
      ▼
┌─────────────┐
│  STAGE 1    │  ContextBuilder
│  Context    │  ─ reads Channel, Customer, Session, Cart, Orders
│  Snapshot   │  ─ loads menu snapshot (top 40 items)
└──────┬──────┘  ─ history (last 8 turns), pending slots, state
       │
       ▼
┌─────────────┐
│  STAGE 2    │  Planner (Gemini LLM, JSON mode, temp=0.1)
│  Planner    │  ─ reads full context snapshot
│             │  ─ outputs: goal, missing_slots, proposed_actions
│             │  ─ never writes to DB
└──────┬──────┘
       │
       ▼  (short-circuit if only missing slots, no actions)
┌─────────────┐
│  STAGE 3    │  Executor (pure Python, no LLM)
│  Executor   │  ─ runs each proposed_action sequentially
│             │  ─ validates inputs, checks policies
│             │  ─ commits to DB (cart, order, reservation, payment)
│             │  ─ reads DB prices — never trusts LLM-supplied prices
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  STAGE 4    │  Verifier (pure Python, no LLM)
│  Verifier   │  ─ re-reads committed data from DB
│             │  ─ recomputes order total, checks item availability
│             │  ─ annotates kitchen load signal (NORMAL/BUSY/FULL)
│             │  ─ builds VerifiedSummary for Responder
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  STAGE 5    │  Responder (Gemini LLM, temp=0.7)
│  Responder  │  ─ converts VerifiedSummary → friendly reply
│             │  ─ mirrors customer language (EN / TE+EN / HI+EN)
│             │  ─ injects tone from Persona module
│             │  ─ one gentle upsell per session max
└──────┬──────┘
       │
       ▼
  Reply sent to Telegram
  Session persisted to PostgreSQL
```

### Key Principle: LLM Proposes, Code Commits

The LLM (Planner) **only proposes actions**. The Executor (pure Python) decides whether to execute them based on hard rules:
- Are operating hours valid?
- Is the item actually available in DB right now?
- Is the cart non-empty before placing an order?
- Does the delivery address fall within the serviceable radius?
- Is the Razorpay provider actually supported?

The LLM cannot bypass these checks. It cannot fabricate prices — the Executor re-reads `MenuItem.price_cents` from DB for every order.

### State Machine

The bot tracks each customer's journey through defined states:

```
IDLE → BROWSING_MENU → BUILDING_CART → COLLECTING_DETAILS
     → AWAITING_CONFIRMATION → AWAITING_PAYMENT → ORDER_PLACED
     → ORDER_IN_PROGRESS → OUT_FOR_DELIVERY → COMPLETED
```

State transitions happen based on tool outcomes (e.g. `add_to_cart` → BUILDING_CART, `place_order` → ORDER_PLACED or AWAITING_PAYMENT if Razorpay is triggered).

### Persona Module

Before the Responder runs, the Persona module provides:
- **Language detection**: scans message + history for Telugu/Hindi trigger words → returns `en`, `te+en`, or `hi+en`
- **Tone calibration**: maps bot state + time-of-day to a tone instruction (e.g. AWAITING_PAYMENT → "reassuring and calm", late night → "hint of humour")
- **Alternatives**: if an item is unavailable, finds similar available items by name overlap + price proximity
- **Kitchen ETA**: maps kitchen load ratio to NORMAL/BUSY/VERY_BUSY/FULL → builds apology-aware ETA string
- **Delivery radius**: validates address against a known Hyderabad area list

---

## 6. Swiggy & Zomato MCP Integration

**MCP = Model Context Protocol** — an open standard for connecting LLMs to external tools over HTTP.

When a customer selects Zomato or Swiggy from the platform menu:

```
Customer picks "Swiggy" in Telegram
        │
        ▼
Bot saves ordering_platform = "Swiggy" to session
        │
        ▼  (if MCP_ENABLED=true in .env)
LangGraph ReAct Agent boots for platform="Swiggy"
        │
        ├── Loads Swiggy MCP tools via npx mcp-remote
        │   (spawns subprocess: npx -y mcp-remote https://mcp.swiggy.com/food)
        │   (mcp-remote handles OAuth, caches token under ~/.mcp-auth/)
        │
        ├── Filters tools by prefix ("swiggy_*")
        │
        ├── Binds tools to Gemini 2.5-flash LLM
        │
        └── Runs create_react_agent(llm, swiggy_tools)
               │
               ▼ per-message turn
        Agent receives: SystemPrompt + rolling history + user message
               │
               ▼
        LLM reasons → calls Swiggy tools (search, menu, cart, checkout)
               │
               ▼
        Tool results feed back → LLM summarises
               │
               ▼
        Reply saved to session history → sent to customer
```

### MCP Reliability Controls

| Control | Default | What it does |
|---|---|---|
| `MCP_TOOL_TIMEOUT_S` | 45s | Each tool call has a hard timeout |
| `MCP_RECONNECT_COOLDOWN_S` | 30s | After a failure, waits before reconnecting |
| Per-platform lock | asyncio.Lock | Prevents stampede reconnects |
| Tool cache | `_tools_caches` dict | Tools loaded once, reused |
| Fallback | Legacy redirect link | If MCP unavailable, sends Swiggy/Zomato URL |

### Gemini Schema Patch

Gemini's function-calling API does not accept `additionalProperties` in tool schemas and requires enum values to be strings (not integers). A one-time monkey-patch (`_patch_gemini_schema_converter`) is applied at import time to sanitise all tool schemas before they reach the Gemini API.

### Fallback Chain

If the Gemini API itself is under load:
```
gemini-2.5-flash (primary)
  └─ gemini-2.0-flash (fallback 1)
       └─ gemini-1.5-flash (fallback 2)
```
LangChain's `.with_fallbacks()` automatically promotes to the next model on failure.

---

## 7. Real-Time Layer — WebSocket

When the Executor commits an order to PostgreSQL:

```
place_order tool commits Order + OrderItems to DB
        │
        ├── OutboxEvent written (same transaction)
        │
        ├── ws_manager.broadcast(restaurant_id, {
        │       "event_type": "order.created",
        │       "payload": {"order_id": 123, "order_ref": "DZK-A1B2C3"},
        │       "timestamp": "2026-05-07T14:23:11+05:30"
        │   })
        │
        ▼
All connected POS clients receive the event
        │
        ├── Admin Dashboard: toast + refresh order list
        ├── KDS: new card appears in Pending column
        └── Waiter Portal: session orders refresh
```

The frontend WebSocket hook handles all of this transparently. Components just call `on('order.created', callback)` and get notified live.

---

## 8. Full Ordering Flow: Scenario-Based Bot Conversations

These are real conversation scenarios showing how the NourishBot bot handles natural, human-like ordering using context, emotion, and intelligence.

---

### Scenario A: Weather-Driven Comfort Food

> **Context**: Rainy evening in Hyderabad, customer opens Telegram

---

**Customer:** Hey NourishBot!

**Bot:** 👋 Hey Krishna! Welcome to NourishBot Restaurant 🍽️
_Where every bite hits different ❤️_

Quick question — would you like to order through **NourishBot Bot** (right here), or via **Zomato** or **Swiggy**?

**Customer:** *(taps "Order via NourishBot Bot")*

**Bot:** 🍽️ Awesome — let's get you sorted right here! ❤️

---

**Customer:** It's raining so hard outside, I want something really warm and comforting

**Bot:** Ohhh rainy evening in Hyderabad — PERFECT food weather! 🌧️

Here's what'll hit different right now:
- **Mutton Biryani** (Non-Veg) — ₹280 *(rich, aromatic, soul food)*
- **Butter Chicken + Naan** (Non-Veg) — ₹320 *(creamy, warm, ideal)*
- **Masala Chai + Samosa** (Veg) — ₹80 *(light, cosy snack)*
- **Dal Makhani + Rice** (Veg) — ₹160 *(slow-cooked, heartwarming)*

What's calling your name tonight? 😊

**Customer:** Butter chicken sounds perfect. Add 1 Butter Chicken + 2 Garlic Naan

**Bot:** Done! Added to your cart 🛒

```
1x Butter Chicken  — ₹220
2x Garlic Naan     — ₹100
─────────────────────────
Total              — ₹320
```

*A Lassi pairs beautifully with Butter Chicken on a rainy night — want one?* 🥛

**Customer:** Yes add a Sweet Lassi

**Bot:** Perfect combo for the weather! ☁️

```
1x Butter Chicken  — ₹220
2x Garlic Naan     — ₹100
1x Sweet Lassi     — ₹60
─────────────────────────
Total              — ₹380
```

Is this delivery or pickup tonight?

**Customer:** Delivery please. My address is Kondapur, near JNTU Metro

**Bot:** Kondapur — we deliver there! 📍

Could I get your name and mobile number to confirm the order?

**Customer:** Krishna, 9876543210

**Bot:** Got it! Confirming your order:

```
Order Summary for Krishna
─────────────────────────
1x Butter Chicken  — ₹220
2x Garlic Naan     — ₹100
1x Sweet Lassi     — ₹60
─────────────────────────
Total              — ₹380
Delivery to: Kondapur, near JNTU Metro
```

Shall I place this? 😊

**Customer:** Yes go ahead!

**Bot:** 🎉 Order confirmed! Keeping you warm tonight!

```
Order ID:  #DZK-K7M2P9
Total:     ₹380
ETA:       ~25-35 mins
```

I'll send you updates as your food gets prepared. Stay cosy! 🌧️❤️

> **What happened behind the scenes:**
> - Planner detected goal = ORDER_ONLINE, order_type = DELIVERY
> - Executor added items to Cart (read prices from DB, not from LLM)
> - Executor validated Kondapur is within Hyderabad delivery radius
> - place_order created Order + OrderItems in PostgreSQL
> - Razorpay payment intent auto-created
> - WebSocket broadcast → Admin KDS shows new order instantly
> - Telegram notification queued for status updates

---

### Scenario B: Mood-Based Ordering

> **Context**: Customer had a bad day, evening time

---

**Customer:** Not having a great day honestly 😔

**Bot:** Aw, I'm sorry to hear that 💙 Those days are tough.

You know what always helps? Good food 😄

For bad days, I'd personally recommend:
- **Gulab Jamun** — warm, sweet, instantly comforting
- **Kheer** — soft, milky, feels like a hug in a bowl
- **Butter Chicken + Naan** — the ultimate emotional support meal
- **Chocolate Brownie** (if we have it today) — let me check...

Want to go the full comfort route? Or something lighter?

**Customer:** Full comfort route. What's the best combo you'd suggest?

**Bot:** For a proper "treating yourself" evening, here's the combo I'd build:

```
🍛  Butter Chicken          — ₹220  (rich, creamy, deeply satisfying)
🫓  Garlic Naan × 2         — ₹100  (perfect for the gravy)
🍚  Steamed Rice            — ₹60   (for the serious dipping)
🍮  Gulab Jamun (2 pcs)     — ₹80   (finish strong)
─────────────────────────────────────
Comfort Combo Total          — ₹460
```

This is the "you deserve it" order. Should I add all of these? 😊

**Customer:** Yes! That's exactly what I need

**Bot:** Added everything! ❤️

Quick question — delivery or pickup? And your name + number?

**Customer:** Delivery. I'm Priya, 9988776655, Madhapur area, Ayyappa Society

**Bot:** Madhapur — absolutely in our zone!

```
Order Summary for Priya
──────────────────────────
1x Butter Chicken  — ₹220
2x Garlic Naan     — ₹100
1x Steamed Rice    — ₹60
2x Gulab Jamun     — ₹80
──────────────────────────
Total              — ₹460
Delivery to: Ayyappa Society, Madhapur
```

Place this order? 🍛

**Customer:** Place it

**Bot:** Done! 🎉 Priya, your comfort meal is on its way!

```
Order ID:  #DZK-P4R8Q1
Total:     ₹460
ETA:       ~28-38 mins
```

Hope this turns your day around — it will 😄 Sending warm food and good vibes! ❤️

---

### Scenario C: Budget / Cost Optimization

> **Context**: Customer is price-conscious, wants to maximise value

---

**Customer:** Hey I want to order but I have a budget of ₹200 only. What can I get?

**Bot:** ₹200 budget — challenge accepted! Let me find you the best value 😄

**Best options under ₹200:**

```
Option 1 — Full Meal (₹180)
   Dal Tadka + Rice     — ₹120
   Roti × 2            — ₹60

Option 2 — Comfort Snack (₹160)
   Veg Biryani (half)   — ₹120
   Lassi                — ₹60  ← this puts you at ₹180

Option 3 — Light + Filling (₹140)
   Masala Dosa          — ₹100
   Filter Coffee        — ₹40
```

Which sounds good? Or should I suggest the single best item for ₹200? 🍽️

**Customer:** Option 1 looks good. Dal Tadka, Rice and 2 Rotis

**Bot:** Great choice — a proper home-style meal! Added to cart:

```
1x Dal Tadka    — ₹120
1x Steamed Rice — ₹40
2x Roti         — ₹40
─────────────────────
Total           — ₹200  ✓
```

Is this delivery or pickup? (Note: delivery has a ₹200 minimum — you're right at it!)

**Customer:** Pickup. I'll come collect it. Name is Ravi, 9876001234

**Bot:** Perfect — pickup it is! That saves you on delivery wait too.

```
Order Summary for Ravi (PICKUP)
─────────────────────────────────
1x Dal Tadka    — ₹120
1x Steamed Rice — ₹40
2x Roti         — ₹40
─────────────────────────────────
Total           — ₹200
```

Confirm order? 😊

**Customer:** Yes

**Bot:** Order placed! 🎉

```
Order ID:  #DZK-R2V7W5
Total:     ₹200
Type:      PICKUP
Ready in:  ~15-20 mins
```

Come by when ready — we'll have it packed and waiting! 😊

> **What happened behind the scenes:**
> - Planner identified goal = TAKEAWAY, suggested items within budget
> - Executor validated order_type = PICKUP (no delivery minimum applies)
> - No Razorpay payment intent auto-created for PICKUP (COD flow)
> - WS event fired → KDS shows new pickup ticket

---

### Scenario D: "What's Delicious Today?" — Menu Exploration + Upsell

> **Context**: Lunch time, regular customer, not sure what to order

---

**Customer:** Krishna cheppandi, what's delicious today? (Telugu+English mix)

**Bot:** Haha, perfect timing Krishna garu! 😄 Lunch lo best ga untayi ivi:

**Today's Top Picks:**

```
🌟  Hyderabadi Chicken Biryani   — ₹250  (always a hit!)
🌟  Paneer Butter Masala + Naan  — ₹280  (rich, creamy)
🌶️  Chilli Chicken + Fried Rice  — ₹240  (lunch special)
🥗  Kadai Paneer + Roti × 3      — ₹200  (value meal)
```

Lunch time undi, biryani best choice ga untundi! 🍛

**Customer:** Biryani veyandi 2 plates

**Bot:** 2x Biryani added! 🍛 Cart lo:

```
2x Hyderabadi Chicken Biryani — ₹500
```

Biryani ki raita ledha salan want cheytalantara? (Comes with it, just checking!)

**Customer:** Yes raita too

**Bot:** Done! 2x Raita (complimentary with Biryani) added. ✅

Delivery ledha pickup?

**Customer:** Delivery. My address is Gachibowli, Microsoft Campus road

**Bot:** Gachibowli — we deliver there! 📍 Meeru peru, number?

**Customer:** Suresh, 9988001122

**Bot:** Summary:

```
Order for Suresh — DELIVERY
───────────────────────────────────
2x Hyderabadi Chicken Biryani — ₹500
2x Raita (complimentary)      — ₹0
───────────────────────────────────
Total                         — ₹500
Deliver to: Gachibowli, Microsoft Campus road
```

Place cheyyalama? 😊

**Customer:** ha ayithe veyyandi!

**Bot:** 🎉 Order confirmed! Enjoy your lunch, Suresh garu!

```
Order ID:  #DZK-S9U3X7
Total:     ₹500
ETA:       ~25-35 mins
```

Hot ga vachestundi — enjoy! 🍛❤️

> **Language detection in action:**
> - Persona module detected `te+en` from tokens: "cheppandi", "veyandi", "ha", "ayithe", "garu"
> - All slot questions and CTAs automatically switched to Telugu+English templates
> - Responder was instructed to mirror TE+EN register throughout

---

## 9. Order Lifecycle: End-to-End

From customer tapping "yes" to food being delivered:

```
1. Customer says "yes, place it" in Telegram
         │
         ▼
2. Planner generates: { tool: "place_order", args: {order_type, address} }
         │
         ▼
3. Executor runs place_order tool:
   - Validates: restaurant is open, customer has name+phone, cart non-empty
   - Validates: delivery address is in range
   - Re-reads ALL item prices from MenuItem table (never trusts LLM)
   - Creates Order + OrderItems in DB (single transaction)
   - Marks Cart as CONVERTED
   - Auto-creates Razorpay payment intent
   - Writes OutboxEvent
         │
         ▼
4. Verifier re-reads Order from DB:
   - Recomputes total (price × qty for all items)
   - Checks all items still available (race condition guard)
   - Corrects DB total if mismatch found
   - Builds VerifiedSummary with order_ref, items, total, eta
         │
         ▼
5. Responder composes confirmation message in customer's language
         │
         ▼
6. WebSocket broadcast fires to all POS clients:
   event_type: "order.created"
         │
         ├── Admin Dashboard: order appears, pending count +1
         ├── KDS: new ticket in "Pending" column
         └── Waiter Portal (if dine-in): session orders refresh
         │
         ▼
7. Kitchen staff sees ticket on KDS:
   - Clicks "Accept" → order → PREPARING
   - WebSocket fires → admin sees "Preparing" chip
   - Telegram notification sent to customer: "Your order is being prepared!"
         │
         ▼
8. Item-level updates:
   - Kitchen clicks "Start cooking" per item → item → IN_PROGRESS
   - Kitchen clicks "Done" per item → item → DONE
   - When ALL items DONE → "Mark Ready" button enables
         │
         ▼
9. Order marked READY → Driver assigned:
   - Admin assigns driver from Deliveries tab
   - Delivery row created: Driver + Order linked
   - WebSocket fires delivery event
   - Customer notified: "Driver assigned!"
         │
         ▼
10. Delivery flow:
    - Driver picks up → PICKED_UP → customer notified
    - Driver en route → EN_ROUTE
    - /track/DZK-XXXXX shows live progress to customer
    - Delivered → proof submitted → order COMPLETED
         │
         ▼
11. Settlement:
    - If Zomato/Swiggy order: fee_breakdown calculated
    - Invoice generated
    - Visible in Settlements tab
```

---

## 10. Data & Security Design

### Authentication
- Staff login: `POST /api/v1/auth/login` → bcrypt password verify → HS256 JWT (8-hour expiry)
- JWT contains: `user_id`, `restaurant_id`, `role`, `email`
- Frontend auto-clears token on 401 response (via `nourishbot-auth-expired` event)

### Payment Security
- Razorpay order IDs created server-side only — frontend never creates payment amounts
- Idempotency keys on every order and payment creation
- Razorpay webhook signature verified before marking payment captured

### Bot Security
- LLM cannot write to DB — only the deterministic Executor can
- Prices always read from DB, never from LLM output
- Delivery radius validated before order placement
- Operating hours enforced on every `place_order` call
- All tool results pass through Verifier before being shown to customer

### CORS
- API: allows `localhost:5173` and `localhost:3000` for development
- WebSocket: same CORS policy

### Environment Variables (`.env`)
- `DATABASE_URL` — PostgreSQL async connection string
- `TELEGRAM_TOKEN` — Bot token
- `GEMINI_API_KEY` — Google AI API key
- `JWT_SECRET` — Change from default in production
- `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` / `RAZORPAY_WEBHOOK_SECRET`
- `MCP_ENABLED` — Enable/disable live Swiggy/Zomato ordering
- `GEMINI_PRIMARY_MODEL` — Default: `gemini-2.5-flash`

---

## Video Recording Guide

### Suggested Walkthrough Order

1. **Show `.env` and explain config** (1 min) — keys, models, MCP flags
2. **Start the backend** (`python main.py`) — show Telegram bot + FastAPI starting in one process
3. **Open POS frontend** (`npm run dev`) — show login page
4. **Login as admin** — show dashboard with live stats
5. **Walk through Admin tabs** (3 min) — Orders → KDS → Tables → Menu → Employees
6. **Login as waiter** — switch role, show table map, open a session, fire to kitchen
7. **Switch to Kitchen** — show KDS v2, item-level status, expedite queue
8. **Open Telegram bot** — go through Scenario A (rainy weather order)
   - Show platform selection
   - Show natural language ordering
   - Watch KDS update live as order comes in
9. **Show Swiggy/Zomato flow** — select Swiggy, show MCP agent connecting
10. **Track order page** — open `/track/DZK-XXXXX` in browser, show progress
11. **Show bot in Telugu** — Scenario D (te+en language detection)
12. **Show code highlights** (2 min):
    - `pipeline.py` — 5 stages
    - `executor.py` — price re-read from DB
    - `persona.py` — language detection
    - `mcp_clients.py` — per-platform connection with cooldown

---

*Document generated: May 2026 | NourishBot Restaurant Platform v3.0*
