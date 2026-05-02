# Project Dzukku — Product & Engineering Documentation

> *"Where every bite hits different ❤️"*

A conversational, agentic restaurant operating system that turns a chat window into a full ordering, reservation, and POS experience. Dzukku replaces the cluttered scroll-and-tap journey of Swiggy and Zomato with a single, friendly assistant that *understands* the customer and a clean operations dashboard for the restaurant.

---

## 1. Executive Summary

Project Dzukku is composed of two cooperating products:

1. **DzukkuBot** — a Python (FastAPI + Telegram + WhatsApp) backend powered by Google Gemini 2.5 Flash with function-calling tools. It is the customer-facing **AI restaurant agent**.
2. **Dzukku POS Frontend** — a React 19 + Vite dashboard for the restaurant team: live order board, KDS (Kitchen Display System), table map, reservations, menu, analytics, employees, and invoices.

The customer never opens an app, picks a delivery partner, or browses 40 visually identical dishes. They simply chat — in English, Telugu+English, or Hindi+English — and the bot guides them through menu, ordering, and reservations. Behind the scenes, Gemini orchestrates a tool loop (`get_menu`, `add_to_cart`, `update_customer_info`, `place_order`, `make_reservation`, …) backed by SQLite, an Excel master sheet, and Google Sheets sync.

---

## 2. What the Application Does — Feature List

### 2.1 Customer-facing (DzukkuBot)
- **Multi-channel chat**: Telegram (polling) and WhatsApp (Twilio webhook).
- **Natural-language menu browsing** — "what's good for dinner?", "show me veg starters under 200".
- **Time-aware suggestions** — morning → dosa/chai, lunch → biryani/thali, dinner → butter chicken, late-night → comfort food.
- **Mood-aware suggestions** — "had a rough day" → kheer, gulab jamun, butter chicken; "celebrating" → premium family pack, prawn, mutton.
- **Cart management** — add, view, modify, clear, with running totals.
- **Order placement** with name + phone + explicit confirmation, returns a `DZK-XXXXXX` order reference.
- **Table reservations** — date, time, guests, special request → `RSV-XXXXXX` reference.
- **Persistent session memory** — remembers your name and phone across messages so it never asks twice.
- **Rolling conversation history** (last 16 turns) so a bare "9999999999" reply still resolves to "phone number for that order".
- **Inline keyboards & quick action buttons** in Telegram for one-tap menu/order/reserve flows.
- **Location sharing hook** — share a Telegram pin → bot can answer "find restaurants near me".
- **Multilingual mirroring** — replies in the same code-mix the user used.
- **One smart upsell per session** (beverage/dessert/combo) — never pushy.
- **Idempotent order creation** — duplicate submissions return the existing order, never double-charge.

### 2.2 Restaurant-facing (POS Frontend)
- **Dashboard** — live KPIs, sparkline trends, top dishes, revenue charts (recharts).
- **POS & Orders** — manual order entry, real-time list of inbound chat orders.
- **Kitchen Display System (KDS)** — column board (Pending → Accepted → Preparing → Ready → Delivered) with status transitions wired to the backend.
- **Table map** — table state and walk-in management.
- **Reservations** — view, accept, cancel; synced from chat bookings.
- **Menu** — browse and edit items loaded from `Project_Dzukku.xlsx` (Master_Menu sheet).
- **Employees & Shifts**, **Invoices**, **Settlements** & payment intents.
- **n8n integration** — POS events (`order.created`, `order.status_updated`) can publish to an n8n webhook for downstream automations.
- **Excel-driven historical data** — analytics reads from `public/Project_Dzukku.xlsx`.

### 2.3 Cross-cutting
- **Single source of truth** in SQLite (`storage/dzukku.db`), seeded from the Excel master sheet, mirrored to Google Sheets for non-tech operators.
- **Idempotency keys** on order creation.
- **Structured logging** to `logs/dzukku.log`.
- **CORS-friendly REST API** at `/api/*` for the frontend.

---

## 3. Agentic Style — How the Bot Thinks

### 3.1 Architecture: Tool-using LLM, not a scripted FAQ
DzukkuBot is **not** a finite-state chatbot with `if/elif` intent rules. It is an **agent loop**:

```
User message
   │
   ▼
build_system_prompt(session)   ← injects time-of-day, name/phone on file, current cart
   │
   ▼
Gemini 2.5 Flash  ──────────►  picks 0..N tool calls  (function-calling)
   ▲                                    │
   │                                    ▼
   └── tool results (JSON) ───── execute_tool(name, args, session)
                                       (deterministic Python)
   │
   ▼  (loop, max 6 iterations)
Final natural-language reply  →  user
```

Key properties:
- **The LLM never writes to the database.** It only proposes tool calls. The orchestrator validates inputs and runs Python code that touches SQLite / Excel / Sheets. This is the "guardrails-on-tools" pattern, not "trust the LLM with SQL".
- **Stateful sessions** are stored per `chat_id` in the `sessions` table (cart, customer name/phone, rolling history, FSM state).
- **Up to 6 reasoning iterations** per turn — enough to chain `get_menu → add_to_cart → view_cart → reply`, while bounded so a hallucinating loop cannot run away.
- **Two-model resilience** — primary `gemini-2.5-flash`, fallback `gemini-flash-latest`. On primary failure the agent transparently retries on the fallback before degrading to a friendly error.
- **System prompt is dynamically rebuilt every turn** so the LLM always sees fresh context: time of day, what's in the cart, what we already know about the customer.

### 3.2 Available tools (function declarations)

| Tool | Purpose |
|---|---|
| `get_menu` | Fetch live menu (with type/category filters). |
| `add_to_cart` | Add item(s) by fuzzy-matched name; merges quantities. |
| `view_cart` | Return current cart + total. |
| `clear_cart` | Empty the cart. |
| `update_customer_info` | Persist name/phone to session as soon as known. |
| `place_order` | Final commit — only after explicit user confirmation. |
| `make_reservation` | Commit a reservation with date/time/guests. |
| `get_restaurant_info` | Static info: timings, location, delivery, contact. |

### 3.3 Why this style is different from a typical chatbot
- **Determinism where it matters** (price, availability, order persistence) and **flexibility where it helps** (small talk, mood matching, language mirroring).
- **No hand-written intent classifier** — Gemini decides which tool to call. New capabilities = add a new tool.
- **No leaking of internal state** — tool tags and JSON are stripped from the user's reply.
- **Memory by design** — the same chat across days continues the same session.

---

## 4. How Dzukku Differs from Swiggy / Zomato

| Dimension | Swiggy / Zomato | Dzukku |
|---|---|---|
| **Entry point** | Install app, sign up, OTP, share location, pick restaurant | Send a Telegram or WhatsApp message — that's it. |
| **Discovery UX** | Infinite scroll of cards, ratings, photos, ads | Conversational: "what's good right now?" → 3 contextual suggestions. |
| **Cognitive load** | Compare 20 restaurants, 200 dishes, filters, badges | Single restaurant, single thread, single decision at a time. |
| **Personalization** | "Recommended for you" based on past orders | Real-time mood + time-of-day + language matching, plus session memory. |
| **Re-ordering** | Tap History → Reorder → Pay | "Same as last time?" Yes. Done. |
| **Reservation** | Separate flow / not always available | Built into the same chat, same agent. |
| **Restaurant economics** | 18–30% commission per order | Direct chat orders → no aggregator commission, restaurant keeps margin. |
| **Customer data** | Owned by the aggregator | Owned by the restaurant (SQLite + Sheets). |
| **Onboarding for the restaurant** | Listing fees, photo shoots, ad credits | Drop a menu Excel, point a Telegram bot, you're live. |
| **Languages** | English / Hindi UI labels | Free-form code-mix (Tenglish, Hinglish) handled natively by the LLM. |
| **Delivery** | Owned fleet | Configurable — Dzukku still surfaces Swiggy/Zomato as a delivery channel when needed. |
| **Operations** | Aggregator dashboard, partner app | Built-in POS + KDS + reservations + analytics + invoicing in one React app. |

The product is **not** trying to out-Swiggy Swiggy at scale. It is a **direct-to-customer ordering layer** for a restaurant, removing the marketplace middle layer for the 60–80% of customers who already know which restaurant they want.

---

## 5. User Interaction & Usability

### 5.1 What the customer experiences
1. They scan a QR or tap a link → opens Telegram/WhatsApp chat with Dzukku.
2. First message: a warm, name-personalized greeting + 6 quick-action buttons (Menu, Specials, Order, Reserve, Cart, Info).
3. They type or tap. Replies are **2–4 lines for chat, longer only for menu listings or bills**.
4. Every reply ends with a **clear next action** — no dead ends.
5. Cart, name, phone are remembered. Confirmation is always explicit.
6. On placement, they get a formatted bill:

```
🧾 Order Confirmed!
Order ID: #DZK-A1B2C3
────────────────
2x Chicken Biryani — ₹520
1x Gulab Jamun     — ₹80
────────────────
Total: ₹600
ETA: ~20-30 mins
Thank you for choosing Dzukku 🙏❤️
```

### 5.2 Usability principles baked into the system prompt
- Warm, witty, concise — like a friend who runs a great restaurant.
- Match the user's language and pace. Rushed → short. Chatty → playful.
- Emojis used naturally, not as decoration.
- Hard rules: never invent a price, never place an unconfirmed order, never wall-of-text, always end with a question or CTA.
- Off-topic queries are deflected gracefully ("I'm a food-only expert 😄").

### 5.3 Restaurant operator UX
- Single React app. Sidebar navigation, ⌘K-style search bar, live clock.
- Color-coded KDS columns for the kitchen.
- One-click status transitions (Accept → Prepare → Ready → Deliver).
- Excel-first content management — chefs and managers edit a familiar `.xlsx`, the system reseeds.

---

## 6. Technical Documentation

### 6.1 Repository layout
```
Project-Duzukku/
├── DzukkuBot/                        # Python backend + AI agent
│   ├── main.py                       # uvicorn entrypoint (FastAPI + Telegram thread)
│   ├── requirements.txt
│   ├── app/
│   │   ├── api/main.py               # FastAPI app, REST endpoints, CORS
│   │   ├── agent/orchestrator.py     # Gemini agent loop + tool definitions
│   │   ├── agent/legacy_brain.py     # Legacy Groq/regex brain (WhatsApp fallback)
│   │   ├── bot/telegram.py           # python-telegram-bot handlers
│   │   ├── bot/whatsapp.py           # Flask + Twilio webhook
│   │   ├── core/config.py            # Settings (env-driven)
│   │   ├── core/database.py          # SQLite schema, CRUD, sessions, menu seed
│   │   ├── core/excel_sink.py        # Append rows to Project_Dzukku.xlsx
│   │   ├── core/sheets.py            # Google Sheets sync
│   │   └── core/logging_config.py
│   ├── data/Project_Dzukku.xlsx      # Master menu + historical data
│   ├── storage/dzukku.db             # SQLite runtime DB
│   ├── docs/                         # Architecture deep-dives (CTO, Production)
│   └── logs/dzukku.log
└── restaurant-pos-frontend/          # React 19 + Vite POS dashboard
    ├── src/App.jsx                   # Single-file dashboard (≈1.4k LOC)
    ├── src/services/platformApi.js   # Backend REST client
    ├── src/services/n8nService.js    # n8n webhook publisher
    ├── src/hooks/useExcelData.js     # XLSX loader for historical data
    ├── public/Project_Dzukku.xlsx
    └── package.json
```

### 6.2 Tech stack

**Backend (DzukkuBot)**
- Python 3.9+
- FastAPI + uvicorn (REST API for the POS)
- python-telegram-bot 21 (polling)
- Flask + twilio (WhatsApp webhook, legacy)
- google-generativeai (Gemini 2.5 Flash, function calling)
- SQLite (stdlib `sqlite3`, WAL mode)
- pandas + openpyxl (Excel master sheet)
- gspread + google-auth (Google Sheets sync)

**Frontend**
- React 19, Vite 8
- recharts (analytics), framer-motion (animations), lucide-react (icons)
- react-hot-toast (notifications)
- xlsx (client-side Excel parsing for historical/static data)

**Integrations**
- Telegram Bot API
- Twilio WhatsApp
- Google Sheets (Orders / Reservations tabs)
- n8n webhooks (`order.created`, `order.status_updated`, integration test)

### 6.3 Runtime topology
```
                 ┌──────────────────────────────────────────────┐
                 │            DzukkuBot process (Python)        │
   Telegram      │                                              │
   ──────────►   │  Telegram polling thread ─┐                  │
                 │                            ├─► agent.orchestrator
   WhatsApp      │  Flask /whatsapp ──────────┘                  │
   ──────────►   │                            ▼                  │
                 │                       Gemini 2.5 Flash         │
                 │                            │                   │
                 │                            ▼                   │
   POS Frontend  │  FastAPI /api/* ◄──► SQLite + Excel + Sheets  │
   ──────────►   │                                                │
                 └──────────────────────────────────────────────┘
                                  │
                                  ▼
                            n8n workflows (optional)
```

### 6.4 Data model (SQLite)
- `menu(item_no PK, item_name, description, type, category, price, available, stock, special_price)`
- `customers(id PK, name, phone UNIQUE, first_seen)`
- `orders(id PK, order_ref UNIQUE, customer_name, customer_phone, items_ordered JSON, total_price, platform, status, eta, idempotency_key UNIQUE, order_time)`
- `reservations(id PK, reservation_ref UNIQUE, customer_name, customer_phone, date, time, guests, special_request, status, booked_on)`
- `sessions(chat_id PK, state, user_name, cart JSON, customer_name, customer_phone, history JSON, updated_at)`

Auto-migrations: on startup `create_tables()` ensures missing columns are added (`platform`, `eta`, `idempotency_key`, `history`, `stock`, `special_price`).

### 6.5 REST API (FastAPI, served on `:8000`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Service heartbeat. |
| GET | `/api/menu` | Live menu list. |
| GET | `/api/orders?limit=200` | Recent orders for the POS dashboard. |
| POST | `/api/orders` | Create an order (supports `Idempotency-Key` header). |
| PATCH | `/api/orders/{order_id}/status` | Status transition (Accept/Prepare/Ready/Deliver/Cancel). Accepts numeric id or `DZK-…`/`DZX-…` ref. |

Order JSON shape:
```json
{
  "id": "DZK-A1B2C3",
  "dbId": 42,
  "customer": "Krishna",
  "phone": "9999999999",
  "items": [{ "item_name": "Chicken Biryani", "qty": 2, "price": 260 }],
  "total": 520,
  "platform": "Telegram",
  "status": "CREATED",
  "eta": "20 min",
  "orderTime": "2026-04-28 16:11:09"
}
```

### 6.6 Environment variables
```
TELEGRAM_TOKEN=...
GEMINI_API_KEY=...
GEMINI_PRIMARY_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-flash-latest
GOOGLE_SHEET_ID=...
GOOGLE_CREDENTIALS={...}              # or place credentials.json under config/
PORT=8000
LOG_LEVEL=INFO
MENU_SHEET=Master_Menu
N8N_WEBHOOK_URL=...                   # optional
N8N_WEBHOOK_SECRET=...                # optional
```
Frontend `.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```

### 6.7 Local run

Backend:
```bash
cd DzukkuBot
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
python main.py            # FastAPI on :8000 + Telegram polling
```

Frontend:
```bash
cd restaurant-pos-frontend
npm install
npm run dev               # Vite on :5173
npm run lint && npm run build
```

### 6.8 Operational notes
- SQLite is opened with `journal_mode=WAL` and `check_same_thread=False` — safe for the POS GET traffic alongside Telegram writes.
- Sheets and n8n syncs are **best-effort** — failures are logged but never block the user reply or order persistence.
- The agent loop is bounded at 6 iterations to prevent runaway tool chains.
- `reset_session(chat_id)` on `/start` and `/reset` clears cart + customer info + history.
- Telegram inline button text is mapped to natural-language intents (`"📋 Menu" → "Show me the full menu please"`) so the LLM always receives prose.

---

## 7. Functional Documentation

### 7.1 Customer journeys

**A. New customer, ordering food (Telegram)**
1. `/start` → greeting with quick-action keyboard.
2. Tap "Menu" → bot calls `get_menu`, replies with categorised list.
3. "I want 2 chicken biryani and a kheer" → `add_to_cart` (fuzzy match), confirms with running total.
4. "That's it" → bot asks for name (if not on file).
5. User types "Krishna" → `update_customer_info(customer_name="Krishna")`.
6. Bot asks for phone. User types "9999999999" → `update_customer_info(customer_phone=...)`.
7. Bot summarises and asks "Shall I place this? ✅".
8. "Yes" → `place_order` → returns ref `DZK-A1B2C3`, ETA 20–30 min.
9. Order appears live on the POS dashboard and in the KDS "Pending" column.
10. Restaurant moves it through KDS → backend PATCH updates status; frontend reflects in real time on next refresh.

**B. Returning customer**
- `customer_name` and `customer_phone` already stored in `sessions` for this `chat_id`. Bot skips those questions and goes straight to confirmation.

**C. Reservation**
- "Book a table for 4 on Friday at 8 PM" → bot asks any missing fields → `make_reservation` → `RSV-XXXXXX`.

**D. Off-topic**
- "Can you book a flight?" → polite deflection without breaking persona.

### 7.2 Restaurant journeys

**Daily flow**
1. Open POS dashboard → see live KPIs and inbound orders.
2. KDS columns auto-populate from chat orders; staff click to advance status.
3. Reservations view shows the day's bookings with guest count.
4. Manager edits `Project_Dzukku.xlsx` to add/remove items or change prices; the menu reseeds.
5. Settlements & invoices roll up at end of shift.

**Configuration**
- Menu: edit `data/Project_Dzukku.xlsx` → `Master_Menu` sheet.
- Restaurant info: `app/core/config.py` (`RESTAURANT_*` constants) — easily moved to env.
- Bot persona: `build_system_prompt` in `app/agent/orchestrator.py` and `docs/DZUKKU_BOT_SYSTEM_PROMPT.txt`.

### 7.3 Failure modes & fallbacks

| Failure | Behaviour |
|---|---|
| Primary Gemini model errors | Auto-fallback to `gemini-flash-latest`. |
| Both models fail | Friendly user message, error logged, no crash. |
| Google Sheets sync fails | Logged warning; order still saved locally. |
| n8n webhook unreachable | Order succeeds; webhook result reported in API response. |
| Duplicate order submission with same `Idempotency-Key` | Returns the previously-created order; no double charge. |
| Telegram polling drops | Thread is daemonised; restart of the process recovers; webhooks unaffected. |

---

## 8. Head-to-Head: Zomato vs Swiggy vs Dzukku

### 8.1 Side-by-side comparison

| Capability | Zomato | Swiggy | **Dzukku** |
|---|---|---|---|
| **Primary surface** | Native mobile app | Native mobile app | Telegram / WhatsApp chat (no install) |
| **Onboarding** | Sign-up, OTP, address, payment setup | Sign-up, OTP, address, payment setup | Send a message — done |
| **Discovery** | Marketplace feed of 1000s of restaurants, ads, badges | Marketplace feed, "Daily" stories, ads | Single restaurant, conversational, contextual |
| **Search & filter** | Tabs, chips, filters, sorting | Tabs, chips, filters, sorting | Free-text in any language ("spicy and creamy under 200") |
| **Personalization** | Collaborative filtering on past orders | Collaborative filtering + delivery patterns | Real-time mood + time-of-day + language mirroring + per-session memory |
| **Reordering** | History → Reorder → Pay (3 taps) | History → Reorder → Pay (3 taps) | "Same as last time?" → "Yes" |
| **Reservation / table booking** | Dineout (separate product) | Not native, partial | Same chat, same agent, one ref system |
| **Languages supported** | English, Hindi (UI strings) | English, Hindi (UI strings) | Free-form code-mix: Tenglish, Hinglish, Tanglish (LLM-native) |
| **Customer acquisition cost** | Aggregator absorbs, then charges restaurant | Aggregator absorbs, then charges restaurant | Near-zero — QR / link / sticker → chat |
| **Commission per order** | ~18–30% | ~18–30% | 0% (direct) — restaurant keeps full margin |
| **Customer data ownership** | Aggregator | Aggregator | Restaurant (SQLite + Sheets, on-prem possible) |
| **Menu update latency** | Partner dashboard, propagation delay | Partner dashboard, propagation delay | Edit `Master_Menu` Excel → immediate reseed |
| **Restaurant-side tooling** | Partner app + dashboard | Partner app + dashboard | Built-in POS + KDS + reservations + analytics + invoicing |
| **Delivery** | Owned fleet | Owned fleet | Channel-agnostic — direct, Swiggy, Zomato, or pick-up |
| **Outage behaviour** | App down → restaurant invisible | App down → restaurant invisible | Telegram + WhatsApp + REST — multi-channel resilience |
| **Marketing & loyalty** | Aggregator-controlled coupons / Gold | Aggregator-controlled offers / One | Restaurant-controlled — bot can apply any rule, any day |
| **Cold-start** | Listing fee + ad spend to be visible | Listing fee + ad spend to be visible | Print one QR, ship one Telegram link |
| **Dispute / refund** | Aggregator support, slow | Aggregator support, slow | Direct restaurant-customer, instant |
| **Tech model** | Marketplace platform | Marketplace platform | Vertical SaaS + AI agent for one restaurant |

### 8.2 Where each model wins

- **Zomato / Swiggy win when:** the customer doesn't yet know which restaurant they want, wants to compare ratings, or needs ultra-broad geographic reach.
- **Dzukku wins when:** the customer already trusts a restaurant, wants speed, wants conversation, wants their order remembered, and the restaurant wants to keep its margin and customer relationship.

### 8.3 Why this matters strategically
Aggregators have trained customers to scroll. Dzukku flips it back to a relationship: the **restaurant talks to its customer directly**, in the channel they already use 50 times a day (Telegram/WhatsApp), with an agent that is warm, multilingual, and never forgets. Aggregators stay useful for discovery; Dzukku captures repeat business — which is 70%+ of any healthy restaurant's revenue.

---

## 9. Phase-wise Software Development Plan (Agile + RAD)

### 9.1 Methodology choice
Project Dzukku uses a **hybrid Agile + RAD (Rapid Application Development)** model:

- **RAD** for the customer-facing bot and POS UI — these are highly visual / interactive and benefit from rapid prototyping, demoing to the restaurant owner, and immediate feedback.
- **Scrum (Agile)** for backend, data, and integration work — clear sprint boundaries, backlog grooming, retrospectives.
- **Two-week sprints**, sprint review + demo every other Friday, retro on the same day.
- **DoD (Definition of Done):** code merged, lint+build pass, manual smoke test on Telegram + POS, env vars documented, log lines added.

### 9.2 Roles
| Role | Responsibility |
|---|---|
| Product Owner | Restaurant owner / operator — prioritises backlog. |
| Scrum Master | Tech lead — runs ceremonies, removes blockers. |
| AI/Backend Dev | Gemini orchestrator, FastAPI, SQLite, Sheets/Excel sync. |
| Frontend Dev | React POS dashboard, KDS, analytics. |
| QA / Ops | Telegram/WhatsApp smoke testing, deployment, monitoring. |

### 9.3 Phase plan

#### Phase 0 — Inception & Requirements (Week 0, 1 week)
*RAD: Requirements Planning workshop with the restaurant owner.*
- Capture menu structure, pricing, opening hours, delivery model.
- Map customer personas (new walk-in, regular, late-night, family).
- Pick channels (Telegram first, WhatsApp second).
- Acceptance criteria: signed-off feature list + sample chat scripts.

#### Phase 1 — Prototype / Walking Skeleton (Sprint 1, weeks 1–2)
*RAD: User Design + Construction loop, throwaway prototype OK.*
- Telegram bot echoing replies via Gemini.
- SQLite schema (menu, orders, sessions).
- Seed menu from Excel.
- One end-to-end happy path: greet → menu → add → order ref.
- **Demo:** owner places one order from his phone.

#### Phase 2 — Agentic Core (Sprints 2–3, weeks 3–6)
*Scrum.*
- Function-calling tools: `get_menu`, `add_to_cart`, `view_cart`, `clear_cart`, `update_customer_info`, `place_order`.
- Bounded agent loop (max 6 iterations) with primary/fallback model.
- Persistent sessions (cart + customer info + rolling history).
- System prompt with time-of-day & mood awareness.
- Idempotency keys on order create.
- **Demo:** five different customer personas tested live.

#### Phase 3 — Reservations + Multi-channel (Sprint 4, weeks 7–8)
*Scrum.*
- `make_reservation` tool + `reservations` table.
- WhatsApp (Twilio) parity for ordering.
- Google Sheets sync for non-tech operators.
- Multilingual mirroring tested for Telugu+English, Hindi+English.
- **Demo:** reservation booked from WhatsApp, appears in Sheets + POS.

#### Phase 4 — POS Frontend (Sprints 5–6, weeks 9–12)
*RAD: rapid UI iteration with Vite + React + lucide.*
- Dashboard, POS & Orders, KDS columns, Tables, Reservations, Menu, Analytics.
- Backend REST: `/api/menu`, `/api/orders`, `/api/orders/{id}/status`.
- Live polling / refresh of orders from chat into KDS.
- Excel-driven historical analytics.
- **Demo:** kitchen runs a full lunch service from KDS.

#### Phase 5 — Operations & Integrations (Sprint 7, weeks 13–14)
*Scrum.*
- Invoices, settlements, payment intents (stubs / Razorpay sandbox).
- n8n webhooks for `order.created`, `order.status_updated`.
- Structured logging, log rotation, error alerting.
- Deployment scripts, `.env.example`, runbook.
- **Demo:** end-to-end with a downstream n8n workflow (e.g., SMS to delivery partner).

#### Phase 6 — Hardening & Pilot (Sprint 8, weeks 15–16)
*Scrum + UAT.*
- Load test agent loop (concurrent chats).
- Auto-migrations, idempotency edge cases, duplicate-order tests.
- Pilot with one real restaurant for 2 weeks; collect feedback in a shared backlog.
- **Demo / Go-live readiness review.**

#### Phase 7 — Launch & Iterate (Continuous)
*Kanban + monthly increments.*
- Production launch with QR + WhatsApp number printed on table tents.
- Weekly metrics review: chats started, orders placed, AOV, cart-abandonment, top dishes.
- Backlog-driven enhancements: payments live, vector search, multi-tenant mode, observability.

### 9.4 Sprint cadence (per sprint, 2 weeks)
| Day | Activity |
|---|---|
| Mon W1 | Sprint planning (story estimation, capacity) |
| Daily | 15-min stand-up |
| Wed W2 | Backlog refinement |
| Fri W2 AM | Sprint review / demo to owner |
| Fri W2 PM | Retrospective + next-sprint goal lock |

### 9.5 Why this hybrid (not pure Waterfall, not pure Scrum)
- **Restaurant requirements evolve** as the owner sees the bot in their hand — pure waterfall would freeze the wrong specs.
- **AI behaviour is empirical**, not specifiable upfront — you have to prompt, demo, adjust. RAD's prototype-feedback-iterate loop is a perfect fit.
- **Backend integrations** (Sheets, payments, Twilio) have real contracts and real quotas — Scrum's sprint discipline keeps them on track.
- **Two ceremonies, one team** — RAD workshops at phase boundaries, Scrum ceremonies inside each phase.

### 9.6 Risk register (live, reviewed each retro)
| Risk | Mitigation |
|---|---|
| LLM hallucinates a price or item | All data via tools; LLM cannot bypass `get_menu`. |
| Gemini quota / outage | Dual-model fallback + graceful error message. |
| Telegram polling drops | Process-supervised; future: webhook mode. |
| Sheets API rate limit | Best-effort sync, never blocks order persistence. |
| Restaurant changes menu mid-service | Excel reseed re-runs in seconds. |
| Multi-instance session collisions | Session keyed by `chat_id` in SQLite (single-writer); future: Redis. |

---

## 10. Roadmap (suggested next steps)
- Move in-memory WhatsApp histories to SQLite (parity with Telegram).
- Replace polling with Telegram webhooks for horizontal scaling.
- Streaming responses for menu listings (lower perceived latency).
- Vector retrieval over menu descriptions for semantic search ("something spicy and creamy").
- Payment integration (Razorpay / UPI deep links) so the entire flow including pay-out happens in chat.
- Multi-tenant mode — same backend serves multiple restaurants by `restaurant_id`.
- Observability: OpenTelemetry traces around the agent loop.

---

## 11. One-line pitch
**Dzukku replaces a marketplace app with a conversation. The customer chats once; the restaurant gets the order, the cash, the data, and a dashboard that runs the kitchen.**
