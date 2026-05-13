# Dzukku — Implemented Features & Testing Guide

---

## Part 1: Implemented Features

### 1. Telegram Bot (Dzukku Bot)

**Commands**
| Command | What it does |
|---------|-------------|
| `/start` | Resets session, greets user, shows platform picker (Dzukku / Zomato / Swiggy) |
| `/menu` | Shows full menu with prices |
| `/order` | Initiates an order flow |
| `/reserve` | Initiates a table reservation flow |
| `/cart` | Shows current cart contents + total |
| `/reset` | Clears session and starts fresh |
| `/help` | Lists all commands |

**Keyboards & UI**
- Persistent reply keyboard: Menu, Specials, Order, Reserve a Table, My Cart, Info
- Inline quick-action buttons (same set as above, callback-driven)
- Platform selection inline keyboard: Dzukku Bot / Zomato / Swiggy
- Post-order star rating keyboard (1–5 stars per order)
- Typing indicator while the agent thinks

**Greetings**
- Detects 30+ greeting words/phrases (hi, hello, namaste, salaam, gm, etc.)
- Re-shows platform selection only before a platform is chosen
- Greetings during active Zomato/Swiggy sessions continue that session

---

### 2. Dzukku In-House Ordering (5-Stage Pipeline)

**Stage 1 — Context Builder**
- Loads customer profile (name, phone, language, marketing opt-in)
- Live cart contents + cart total
- Active orders in flight (last 5)
- Kitchen load (orders being prepared)
- Full menu snapshot (available items only)
- Last 8 conversation turns (rolling history)
- Session meta: pending goal, pending slots, upsell count, order type
- Restaurant open/closed status, time-of-day label
- User memory summary + top cravings (personalisation)

**Stage 2 — Planner (LLM)**
- Gemini 2.5 Flash decides what tools to call
- Accepts slot hints for name/phone/address (pre-filled from message parsing)
- Produces a JSON plan: goal, slots, proposed_actions

**Stage 3 — Executor (Deterministic)**
| Tool | Function |
|------|----------|
| `add_to_cart` | Add items; price always read from DB, not from LLM |
| `remove_from_cart` | Remove item by name/id |
| `update_cart_item` | Change quantity |
| `clear_cart` | Empty the cart |
| `view_cart` | Return cart snapshot |
| `set_order_type` | DELIVERY / PICKUP / DINE_IN |
| `set_delivery_address` | Validate + save delivery address |
| `update_customer` | Save name, phone, address |
| `place_order` | Re-reads DB prices, creates Order + OrderItems, syncs Excel/Sheets |
| `track_order` | Returns live order status + ETA |
| `cancel_order` | Cancels within policy window |
| `make_reservation` | Creates reservation with date/time/guests/name/phone |
| `get_menu` | Returns filtered/full menu |
| `search_menu` | Keyword/category search |
| `get_item_details` | Item info + modifiers |
| `get_kitchen_eta` | Estimated prep time based on kitchen load |
| `get_restaurant_info` | Address, timings, delivery radius |
| `create_payment_intent` | Razorpay order creation |
| `check_payment_status` | Poll Razorpay for payment status |
| `open_table_session` | Start dine-in table session |
| `add_table_order` | Add order to existing table session |
| `close_table_session` | Close table and generate bill |
| `generate_invoice` | PDF invoice for order/session |
| `set_item_availability` | Toggle menu item on/off |
| `update_stock` | Adjust stock level |
| `update_order_status` | Move order through lifecycle |
| `assign_driver` | Assign delivery driver |
| `update_delivery_status` | Update driver/delivery progress |

**Stage 4 — Verifier**
- Re-reads DB after execution
- Validates cart totals match DB prices
- Confirms item availability
- Checks policy compliance (operating hours, cancellation windows, kitchen capacity)

**Stage 5 — Responder (LLM)**
- Converts verified facts into a friendly Telegram reply
- Multilingual: English, Telugu (te), Hindi (hi)
- Language persisted in DB (`Customer.language_pref`)

**Order State Machine**
```
IDLE → BROWSING_MENU → BUILDING_CART → COLLECTING_DETAILS
→ AWAITING_CONFIRMATION → AWAITING_PAYMENT → ORDER_PLACED
→ ORDER_IN_PROGRESS → OUT_FOR_DELIVERY → COMPLETED
```
Support case, session-closed, and payment-failed paths also handled.

**Order Types**
- DELIVERY — requires address collection
- PICKUP — no address required
- DINE_IN — table session flow

**Personalization (Memory Agent)**
- Taste vector updated after every completed order
- Implicit signals: reorders (+0.1), cancellation (−0.15), 5 stars (+0.2), 1 star (−0.3)
- Rating from post-order inline keyboard updates taste vector immediately
- Top cravings + memory summary injected into planner prompt

**Policies (Enforced)**
- Operating hours check (restaurant open/closed)
- Kitchen capacity gate (max concurrent orders)
- Cancellation window enforcement
- Language detection + persistence

---

### 3. Zomato / Swiggy MCP Integration

- LangGraph ReAct agent with Gemini 2.5 Flash
- Tools loaded live from Zomato and Swiggy MCP servers via `npx mcp-remote`
- OAuth handled by `mcp-remote` on loopback (no public callback needed)
- Per-platform stdio MCP client (one process per platform per session)
- Rolling conversation history replayed into prompt (same PostgreSQL session)
- Error classification: rate-limit, auth failure, service unavailable, timeout
- Graceful fallback: MCP down → external app link (Zomato/Swiggy URL)
- Toggle: controlled by `MCP_ENABLED` env flag

---

### 4. REST API (FastAPI)

| Module | Endpoints |
|--------|-----------|
| Auth | `POST /auth/login` |
| Orders | `GET/PATCH /orders`, order state, mark-paid, per-item status |
| Kitchen | `GET /kitchen/orders` (KDS feed) |
| Menu | `GET/POST/PATCH /menu/items`, toggle availability, upload images |
| Tables | Full CRUD: tables, sessions, session orders, fire-to-kitchen, invoice |
| Reservations | `GET/PATCH /reservations` |
| Deliveries | List, drivers, assign, status update, proof upload, live tracking |
| Payments | Razorpay payment intent creation, webhook handler |
| Staff | List, create, toggle active |
| Invoices | `GET /invoices` |

All endpoints require JWT auth (except login).

---

### 5. Frontend — Next.js Dashboards

**Login Page**
- Role-based login (ADMIN / WAITER / KITCHEN)
- JWT stored in AuthContext
- Redirects to role-specific dashboard

**Admin Dashboard**
- Sidebar navigation with 11 sections
- Real-time clock
- Views: Dashboard, Orders, Deliveries, KDS, Tables, Reservations, Menu, Employees, Invoices, Settlements, Analytics
- Analytics: revenue-by-hour area chart (Recharts)
- WebSocket live updates

**Waiter Portal**
- Table map with colour-coded status (Available / Occupied / Reserved / Inactive)
- Open table sessions, add items from live menu
- Fire-to-kitchen button
- Per-item status tracking (Pending / Cooking / Done / Cancelled)
- Generate invoice / billing flow
- **Offline mode**: caches last floor data, queues orders locally, syncs when back online

**Kitchen Display System (KDS)**
- Live incoming orders with item-level status
- Station filters: All / DINE_IN / DELIVERY / PICKUP
- Per-item status update: Pending → Cooking → Done / Cancelled
- Mark full order as READY
- WebSocket real-time push updates

**Order Tracking Page**
- Public-facing order status page (by order ref)

---

### 6. Infrastructure & Background

- **PostgreSQL** + Alembic migrations (fully async via SQLAlchemy + asyncpg)
- **Redis** — session cache + Celery broker
- **Celery** — async task queue (notifications, reports, scheduled jobs)
- **pgvector** — vector embeddings for semantic menu/preference search
- **Google Sheets sync** — orders mirrored to spreadsheet (best-effort)
- **Excel sink** — orders written to `data/Project_Dzukku.xlsx`
- **Razorpay** — payment gateway (intents + webhook)
- **WebSocket** — real-time push to all frontend dashboards (via `app/realtime`)
- **Docker** — Python 3.11 + Node 20 image; `docker-compose.yml` included
- **CI/CD** — Azure pipelines + GitHub Actions scaffolded (Sprint 0 complete)
- **Logging** — structured logging with rotation (`app/core/logging_config.py`)

---

## Part 2: Testing Questions

> Use these to manually test each surface. Run the bot in Telegram and the frontend at `http://localhost:3000`.

---

### A. Dzukku Bot — Basic Flow

1. Send `/start` — does it greet you by first name and show the platform picker?
2. Tap "🍽️ Order via Dzukku Bot" — does it show the quick-action keyboard?
3. Type "hi" before choosing a platform — does it re-show the platform picker?
4. Type "hi" after choosing Dzukku — does it NOT re-show the picker and instead route to the agent?
5. Send `/menu` — do you see a formatted menu with item names and prices in ₹?
6. Type "show me vegetarian items" — does it filter and show only veg items?
7. Type "What are today's specials?" — does the bot respond with featured/special items?
8. Send `/help` — are all 7 commands listed?
9. Send `/reset` — does it confirm session reset and start fresh?
10. Send a random message like "Are you open now?" — does the bot respond with opening hours?

---

### B. Dzukku Bot — Ordering Flow

11. Type "I want 2 Chicken Biryani" — does the bot add them to cart and confirm?
12. Type "add 1 Butter Naan" — does it add to the existing cart without clearing it?
13. Type "remove Butter Naan" — is it removed and cart total updated?
14. Send `/cart` — is the cart accurate with correct per-item prices?
15. Type "change Chicken Biryani to 3" — does quantity update correctly?
16. Type "I want delivery" — does it ask for your delivery address?
17. Provide a delivery address — does the bot accept it and move to name/phone collection?
18. Provide your name and phone number — does it show an order summary for confirmation?
19. Confirm the order — does it place the order, give you an order reference number, and show a rating keyboard?
20. Type "track my order" — does it show the current status?
21. Type "cancel my order" — does it cancel (within window) and confirm?
22. Type "I want to pick up my order" — does it switch to PICKUP and skip address collection?
23. Order an item that does not exist on the menu — does the bot say it's not available and suggest alternatives?

---

### C. Dzukku Bot — Reservation Flow

24. Type "book a table for 4 people on Saturday at 7 PM" — does it collect all missing slots?
25. Provide only date and time (no guest count) — does it ask for number of guests?
26. Provide your name and phone — does it confirm the reservation with a reference?
27. Ask "do I have any upcoming reservations?" — does it show your booking?

---

### D. Dzukku Bot — Multilingual

28. Type "నమస్కారం, మీ menu చూపించండి" (Telugu) — does the bot reply in Telugu?
29. Type "नमस्ते, मुझे menu दिखाओ" (Hindi) — does the bot reply in Hindi?
30. After a Telugu conversation, send `/reset` and type in English — does the bot switch back to English?

---

### E. Dzukku Bot — Edge Cases

31. Send a message when the restaurant is closed (after hours) — does the bot mention closed status?
32. Order more than the kitchen capacity limit — does the bot warn about high kitchen load?
33. Send a location using Telegram's location share — does the bot acknowledge your coordinates?
34. Type nonsense like "xyzabc123!@#" — does the bot handle gracefully without crashing?
35. Close and reopen the chat — is your cart still intact from the previous session?
36. Rate your last order with 1 star — does the bot confirm and update preferences?
37. Rate your last order with 5 stars — does the bot confirm?

---

### F. Zomato / Swiggy MCP Integration

38. Send `/start`, tap "🟥 Zomato" — does it connect and greet you in Telegram?
39. After selecting Zomato, type "show me biryani places near Hyderabad" — does it return live results?
40. Ask Zomato "add chicken biryani to my cart" — does it confirm item added?
41. Ask "what's in my cart?" — does it return the Zomato cart state?
42. Type "checkout" — does it initiate the Zomato checkout flow?
43. Repeat steps 38–42 for Swiggy.
44. Send `/start`, pick Zomato, then type "switch to Dzukku" — does selecting "↩️ Order via Dzukku Bot instead" work?
45. With `MCP_ENABLED=false`: tap Zomato — do you get the "Open Zomato" link button instead of an agent?
46. Simulate MCP server down (stop the MCP process) — does the bot show the fallback link gracefully?

---

### G. Frontend — Login

47. Open `http://localhost:3000` without being logged in — are you redirected to `/login`?
48. Log in with role ADMIN — are you directed to `/admin`?
49. Log in with role WAITER — are you directed to `/waiter`?
50. Log in with role KITCHEN — are you directed to `/kitchen`?
51. Try to access `/admin` while logged in as WAITER — are you redirected away?

---

### H. Frontend — Admin Dashboard

52. Log in as ADMIN — do all 11 sidebar items render without errors?
53. Does the live clock tick in the header?
54. Go to "Orders" — do active orders load from the API?
55. Change an order's status (e.g., ACCEPTED → PREPARING) — does it update in the list?
56. Go to "Menu" — are all menu items listed with prices and availability toggles?
57. Toggle a menu item's availability — does it update and reflect immediately?
58. Go to "Tables" — are tables displayed with correct status colours?
59. Go to "Reservations" — are upcoming reservations listed?
60. Go to "Analytics" — does the revenue-by-hour chart render with real data?
61. Go to "Employees" — are staff records listed?
62. Go to "Deliveries" — are active deliveries shown with driver details?
63. Go to "Invoices" — are invoices listed and filterable?
64. Log out — are you redirected to `/login` and JWT cleared?

---

### I. Frontend — Waiter Portal

65. Log in as WAITER — does the table map load with coloured table tiles?
66. Click an available table — can you open a new session?
67. Add 2 items to the cart from the menu — does the cart total update correctly?
68. Add the same item twice — does the quantity increment instead of duplicating?
69. Fire the order to kitchen — does the table status change to OCCUPIED?
70. Check that the fired order appears in the Kitchen Display.
71. Click "Billing" on an occupied table — does it show the correct total?
72. Generate an invoice — does it create an invoice in the system?
73. Turn off your internet / simulate offline — does the "Offline mode" banner appear?
74. In offline mode, add an item — is it queued locally?
75. Restore internet — does the offline queue sync automatically?

---

### J. Frontend — Kitchen Display (KDS)

76. Log in as KITCHEN — do pending orders appear with item details?
77. Change a single item status to "Cooking" — does only that item update?
78. Change a single item status to "Done" — does it reflect immediately?
79. When all items are done, mark the order as READY — does it leave the active queue?
80. Filter by "DELIVERY" station — do only delivery orders show?
81. Filter by "DINE_IN" — do only dine-in orders show?
82. Place a new order via the bot — does it appear in the KDS within seconds (WebSocket push)?

---

### K. Payments

83. Initiate an order requiring payment — does the bot provide a payment link / Razorpay flow?
84. Simulate payment success via Razorpay test mode — does the order status update to PAID?
85. Simulate payment failure — does the bot inform the user and allow retry?
86. Check `/admin` → Orders after payment — is the order marked paid with correct amount?

---

*Last updated: May 2026*
