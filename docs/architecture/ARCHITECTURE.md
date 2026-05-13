# Dzukku Bot — Architecture & Workflow

## 1. System Overview

Dzukku Bot is a Telegram-based AI restaurant assistant for **Dzukku Restaurant (Hyderabad)**. It lets users:
- Order food in-chat via the bot (Excel menu + SQLite backend)
- Order via Zomato / Swiggy using live LLM → MCP integration
- Book table reservations
- Sync orders to Google Sheets + Excel

---

## 2. Tech Stack

| Layer | Technology |
|-------|-----------|
| Bot transport | `python-telegram-bot 21` (polling) |
| API server | FastAPI + Uvicorn |
| AI model | Google Gemini 2.5 Flash (primary) |
| Agent framework | LangGraph `create_react_agent` |
| LLM integration | `langchain-google-genai` |
| MCP bridge | `langchain-mcp-adapters` + `npx mcp-remote` |
| Database | SQLite (`storage/dzukku.db`) |
| Menu source | Excel (`data/Project_Dzukku.xlsx` → Master_Menu sheet) |
| Orders sink | Excel (Orders sheet) + Google Sheets (best-effort) |
| Language | Python 3.11 (venv at `env/`) |
| Node.js | v22 (required for `npx mcp-remote`) |
| Containerisation | Docker (Python 3.11 + Node 20) |

---

## 3. High-Level Architecture

```
Telegram User
      │
      ▼
┌─────────────────────────────────────────────┐
│          python-telegram-bot (polling)       │
│              app/bot/telegram.py             │
│                                              │
│  /start → welcome + platform prompt          │
│  text/buttons → _think_and_reply()           │
│  callbacks → handle_callback()               │
└──────────────┬──────────────────────────────┘
               │
               │  platform_choice = session["ordering_platform"]
               │
       ┌───────┴────────────────────────┐
       │                                │
       ▼                                ▼
Zomato / Swiggy                    Dzukku Bot
(MCP_ENABLED=true)            (in-house ordering)
       │                                │
       ▼                                ▼
┌──────────────┐              ┌──────────────────────┐
│  mcp_agent   │              │   dzukku_agent        │
│  (LangGraph  │              │   (LangGraph ReAct)   │
│   ReAct)     │              │   8 local tools        │
└──────┬───────┘              └──────────┬────────────┘
       │                                 │
       ▼                                 ▼
mcp_clients.py                    Local tools (SQLite
(per-platform                     + Excel + Sheets)
 stdio client)
       │
       ▼
npx mcp-remote
(OAuth bridge)
       │
       ├──▶ https://mcp-server.zomato.com/mcp
       └──▶ https://mcp.swiggy.com/food
```

---

## 4. Platform Selection Flow (User Journey)

```
User sends /start or greets ("hi", "hello", "namaste", ...)
        │
        ▼
Bot sends welcome + inline keyboard:
  ┌─────────────────────────────┐
  │  🍽️ Order via Dzukku Bot    │
  │  🟥 Zomato  │  🟧 Swiggy   │
  └─────────────────────────────┘
        │
        ├── Clicks "Dzukku Bot"
        │       └─ session.ordering_platform = "Dzukku"
        │          → Quick-action keyboard shown
        │          → All text goes to dzukku_agent
        │
        ├── Clicks "Zomato"  (MCP_ENABLED=true)
        │       └─ session.ordering_platform = "Zomato"
        │          → "Connecting to Zomato..." message
        │          → mcp_agent greets user via Zomato MCP
        │
        ├── Clicks "Swiggy"  (MCP_ENABLED=true)
        │       └─ session.ordering_platform = "Swiggy"
        │          → mcp_agent greets user via Swiggy MCP
        │
        └── Clicks Zomato/Swiggy  (MCP_ENABLED=false)
                └─ Sends redirect button → user opens Zomato/Swiggy app
```

---

## 5. Dzukku Bot Flow (In-House Ordering)

```
User: "I want 2 Chicken Biryani and a Mango Lassi"
                │
                ▼
        dzukku_agent.get_dzukku_response()
                │
                ▼
     LangGraph ReAct loop:
     ┌─────────────────────────┐
     │  System prompt          │  ← restaurant persona + session state
     │  + rolling history      │     (customer_name, phone, cart, time)
     │  + user message         │
     └──────────┬──────────────┘
                │
                ▼
         Gemini 2.5 Flash   ─── (fallback: 2.0-flash → 1.5-flash)
                │
      ┌─────────▼──────────────────────────────────────┐
      │  Thinks: "I need to add items to cart"          │
      │  Calls:  add_to_cart([{item_name, qty}, ...])   │
      └─────────┬──────────────────────────────────────┘
                │  tool executes via ContextVar → session
                ▼
      ┌─────────────────────────────────────────────────┐
      │  Tool result: {added: [...], cart_total: 340}   │
      └─────────┬───────────────────────────────────────┘
                │
      ┌─────────▼──────────────────────────────────────┐
      │  Thinks: "Should ask for name/phone to confirm" │
      │  Responds: "Added! Your total is ₹340...        │
      │             What's your name and number?"       │
      └────────────────────────────────────────────────┘
                │
                ▼
     User: "Krishna Reddy, 9999999999"
                │
                ▼
      Calls: update_customer_info(name, phone)
      Calls: view_cart()   ← confirms items
      Asks:  "Confirm your order?"
                │
                ▼
      User: "Yes"
                │
                ▼
      Calls: place_order()
              │
              ├─ save_order()          → SQLite
              ├─ excel_append_order()  → data/Project_Dzukku.xlsx (Orders sheet)
              └─ sync_order_to_sheet() → Google Sheets (best-effort)
                │
                ▼
      Reply: "🧾 Order Confirmed! #DZK-XXXX ..."
```

### Dzukku Agent Tools (8 total)

| Tool | Purpose |
|------|---------|
| `get_menu` | Reads from SQLite (seeded from Excel Master_Menu). Supports type/category filters. |
| `add_to_cart` | Adds items by name (fuzzy match). Stores in session. |
| `view_cart` | Returns current cart + total. |
| `clear_cart` | Empties cart. |
| `update_customer_info` | Saves name + phone to session. |
| `place_order` | Finalises order → SQLite + Excel + Sheets. |
| `make_reservation` | Books table → SQLite + Excel + Sheets. |
| `get_restaurant_info` | Returns static info (timings, location, cuisine). |

---

## 6. MCP Flow (Zomato / Swiggy Live Ordering)

```
User clicks Zomato  (MCP_ENABLED=true)
        │
        ▼
mcp_clients.get_mcp_tools_async("Zomato")
        │
        ├── [Cache hit] return cached tools
        │
        └── [First time]
               │
               ▼
        MultiServerMCPClient({
          "zomato": {
            transport: "stdio",
            command: "npx",
            args: ["-y", "mcp-remote", "https://mcp-server.zomato.com/mcp"]
          }
        })
               │
               ▼
        npx mcp-remote  ──OAuth──▶  mcp-server.zomato.com
          (subprocess)              (tokens cached ~/.mcp-auth/)
               │
               ▼
        tools/list  →  11 Zomato tools returned
        Session closes (normal — per-tool sessions)
               │
               ▼
        Schema sanitized:
          - integer enum values → strings  (Gemini requirement)
          - additionalProperties stripped  (Gemini requirement)
               │
               ▼
        create_react_agent(llm_with_fallbacks, tools)
               │
               ▼
        Agent greets user and starts ordering conversation
               │
               ▼
        Each tool call:  npx mcp-remote spawns → executes → returns → exits
```

### MCP Token Strategy (Shared Bot-Owner Token)

```
One-time setup (on the bot host):
  npx -y mcp-remote https://mcp-server.zomato.com/mcp
  npx -y mcp-remote https://mcp.swiggy.com/food
    │
    ▼
  Browser opens → bot owner logs in
  Tokens cached at ~/.mcp-auth/

All Telegram users share the bot owner's Zomato/Swiggy account.
Suitable for: demo, family bot, single-restaurant showcase.
```

---

## 7. Routing Matrix (`telegram.py:_think_and_reply`)

```
platform_choice = session["ordering_platform"]

platform = "Zomato" or "Swiggy"
AND MCP_ENABLED = true
        │
        └─▶  mcp_agent.get_mcp_response()
                  └─ fallback: legacy orchestrator (if agent returns None)

platform = "Dzukku" (or empty)
        │
        └─▶  dzukku_agent.get_dzukku_response()
                  └─ fallback: legacy orchestrator (if LangGraph unavailable)

Any agent returns None
        │
        └─▶  orchestrator.get_bot_response()  (legacy Gemini function-calling loop)

platform = "Zomato" or "Swiggy"
AND MCP_ENABLED = false
        │
        └─▶  Send redirect button → open Zomato/Swiggy app
```

---

## 8. LLM Fallback Chain

Every agent uses the same 3-tier fallback:

```
Request
   │
   ▼
gemini-2.5-flash  (primary — fast, capable)
   │  503 / 429 / SDK retries exhausted
   ▼
gemini-2.0-flash  (fallback 1 — stable, less traffic)
   │  fails again
   ▼
gemini-1.5-flash  (fallback 2 — old reliable)
```

Override in `.env`:
```
GEMINI_PRIMARY_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-2.0-flash
GEMINI_FALLBACK_2_MODEL=gemini-1.5-flash
```

---

## 9. Session State (SQLite `sessions` table)

| Column | Type | Purpose |
|--------|------|---------|
| `chat_id` | INTEGER PK | Telegram chat ID |
| `user_name` | TEXT | First name |
| `state` | TEXT | Conversation state |
| `history` | JSON | Last 16 turns (role + content) |
| `cart` | JSON | `[{item_name, qty, price, type}]` |
| `customer_name` | TEXT | Name for orders/reservations |
| `customer_phone` | TEXT | Phone for orders/reservations |
| `ordering_platform` | TEXT | `"Dzukku"` / `"Zomato"` / `"Swiggy"` |
| `updated_at` | DATETIME | Last activity |

---

## 10. File Structure

```
DzukkuBot/
├── main.py                        # Uvicorn entrypoint
├── requirements.txt               # Python deps (Python 3.11+)
├── Dockerfile                     # Python 3.11 + Node 20 image
├── docker-compose.yml             # cloud profile (MCP off) + mcp profile (MCP on)
├── .env                           # secrets + feature flags (not in git)
├── .env.example                   # template
├── scripts/
│   └── export_mcp_auth.sh         # packages ~/.mcp-auth for cloud upload
│
├── app/
│   ├── api/
│   │   └── main.py                # FastAPI app, lifespan hooks
│   │
│   ├── bot/
│   │   └── telegram.py            # Telegram handlers, routing matrix
│   │
│   ├── agent/
│   │   ├── mcp_clients.py         # Per-platform MultiServerMCPClient (lazy, cached)
│   │   ├── mcp_agent.py           # LangGraph ReAct agent — Zomato/Swiggy via MCP
│   │   ├── dzukku_agent.py        # LangGraph ReAct agent — in-house Dzukku tools
│   │   └── orchestrator.py        # Legacy Gemini function-calling loop (fallback)
│   │
│   └── core/
│       ├── config.py              # All settings (env-backed)
│       ├── database.py            # SQLite CRUD + menu seeding
│       ├── sheets.py              # Google Sheets sync
│       ├── excel_sink.py          # Excel Orders/Reservations append
│       └── logging_config.py      # Rotating file + console logger
│
├── data/
│   └── Project_Dzukku.xlsx        # Master_Menu, Orders, Reservations sheets
│
├── config/
│   └── credentials.json           # Google service account key (not in git)
│
└── storage/
    └── dzukku.db                  # SQLite database (auto-created)
```

---

## 11. Deployment

### Local Development

```bash
cd DzukkuBot
source env/bin/activate            # Python 3.11 venv
python main.py
```

### Docker — Cloud (redirect-link UX, no MCP)

```bash
cp .env.example .env               # fill TELEGRAM_TOKEN + GEMINI_API_KEY
docker compose --profile cloud up -d --build
```

### Docker — MCP (live ordering, single-tenant)

```bash
# One-time OAuth (on the host)
npx -y mcp-remote https://mcp-server.zomato.com/mcp
npx -y mcp-remote https://mcp.swiggy.com/food

# Deploy
docker compose --profile mcp up -d --build
```

### Moving MCP tokens to another host

```bash
./scripts/export_mcp_auth.sh          # creates mcp-auth-export.tar.gz
# Upload as a secret, extract to the dir set in MCP_AUTH_DIR
```

---

## 12. Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TELEGRAM_TOKEN` | — | Bot token from @BotFather |
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `GOOGLE_SHEET_ID` | — | Google Sheet for order sync |
| `MCP_ENABLED` | `false` | `true` = live Zomato/Swiggy ordering via MCP |
| `MCP_AUTH_DIR` | `""` | Path to mcp-remote token cache (Docker/cloud) |
| `GEMINI_PRIMARY_MODEL` | `gemini-2.5-flash` | Primary LLM |
| `GEMINI_FALLBACK_MODEL` | `gemini-2.0-flash` | First fallback |
| `GEMINI_FALLBACK_2_MODEL` | `gemini-1.5-flash` | Second fallback |
| `MCP_ZOMATO_ENABLED` | `true` | Enable/disable Zomato MCP |
| `MCP_SWIGGY_FOOD_ENABLED` | `true` | Enable/disable Swiggy MCP |
| `NPX_BIN` | `npx` | Path to npx executable |

---

## 13. Greeting Re-Prompt Logic

When a user sends a greeting (`hi`, `hello`, `namaste`, etc.) and their **cart is empty**, the bot re-shows the platform selection prompt. This handles:
- New users starting a fresh session
- Returning users wanting to switch platforms

If the cart is non-empty (order in progress), the greeting is passed to the active agent instead so the conversation isn't interrupted.

---

## 14. Schema Compatibility Patches

The Zomato MCP server returns tool schemas with **integer enum values** (e.g. sort options `[1, 2, 3]`). Gemini requires all enum values to be strings. A one-time monkey-patch in `mcp_agent._patch_gemini_schema_converter()` rewrites the `langchain_google_genai` schema converter to:
1. Convert `enum: [1, 2, 3]` → `enum: ["1", "2", "3"]`
2. Strip `additionalProperties` (unsupported by Gemini)

This is applied once at import time and covers all recursive schema processing.
