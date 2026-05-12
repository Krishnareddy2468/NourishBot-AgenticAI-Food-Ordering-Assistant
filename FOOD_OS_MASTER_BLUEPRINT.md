# DZUKKU → FOOD OPERATING SYSTEM

## Master Blueprint — Agentic AI Food OS for India

> **Document Type:** Strategic + Technical Blueprint
> **Status:** Planning Phase
> **Built on:** Current Dzukku stack (FastAPI + Gemini 2.5 + Telegram + PostgreSQL)
> **Vision horizon:** 18–36 months

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Current State Audit](#2-current-state-audit)
3. [Product Strategy](#3-product-strategy)
4. [Agentic AI Architecture](#4-agentic-ai-architecture)
5. [WhatsApp-First Experience](#5-whatsapp-first-experience)
6. [Personalization Engine](#6-personalization-engine)
7. [Savings Optimization Engine](#7-savings-optimization-engine)
8. [Nutrition + Health Layer](#8-nutrition--health-layer)
9. [Local Commerce Intelligence](#9-local-commerce-intelligence)
10. [Business Model](#10-business-model)
11. [India-Specific Advantages](#11-india-specific-advantages)
12. [Go-To-Market Strategy](#12-go-to-market-strategy)
13. [Competitive Analysis](#13-competitive-analysis)
14. [Tech Stack Evolution](#14-tech-stack-evolution)
15. [Execution Roadmap](#15-execution-roadmap)
16. [Future Vision](#16-future-vision)

---

## 1. EXECUTIVE SUMMARY

**What we are building:**
Dzukku is not a food delivery app. It is the Operating System for how India thinks about, orders, and experiences food — running natively through WhatsApp and Telegram, powered by agentic AI that learns, remembers, negotiates, and acts on behalf of the user.

**The core shift:**
From a restaurant bot (current) → to a personal AI food concierge that serves millions of users across thousands of restaurants, home chefs, cloud kitchens, and local vendors.

**The single sentence:**
"Dzukku is the AI food brain in your phone — it knows what you want before you do, finds the best deal, orders it, and makes you healthier while doing it."

**Why now:**

- Gemini 2.5 / GPT-4o multimodal capabilities are mature enough for production agentic systems
- WhatsApp Business API is now available at scale in India
- UPI is ubiquitous — payment friction is solved
- India has 500M+ smartphone users with no truly intelligent food layer
- The current Dzukku stack (Gemini + FastAPI + PostgreSQL + Telegram pipeline) is the exact foundation to build this on

---

## 2. CURRENT STATE AUDIT

### What exists today (Dzukku v1)

```
Telegram Bot (single restaurant)
    ↓
5-Stage Pipeline:
  ContextBuilder → Planner → Executor → Verifier → Responder
    ↓
PostgreSQL (orders, sessions, menu, reservations)
    ↓
React POS Frontend (Admin / Waiter / Kitchen / Tracking)
    ↓
Razorpay (payments)
    ↓
MCP Bridge → Zomato / Swiggy ordering
```

### Strengths to carry forward

- Proven agentic pipeline (Planner/Executor/Verifier pattern) — production-grade
- LLM-never-writes-to-DB safety architecture
- Gemini 2.5 Flash function-calling integration
- Multi-agent via MCP (Zomato, Swiggy connectors)
- PostgreSQL schema with `restaurant_id` multi-tenant foundation
- Role-based POS frontend (Admin/Waiter/Kitchen)
- State machine for conversation flow

### Gaps to close

- Single restaurant only → needs multi-tenant, multi-vendor
- No persistent user memory across sessions
- No personalization engine — each session starts cold
- No WhatsApp channel (only Telegram)
- No nutrition/health layer
- No savings optimization
- No proactive / push AI behavior
- No voice input
- No vendor negotiation intelligence
- No hyperlocal discovery

---

## 3. PRODUCT STRATEGY

### 3.1 Positioning

**NOT:** A food delivery app with a chatbot bolted on
**IS:** A conversational AI agent that happens to execute food orders as one of many actions

The user never opens a menu. The AI surfaces the right options at the right time.

### 3.2 Differentiation Matrix

| Dimension       | Zomato/Swiggy           | Dzukku Food OS                         |
| --------------- | ----------------------- | -------------------------------------- |
| Interface       | App (browse-first)      | WhatsApp/Telegram (conversation-first) |
| Memory          | Order history only      | Full behavioral + nutritional memory   |
| Personalization | Collaborative filtering | Individual AI model per user           |
| Proactivity     | Push notifications      | Contextual AI nudges with intent       |
| Savings         | Manual coupons          | Autonomous savings agent               |
| Nutrition       | None                    | Real-time health-aware ordering        |
| Vendor rel.     | Platform → vendor      | AI negotiation + dynamic pricing       |
| Discovery       | Ranked lists            | Predictive concierge                   |
| Voice           | None                    | Native voice ordering                  |
| Multilingual    | Limited                 | Full regional language support         |

### 3.3 Why WhatsApp/Telegram-First Wins in India

1. **Zero install friction** — 500M+ users already have WhatsApp
2. **Trusted surface** — Indians transact with family/businesses over WhatsApp
3. **Low-end device friendly** — No 200MB app, no storage wars
4. **Notification open rates** — WhatsApp: 98% vs App push: 12%
5. **Voice native** — Indians prefer voice messages; WhatsApp has native voice
6. **Group ordering** — WhatsApp groups enable viral group food orders
7. **Tier-2/Tier-3 reach** — Swiggy/Zomato barely penetrate; WhatsApp is universal

### 3.4 Strategic Moats

**Moat 1 — User Memory**
Every interaction deepens the user model. After 30 orders, Dzukku knows more about your food preferences than any app ever will. This is a compounding, proprietary data asset.

**Moat 2 — Vendor Intelligence Network**
Aggregated demand signals across thousands of users give Dzukku negotiating power with restaurants that no individual user has.

**Moat 3 — Conversational UI Lock-in**
Once users experience frictionless AI ordering, going back to scrolling a menu feels like regression. The behavior change is the moat.

**Moat 4 — Nutrition Graph**
A longitudinal food + health graph per user. Impossible for any competitor to replicate without starting over.

**Moat 5 — ONDC + Local Vendor Network**
Early integration with India's ONDC protocol + relationships with home chefs and local tiffin services creates supply no incumbent can quickly replicate.

### 3.5 Retention Loops

```
User orders → AI learns preference → next recommendation is better
     ↑                                            ↓
User trusts AI more ← order satisfaction increases ←
```

```
Savings agent finds deal → user saves money → user orders again sooner
     ↑                                              ↓
Dzukku earns commission ← higher order frequency ←
```

### 3.6 Virality Loops

- Group ordering: one user invites friends → Dzukku handles split orders
- Referral: "My AI food agent saved me ₹800 this month" → organic sharing
- Streak sharing: "I've hit my protein goal 14 days in a row" → WhatsApp status
- Restaurant-side: vendors share "Order via Dzukku for 15% off" → user acquisition

---

## 4. AGENTIC AI ARCHITECTURE

### 4.1 Multi-Agent System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    DZUKKU AGENT ORCHESTRATOR                    │
│                    (LangGraph state machine)                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────────────┐
        │              │                       │
        ▼              ▼                       ▼
┌──────────────┐ ┌──────────────┐    ┌──────────────────┐
│  PERCEPTION  │ │   MEMORY     │    │    EXECUTION     │
│   AGENTS     │ │   AGENTS     │    │    AGENTS        │
└──────┬───────┘ └──────┬───────┘    └────────┬─────────┘
       │                │                     │
       ▼                ▼                     ▼
 Intent Parser    Short-term ctx       Tool Executor
 Slot Extractor   Long-term prefs      DB Writer
 Sentiment        Nutrition graph      Payment Agent
 Voice→Text       Spending history     Logistics Agent
 Language ID      Craving model        Notif Agent
```

### 4.2 Agent Definitions

#### A. Intent Agent

- Classifies every user message into intent categories
- Intents: `order`, `discover`, `health_check`, `savings_query`, `track`, `cancel`, `complaint`, `chitchat`, `voice_order`
- Uses Gemini 2.5 Flash for classification with confidence scoring
- Falls back to rule-based matching for high-frequency intents (speed)

#### B. Memory Agent

```
Short-term memory (per session):
  - Current cart state
  - Active conversation context
  - Slot values being collected

Long-term memory (persistent, per user):
  - Cuisine preferences (weighted vector)
  - Timing patterns (breakfast/lunch/dinner/late-night)
  - Price sensitivity score
  - Spice tolerance
  - Dietary constraints
  - Health goals
  - Past orders with ratings
  - Spending patterns by category/time
  - Emotional state patterns
```

**Implementation:**

- Short-term: Redis (TTL = session lifetime)
- Long-term: PostgreSQL (structured) + pgvector (preference embeddings)
- RAG: Retrieve relevant past orders/preferences to inject into every LLM prompt

#### C. Planner Agent

- Given intent + memory snapshot → produce an action plan
- Outputs: `{goal, slots_needed, proposed_actions[], confidence}`
- Constrained: can only propose, never execute directly
- Current Dzukku Planner is the foundation — extend with memory injection

#### D. Nutrition Agent

```python
NutritionContext:
  daily_calories_consumed: int
  daily_protein_g: float
  daily_fiber_g: float
  health_goals: List[str]  # ["high-protein", "diabetic-friendly", "weight-loss"]
  diet_type: str           # vegetarian | non-veg | vegan
  allergies: List[str]
  bmi_category: str        # optional, user-provided
```

- Scores each menu item against user's nutritional context
- Injects health-aware suggestions into Planner prompt
- Generates nudges: "You've had 1800 cal today. Light dinner?"

#### E. Savings Agent

```
For each order intent:
  1. Query current prices across platforms (Zomato/Swiggy MCP + direct)
  2. Apply available coupons automatically
  3. Identify combo opportunities
  4. Compare equivalent items at nearby restaurants
  5. Check time-based discounts (happy hours, late-night deals)
  6. Apply loyalty points/wallet balance
  7. Output: best_option, savings_amount, rationale
```

#### F. Negotiation Agent (V2)

- Communicates with restaurant-side Dzukku dashboard via API
- When demand is low (predicted via ML), proposes dynamic discount offers to users
- Restaurants set floor prices; agent negotiates above floor
- Creates win-win: restaurants fill capacity, users get deals

#### G. Recommendation Agent

```
Input:
  - Time of day
  - Day of week
  - Weather (API)
  - User mood signal (from message sentiment)
  - Past orders at this time
  - Health context
  - Budget context

Output:
  - Top 3 personalized recommendations
  - Each with: item, restaurant, price, savings, health score, confidence
```

#### H. Logistics Agent

- Post-order: tracks delivery status
- Proactively notifies: "Your order is 5 min away"
- Handles escalations: "Driver is lost, I've shared your location"
- Interfaces: Razorpay (payment), restaurant WebSocket (status), maps API

#### I. Payment Agent

- Validates cart totals
- Selects optimal payment method (UPI > wallet > card)
- Applies cashback automatically
- Generates and sends invoice
- Handles refunds conversationally: "Your ₹180 refund will hit UPI in 2 hours"

### 4.3 Memory Architecture (Deep Dive)

```
USER MESSAGE
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                MEMORY RETRIEVAL LAYER                │
│                                                      │
│  1. Session store (Redis) → last 20 turns            │
│  2. Preference store (PostgreSQL) → top preferences  │
│  3. Vector search (pgvector) → semantic past orders  │
│  4. Nutrition store → today's food log               │
│  5. Spending store → this week/month totals          │
└──────────────────────────────────────────────────────┘
    │
    ▼
ASSEMBLED CONTEXT SNAPSHOT (injected into LLM prompt)
    │
    ▼
PLANNER LLM → action plan
    │
    ▼
EXECUTOR (deterministic) → DB writes
    │
    ▼
RESPONDER LLM → friendly message
```

### 4.4 Long-Term Preference Modeling

Each user has a **Taste Vector** — updated after every order:

```
TasteVector:
  spice_level: float          # 0.0 (no spice) → 1.0 (extra hot)
  cuisine_weights: dict       # {"biryani": 0.8, "pizza": 0.3, "idli": 0.6}
  price_band: str             # budget | mid | premium
  order_timing: dict          # {"breakfast": 0.2, "lunch": 0.7, "dinner": 0.9}
  platform_preference: str    # direct | zomato | swiggy
  dietary_flags: list         # ["no-pork", "less-oil"]
  craving_cycles: dict        # {"biryani": {"last": timestamp, "freq_days": 5}}
  weather_behavior: dict      # {"rainy": "hot_soup", "hot": "cold_drinks"}
```

Updated via a **lightweight feedback loop:**

- Explicit: user rates order 1–5
- Implicit: re-order of same item = positive signal
- Negative: order cancelled / complaint = negative signal
- Reinforcement: A/B test recommendations, measure conversion

---

## 5. WHATSAPP-FIRST EXPERIENCE

### 5.1 Why WhatsApp Beats Apps for India

- 500M Indian WhatsApp users (2025)
- Average Indian opens WhatsApp 25+ times/day
- Trust factor: WhatsApp = personal space; apps = commercial space
- No update fatigue, no storage management, no login friction
- Businesses already transact via WhatsApp (kirana stores, tailors, etc.)
- Indians are comfortable sharing voice notes over WhatsApp

### 5.2 Onboarding Flow (< 60 seconds)

```
User messages: "Hi" or scans QR code

Dzukku: "Hey! I'm Dzukku, your personal food AI 🍛
  Quick setup — takes 30 seconds:
  1. What's your name?
  2. Where are you? [Share Location]
  3. Veg or Non-Veg?
  4. Any allergies? (skip if none)
  
  That's it. I'll handle everything else."

[User responds in any order, any language]

Dzukku: "Perfect, [Name]! I'll remember your preferences.
  What are you in the mood for today?"
```

### 5.3 Conversation Design Principles

**Principle 1 — Never show a menu unless asked**
Instead of: "Here's our menu with 200 items"
Do: "Based on your last order and the time, you might want X or Y. Want either, or something different?"

**Principle 2 — Confirm before charging, always**
Show a clean summary before any payment action. No surprise charges.

**Principle 3 — Personality consistency**
Dzukku has a warm, slightly witty, knowledgeable personality. Not robotic. Not over-the-top cheerful. Like a friend who knows food.

**Principle 4 — Exit gracefully**
If user goes quiet, no spam. One gentle follow-up after 10 min, then silence.

**Principle 5 — Handle confusion naturally**
"I didn't quite get that — did you mean [X] or [Y]?" Never "Invalid input."

### 5.4 Quick Reply Patterns

```
Ordering:
  [Same as last time] [Something new] [What's on offer?]

Post-recommendation:
  [Order this] [See alternatives] [Later]

Checkout:
  [Pay via UPI] [Pay via Card] [Save for later]

After delivery:
  [Rate your meal ⭐] [Order again] [Report an issue]
```

### 5.5 Voice Ordering Flow

```
User sends voice note: "Ek biryani aur raita bhejo, 30 minute mein chahiye"

↓ Whisper / Gemini audio → text transcription
↓ Language detection: Hindi
↓ Intent: order (biryani + raita, time constraint: 30 min)
↓ Slot fill: quantity=1, items=[biryani, raita], delivery_time=30min
↓ Check restaurants with ≤25 min ETA

Dzukku (in Hindi): "Samajh gaya! Paradise Biryani se 1 Chicken Biryani + Raita 
  ₹320 mein, 22 min mein. ₹40 coupon bhi laga diya. 
  Total: ₹280. Order karun?"
```

### 5.6 Multilingual Support

| Language | Priority | Implementation                  |
| -------- | -------- | ------------------------------- |
| English  | P0       | Native (current)                |
| Hindi    | P0       | Gemini multilingual output      |
| Telugu   | P1       | Gemini + regional fine-tune     |
| Tamil    | P1       | Gemini multilingual             |
| Kannada  | P2       | Gemini multilingual             |
| Bengali  | P2       | Gemini multilingual             |
| Hinglish | P0       | Handle code-switching naturally |

The AI detects language from the first message and responds in kind. Hinglish is explicitly supported (most common in urban India).

### 5.7 Proactive Notifications (Non-Spammy)

**Rule:** One proactive message per day maximum, only if confidence > 80%.

```
Triggers:
  - 12:45 PM on a workday → "Lunch time! Same dal rice from yesterday?"
  - Friday 7 PM → "Weekend starts! Your usual biryani order?"
  - User hasn't ordered in 5 days → "Miss you! ₹100 off to come back?"
  - Weather: raining in user location → "Rainy day = masala chai + pakoras?"
  - Festival: "Happy Diwali! Sweet boxes from [local mithai shop]?"
```

---

## 6. PERSONALIZATION ENGINE

### 6.1 Data Signals Collected Per User

```
Behavioral signals (passive collection):
  ✓ Order time of day
  ✓ Day of week patterns
  ✓ Items re-ordered (implicit positive)
  ✓ Items browsed but not ordered
  ✓ Order cancellations (with reason)
  ✓ Complaint patterns
  ✓ Budget range per meal
  ✓ Delivery address patterns (home vs office vs other)
  ✓ Payment method preference
  ✓ Session length before ordering

Explicit signals (collected conversationally):
  ✓ Health goals
  ✓ Dietary restrictions
  ✓ Allergies
  ✓ Spice preference
  ✓ Cuisine favorites
  ✓ Star ratings post-order

Contextual signals (real-time):
  ✓ Time + day
  ✓ Weather at delivery location
  ✓ Day in month (salary week vs end of month)
  ✓ Local events / festivals
  ✓ Device type (voice vs text)
```

### 6.2 Craving Prediction Model

**Approach:** Time-series pattern matching + LLM reasoning

```
Step 1: Extract order history → time-series per food category
Step 2: Identify weekly/monthly cycles (e.g., biryani every 5–7 days)
Step 3: Compute "craving likelihood" per category at current time
Step 4: Cross-reference with health context (e.g., suppress high-cal if user is on diet goal)
Step 5: Generate recommendation with confidence score
```

**Example model output:**

```json
{
  "user_id": "u_12345",
  "timestamp": "2026-05-15T12:30:00",
  "predictions": [
    {"item": "Chicken Biryani", "confidence": 0.82, "reason": "ordered every Friday lunch for 6 weeks"},
    {"item": "Masala Dosa", "confidence": 0.61, "reason": "orders light meals on Fridays recently"},
    {"item": "Cold Coffee", "confidence": 0.45, "reason": "hot day, ordered cold drinks last 3 hot days"}
  ]
}
```

### 6.3 Mood-Based Suggestions

The AI infers mood from message tone using Gemini sentiment analysis:

| Mood signal                       | AI behavior                         |
| --------------------------------- | ----------------------------------- |
| "Stressed", short replies         | Suggest comfort food, fast delivery |
| "Celebrating" / exclamation marks | Suggest premium options, desserts   |
| "Tired", late night               | Suggest light, quick items          |
| Health-goal mention               | Activate nutrition filter           |
| Budget mention                    | Activate savings agent first        |

### 6.4 Weather-Based Ordering Intelligence

```python
weather_food_map = {
    "rainy":   ["hot chai", "pakoras", "maggi", "soup", "samosa"],
    "cold":    ["biryani", "haleem", "hot chocolate", "dal makhani"],
    "hot":     ["lassi", "chaas", "cold coffee", "ice cream", "salad"],
    "humid":   ["light meals", "juices", "grilled items"],
}
```

Weather data: OpenWeatherMap API, checked at notification trigger time.

### 6.5 Festival & Event Intelligence

```
Diwali        → sweets, dry fruits, premium gifting combos
Eid           → biryani, sheer khurma, kebabs
Dussehra      → festive thali, sweets
Exam season   → study snacks, caffeine, quick meals
Cricket match → snacks, finger food, group orders
New Year      → premium orders, late night
```

---

## 7. SAVINGS OPTIMIZATION ENGINE

### 7.1 Design Philosophy

The AI acts as a **financial optimizer for food spend.** It tracks the user's food budget, finds every available discount, and reports monthly savings like a finance app.

### 7.2 Savings Stack (applied in order)

```
For every order intent:

1. Platform comparison
   → Query same item on Zomato, Swiggy, Dzukku direct
   → Pick cheapest after all discounts

2. Coupon auto-apply
   → Maintain live coupon DB per restaurant per platform
   → Apply best non-stackable coupon automatically
   → Try stackable coupons (bank offers + restaurant offers)

3. Combo intelligence
   → "Adding a drink makes this ₹40 cheaper via combo"
   → ML model trained on historical combo savings

4. Timing optimization
   → "Same restaurant is 20% off in 30 minutes (lunch special ends)"
   → "Ordering tomorrow morning saves ₹60 (pre-order discount)"

5. Alternative restaurant detection
   → Semantic similarity: find equivalent item at lower price
   → Quality filter: only suggest if rating ≥ user's threshold

6. Wallet / loyalty stacking
   → Auto-apply Dzukku wallet balance
   → Stack bank cashback offers (HDFC, SBI, etc.)
   → Apply restaurant loyalty points

7. Group order optimization
   → "Order with 2 friends — free delivery + extra 10% off"
```

### 7.3 Monthly Savings Report

Every month, Dzukku sends users a summary:

```
"Your Dzukku Savings Report — April 2026

Total spent on food: ₹6,420
Savings found by AI: ₹1,840 (22% saved!)

  - Auto-coupons applied: ₹680
  - Smart alternatives: ₹420
  - Timing deals: ₹310
  - Combo optimization: ₹430

Your most expensive habit: Late-night pizza (₹1,200/month)
Healthier + cheaper alternative saved: ₹180 last week ✓

This month's streak: 8 days healthy + budget meals 🎯"
```

### 7.4 Price Prediction

- Track price history per item per restaurant per time-of-day
- Alert users: "This item is 15% more expensive than usual right now. Order in 2 hours for a better deal?"
- Surge detection: festival weekends, match nights → warn users proactively

---

## 8. NUTRITION + HEALTH LAYER

### 8.1 Health Profile Setup (Conversational)

```
Dzukku: "Want me to help with healthier choices? Tell me a bit about yourself:"
  → "What's your rough goal?" [Lose weight] [Build muscle] [Eat healthier] [Diabetic-friendly] [No specific goal]
  → "Are you vegetarian?" [Yes] [No] [Vegan]
  → "Any food allergies?" (optional)

That's it. No medical forms. No calorie diaries.
```

### 8.2 Per-Order Nutritional Awareness

For every recommendation, compute:

```
NutritionScore:
  calories: int
  protein_g: float
  carbs_g: float
  fiber_g: float
  fat_g: float
  health_tag: str  # "high-protein" | "light" | "heavy" | "diabetic-safe"
```

Source: nutritional database (ICMR data + crowdsourced for Indian dishes) + restaurant-provided data.

**In-chat display:**

```
"Chicken Biryani — ₹320
  🔥 650 cal | 💪 28g protein | Moderate carbs
  [Good match for your muscle goal]"
```

### 8.3 Health Nudges (Non-Preachy)

**Rule:** Nudge only if user has set a health goal. Never unsolicited.

```
"You've had 1,800 cal today — on track! 
 This biryani would put you at 2,450. 
 Want a lighter version or stick with biryani?"

NOT:
"You're eating too much! You should order salad instead."
```

### 8.4 Weekly Food Report

```
"Your week in food — May 5–11, 2026

  Avg daily calories: 1,980 (goal: 2,000) ✓
  Avg daily protein: 52g (goal: 80g) — you can do better!
  
  Best day: Wednesday (high-protein lunch + light dinner)
  Watch out: Sunday late-night ordering pattern
  
  Top suggestion: Add 1 protein-rich breakfast per week.
  Try: Egg Bhurji from [nearby restaurant] — ₹90, 30g protein"
```

### 8.5 Condition-Specific Intelligence

| Condition   | Behavior                                                        |
| ----------- | --------------------------------------------------------------- |
| Diabetes    | Filter high-GI items, suggest roti over rice, flag sugar        |
| Gym/Muscle  | Surface high-protein options first, suggest protein supplements |
| Weight loss | Show calorie count prominently, suggest smaller portions        |
| BP patients | Flag high-sodium items                                          |
| Pregnant    | Flag unsafe items (raw meat, high-mercury fish)                 |

All self-reported. No medical diagnosis. Clearly positioned as suggestions, not prescriptions.

---

## 9. LOCAL COMMERCE INTELLIGENCE

### 9.1 Vendor Ecosystem

Dzukku serves a wider vendor universe than current apps:

```
Tier 1: Established restaurants (current Dzukku focus)
Tier 2: Cloud kitchens (direct integration)
Tier 3: Home chefs (onboarded via WhatsApp)
Tier 4: Local tiffin services (per-order model)
Tier 5: Street food vendors (with digital payment)
Tier 6: Bakeries, mithai shops, specialty stores
```

### 9.2 Vendor Onboarding (AI-Assisted)

Home chef / small vendor onboarding via WhatsApp:

```
Vendor: "I want to list my tiffin service"

Dzukku (Vendor Bot):
  "Great! Let's set you up:
  1. Business name?
  2. What do you serve? (send photos + menu)
  3. Service area (share pin)?
  4. Timings?
  5. UPI ID for payments?
  
  Done! You'll be live in 2 hours."
```

The AI extracts menu from photos using Gemini Vision. No complex forms.

### 9.3 AI Negotiation with Vendors

**Problem:** Individual users can't negotiate with restaurants. But Dzukku can, on behalf of thousands of users.

**Mechanism:**

```
Dzukku Demand Signal: "200 users in Banjara Hills want biryani tonight"
→ Dzukku proposes to 3 restaurants: "Offer 15% off tonight, I'll send you orders"
→ Restaurant accepts (fills capacity during slow hour)
→ Dzukku routes orders to accepting restaurant
→ Users get deal, restaurant gets volume, Dzukku earns commission
```

This is **demand aggregation as negotiating power** — a moat that grows with user scale.

### 9.4 Surplus Inventory Routing

**Problem restaurants have:** Food prepared but not sold → waste.

**Dzukku solution:**

```
Restaurant signals: "50 portions of dal makhani left, expires in 2 hours"
→ Dzukku AI: "Flash deal! Dal Makhani from [Restaurant] — 40% off, next 2 hours"
→ Sent to users within 3km who have ordered dal makhani before
→ Restaurant recovers cost, reduces waste, Dzukku earns share
```

### 9.5 Restaurant CRM (Dzukku Dashboard)

The existing POS frontend evolves into a full vendor intelligence dashboard:

```
What the restaurant sees:
  - Real-time demand heatmap (by item, time, area)
  - AI predicted busy periods (next 7 days)
  - Customer lifetime value by segment
  - AI-suggested pricing adjustments
  - Inventory alerts
  - Auto-generated offers to push to Dzukku users
  - WhatsApp order notifications
```

---

## 10. BUSINESS MODEL

### 10.1 Revenue Streams

**Stream 1: Commission on orders (core)**

- Direct orders via Dzukku: **2–5% commission** (vs Zomato/Swiggy at 18–30%)
- This is a deliberate strategic choice: far lower friction for restaurant onboarding
- Volume + loyalty drives revenue, not margin extraction
- Restaurants keep more, pass savings to users via better prices/deals — virtuous cycle

**Stream 2: Promoted placements**

- Restaurants pay for prioritized recommendations (clearly labeled as "Promoted")
- Not ad-first; organic recommendation quality must never be compromised
- Cap: 1 promoted result per 3 organic results
- Sold on CPO (cost-per-order) model, not CPM — restaurants only pay when an order is placed

**Stream 3: Embedded finance**

- "Dzukku Pay Later" — order now, pay after salary
- Partner with NBFC, earn interest share
- "Food wallet" with cashback incentives (float income)
- UPI AutoPay for repeat meal plans (no subscription wall — user controls it)

**Stream 4: Nutrition partnerships**

- Partner with fitness apps, gyms, health insurance companies
- "Order 10 healthy meals from Dzukku, get gym discount"
- Revenue via referral fee per converted partnership action — no user charge

**Stream 5: B2B office ordering**

- Corporate lunch automation (bulk ordering with GST invoicing)
- Negotiated fixed rates for offices with 50+ employees
- Commission on bulk orders: 3–4% (within the 2–5% band)

**Stream 6: Flash deal revenue share**

- When Dzukku AI negotiates a surplus inventory deal with a restaurant, a portion of the discount gap is shared with Dzukku as a placement fee
- Restaurant keeps 60%, user saves 30%, Dzukku earns 10% of discount value

### 10.2 Unit Economics (Estimated)

```
Average order value (AOV): ₹350
Commission rate: 3.5% average = ₹12.25/order
Repeat orders/user/month: 8
Monthly commission revenue per active user: ₹98

Promoted placement CPO: avg ₹15/order, 20% of orders are promoted
Monthly promoted revenue per active user: ₹24

Total monthly revenue per active user: ~₹122

LLM cost per order (Gemini 2.5 Flash): ~₹2–4
Net margin per active user/month: ~₹88–106

Target: 100,000 active users × ₹100 = ₹1 Cr/month commission
B2B + promoted: additional ₹30–50 L/month
Run rate at 100k users: ~₹1.3–1.5 Cr/month (Year 2)

Scale driver: Volume is the play.
  500,000 users → ₹6–7 Cr/month
  2M users → ₹25+ Cr/month
```

**Why low commission works:**

- Restaurants prefer Dzukku → faster onboarding → more supply → better user experience → more orders → more total commission despite lower rate
- Zomato/Swiggy's 18–30% drives restaurants to seek alternatives actively — Dzukku is that alternative

### 10.3 Path to Profitability

```
Phase 1 (0–12 months): Commission only, prove retention and order volume
Phase 2 (12–24 months): Add promoted placements, B2B office program
Phase 3 (24–36 months): Embedded finance, nutrition partnerships, flash deal revenue share
```

---

## 11. INDIA-SPECIFIC ADVANTAGES

### 11.1 UPI Ecosystem

- 14 billion UPI transactions/month (2025)
- Dzukku integrates UPI collect requests directly in WhatsApp chat
- Zero card friction — most Indians prefer UPI
- UPI Lite for small orders (< ₹500) — instant, offline capable
- UPI AutoPay for user-opted repeat meal plans (fully user-controlled, not a subscription)

### 11.2 WhatsApp Business API

- Meta's WhatsApp Business API now available to Indian startups
- Supports: text, images, buttons, lists, voice, video
- Conversation-based pricing (₹0.58/user conversation/day) — very affordable at scale
- Green tick verification builds trust
- WhatsApp Pay integration for in-chat payments (phased rollout)

### 11.3 Multilingual Behavior

- India has 22 official languages + hundreds of dialects
- Dzukku must handle code-switching (Hinglish, Tanglish, etc.)
- Gemini 2.5 handles 100+ languages natively
- Regional language support = unlock for Tier-2/3 cities
- Telugu food market (Hyderabad base) is underserved and large

### 11.4 Tier-2/Tier-3 Opportunity

```
Tier-1 cities (Delhi, Mumbai, Bangalore): Saturated by Zomato/Swiggy
Tier-2 (Hyderabad, Pune, Jaipur, Lucknow): Partial coverage, weak loyalty
Tier-3 (Vizag, Warangal, Kochi, Indore): Largely unserved

Dzukku strategy:
  Start Hyderabad (home turf, Telugu speaking) →
  Expand to Telugu-belt Tier-2 cities →
  Expand to Hindi-belt Tier-2 →
  National expansion with multilingual AI
```

### 11.5 Street Food Digitization

- India's street food market: ₹1,60,000 Cr (2025)
- Currently entirely cash-based and undiscoverable
- Dzukku can onboard street vendors with WhatsApp + UPI QR
- User says "Find good pani puri near me" → Dzukku surfaces nearby vendors with ratings
- Creates entirely new inventory that incumbents don't have

### 11.6 Creator-Led Commerce

- Food influencers on Instagram/YouTube recommend specific dishes
- Dzukku can integrate: "Order the [influencer]'s recommended dish from [restaurant]"
- Influencer earns referral commission per order
- Creates a new user acquisition channel that is completely unique

---

## 12. GO-TO-MARKET STRATEGY

### 12.1 Phase 1: Hyderabad Deep Penetration (Months 1–6)

**Goal:** 10,000 active users, 50 restaurants, prove retention

**Actions:**

1. Launch WhatsApp channel alongside existing Telegram bot
2. Onboard 50 restaurants in top Hyderabad neighborhoods (Banjara Hills, Jubilee Hills, Kondapur, Madhapur)
3. Partner with 10 home chefs / tiffin services (unique inventory)
4. College campus activation: BITS Hyderabad, IIT Hyderabad, Osmania University
5. Office park activation: HITEC City, Mindspace, Gachibowli
6. Referral program: ₹50 wallet credit for each friend who orders

**Key metric:** 30-day repeat order rate > 60%

### 12.2 Phase 2: Telugu Belt Expansion (Months 7–12)

**Cities:** Vizag, Warangal, Vijayawada, Tirupati, Karimnagar

**Approach:**

- Telugu-language AI (fully localized, not translated)
- Partner with Telugu food influencers
- Onboard local restaurant chains (Minerva, Rayalaseema Ruchulu, etc.)
- Street food vendor digitization program
- Local language WhatsApp groups → group ordering feature launch

### 12.3 Phase 3: National Scale (Months 13–24)

**Markets:** Bengaluru, Chennai, Pune, Delhi NCR

**Strategy:**

- Hindi + Kannada + Tamil + Marathi language support
- Corporate meal program B2B launch
- ONDC integration (reach vendors on the national open network)
- Promoted placements product launch (restaurant self-serve ad portal)
- Flash deal revenue share program rollout

### 12.4 Virality Mechanics

```
1. Referral: ₹50 credit per friend (both get ₹50)
2. Group ordering: "Order with friends, get free delivery"
3. Streak sharing: WhatsApp status auto-draft for health streaks
4. Savings share: "My AI saved me ₹1,200 this month — try Dzukku"
5. Restaurant sharing: Restaurants share QR → "Order via Dzukku, get 10% off"
6. Influencer API: Food creators get tracked referral links + commission
```

---

## 13. COMPETITIVE ANALYSIS

### 13.1 Incumbent Landscape

| Platform   | Strength                    | Core Weakness                             |
| ---------- | --------------------------- | ----------------------------------------- |
| Zomato     | Brand, coverage, logistics  | App-dependent, no memory, high commission |
| Swiggy     | Logistics, quick commerce   | Same as Zomato, no conversation           |
| Zepto Cafe | Speed (10 min)              | Limited menu, no personalization          |
| Magicpin   | Dine-in discovery           | Weak delivery, coupon-only value prop     |
| ONDC       | Open network, no commission | No UX layer, discovery is broken          |
| Dunzo      | Errand running              | Struggling financially, not food-first    |

### 13.2 Where Incumbents Are Vulnerable

**Zomato/Swiggy vulnerabilities:**

1. 18–30% commission is unsustainable for restaurants — churn risk
2. No memory or personalization — users feel like strangers every time
3. App-first strategy misses WhatsApp-native users (400M+ in India)
4. Decision fatigue is a known UX problem — no AI solution in sight
5. Nutrition and health: complete blind spot
6. Street food / home chefs: ignored supply

**Why agentic AI changes consumer expectations:**

- Once a user has a food AI that knows their preferences and proactively suggests meals, browsing a menu feels like manual labor
- The AI creates a new baseline expectation that incumbents cannot match without rebuilding from scratch
- Memory creates switching cost — "my AI knows me" is a powerful retention mechanic

### 13.3 ONDC as an Opportunity

ONDC (Open Network for Digital Commerce) is India's government-backed open commerce protocol. Rather than competing with it:

```
Dzukku strategy for ONDC:
  - Become a Buyer App on ONDC
  - Access all ONDC-listed restaurants without onboarding each one
  - Layer Dzukku's AI + personalization on top of ONDC inventory
  - Offer the best UX on the open network
  - This gives national inventory without national sales team
```

---

## 14. TECH STACK EVOLUTION

### 14.1 Current Stack (Keep + Extend)

| Component       | Current                                     | Evolution                                           |
| --------------- | ------------------------------------------- | --------------------------------------------------- |
| AI Model        | Gemini 2.5 Flash                            | Gemini 2.5 Flash (keep) + Pro for complex reasoning |
| Agent Framework | Custom pipeline (Planner/Executor/Verifier) | Extend + LangGraph for multi-agent                  |
| Bot transport   | Telegram only                               | Add WhatsApp Business API                           |
| Database        | PostgreSQL                                  | PostgreSQL + pgvector (embeddings)                  |
| Cache           | None                                        | Redis (session state, rate limiting)                |
| Queue           | None                                        | Celery + Redis (async tasks)                        |
| Frontend        | React/Vite POS                              | Extend + vendor dashboard                           |
| Payments        | Razorpay                                    | Razorpay + UPI Collect                              |
| Storage         | Local                                       | GCS / S3 (menu images, invoices)                    |

### 14.2 New Components Required

```
Vector Database:
  → pgvector extension on existing PostgreSQL
  → Store user taste embeddings, past order embeddings
  → Semantic search for "something like my last order but healthier"

Memory Store:
  → Redis for short-term session memory
  → PostgreSQL for long-term preference profiles
  → Structured + semantic hybrid retrieval

WhatsApp Integration:
  → Meta WhatsApp Business API
  → Self-hosted via WhatsApp Business Cloud API
  → Message template approval process (Meta)
  → Webhook handler (new FastAPI route)

Voice Processing:
  → Gemini audio understanding (native, already in API)
  → Fallback: OpenAI Whisper for transcription

Notification System:
  → Celery beat for scheduled proactive messages
  → Time + weather + craving trigger evaluation
  → Rate limiting: max 1 proactive/user/day

Analytics:
  → PostHog (self-hosted) for product analytics
  → Custom food behavior tracking events

Nutrition Database:
  → ICMR nutritional data for Indian dishes
  → Restaurant-provided macros (via onboarding)
  → Crowdsourced + ML-estimated for ungathered items

Weather API:
  → OpenWeatherMap (free tier sufficient for city-level)

Maps / Geo:
  → Google Maps Platform (distance, ETA)
  → Or MapMyIndia (India-first, often more accurate for Tier-2)
```

### 14.3 Target Architecture (v3)

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER CHANNELS                           │
│         WhatsApp           Telegram         Web (future)        │
└──────────────┬──────────────────┬──────────────────────────────┘
               │                  │
               ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MESSAGE GATEWAY                            │
│              FastAPI webhook handlers (per channel)             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT ORCHESTRATOR                           │
│                   (LangGraph multi-agent)                       │
│                                                                 │
│  Intent → Memory → Planner → Executor → Verifier → Responder   │
│                                                                 │
│  Sub-agents: Nutrition | Savings | Recommendation | Logistics   │
└───────┬────────────────────────────────────────────────────────┘
        │
        ├──→ PostgreSQL + pgvector (persistence + embeddings)
        ├──→ Redis (session cache)
        ├──→ Celery (async: notifications, reports)
        ├──→ Razorpay API (payments)
        ├──→ OpenWeatherMap (context)
        ├──→ Google Maps (geo + ETA)
        ├──→ ONDC / Zomato / Swiggy (MCP bridge - existing)
        └──→ Restaurant WebSocket (real-time order status)

┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND PORTALS                            │
│   Admin POS | Waiter App | Kitchen KDS | Vendor Dashboard       │
│                  (React + Vite - existing, extended)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 15. EXECUTION ROADMAP

### Phase 0: Foundation (Weeks 1–4) — Current Sprint

**Goal:** Production-grade single-restaurant system

- [X] PostgreSQL migration (done)
- [X] 5-stage agentic pipeline (done)
- [X] POS frontend with roles (done)
- [X] Razorpay payments (done)
- [ ] Stable WhatsApp Business API integration
- [ ] Redis session store (replace in-memory)
- [ ] pgvector setup for future embeddings

**Exit criteria:** Dzukku Restaurant running stably on production

---

### Phase 1: Memory + Personalization (Weeks 5–12)

**Goal:** AI that remembers users across sessions

**Deliverables:**

1. User preference model schema (PostgreSQL)
2. Memory injection into context builder (extend existing `context_builder.py`)
3. Taste vector computation after each completed order
4. Personalized recommendation in Planner prompt
5. Post-order feedback collection (simple 1–5 rating in chat)
6. Basic craving prediction (top-3 suggestions at order time)

**Key code changes:**

- `context_builder.py`: Add `UserPreferences` to `ContextSnapshot`
- `planner.py`: Inject preference summary into system prompt
- `responder.py`: Personalize tone based on user history
- New: `app/agent/memory_agent.py`
- New: `app/db/models/user_preferences.py`

---

### Phase 2: WhatsApp Channel (Weeks 9–16)

**Goal:** WhatsApp as primary ordering channel

**Deliverables:**

1. WhatsApp Business API integration (Meta Cloud API)
2. Message gateway abstraction (handle Telegram + WhatsApp from same agent)
3. WhatsApp quick reply buttons
4. Voice note processing (Gemini audio)
5. WhatsApp payment link flow
6. Hindi + Telugu language support in agent responses

**Key code changes:**

- New: `app/bot/whatsapp.py` (mirrors `telegram.py` structure)
- New: `app/api/routes/whatsapp.py` (webhook handler)
- Extend: `app/agent/pipeline.py` — channel-agnostic message handling
- Extend: `app/core/config.py` — WhatsApp API settings

---

### Phase 3: Savings + Nutrition Engine (Weeks 13–20)

**Goal:** AI that saves money and improves health

**Deliverables:**

1. Savings agent: auto-coupon + platform comparison
2. Nutrition database integration (ICMR + manual)
3. Per-item macro display in recommendations
4. Health profile setup flow (conversational onboarding)
5. Proactive notification system (Celery beat)
6. Monthly savings report generation

**Key code changes:**

- New: `app/agent/savings_agent.py`
- New: `app/agent/nutrition_agent.py`
- New: `app/db/models/nutrition.py`
- New: `app/workers/notification_worker.py` (Celery)
- Extend: `app/agent/planner.py` — savings + nutrition context injection

---

### Phase 4: Multi-Vendor + Local Commerce (Weeks 17–28)

**Goal:** Beyond single restaurant — full local food ecosystem

**Deliverables:**

1. Multi-restaurant schema (already has `restaurant_id`)
2. Vendor onboarding flow (WhatsApp-based)
3. Home chef / tiffin service support
4. Vendor dashboard (extend POS frontend)
5. AI negotiation for flash deals
6. Surplus inventory routing system
7. ONDC buyer app integration

**Key code changes:**

- Extend: all DB models to be fully multi-tenant
- New: `app/agent/negotiation_agent.py`
- New: `app/api/routes/vendors.py`
- Extend: POS frontend — vendor dashboard tab

---

### Phase 5: Scale + Intelligence (Weeks 25–36)

**Goal:** Platform network effects kick in

**Deliverables:**

1. Group ordering (WhatsApp group integration)
2. B2B office meal program (bulk orders + GST invoicing)
3. Promoted placements self-serve portal for restaurants
4. Flash deal revenue share automation
5. Influencer referral API
6. Advanced craving prediction (ML model, not rule-based)
7. Price prediction + surge alerts
8. City expansion tooling (new city onboarding playbook)

---

## 16. FUTURE VISION

### 16.1 The 5-Year Picture

```
2026: Hyderabad's favorite AI food concierge
2027: Telugu-belt Food OS, 1M+ users
2028: National platform, 10M+ users, ONDC #1 buyer app
2029: Embedded food finance, nutrition partnerships, wearable integration
2030: Autonomous food operating system — the AI buys groceries, 
      plans meals, and manages food spend for Indian families
```

### 16.2 Autonomous Food Purchasing

The end state: users set a weekly meal plan preference + budget. Dzukku:

- Plans the week's meals based on nutrition goals
- Places recurring orders automatically
- Adjusts when restaurant is unavailable
- Manages grocery + restaurant hybrid planning
- Reports weekly: "I planned your meals this week, saved ₹640, hit your protein goal 5/7 days"

### 16.3 Wearable Integration

- Sync with Apple Watch / Fitbit / Garmin
- Post-workout: "You burned 520 cal. Want a high-protein meal? I found one at ₹180 nearby."
- Sleep data: "You slept only 5 hours. Morning coffee + light breakfast?"

### 16.4 Smart Kitchen Integration

- Connect with smart fridges (Samsung, LG Family Hub)
- AI detects what's in the fridge, suggests recipes
- If ingredients are missing: "You have everything except onions. Want me to order from Zepto or get a dish with onions from [restaurant]?"

### 16.5 AI Family Meal Planning

- One family account, multiple profiles (kids, elderly, diabetic, gym-going)
- AI balances all nutritional needs in a single daily order
- "Family dinner tonight: Paneer for kids, grilled chicken for you, soft idli for grandma — all from one order, ₹680 total"

### 16.6 The Operating System Vision

```
TODAY:
  User → opens app → browses → decides → orders

DZUKKU FUTURE:
  Context (time + health + budget + mood + history)
       ↓
  AI decides (or proposes with one-tap confirm)
       ↓
  Order placed → tracked → delivered
       ↓
  Learning loop improves next prediction

The user's job becomes: occasionally approve.
The AI's job becomes: everything else.
```

---

## OPERATIONAL RISKS + MITIGATIONS

| Risk                                          | Severity | Mitigation                                                 |
| --------------------------------------------- | -------- | ---------------------------------------------------------- |
| LLM hallucination on prices/menus             | High     | Executor always reads from DB, never trusts LLM for prices |
| WhatsApp API policy violations                | High     | Strict message template compliance, opt-in only            |
| User data privacy (food habits are sensitive) | High     | DPDP Act compliance, data minimization, local storage      |
| Restaurant churn (they leave for Zomato)      | Medium   | Lower commission + better tools = retention                |
| LLM cost at scale (Gemini API)                | Medium   | Cache common responses, use Flash (cheaper) for most calls |
| Fake reviews / restaurant gaming              | Medium   | Cross-verify with order data, detect anomalies             |
| Food safety liability                         | Medium   | Clear disclaimers, nutritional info is advisory only       |
| Network connectivity (Tier-2/3)               | Low      | Lightweight responses, progressive enhancement             |

---

## INVESTOR NARRATIVE (Summary)

**Market:** India food delivery — ₹8,00,000 Cr total market, ₹45,000 Cr organized delivery
**Problem:** Existing apps are menu browsers, not intelligent food assistants
**Solution:** Agentic AI Food OS on WhatsApp — the food brain Indians don't know they need yet
**Traction:** Dzukku v1 proves the agent pipeline, POS, and restaurant operations
**Moat:** User memory + vendor intelligence network + conversational UI lock-in
**Business model:** Low-commission (2–5%) + Promoted placements (CPO) + Embedded finance + B2B
**Why us:** Built the agentic pipeline from scratch; deep India food + AI expertise; Hyderabad base for Telugu-belt launch
**Ask:** Seed round to hire ML engineer, WhatsApp API costs, and Hyderabad market expansion

---

*Document version: 1.1 — May 2026*
*Next review: After Phase 1 completion*

---

## 17. STEP-BY-STEP IMPLEMENTATION PLAN

This section is the engineering execution guide — one concrete task at a time, in dependency order. Each step specifies what to build, which files to touch, how to verify it is done, and what it unlocks for the next step.

---

### SPRINT 0 — Stabilize Current Stack (Week 1–2)

**Goal:** Get production-grade on the existing Dzukku single-restaurant system before adding anything new.

---

#### Step 0.1 — Add Redis for session state

**Why:** The current session state lives in PostgreSQL but is queried on every message. Redis gives sub-millisecond reads and clean TTL-based cleanup.

**What to do:**

1. Add `redis` and `redis[asyncio]` to `requirements.txt`
2. Create `app/core/redis_client.py`:
   ```python
   import redis.asyncio as aioredis
   from app.core.config import settings

   _pool = None

   async def get_redis():
       global _pool
       if _pool is None:
           _pool = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
       return _pool
   ```
3. Add `REDIS_URL` to `app/core/config.py` and `.env.example`
4. Update `app/db/crud.py` — `get_session()` / `save_session()` to read/write Redis first, fall back to PostgreSQL
5. Add TTL of 24 hours on session keys

**Verify:** Send a message → check Redis with `redis-cli KEYS "session:*"` → key exists with correct TTL

**Unlocks:** Faster bot responses; clean foundation for memory agent

---

#### Step 0.2 — Enable pgvector extension

**Why:** All personalization and semantic search depend on vector embeddings stored in PostgreSQL.

**What to do:**

1. Add to Alembic migration:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. Add `pgvector` to `requirements.txt`
3. Run `alembic revision --autogenerate -m "enable_pgvector"` and apply
4. Verify: `SELECT * FROM pg_extension WHERE extname = 'vector';` returns a row

**Verify:** Migration runs cleanly, extension confirmed in DB

**Unlocks:** Step 1.1 (user preference embeddings)

---

#### Step 0.3 — Harden the agentic pipeline error handling

**Why:** Before scaling users, the pipeline must handle Gemini API timeouts, DB errors, and malformed LLM output gracefully — without crashing the bot.

**What to do:**

1. In `app/agent/planner.py` — wrap Gemini call in try/except, return a safe fallback plan on failure
2. In `app/agent/responder.py` — add truncation detection (already partially done in history); ensure it retries once before giving a user-visible error
3. In `app/bot/telegram.py` — add a top-level `try/except` in `_think_and_reply()` that sends a friendly "I had trouble with that, try again?" message instead of crashing
4. Add structured logging of all pipeline stage timings: `logger.info("stage=%s duration_ms=%d", stage, ms)`

**Verify:** Deliberately send a message while Gemini key is invalid → bot responds gracefully, doesn't hang

**Unlocks:** Confidence to add more users

---

#### Step 0.4 — Add Celery + worker infrastructure

**Why:** Proactive notifications, monthly savings reports, and nutrition summaries all need async scheduled tasks. Celery is the right tool given the Python stack.

**What to do:**

1. Add `celery[redis]` to `requirements.txt`
2. Create `app/workers/__init__.py` and `app/workers/celery_app.py`:
   ```python
   from celery import Celery
   from app.core.config import settings

   celery_app = Celery(
       "dzukku",
       broker=settings.REDIS_URL,
       backend=settings.REDIS_URL,
       include=["app.workers.notification_worker"],
   )
   celery_app.conf.beat_schedule = {}  # populated in later steps
   ```
3. Add Celery worker startup to `docker-compose.yml` as a separate service
4. Verify worker starts: `celery -A app.workers.celery_app worker --loglevel=info`

**Verify:** `celery inspect ping` returns a response from the worker

**Unlocks:** All async/scheduled features in Steps 3+

---

### SPRINT 1 — User Memory + Personalization (Weeks 3–6)

**Goal:** The bot remembers each user across sessions and uses that memory in every recommendation.

---

#### Step 1.1 — User preference DB model

**What to do:**

1. Create `app/db/models/user_preferences.py`:
   ```python
   # Fields: user_id (FK to sessions), spice_level (float), 
   # cuisine_weights (JSONB), price_band (str), order_timing (JSONB),
   # dietary_flags (ARRAY), craving_cycles (JSONB),
   # health_goals (ARRAY), allergies (ARRAY),
   # taste_embedding (vector(384))  ← pgvector column
   ```
2. Add Alembic migration
3. Add `app/db/crud.py` functions: `get_user_preferences(user_id)`, `upsert_user_preferences(user_id, data)`

**Verify:** `pytest tests/test_crud.py::test_user_preferences` passes

---

#### Step 1.2 — Post-order preference update

**What to do:**

1. In `app/agent/executor.py`, after a completed order is saved, call `update_taste_vector(user_id, ordered_items)`
2. Create `app/agent/memory_agent.py` with:
   - `update_taste_vector()` — updates cuisine_weights, order_timing, craving_cycles
   - `compute_embedding()` — generates a 384-dim text embedding of the user's food history using Gemini embedding API
   - `get_user_memory_summary()` — returns a short text block for injection into the Planner prompt
3. Implicit feedback: reorder of same item = +0.1 weight; cancellation = -0.15 weight

**Verify:** Place 3 test orders → check `user_preferences` table updated correctly after each

---

#### Step 1.3 — Inject memory into ContextSnapshot

**What to do:**

1. In `app/agent/context_builder.py`, extend `ContextSnapshot` dataclass:
   ```python
   @dataclass
   class ContextSnapshot:
       # ... existing fields ...
       user_preferences: Optional[UserPreferences] = None
       memory_summary: str = ""         # short text summary for LLM
       top_cravings: list[str] = field(default_factory=list)
   ```
2. In `build_context()`, call `get_user_preferences()` and `get_user_memory_summary()`

**Verify:** Log the assembled context — memory_summary is non-empty after 2+ orders

---

#### Step 1.4 — Personalized Planner prompt

**What to do:**

1. In `app/agent/planner.py`, add a memory block to the system prompt:
   ```
   USER MEMORY:
   {ctx.memory_summary}
   Top cravings right now: {", ".join(ctx.top_cravings)}
   ```
2. The Planner should now produce personalized `proposed_actions[]` based on past behavior

**Verify:** User who always orders biryani → Planner's first proposed action includes biryani recommendation without being asked

---

#### Step 1.5 — Post-order rating collection

**What to do:**

1. After order status changes to `delivered`, trigger a rating request via Telegram/WhatsApp
2. In `app/bot/telegram.py`, add an inline keyboard: ⭐ ⭐⭐ ⭐⭐⭐ ⭐⭐⭐⭐ ⭐⭐⭐⭐⭐
3. Callback handler saves rating to `orders.rating` column
4. Rating feeds back into `update_taste_vector()` — 5 stars = +0.2 weight, 1 star = -0.3

**Verify:** Complete an order → rating keyboard appears → selecting a star saves to DB

---

### SPRINT 2 — WhatsApp Channel (Weeks 5–10)

**Goal:** WhatsApp becomes the primary user channel with feature parity to Telegram.

---

#### Step 2.1 — WhatsApp Business API setup

**What to do:**

1. Apply for Meta WhatsApp Business API access at business.facebook.com
2. Set up a WhatsApp Business Account with a dedicated phone number
3. Create message templates for: welcome, order confirmation, delivery notification, rating request — submit for Meta approval (takes 24–48 hrs)
4. Add to `app/core/config.py`:
   ```python
   WHATSAPP_TOKEN: str = os.getenv("WHATSAPP_TOKEN", "")
   WHATSAPP_PHONE_ID: str = os.getenv("WHATSAPP_PHONE_ID", "")
   WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "dzukku-webhook-verify")
   ```

**Verify:** Webhook verification URL responds correctly to Meta's GET challenge

---

#### Step 2.2 — WhatsApp webhook handler

**What to do:**

1. Create `app/api/routes/whatsapp.py`:
   - `GET /whatsapp/webhook` — handles Meta's verification challenge
   - `POST /whatsapp/webhook` — receives incoming messages, calls pipeline
2. Parse WhatsApp message payload:
   - Text messages → extract `body`
   - Audio messages → download audio, pass to Gemini audio API for transcription
   - Button replies → map to intent directly
   - Location share → extract lat/lng for geo features
3. Register router in `app/api/main.py`

**Verify:** Send "Hi" on WhatsApp → webhook receives it → pipeline processes → reply sent back via WhatsApp API

---

#### Step 2.3 — Channel-agnostic message gateway

**What to do:**

1. Create `app/bot/gateway.py` with a unified `Message` and `Channel` abstraction:
   ```python
   @dataclass
   class IncomingMessage:
       channel: str           # "telegram" | "whatsapp"
       chat_id: str
       user_name: str
       text: str
       audio_url: Optional[str] = None
       location: Optional[tuple] = None

   async def send_reply(channel: str, chat_id: str, text: str, buttons: list = None):
       if channel == "telegram":
           await _send_telegram(chat_id, text, buttons)
       elif channel == "whatsapp":
           await _send_whatsapp(chat_id, text, buttons)
   ```
2. Refactor `app/agent/pipeline.py` — `process_message()` accepts `IncomingMessage`, returns plain text
3. Both `telegram.py` and `whatsapp.py` construct an `IncomingMessage` and call `process_message()`

**Verify:** Same "order biryani" message via Telegram and WhatsApp both process identically

---

#### Step 2.4 — Voice note processing

**What to do:**

1. In the WhatsApp webhook handler, detect `message.type == "audio"`
2. Download audio file using WhatsApp media API
3. Pass audio bytes to Gemini: `model.generate_content([audio_part, "Transcribe this food order in English"])`
4. Feed transcribed text into the normal pipeline
5. In `app/agent/planner.py`, add voice-specific slot: `input_mode: "voice"` → Planner is more lenient with missing slots

**Verify:** Send a Hindi voice note "Biryani order karo" → transcribed → order processed correctly

---

#### Step 2.5 — WhatsApp interactive buttons

**What to do:**

1. Create `app/bot/whatsapp.py` helper: `send_interactive_buttons(phone, body_text, buttons)`
2. Map existing Telegram inline keyboards to WhatsApp `interactive.type = "button"` (max 3 buttons) and `interactive.type = "list"` (max 10 options) for menu browsing
3. Quick replies: [Same as last time] [Something new] [What's on offer?]

**Verify:** Order flow from WhatsApp shows buttons at every decision point

---

#### Step 2.6 — Multilingual response support (Hindi + Telugu)

**What to do:**

1. In `app/agent/context_builder.py`, add `user_language: str = "en"` to `ContextSnapshot`
2. In the gateway, detect language from first message using Gemini: `"Detect language code for: {text}"`; save to user preferences
3. In `app/agent/responder.py`, append to system prompt: `"Respond in {ctx.user_language}. If Hinglish, mix Hindi and English naturally."`
4. Telugu: treat as language code `"te"` — Gemini handles it natively

**Verify:** Send "Biryani order cheyyali" (Telugu) → bot responds in Telugu

---

### SPRINT 3 — Savings + Nutrition Engine (Weeks 9–16)

**Goal:** Every order recommendation includes the best price and a health score.

---

#### Step 3.1 — Nutrition database setup

**What to do:**

1. Create `app/db/models/nutrition.py`:
   ```python
   # Table: item_nutrition
   # Columns: menu_item_id (FK), calories, protein_g, carbs_g, 
   #          fat_g, fiber_g, sodium_mg, health_tags (ARRAY), source (str)
   ```
2. Seed with ICMR nutritional values for top 100 Indian dishes (CSV import script in `scripts/`)
3. For restaurant menu items without nutrition data: use Gemini to estimate — `"Estimate macros for: {item_name}, {description}. Return JSON."`
4. Add `app/db/crud.py`: `get_nutrition(menu_item_id)`, `bulk_upsert_nutrition(records)`

**Verify:** Query nutrition for "Chicken Biryani" → returns calories ~650, protein ~28g

---

#### Step 3.2 — Health profile conversational onboarding

**What to do:**

1. Add `health_onboarding_done: bool` flag to user preferences model
2. In `app/agent/state_machine.py`, add state: `health_onboarding`
3. Trigger after 3rd order: "Want me to help you eat healthier? Takes 30 seconds."
4. Collect via quick replies: health goal, diet type, allergies
5. Save to `user_preferences.health_goals`, `dietary_flags`, `allergies`

**Verify:** 3rd order triggers onboarding → answers saved correctly to DB

---

#### Step 3.3 — Nutrition Agent

**What to do:**

1. Create `app/agent/nutrition_agent.py`:
   ```python
   class NutritionAgent:
       async def score_item(self, menu_item_id: int, user_prefs: UserPreferences) -> NutritionScore
       async def get_daily_summary(self, user_id: str) -> DailySummary
       async def generate_nudge(self, ctx: ContextSnapshot) -> Optional[str]
       async def filter_by_health(self, items: list, user_prefs: UserPreferences) -> list
   ```
2. `score_item()`: fetch nutrition data, compare against user health goals, return score 0–10 + health_tag
3. `generate_nudge()`: if daily calories > 80% of goal, inject a nudge sentence into Responder
4. In `app/agent/planner.py`: call `NutritionAgent.filter_by_health()` when health goals are set

**Verify:** User with "weight loss" goal → high-cal items ranked lower in recommendations

---

#### Step 3.4 — Savings Agent

**What to do:**

1. Create `app/agent/savings_agent.py`:
   ```python
   class SavingsAgent:
       async def find_best_price(self, item_name: str, location: str) -> SavingsResult
       async def apply_coupons(self, cart: Cart, user_id: str) -> Cart
       async def find_alternatives(self, item: MenuItem, max_price: int) -> list[MenuItem]
       async def get_monthly_savings(self, user_id: str) -> MonthlySavings
   ```
2. `find_best_price()`: queries Dzukku direct price + Zomato/Swiggy via MCP bridge; returns cheapest
3. `apply_coupons()`: maintain `coupons` table (restaurant_id, code, discount_type, value, expiry); auto-apply best valid coupon
4. `find_alternatives()`: use pgvector similarity search on menu item embeddings to find semantically similar cheaper items
5. Monthly savings tracking: `savings_log` table — record every coupon applied, every alternative suggested

**Verify:** Order biryani → savings agent finds a coupon → total reduced → savings logged

---

#### Step 3.5 — Inject savings + nutrition into Planner

**What to do:**

1. In `app/agent/context_builder.py`, extend `ContextSnapshot`:
   ```python
   savings_available: Optional[SavingsResult] = None
   nutrition_nudge: Optional[str] = None
   daily_nutrition: Optional[DailySummary] = None
   ```
2. In `build_context()`, call both agents in parallel using `asyncio.gather()`
3. In `app/agent/planner.py`, add to system prompt block:
   ```
   SAVINGS CONTEXT: {ctx.savings_available}
   NUTRITION TODAY: {ctx.daily_nutrition}
   NUDGE: {ctx.nutrition_nudge}
   ```

**Verify:** Planner output now includes savings amount and health tag in `proposed_actions`

---

#### Step 3.6 — Proactive notification system

**What to do:**

1. Create `app/workers/notification_worker.py`:
   ```python
   @celery_app.task
   def evaluate_and_send_proactive_notifications():
       # For each active user:
       #   1. Check last message time (skip if < 3 hours)
       #   2. Evaluate triggers: lunch time, dinner time, craving cycle, weather
       #   3. If trigger fires AND confidence > 0.8: generate message, send
       #   4. Max 1 proactive per user per day — enforce via Redis flag
   ```
2. Register in Celery beat: run every 15 minutes
3. Weather trigger: call OpenWeatherMap API for user's saved location
4. Craving trigger: check `craving_cycles` — if days since last order of that item >= avg frequency, trigger

**Verify:** Set a test user's craving_cycle for biryani to have last_ordered = 7 days ago → notification fires

---

### SPRINT 4 — Multi-Vendor Platform (Weeks 15–24)

**Goal:** Open Dzukku to multiple restaurants, home chefs, and cloud kitchens.

---

#### Step 4.1 — Multi-tenant DB audit

**What to do:**

1. Audit all DB models in `app/db/models/` — confirm every table has `restaurant_id` (already in schema per vNext design)
2. Audit all queries in `app/db/crud.py` — confirm every query filters by `restaurant_id`
3. Add `app/db/models/vendor.py` for the new vendor profile:
   ```python
   # Table: vendors
   # Columns: id, name, type (restaurant|home_chef|cloud_kitchen|tiffin|street_food),
   #          whatsapp_number, upi_id, service_area (PostGIS polygon or city_ids),
   #          onboarding_status, commission_rate (float, 2–5%), is_active
   ```
4. Commission rate stored per vendor — allows negotiating different rates per partner

**Verify:** Run existing tests — all pass with `restaurant_id` filters in place

---

#### Step 4.2 — Vendor onboarding via WhatsApp

**What to do:**

1. Create a separate WhatsApp number / bot persona for vendor onboarding: "Dzukku Partner"
2. Add `app/api/routes/vendor_onboarding.py` — dedicated webhook for vendor messages
3. Build a WhatsApp-native onboarding state machine:
   - Collect: business name, vendor type, menu (photos OK), service area, timings, UPI ID
   - Menu photo → Gemini Vision: extract item names, prices, descriptions → populate `menu_items` table
   - On completion: set `onboarding_status = "pending_review"` → admin reviews → approve via POS dashboard
4. Approval triggers: activate vendor, send welcome message, vendor appears in user-facing discovery

**Verify:** Onboard a test home chef via WhatsApp → menu items appear in DB → vendor is searchable

---

#### Step 4.3 — Vendor dashboard (extend POS frontend)

**What to do:**

1. In `restaurant-pos-frontend/src/pages/`, add `vendor/VendorDashboard.jsx`
2. Tabs:
   - **Orders** (existing KDS — reuse)
   - **Menu** (manage items, toggle availability, upload photos)
   - **Analytics** (orders/day, popular items, revenue)
   - **Offers** (create flash deals, set surplus inventory alerts)
   - **Settings** (service area, timings, UPI ID)
3. Add role `VENDOR` to `AuthContext` and protected routes
4. Backend: extend `app/api/routes/` with vendor-scoped endpoints

**Verify:** Vendor logs in → sees only their own orders + menu; no access to other vendors' data

---

#### Step 4.4 — Hyperlocal discovery

**What to do:**

1. Add PostGIS extension to PostgreSQL (or use simple lat/lng + Haversine for V1)
2. In `app/agent/executor.py`, add tool: `discover_nearby_vendors(lat, lng, radius_km, filters)`
3. Filters: cuisine type, price range, delivery time, health tags, rating
4. In `app/agent/recommendation_agent.py`, use discovery tool when user says "find something nearby" or "what's available?"

**Verify:** User shares location → bot returns 3–5 nearby vendors with ETAs and prices

---

#### Step 4.5 — AI negotiation for flash deals

**What to do:**

1. Create `app/agent/negotiation_agent.py`:
   ```python
   class NegotiationAgent:
       async def detect_low_demand_window(self, vendor_id: int) -> bool
       async def propose_flash_deal(self, vendor_id: int, discount_pct: int) -> FlashDeal
       async def route_to_deal_users(self, deal: FlashDeal) -> int  # returns notified count
   ```
2. Low demand detection: if orders in next 2 hours are predicted < 30% of avg → trigger negotiation
3. Prediction: simple rolling average of historical orders by hour-of-day, day-of-week
4. Flash deal proposal: sent to vendor via WhatsApp notification + shown in vendor dashboard
5. On vendor acceptance: `route_to_deal_users()` sends targeted notification to users who ordered that cuisine in the last 30 days, within delivery radius

**Verify:** Manually trigger a flash deal for a test vendor → correct users receive notification → deal orders are attributed correctly

---

#### Step 4.6 — ONDC buyer app integration

**What to do:**

1. Register Dzukku as a Buyer Network Participant (BNP) on ONDC
2. Implement ONDC protocol APIs: `search`, `select`, `init`, `confirm`, `track`, `cancel`
3. Create `app/agent/ondc_client.py` — wraps ONDC HTTP API
4. Add ONDC as a vendor source in `discover_nearby_vendors()` — ONDC results appear alongside native vendors
5. Orders placed via ONDC are tracked in Dzukku's order table with `source = "ondc"`

**Verify:** Search for "biryani near Kondapur" → ONDC results appear in bot recommendations alongside Dzukku-native vendors

---

### SPRINT 5 — Promoted Placements + B2B (Weeks 22–30)

**Goal:** Activate the second and third revenue streams.

---

#### Step 5.1 — Promoted placements backend

**What to do:**

1. Add `promotions` table: `vendor_id, campaign_name, budget_inr, cpo_bid_inr, start_date, end_date, status, orders_generated, spend_to_date`
2. In `app/agent/recommendation_agent.py`, after assembling organic recommendations: check if any vendor has an active promotion matching the user's intent → insert 1 promoted result with label `[Promoted]`
3. On order completion: if order came from a promoted slot, deduct `cpo_bid_inr` from campaign budget, log to `promotions`
4. Auto-pause campaign when `spend_to_date >= budget_inr`

**Verify:** Create a test promotion → recommendation includes promoted result → order placed → budget decremented

---

#### Step 5.2 — Promoted placements self-serve portal

**What to do:**

1. In vendor dashboard: add "Promote My Restaurant" tab
2. Form: campaign name, budget, CPO bid, date range, target cuisine/area
3. Backend: `POST /api/promotions` creates campaign in `pending` state
4. Admin review (simple approve button in Admin POS) → campaign goes `active`
5. Analytics tab shows: orders generated, spend, CPO achieved vs bid

**Verify:** Vendor creates a campaign → admin approves → campaign appears in recommendations

---

#### Step 5.3 — B2B office meal program

**What to do:**

1. Add `corporate_accounts` table: `company_name, billing_email, gst_number, monthly_budget_inr, contact_person, whatsapp_group_id`
2. Build a group ordering flow:
   - Company admin adds Dzukku to a WhatsApp group
   - Employees message the group with their orders
   - Dzukku collects all orders for a slot (e.g., lunch 12–1 PM)
   - Consolidates into a single order to the vendor
   - Generates a GST invoice for the company
3. Commission on B2B orders: 3–4% (within 2–5% band)

**Verify:** Simulate 3 employees ordering in a group → single consolidated order placed → invoice generated with correct totals

---

### SPRINT 6 — Intelligence + Scale (Weeks 28–36)

**Goal:** Replace rule-based predictions with ML models; prepare for city expansion.

---

#### Step 6.1 — ML-based craving prediction

**What to do:**

1. Extract training data: `(user_id, timestamp, ordered_items[], context_features[])` from order history
2. Context features: hour, day_of_week, weather, days_since_last_biryani, etc.
3. Train a simple gradient boosting model (XGBoost) per cuisine category: P(order biryani today) given context
4. Serve predictions via `app/agent/prediction_service.py`
5. Replace the rule-based `craving_cycles` logic in `memory_agent.py` with ML predictions

**Verify:** Backtest: model predicts next order item correctly > 55% of the time (baseline: random = 10%)

---

#### Step 6.2 — Price history + surge detection

**What to do:**

1. Add `price_history` table: `menu_item_id, price_inr, recorded_at, platform`
2. Scheduled Celery task: record prices for top 50 items across platforms daily
3. Surge detection: if current price > 1.15× 30-day average → flag as surge
4. In `app/agent/savings_agent.py`, add: if surge detected, warn user and suggest ordering later or alternative

**Verify:** Manually insert a price spike → savings agent flags it in next recommendation

---

#### Step 6.3 — City expansion tooling

**What to do:**

1. In `app/core/config.py`: add `ACTIVE_CITIES: list[str]` config
2. In `app/db/models/vendor.py`: add `city: str` field
3. Build `scripts/city_onboarding.py`: CLI tool that:
   - Seeds a new city with initial vendor data (from CSV)
   - Sets up geo boundaries
   - Configures city-specific notification triggers (local festivals, events)
4. Document the city launch playbook in `docs/city_launch_playbook.md`

**Verify:** Run `python scripts/city_onboarding.py --city vizag` → Vizag vendors appear in discovery for users in that city

---

#### Step 6.4 — Analytics dashboard (internal)

**What to do:**

1. Deploy PostHog (self-hosted via Docker) for product analytics
2. Add `posthog-python` to `requirements.txt`
3. Instrument key events in the pipeline:
   - `order_completed` (with AOV, vendor, cuisine, channel, savings_applied)
   - `recommendation_clicked`
   - `proactive_notification_converted`
   - `user_onboarded`
   - `flash_deal_claimed`
4. Build a simple internal dashboard: DAU, MAU, orders/day, avg savings/order, top vendors, conversion by channel

**Verify:** Place 5 test orders → PostHog dashboard shows 5 `order_completed` events with correct properties

---

### IMPLEMENTATION SUMMARY TABLE

| Sprint | Weeks  | Key Output                                              | Revenue Impact                  |
| ------ | ------ | ------------------------------------------------------- | ------------------------------- |
| 0      | 1–2   | Redis, pgvector, Celery, hardened pipeline              | Foundation only                 |
| 1      | 3–6   | User memory, taste vector, personalized recommendations | Better retention → more orders |
| 2      | 5–10  | WhatsApp channel, voice, multilingual                   | 3–5x user surface expansion    |
| 3      | 9–16  | Savings agent, nutrition agent, proactive notifications | Higher order frequency          |
| 4      | 15–24 | Multi-vendor, ONDC, flash deals, vendor dashboard       | Commission revenue at scale     |
| 5      | 22–30 | Promoted placements, B2B office meals                   | New revenue streams activated   |
| 6      | 28–36 | ML predictions, price history, city expansion tooling   | 2nd city launch ready           |

**Overlapping sprints are intentional** — Sprint 2 (WhatsApp) starts before Sprint 1 (Memory) is complete because they are independent workstreams. Assign team members accordingly.

---

### DEFINITION OF DONE (per feature)

A feature is "done" when:

1. Unit tests pass (`pytest` — test coverage > 70% for new code)
2. Integration test passes end-to-end on staging bot
3. No regressions in existing Telegram ordering flow
4. Structured log confirms the feature fires correctly in production
5. The change is documented in the relevant architecture section of this doc

---

*Document version: 1.1 — May 2026*
*Implementation tracking: Update sprint checkboxes weekly*
