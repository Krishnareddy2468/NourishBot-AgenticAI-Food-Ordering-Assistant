# NourishBot Refined Product Direction

## One-line definition

NourishBot is an AI-assisted pre-order and pickup system for fictional campus and workplace
canteens.

## The problem

Canteen demand is concentrated into short breaks. Customers lose time in queues, kitchen
teams receive bursts of orders with little visibility, and popular items can become
unavailable before customers reach the counter.

NourishBot lets customers decide and place orders before arriving. The system gives kitchen
and pickup teams a shared, verified view of demand.

## Primary users

### Customer

- Explains a meal preference conversationally
- Reviews a proposed cart with quantities and prices
- Confirms a pre-order
- Tracks queued, cooking, ready, and collected states

### Kitchen team

- Sees the verified preparation queue
- Updates individual line items
- Marks complete orders ready for pickup

### Pickup operator

- Finds an order by reference
- Verifies readiness
- Records customer collection

### Canteen administrator

- Maintains the synthetic menu and availability
- Reviews order throughput and pickup performance
- Manages demonstration staff accounts

## Differentiating project features

1. **Safe agent execution** — the model proposes; deterministic code validates and commits.
2. **Capacity-aware pickup** — future pickup windows cap accepted preparation demand.
3. **Dietary transparency** — menu metadata supports preference filters and explicit warnings.
4. **Group lunch coordination** — future shared carts retain individual confirmation.
5. **Graceful degradation** — menu browsing and staff queues remain usable without the LLM.

## Explicit non-goals

- Building a food-delivery marketplace
- Comparing or ordering from third-party aggregators
- Operating a delivery fleet
- Processing production payments or settlements
- Creating a restaurant advertising or customer-acquisition platform
- Storing real health profiles or making medical nutrition recommendations
- Publishing confidential business strategy, customer information, or proprietary workflows

## Core state flow

```text
DRAFT → PROPOSED → CONFIRMED → QUEUED → PREPARING → READY → COLLECTED
             │          │
             └──────────┴──▶ CANCELLED
```

Every state transition must be allow-listed, authorized, idempotent, and recorded.

## Data policy for the portfolio repository

- Use fictional names and locations.
- Generate menu and order fixtures specifically for the project.
- Store no phone numbers, payment identifiers, access tokens, or real transaction exports.
- Describe architecture using implementation facts, not private commercial strategy.
- Keep optional integrations disabled in public demonstrations.

## Delivery plan

### Milestone 1 — Product alignment

- Update public documentation and staff-interface terminology
- Make pickup the primary order type
- Remove delivery, settlement, and marketplace features from primary navigation

### Milestone 2 — Pickup windows

- Add pickup-slot availability and capacity
- Validate slot capacity during order confirmation
- Display pickup time in kitchen and tracking views

### Milestone 3 — Dietary-aware catalogue

- Add vegetarian, vegan, ingredient, and allergen metadata
- Add explicit uncertainty and cross-contact warnings
- Test hard constraints separately from conversational responses

### Milestone 4 — Group orders

- Add invitation links and member carts
- Require each member to confirm their own items
- Consolidate preparation while preserving member attribution

