# Dzukku Production Architecture and Product Design

Version: 1.0  
Date: 2026-04-06  
Status: Architecture Blueprint (Build-Ready)

## 1) Executive Summary

Dzukku should be positioned as a direct restaurant commerce and digital identity platform running on WhatsApp, with an agentic AI layer that handles discovery, ordering, identity trust, and settlement.

Core business principle:
- Customer pays restaurant directly (or escrow then direct settlement)
- Platform takes a transparent, lower commission cut than aggregator platforms
- Restaurant keeps ownership of customer relationship and digital identity

This architecture is designed for production from day one, with clear boundaries for scale, compliance, cost control, observability, and AI safety.

## 2) Product Vision

Dzukku is not just a chatbot. It is:
- A Digital Identity + Commerce Layer for local restaurants
- A WhatsApp-native ordering and service agent
- A trust and settlement orchestration platform

Primary actors:
- Customer (WhatsApp user)
- Restaurant (merchant identity owner)
- Dzukku Platform (identity, AI orchestration, settlement logic)
- Payment Provider (UPI/cards/wallets)
- Delivery Partner (optional, future phase)

## 3) Problem and Opportunity

Today:
- Aggregator platforms control discovery and data
- Restaurants pay high commissions
- Customers face fragmented loyalty and identity

Dzukku opportunity:
- Give restaurants direct access to customers on WhatsApp
- Reduce commission by removing heavy marketplace overhead
- Create a portable digital identity for users and merchants
- Use AI to provide concierge-like ordering with lower support cost

## 4) Core Design Principles

- Digital identity first (customer + restaurant identities are first-class entities)
- API-first and event-driven design
- Strong tenant isolation for multi-restaurant operations
- Explainable AI decisions for sensitive workflows (payments, policy)
- Security and compliance by default
- Replaceable components (LLM provider, payment provider, channel providers)

## 5) High-Level System Architecture

## 5.1 Logical Layers

1. Experience Layer
- WhatsApp Business API (primary)
- Restaurant dashboard (web)
- Internal ops console

2. Application/API Layer
- API Gateway
- Auth and Identity service
- Conversation and Agent Orchestrator service
- Menu and Catalog service
- Order Management service
- Pricing and Commission service
- Payment and Settlement service
- Notification service

3. Intelligence Layer
- Agentic orchestration engine
- Retrieval system (menu, offers, policies, history)
- Model router (Vertex AI + fallback options)
- Guardrails and policy engine

4. Data Layer
- OLTP DB (PostgreSQL)
- Cache (Redis)
- Message/event bus (Pub/Sub)
- Object storage (GCS)
- Analytics warehouse (BigQuery)

5. Platform/SRE Layer
- Kubernetes (GKE) or Cloud Run
- CI/CD, IaC, secrets, observability, security scanning

## 5.2 Architecture Diagram (Logical View)

```mermaid
flowchart TB
	subgraph Experience[Experience Layer]
		WA[WhatsApp Business API]
		RD[Restaurant Dashboard]
		OPS[Ops Console]
	end

	subgraph API[Application and API Layer]
		GW[API Gateway]
		ID[Identity Service]
		CS[Conversation Service]
		AO[Agent Orchestrator]
		CAT[Catalog Service]
		ORD[Order Service]
		PAY[Payment Service]
		SET[Commission and Settlement Service]
		NOTI[Notification Service]
	end

	subgraph AI[Intelligence Layer]
		RET[Retrieval Layer]
		MR[Model Router]
		GRD[Guardrails and Policy Engine]
		VTX[Vertex AI Models]
	end

	subgraph DATA[Data Layer]
		PG[(PostgreSQL)]
		RED[(Redis)]
		PS[(Pub/Sub)]
		GCS[(GCS)]
		BQ[(BigQuery)]
	end

	subgraph PLATFORM[Platform and SRE Layer]
		CR[Cloud Run or GKE]
		CICD[CI/CD + IaC]
		OBS[Observability]
		SEC[Security and Secrets]
	end

	WA --> GW
	RD --> GW
	OPS --> GW

	GW --> ID
	GW --> CS
	CS --> AO
	AO --> CAT
	AO --> ORD
	AO --> PAY
	AO --> SET
	AO --> NOTI

	AO --> RET
	AO --> MR
	MR --> GRD
	GRD --> VTX

	ID --> PG
	CS --> RED
	ORD --> PG
	PAY --> PG
	SET --> PG
	AO --> PS
	PS --> BQ
	CAT --> GCS

	CR --- API
	CR --- AI
	CICD --- CR
	OBS --- API
	OBS --- AI
	SEC --- API
	SEC --- DATA
```

## 5.3 End-to-End Request Flow (WhatsApp order)

1. Customer sends message in WhatsApp
2. Channel webhook enters API Gateway
3. Identity service resolves customer and restaurant tenant context
4. Agent Orchestrator calls tools:
- Menu retrieval
- Pricing and commission preview
- Payment intent creation
5. Agent responds with structured choices and payment CTA
6. Payment confirmation event updates Order service
7. Settlement service computes split and schedules payout
8. Notification service informs customer and restaurant

## 5.4 Architecture Diagram (Order and Payment Data Flow)

```mermaid
flowchart LR
	U[Customer on WhatsApp] --> CH[Channel Ingress]
	CH --> AO[Agent Orchestrator]
	AO --> O[Order Service]
	AO --> P[Payment Service]
	P --> PGW[Payment Gateway]
	PGW -->|payment webhook| P
	P -->|PaymentConfirmed event| E[(Pub/Sub)]
	E --> S[Settlement Service]
	S --> L[(Settlement Ledger)]
	S --> R[Restaurant Payout Processor]
	AO --> N[Notification Service]
	N --> U
	N --> M[Restaurant]
```

## 5.5 Architecture Diagram (Trust Boundaries)

```mermaid
flowchart TB
	subgraph EXT[External Zone]
		WA[WhatsApp]
		PG[Payment Gateway]
		RB[Restaurant Bank]
	end

	subgraph EDGE[Edge Zone]
		WAF[Cloud Armor and WAF]
		APIGW[API Gateway]
	end

	subgraph CORE[Core Private Zone]
		ING[Ingress Service]
		ID[Identity]
		AO[Agent Orchestrator]
		ORD[Order]
		PAY[Payment]
		SET[Settlement]
		DB[(PostgreSQL)]
		CACHE[(Redis)]
		BUS[(Pub/Sub)]
	end

	subgraph AI[AI Zone]
		VTX[Vertex AI]
		POL[Policy Guardrails]
	end

	WA --> WAF --> APIGW --> ING
	ING --> ID
	ING --> AO
	AO --> ORD
	AO --> PAY
	PAY --> PG
	PG --> PAY
	PAY --> BUS --> SET
	SET --> RB
	ID --> DB
	ORD --> DB
	SET --> DB
	AO --> CACHE
	AO --> POL --> VTX
```

## 5.6 Architecture Diagram (GCP Deployment View)

```mermaid
flowchart TB
	subgraph GCP[GCP Project]
		subgraph NET[VPC]
			LB[External Load Balancer]
			APIGW[API Gateway]
			RUN[Cloud Run Services]
			SQL[Cloud SQL PostgreSQL]
			RED[Memorystore Redis]
			PUB[Pub/Sub]
			SEC[Secret Manager + KMS]
		end

		subgraph AI[Vertex AI]
			MODELS[Model Endpoints]
			EVAL[Prompt and Eval Pipelines]
		end

		subgraph OBS[Observability]
			LOG[Cloud Logging]
			MET[Managed Prometheus]
			TRC[Cloud Trace]
			GRAF[Grafana]
		end

		subgraph ANALYTICS[Analytics]
			BQ[BigQuery]
			GCS[Cloud Storage]
		end
	end

	LB --> APIGW --> RUN
	RUN --> SQL
	RUN --> RED
	RUN --> PUB
	RUN --> SEC
	RUN --> MODELS
	EVAL --> MODELS
	PUB --> BQ
	RUN --> LOG
	RUN --> MET
	RUN --> TRC
	MET --> GRAF
	LOG --> GCS
```

## 6) Digital Identity Architecture

Digital identity is strategic IP for Dzukku.

## 6.1 Identity Objects

Customer Identity
- Stable internal customer_id
- WhatsApp phone binding
- Consent records
- Preference graph (cuisine, repeat patterns)
- Trust score (fraud/risk)

Restaurant Identity
- merchant_id and legal entity mapping
- KYC/KYB status
- Bank settlement profile
- Branch/store hierarchy
- SLA and policy profile

Conversation Identity
- session_id and linked intent history
- language profile
- context memory policy

### Diagram: Digital Identity Model

```mermaid
erDiagram
    CUSTOMER {
        uuid customer_id PK
        string phone_number UK
        string display_name
        string language_pref
        float trust_score
        timestamp created_at
    }
    CUSTOMER_CONSENT {
        uuid consent_id PK
        uuid customer_id FK
        string consent_type
        boolean granted
        timestamp granted_at
        string legal_basis
    }
    PREFERENCE_GRAPH {
        uuid pref_id PK
        uuid customer_id FK
        string cuisine_tags
        string repeat_items
        string allergy_flags
        timestamp last_updated
    }
    RESTAURANT {
        uuid merchant_id PK
        string legal_name
        string kyc_status
        string plan_tier
        string wa_phone_number UK
    }
    RESTAURANT_BRANCH {
        uuid branch_id PK
        uuid merchant_id FK
        string address
        json operating_hours
        boolean is_active
    }
    SETTLEMENT_PROFILE {
        uuid profile_id PK
        uuid merchant_id FK
        string bank_account_no
        string ifsc_code
        string upi_id
        string vpa
    }
    CONVERSATION_SESSION {
        uuid session_id PK
        uuid customer_id FK
        uuid branch_id FK
        json context_memory
        timestamp started_at
        timestamp last_active
    }

    CUSTOMER ||--o{ CUSTOMER_CONSENT : "has"
    CUSTOMER ||--|| PREFERENCE_GRAPH : "has"
    CUSTOMER ||--o{ CONVERSATION_SESSION : "starts"
    RESTAURANT ||--o{ RESTAURANT_BRANCH : "owns"
    RESTAURANT ||--|| SETTLEMENT_PROFILE : "has"
    RESTAURANT_BRANCH ||--o{ CONVERSATION_SESSION : "hosts"
```

## 6.2 Identity Capabilities

- Progressive profile enrichment from conversations
- Consent-driven personalization
- Cross-restaurant interoperability where allowed by policy
- Identity federation with external providers (future)

## 6.3 Security and Compliance Controls

- OAuth 2.1 / OpenID Connect for dashboard users
- JWT with short TTL + refresh rotation
- PII encryption at rest and field-level encryption for sensitive fields
- Signed webhooks and replay protection
- RBAC + ABAC for operations
- Data retention and right-to-delete workflows

### Diagram: Identity and Auth Flow

```mermaid
sequenceDiagram
    actor Customer
    participant WA as WhatsApp
    participant GW as API Gateway
    participant ID as Identity Service
    participant Redis
    participant DB as PostgreSQL

    Customer->>WA: Sends first message
    WA->>GW: Webhook POST (phone, message)
    GW->>GW: Verify WA signature (HMAC)
    GW->>ID: Resolve identity (phone_number)
    ID->>DB: SELECT customer WHERE phone = ?
    alt New Customer
        DB-->>ID: Not found
        ID->>DB: INSERT new customer record
        ID->>DB: INSERT consent capture record
        ID-->>GW: customer_id (new) + onboarding flag
    else Returning Customer
        DB-->>ID: customer_id + profile
        ID->>Redis: Cache session context (TTL 30min)
        ID-->>GW: customer_id + enriched context
    end
    GW->>GW: Attach JWT (short-lived)
    GW-->>ID: Forward to Agent Orchestrator
```

## 7) Commission and Settlement Design

Goal: transparent, low-cut commission while guaranteeing payouts.

## 7.1 Money Flow Model

Option A (recommended): direct-to-restaurant payment with platform fee invoice
- Customer pays restaurant account or merchant gateway
- Dzukku generates periodic fee reconciliation
- Lowest compliance burden for handling customer funds

Option B: split settlement through payment gateway
- Payment captured in a master arrangement
- Real-time split sends restaurant share + platform fee
- Better automation but higher compliance and provider dependency

### Diagram: Commission and Money Flow (Option A vs Option B)

```mermaid
flowchart TB
    subgraph A ["Option A — Recommended (Direct to Restaurant)"] 
        CA([Customer]) -->|"Pays full amount"| RA([Restaurant Payment Account])
        RA -->|"Periodic fee invoice (monthly/weekly)"| DZA(["Dzukku Platform\nFee Collection"])
        DZA -->|"Net settlement to restaurant"| BKA([Restaurant Bank Account])
    end

    subgraph B ["Option B — Split Settlement (via Gateway)"]
        CB([Customer]) -->|"Full payment captured"| PGW(["Payment Gateway\n(Razorpay / Stripe)"])
        PGW -->|"Instant platform fee split"| DZB(["Dzukku Platform\nWallet/Ledger"])
        PGW -->|"Instant restaurant share"| BKB([Restaurant Bank Account])
    end
```

## 7.2 Commission Rule Engine

Inputs:
- Restaurant plan tier
- Order channel (direct repeat vs new discovery)
- Campaign attribution
- Delivery involvement

Outputs:
- Gross amount
- Taxes
- Platform fee
- Net payable to restaurant

Sample formula:
- net_restaurant = gross - tax - platform_fee - payment_processing_fee

### Diagram: Commission Calculation Engine

```mermaid
flowchart LR
    ORD(["Order Placed\nGross: ₹500"]) --> CE["Commission Rule Engine"]

    subgraph CE ["Commission Engine Inputs"]
        TI["Plan Tier: Growth (8%)"] 
        CH["Channel: Repeat Customer (-2%)"] 
        AT["Attribution: Organic (no campaign fee)"] 
        DL["Delivery: Self (no delivery fee)"]
    end

    CE --> CALC["Calculate"]
    CALC --> BK["Breakdown"]

    subgraph BK ["Fee Breakdown"]
        G["Gross: ₹500.00"]
        TX["GST (5%): -₹25.00"]
        PF["Platform Fee (6%): -₹30.00"]
        PX["Payment Processing (2%): -₹10.00"]
        NET["Net to Restaurant: ₹435.00"]
    end

    BK --> LD[("Settlement Ledger")]
    BK --> DASH(["Restaurant Dashboard\nFee Breakdown"])
```

### Diagram: Settlement Sequence

```mermaid
sequenceDiagram
    actor Customer
    participant PAY as Payment Service
    participant PGW as Payment Gateway
    participant PS as Pub/Sub
    participant SET as Settlement Service
    participant LED as Ledger DB
    participant PAYOUT as Payout Processor
    participant BANK as Restaurant Bank
    actor Restaurant

    Customer->>PAY: Confirm order ₹500
    PAY->>PGW: Create payment intent
    PGW-->>Customer: Payment link / UPI
    Customer->>PGW: Completes payment
    PGW->>PAY: Webhook: PaymentSucceeded
    PAY->>PS: Emit event: OrderPaid {order_id, amount}
    PS->>SET: Consume OrderPaid event
    SET->>SET: Apply commission rules
    SET->>LED: Write debit/credit lines
    SET->>PAYOUT: Schedule payout (T+1 or T+2)
    PAYOUT->>BANK: Transfer net ₹435
    PAYOUT->>PS: Emit PayoutCompleted event
    PS-->>Restaurant: WhatsApp / Dashboard notification
    PS-->>Customer: "Your order is confirmed!"
```

## 7.3 Transparency Features

- Per-order fee breakdown in restaurant dashboard
- Settlement timeline tracker
- Dispute and adjustment ledger

## 8) Agentic AI System Design

## 8.1 Agent Roles

- Concierge Agent: discover menu and recommend
- Transaction Agent: cart, pricing, and order confirmation
- Policy Agent: refunds, substitutions, SLA handling
- Ops Agent: restaurant-side insights and automation

## 8.2 Agent Orchestration Pattern

- LLM does planning and response synthesis
- Tool execution remains deterministic in backend services
- Every tool call logged with policy checks
- Human fallback when confidence drops below threshold

### Diagram: Agentic AI Reasoning Loop (ReAct Pattern)

```mermaid
flowchart TD
    A(["User Message via WhatsApp"]) --> B["FastAPI Webhook Handler"]
    B --> C["Load Session Context\n(Redis)"] 
    C --> D["Agent Orchestrator\n(LangGraph / Vertex AI)"]

    D --> E{"Reason: What tool do I need?"}

    E -->|"Intent: Browse Menu"| F["Tool: search_menu()\n(pgvector + SQL)"]
    E -->|"Intent: Check Price"| G["Tool: get_item_price()\n(Catalog Service)"]
    E -->|"Intent: Place Order"| H["Tool: create_order()\n(Order Service)"]
    E -->|"Intent: Pay"| I["Tool: generate_payment_link()\n(Payment Service)"]
    E -->|"Confused / Low Confidence"| J["Tool: escalate_to_human()\n(Notification → Restaurant WA)"]

    F --> K["Observe Tool Result"]
    G --> K
    H --> K
    I --> K

    K --> L{"Is task complete?"}
    L -->|"No — need more info"| E
    L -->|"Yes"| M["Synthesize Final Response\n(Vertex AI — Gemini)"]

    M --> N["Guardrails Check\n(Policy Engine + PII Redact)"]
    N --> O["Send WhatsApp Reply"]
    O --> P["Append to Session History\n(Redis)"]  
    P --> Q["Emit Events to Pub/Sub → BigQuery"]
```

## 8.3 Retrieval and Grounding

Knowledge sources:
- Restaurant menu and metadata
- Operating hours and dynamic availability
- Promotions and constraints
- Policy and FAQ knowledge base

Use hybrid retrieval:
- Keyword + vector retrieval
- Metadata filtering by tenant_id and branch_id
- Strict citation mode for policy-sensitive responses

### Diagram: Hybrid Retrieval Architecture (RAG)

```mermaid
flowchart LR
    Q(["User Query: 'spicy chicken for 2'"]) --> EMB["Embedding Model\n(Vertex AI text-embedding)"]
    EMB --> VEC[("pgvector\nMenu Embeddings")]

    Q --> KW["Keyword Tokenizer"]
    KW --> SQL[("Cloud SQL\nMenu Items Table")]

    VEC --> MRG["Hybrid Merge + Re-ranker"]
    SQL --> MRG

    MRG --> FLT["Metadata Filter\n(tenant_id, branch_id,\navailability, allergens)"]
    FLT --> CTX["Retrieved Context\n(Top-K menu items)"]
    CTX --> GEM["Gemini — Grounded Response\nCan only quote items in context"]
```

## 8.4 AI Guardrails

- Prompt injection detection
- Output schema validation
- PII and payment data redaction
- Toxicity/safety classifier
- Hallucination risk controls by forcing tool-backed answers on transactional intents

### Diagram: Guardrails Pipeline

```mermaid
flowchart LR
    IN(["Raw User Input"]) --> INJ["Prompt Injection\nDetector"]
    INJ -->|"Flagged"| BLOCK1(["Block + Alert"])
    INJ -->|"Clean"| TOX["Toxicity Classifier\n(Vertex AI Safety)"]
    TOX -->|"Flagged"| BLOCK2(["Block + Log"])
    TOX -->|"Clean"| LLM["LLM Processing\n(Agent Orchestrator)"]
    LLM --> OUT["Raw LLM Output"]
    OUT --> SCH["Output Schema\nValidation (Pydantic)"]
    OUT --> PII["PII Scrubber\n(card no, bank, phone redaction)"]
    SCH -->|"Valid"| HAL["Hallucination Check:\nTransactional claim backed by tool?"]
    PII --> HAL
    HAL -->|"Grounded"| SEND(["Send to User"])
    HAL -->|"Ungrounded"| RETRY["Force Re-Attempt\nwith tool citation"] --> LLM
```

## 9) Technology Choices (Modern, Production-Oriented)

## 9.1 Backend Framework: Flask vs FastAPI

Recommendation: FastAPI

Why FastAPI is better for this architecture:
- Native async support for high-concurrency webhook and AI tool workflows
- Strong request/response validation via Pydantic
- Auto-generated OpenAPI docs for partner integrations
- Better performance for I/O heavy operations

When Flask is still acceptable:
- Small monolithic MVP with limited integrations
- Minimal async requirements

Final call:
- Use FastAPI for core platform services
- Flask can remain for legacy scripts/internal tools only

## 9.2 Cloud Platform (GCP + Vertex AI)

Recommended stack:
- Compute: Cloud Run (early) then GKE (scale/complex orchestration)
- API Management: API Gateway or Apigee (enterprise)
- Database: Cloud SQL for PostgreSQL
- Cache: Memorystore (Redis)
- Messaging: Pub/Sub
- Files: Cloud Storage
- Analytics: BigQuery
- Secrets: Secret Manager
- IAM and network controls: VPC + private service connect
- AI: Vertex AI (Gemini models + model tuning + safety + eval)

Code or no-code with Vertex AI?
- No-code (Vertex Studio) is great for fast prompt iteration and evaluation
- Code (Vertex SDK + orchestration service) is required for production agent workflows

Recommendation:
- Hybrid approach: no-code for prototyping, code-first for production runtime

## 9.3 Additional Modern Technologies

- Workflow orchestration: Temporal or Google Workflows (settlement/retries)
- Feature flags: OpenFeature compatible system
- Policy engine: OPA (Open Policy Agent)
- Event contracts: AsyncAPI + schema registry
- Infrastructure as Code: Terraform
- CI/CD: GitHub Actions + Cloud Deploy
- Observability: OpenTelemetry + Prometheus + Grafana + Cloud Logging
- Security scanning: SAST, dependency scanning, container scanning

## 10) Proposed Microservice Boundaries

1. Identity Service
- Customer and merchant profiles, consent, authn/authz

2. Conversation Service
- Session state, WhatsApp event normalization

3. Agent Orchestrator Service
- Planning, tool routing, safety enforcement

4. Catalog Service
- Menus, variants, availability, pricing metadata

5. Order Service
- Cart, order lifecycle, status transitions

6. Payment Service
- Payment intents, confirmation, reconciliation hooks

7. Commission and Settlement Service
- Commission calculation, payout schedule, ledger

8. Notification Service
- WhatsApp confirmations, fallback channels

9. Analytics Service
- KPIs, cohort, restaurant intelligence

### Diagram: Microservice Interaction Map

```mermaid
flowchart TD
    subgraph EDGE ["Edge"]
        GW["API Gateway\n(Cloud Armor + Rate Limit)"]
    end

    subgraph CORE ["Core Services — Cloud Run"]
        IDS["Identity Service"]
        CVS["Conversation Service"]
        AOS["Agent Orchestrator Service"]
        CATS["Catalog Service"]
        ORDS["Order Service"]
        PAYS["Payment Service"]
        SETS["Settlement Service"]
        NOTS["Notification Service"]
        ANS["Analytics Service"]
    end

    subgraph INFRA ["Infrastructure"]
        PG[("Cloud SQL\nPostgreSQL")]
        RD[("Memorystore\nRedis")]
        PS[("Pub/Sub")]
        BQ[("BigQuery")]
        VTX["Vertex AI\n(Gemini)"]
        GCS[("Cloud Storage")]
    end

    GW --> IDS
    GW --> CVS
    CVS --> RD
    CVS --> AOS
    AOS --> VTX
    AOS --> CATS
    AOS --> ORDS
    AOS --> PAYS
    AOS --> NOTS
    IDS --> PG
    CATS --> PG
    CATS --> GCS
    ORDS --> PG
    PAYS --> PG
    PAYS --> PS
    PS --> SETS
    SETS --> PG
    SETS --> PS
    PS --> ANS
    ANS --> BQ
    NOTS --> PS

    style AOS fill:#1a3a5c,color:#fff
    style VTX fill:#0f5132,color:#fff
    style PG fill:#3d1f5c,color:#fff
    style PS fill:#5c3d00,color:#fff
```

## 11) Data Model (Core Entities)

- customer
- customer_consent
- restaurant
- restaurant_branch
- menu
- menu_item
- conversation_session
- message_event
- cart
- order
- payment_intent
- settlement_ledger
- commission_rule
- payout
- audit_event

### Diagram: Core Order Data Model (ERD)

```mermaid
erDiagram
    MENU_ITEM {
        uuid item_id PK
        uuid branch_id FK
        string name
        string description
        float price
        string tags
        boolean is_available
        vector embedding
    }
    CART {
        uuid cart_id PK
        uuid session_id FK
        uuid branch_id FK
        json line_items
        float subtotal
        string status
    }
    ORDER {
        uuid order_id PK
        uuid cart_id FK
        uuid customer_id FK
        uuid branch_id FK
        string status
        float gross_amount
        float platform_fee
        float net_amount
        timestamp placed_at
    }
    PAYMENT_INTENT {
        uuid intent_id PK
        uuid order_id FK
        string gateway_ref
        string status
        float amount
        string currency
        timestamp created_at
        timestamp paid_at
    }
    SETTLEMENT_LEDGER {
        uuid ledger_id PK
        uuid order_id FK
        uuid merchant_id FK
        float gross
        float platform_fee
        float gateway_fee
        float net_payout
        string payout_status
        timestamp settled_at
    }

    MENU_ITEM ||--o{ CART : "added to"
    CART ||--|| ORDER : "becomes"
    ORDER ||--|| PAYMENT_INTENT : "has"
    ORDER ||--|| SETTLEMENT_LEDGER : "generates"
```

## 12) API Design Standards

- REST + event-driven callbacks
- Idempotency keys for payment/order creation
- Correlation IDs on every request
- Versioned APIs (/v1, /v2)
- Strict schema contracts and backward compatibility policy

## 13) Reliability, Performance, and Scale Targets

SLO targets:
- API availability: 99.95%
- Median webhook processing latency: < 300 ms
- P95 transaction response latency: < 1.5 s (excluding user wait states)
- Order creation reliability: > 99.99% successful state persistence

Scalability patterns:
- Stateless services with horizontal autoscaling
- Queue-based backpressure for downstream dependencies
- Read replicas for heavy read paths
- Cache hot menu and profile objects

## 14) Security and Risk Architecture

- OWASP ASVS controls baseline
- Zero trust internal service auth (mTLS/service identity)
- WAF and rate limiting at edge
- Fraud signals for repeated failed payments / abuse
- Immutable audit logs for payout-affecting events
- Periodic penetration testing and threat modeling

## 15) DevOps and Environment Strategy

Environments:
- dev, staging, pre-prod, prod

Release strategy:
- Trunk-based development
- Canary deployments for agent changes
- Blue-green for critical payment/settlement services

Operational readiness:
- Runbooks per service
- Error budget policy
- On-call escalation matrix

## 16) Rollout Plan (Phase-wise)

Phase 1: Foundation (0-8 weeks)
- FastAPI monolith with modular architecture
- WhatsApp integration
- Basic order + payment + commission calculator
- Identity baseline and consent capture

Phase 2: Production hardening (8-16 weeks)
- Service extraction (Identity, Order, Settlement)
- Event-driven architecture with Pub/Sub
- Observability, SLO dashboards, alerting
- Vertex AI agent orchestration with guardrails

Phase 3: Scale and intelligence (16-32 weeks)
- Advanced personalization and memory
- Restaurant intelligence dashboards
- Multi-region failover design
- Dynamic commission optimization

### Diagram: Phased Rollout Timeline

```mermaid
gantt
    title DzukkuBot — Phased Rollout Plan
    dateFormat  YYYY-MM-DD
    section Phase 1 — Foundation
    FastAPI monolith setup           :p1a, 2026-04-07, 14d
    WhatsApp webhook integration     :p1b, after p1a, 7d
    Basic menu + order flow          :p1c, after p1b, 10d
    UPI payment integration          :p1d, after p1c, 7d
    Identity + consent baseline      :p1e, after p1d, 7d
    Commission calculator v1         :p1f, after p1e, 5d

    section Phase 2 — Production Hardening
    Extract Identity Service         :p2a, after p1f, 14d
    Extract Order + Settlement Svc   :p2b, after p2a, 14d
    Pub/Sub event-driven arch        :p2c, after p2b, 10d
    Vertex AI agent integration      :p2d, after p2c, 14d
    Observability + SLOs             :p2e, after p2d, 7d
    Security hardening               :p2f, after p2e, 7d

    section Phase 3 — Scale and Intelligence
    Personalization + memory layer   :p3a, after p2f, 21d
    Restaurant analytics dashboard   :p3b, after p3a, 14d
    Multi-region failover design     :p3c, after p3b, 21d
    Dynamic commission optimization  :p3d, after p3c, 14d
```

### Diagram: Full End-to-End Happy Path (Sequence)

```mermaid
sequenceDiagram
    actor C as Customer (WhatsApp)
    actor R as Restaurant (WhatsApp / Dashboard)
    participant WA as WhatsApp Business API
    participant GW as API Gateway + WAF
    participant ID as Identity Service
    participant CV as Conversation Service
    participant AO as Agent Orchestrator
    participant VTX as Vertex AI Gemini
    participant CAT as Catalog Service
    participant ORD as Order Service
    participant PAY as Payment Service
    participant SET as Settlement Service
    participant NOT as Notification Service

    C->>WA: "I want spicy chicken for 2"
    WA->>GW: Webhook POST (verified)
    GW->>ID: Resolve customer identity
    ID-->>GW: customer_id + profile
    GW->>CV: Attach session context (Redis)
    CV->>AO: Forward message + context

    AO->>VTX: Reason: what tool to call?
    VTX-->>AO: Call search_menu("spicy chicken")
    AO->>CAT: search_menu(query, branch_id)
    CAT-->>AO: [Peri Peri Chicken ₹280, Spicy Wings ₹240]

    AO->>VTX: Synthesize reply
    VTX-->>AO: "We have Peri Peri Chicken or Spicy Wings! Add to cart?"
    AO->>WA: Send reply to customer
    C->>WA: "Add Peri Peri Chicken x2"

    AO->>VTX: Reason: call create_cart()
    AO->>ORD: create_cart(items, customer_id, branch_id)
    ORD-->>AO: cart_id, total ₹560

    AO->>PAY: generate_payment_link(cart_id, ₹560)
    PAY-->>AO: payment_url
    AO->>WA: Send payment link to customer

    C->>PAY: Pays via UPI
    PAY->>ORD: Confirm + create order
    ORD->>SET: Trigger commission calc
    SET-->>ORD: net ₹487 to restaurant

    NOT->>C: "Order confirmed! ETA 30 mins"
    NOT->>R: "New order: Peri Peri Chicken x2 — ₹560"
```

## 17) What Must Be Addressed Next (Gaps and Improvements)

Critical open items:
- Regulatory review for payment flow model by geography
- Legal design for fee invoicing vs split settlement
- WhatsApp template and policy compliance at scale
- Full data governance policy (consent, deletion, retention)
- Incident response drills for payment and payout events

High-impact improvements:
- Add human-in-the-loop console for low-confidence AI transactions
- Add evaluation harness for prompt and model drift
- Add simulation environment for commission rule changes
- Add customer identity portability framework

## 18) Architecture Decision Summary

- Use FastAPI as the primary backend framework
- Use GCP as the platform, Vertex AI for model and safety stack
- Start with Cloud Run for velocity, evolve to GKE when service graph complexity increases
- Design around digital identity and transparent low-commission settlement
- Keep transactional actions deterministic, with AI only orchestrating and assisting

## 19) PDF Export Instructions

This document is intentionally written in a PDF-ready format.

Simple export options:
- Open this Markdown in VS Code and export using a Markdown PDF extension
- Or convert via pandoc if available in your environment

Suggested output file name:
- Dzukku_Production_Architecture_v1.pdf

## 20) Final Recommendation to Your Question

For your goal (agentic AI chatbot, direct restaurant flow, digital identity, and production-level system design), FastAPI is the stronger foundation than Flask.

Use Vertex AI on GCP with a hybrid approach:
- No-code for rapid prompt experiments and evaluation
- Code-first orchestration for real production behavior, governance, and scalability

## 21) Pre-Project Learning Resources and Roadmap

Use this learning track before implementation so the team starts with shared foundations.

### 21.1 Core Backend Foundation (Week 1)

1. FastAPI official docs  
    https://fastapi.tiangolo.com/
2. Python async and asyncio  
    https://docs.python.org/3/library/asyncio.html
3. Pydantic docs  
    https://docs.pydantic.dev/latest/
4. SQLAlchemy 2.0 docs  
    https://docs.sqlalchemy.org/en/20/

### 21.2 Webhooks and Messaging (Week 1)

1. Meta WhatsApp Cloud API docs  
    https://developers.facebook.com/docs/whatsapp/cloud-api
2. Webhook reliability and signature best practices  
    https://stripe.com/docs/webhooks
3. HTTP idempotency reference  
    https://developer.mozilla.org/en-US/docs/Glossary/Idempotent

### 21.3 GCP and Production Deployment (Week 2)

1. Cloud Run docs  
    https://cloud.google.com/run/docs
2. API Gateway docs  
    https://cloud.google.com/api-gateway/docs
3. Cloud SQL for PostgreSQL docs  
    https://cloud.google.com/sql/docs/postgres
4. Pub/Sub docs  
    https://cloud.google.com/pubsub/docs
5. Secret Manager docs  
    https://cloud.google.com/secret-manager/docs
6. IAM overview  
    https://cloud.google.com/iam/docs/overview

### 21.4 Agentic AI and Vertex AI (Week 2)

1. Vertex AI documentation  
    https://cloud.google.com/vertex-ai/docs
2. Gemini on Vertex AI reference  
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini
3. RAG architecture patterns  
    https://cloud.google.com/architecture/gen-ai-rag-application-patterns
4. Prompt design fundamentals  
    https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/introduction-prompt-design

### 21.5 System Design and Reliability (Week 3)

1. Google SRE book  
    https://sre.google/sre-book/table-of-contents/
2. Designing Data-Intensive Applications (book)
3. OpenTelemetry documentation  
    https://opentelemetry.io/docs/

### 21.6 Payments and Settlement Domain (Week 3)

1. Razorpay documentation  
    https://razorpay.com/docs/
2. Stripe documentation (good architecture patterns)  
    https://stripe.com/docs
3. Accounting and ledger concepts for developers  
    https://www.moderntreasury.com/journal/accounting-for-developers-part-i

### 21.7 Security and Compliance Basics (Week 4)

1. OWASP ASVS  
    https://owasp.org/www-project-application-security-verification-standard/
2. OWASP API Security Top 10  
    https://owasp.org/www-project-api-security/
3. OpenID Connect overview  
    https://openid.net/developers/how-connect-works/

### 21.8 Suggested 30-Day Learning Plan

1. Days 1-7: FastAPI, async patterns, webhook fundamentals
2. Days 8-14: GCP runtime stack (Cloud Run, SQL, Pub/Sub, Secrets)
3. Days 15-21: Vertex AI, RAG grounding, safety guardrails
4. Days 22-30: Payments, security hardening, and SRE operations

### 21.9 Skill Validation Milestones

1. Build and test one idempotent webhook endpoint
2. Build one async API flow with external dependency timeout handling
3. Implement one event-driven order-to-settlement sample flow
4. Build one RAG response with citation-backed answers
5. Publish one dashboard with latency, error, and payment metrics
