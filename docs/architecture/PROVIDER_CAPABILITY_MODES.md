# Provider Capability Modes

This contract defines how NourishBot discovers, hands off, and transacts with food providers. External provider access is deny-by-default: a provider can perform only the operations recorded in its latest immutable capability version.

## Operating modes

| Mode | User experience | Permitted behavior |
|---|---|---|
| A — Discovery | NourishBot searches and compares provider data. | Read-only restaurant, menu, price, and availability tools. No provider mutation. |
| B — User-controlled handoff | NourishBot prepares context and sends the user to the provider. | Discovery plus a provider handoff. Checkout remains on the provider surface. |
| C — Verified transactional | NourishBot submits a confirmed scheduled order and returns a receipt. | Only explicitly verified transaction, status, tracking, and cancellation capabilities. |

Swiggy and Zomato start in Mode A. Observing tools does not verify them. NourishBot's in-house provider is seeded as Mode C because its scheduling, status, cancellation, and idempotency behavior is controlled by this application.

## Registry and verification

`provider_capability_versions` is an append-only registry. Each version records:

- Provider, adapter version, operating mode, and health state.
- Discovery, menu, live-price, scheduling, fulfillment, cancellation, status, authentication, and idempotency capabilities.
- The redacted tool inventory and its canonical SHA-256 hash.
- Verification time, verifier, and notes.

The four states are `UNVERIFIED`, `VERIFIED`, `DEGRADED`, and `DISABLED`. Mode C is valid only when the state is `VERIFIED`, a tool-schema hash exists, and scheduled ordering, status lookup, and stable idempotency are all verified.

When a live MCP tool inventory differs from the registered schema hash, NourishBot creates a new `UNVERIFIED` Mode A version. The old verified record remains available for audit, but the changed schema cannot inherit its authority.

An administrator may create a new capability version with `POST /api/v1/providers/{provider}/capabilities`. The endpoint cannot modify an existing version and rejects an invalid Mode C configuration.

## Runtime enforcement

Tool exposure is computed from the current registry version:

- Disabled providers expose no tools.
- Unknown tools and mutation-like tools are blocked.
- Transaction tools require an explicit confirmation turn, verified Mode C, schedule and idempotency arguments on the actual tool schema, and the required registered capabilities.
- Status and cancellation tools require their matching verified capability.
- Configuration flags cannot promote an unverified provider to Mode C.

Telegram and WhatsApp show the current provider mode during provider selection, so a user can distinguish discovery, handoff, and transactional behavior.

## Receipts and uncertain outcomes

Every parsed provider tool response is stored in the append-only `provider_receipts` table with a request fingerprint, response hash, capability version, schema hash, operation, outcome, and provider references. Transaction attempts also keep a minimal redacted receipt so a separate receipt-write failure never causes an automatic resubmission.

A Mode C response without a verifiable provider order ID becomes `UNKNOWN`; it is never reported as success. `UNKNOWN` sets `reconciliation_required` and `RECONCILE_BEFORE_RETRY`. On a later confirmation, NourishBot first calls the verified provider status adapter:

- `CONFIRMED`: return the existing order and do not resubmit.
- `UNKNOWN`: keep the attempt blocked from retry.
- Definite failure: allow a new submission with the same stable idempotency key.

Tracking and cancellation use the same adapter and receipt path. An ambiguous cancellation becomes `UNKNOWN` and requires reconciliation.

## Data minimization

Provider request and response records are recursively redacted before persistence or application logging. Sensitive keys and common text forms for credentials, authorization headers, cookies, addresses, phone numbers, email addresses, coordinates, and payment data are replaced. Chat identifiers are stored in provider receipts only as SHA-256 fingerprints.

Redacted payloads are diagnostic evidence, not a source of customer contact or payment data.

## Operational API

| Endpoint | Access | Purpose |
|---|---|---|
| `GET /api/v1/providers/capabilities` | Staff | Current public capability versions and modes. |
| `POST /api/v1/providers/{provider}/capabilities` | Admin | Append a reviewed capability version. |
| `GET /api/v1/providers/attempts` | Manager | Tenant-scoped provider attempts and recovery state. |
| `GET /api/v1/providers/attempts/{id}/receipts` | Manager | Tenant-scoped redacted response receipts. |
| `POST /api/v1/providers/attempts/{id}/reconcile` | Manager | Run verified status-before-retry reconciliation. |
| `GET /api/v1/providers/attempts/{id}/tracking` | Manager | Fetch status through a verified tracking adapter. |
| `POST /api/v1/providers/attempts/{id}/cancel` | Manager | Request cancellation through a verified adapter. |

## Deployment

Apply Alembic migration `0013` before starting the updated application. It creates the registry and receipt tables, extends provider attempts with reconciliation metadata, seeds conservative defaults, and installs PostgreSQL append-only triggers.

Enabling `MCP_TRANSACTIONAL_ENABLED` only permits the runtime to consider Mode C. It does not override the registry. External transactional access must remain disabled until the exact live schema and all claimed capabilities have been tested and recorded as a new verified version.
