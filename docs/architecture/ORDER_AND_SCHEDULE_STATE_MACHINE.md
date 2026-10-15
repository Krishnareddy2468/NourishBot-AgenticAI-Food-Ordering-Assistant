# Order and Schedule State Machine

Phase 2 separates an order's commercial lifecycle from scheduling, provider submission, and fulfillment. All runtime state changes must use `app.domain.order_state`; callers must not assign lifecycle fields directly.

## State dimensions

### Order

The canonical order lifecycle is:

```text
CREATED -> CONFIRMED -> ACCEPTED -> PREPARING -> READY
                                             -> OUT_FOR_DELIVERY -> DELIVERED -> COMPLETED
                                             -> DELIVERED -> COMPLETED
                                             -> COMPLETED
```

The exact permitted transitions are defined by `ORDER_TRANSITIONS`. `CANCELLED`, `FAILED`, and `EXPIRED` are terminal and deliberately distinct:

- `CANCELLED` means an authorized actor intentionally stopped the order.
- `FAILED` means fulfillment or provider execution could not complete.
- `EXPIRED` means a time-bounded confirmation, schedule, or submission opportunity elapsed.

A delivery failure transitions the order to `FAILED`, never `CANCELLED`. A provider rejection records the provider transition and then fails a non-terminal order with reason `PROVIDER_REJECTED` unless a more specific reason is supplied.

### Schedule

```text
NOT_SCHEDULED -> SCHEDULED -> ACTIVATED -> FULFILLED
                         |            |-> CANCELLED
                         |            `-> FAILED
                         |-> CANCELLED
                         `-> EXPIRED
```

Schedule state answers whether the future meal is waiting, active, or finished. It does not replace order status. A terminal order is mirrored to the active schedule through `sync_terminal_schedule`.

### Provider

```text
PENDING -> SUBMITTING -> CONFIRMED -> CANCELLED
                     |-> REJECTED
                     `-> UNKNOWN -> CONFIRMED | REJECTED | CANCELLED
PENDING -> CANCELLED | EXPIRED
```

In-house NourishBot orders use `NOT_REQUIRED`. `UNKNOWN` is a durable uncertain outcome: the system must reconcile it before retrying or claiming success.

### Fulfillment

`fulfillment_state` is derived by the central order transition service and provides an operational projection:

```text
NOT_STARTED, ACCEPTED, PREPARING, READY, OUT_FOR_DELIVERY,
DELIVERED, COMPLETED, CANCELLED, FAILED, EXPIRED
```

## Time fields

- `desired_for`: the customer-requested fulfillment time.
- `ready_by`: the target time by which preparation should be complete.
- `provider_slot_start` and `provider_slot_end`: a slot verified by an external provider.
- `submission_window_start` and `submission_window_end`: the safe window for submitting the order.

The legacy `scheduled_for` column is retained for compatibility during this phase. Migration `0011` copies it to `desired_for`; new scheduling logic should use `desired_for`.

## Validation and atomic auditing

`transition_order`, `transition_schedule`, and `transition_provider` validate both the current and requested states. Repeated, unknown, and unsupported transitions raise `InvalidStateTransition` before mutation.

Each transition appends an `OrderEvent` or `ScheduleEvent` to the caller's SQLAlchemy session. The transition service never commits, so the lifecycle mutation and audit event commit or roll back together.

Event records are append-only at two layers:

- SQLAlchemy rejects ORM update and delete operations.
- PostgreSQL triggers reject direct update and delete statements.

Every event identifies its source, actor type, optional actor ID, before/after state, reason code, metadata, and occurrence time.

## Migration contract

Migration `0011` follows website-auth migration `0010`. It preserves all existing orders and maps only legacy aliases:

| Legacy order status | Canonical status |
|---|---|
| `SCHEDULED` | `CONFIRMED` |
| `PAID` | `CONFIRMED` |
| `CONFIRMED_BY_RESTAURANT` | `ACCEPTED` |

It creates a `MealSchedule` for each existing scheduled order that has a customer, retains `scheduled_for`, copies the timestamp into `desired_for`, derives the independent state dimensions, and appends migration audit events.

Apply the migration before starting updated API or worker processes:

```bash
cd backend
alembic upgrade head
```

The downgrade removes Phase 2 records and fields. It maps still-scheduled canonical orders back to legacy `SCHEDULED`, but the new planning and audit data is necessarily lost; take a backup before downgrading a populated environment.

## Agent entry point

`app.agent.pipeline.process_message` is the only supported conversational agent entry point. `app.agent.entrypoint` documents that contract. The former `nourishbot_agent` and `orchestrator` entry points fail closed so a channel cannot silently run an older, unaudited execution path.
