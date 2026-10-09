# NourishBot — Interview Prep

Written for: me, before the call. Read the "Corrections" section first — that is
the part that stops me getting caught out.

Everything here was checked against the code on 15 Sep 2026, not from memory.
File paths and line numbers are real. Where my prepared answer did **not** match
the code, I wrote the truth and marked it.

---

## 0. The 60-second fact card

Memorise these. They are the numbers I will be asked for.

| Thing | Real value | Where |
|---|---|---|
| Backend entry | `app.api.main:api` | `backend/main.py` |
| Bot + API + worker | all in **one** process, one event loop | `backend/app/api/main.py:57-98` |
| Pipeline stages | Context → Plan → Execute → Verify → Respond | `backend/app/agent/pipeline.py` |
| Loop bound | `AGENT_MAX_ITERATIONS = 6` | `backend/app/core/config.py:49` |
| States | 11 (`IDLE` … `SUPPORT_CASE`) | `backend/app/agent/state_machine.py:20-31` |
| Events | 18 | `state_machine.py:42-66` |
| Models | `gpt-4o` → `gpt-4o-mini` → `gpt-3.5-turbo` | `config.py:46-48` |
| MCP timeout | 45s per tool, 90s per agent turn | `config.py:93`, `mcp_agent.py:598` |
| DB pool | 10 + 20 overflow (main loop) | `backend/app/db/session.py:22-28` |
| Outbox poll | every 1.0s, 100 rows a batch | `backend/app/workers/outbox_worker.py:23-24` |
| Tests | **zero** | `tests/` holds only `.gitkeep` files |

---

## 1. The shape of the system

```
                         Telegram user
                              │
                              ▼
                  ┌───────────────────────┐
                  │  bot/telegram.py      │
                  │  _do_think_and_reply  │   ← the router, line 192
                  └───────────┬───────────┘
                              │ reads session["ordering_platform"]
              ┌───────────────┼────────────────┐
              │               │                │
     platform = NourishBot   Zomato/Swiggy    MCP is down
        (or unset)       + MCP_ENABLED    (last resort)
              │               │                │
              ▼               ▼                ▼
   ┌────────────────┐  ┌──────────────┐  ┌──────────────┐
   │ agent/pipeline │  │ agent/       │  │ agent/       │
   │ 5 stages       │  │ mcp_agent.py │  │ orchestrator │
   │ our own DB     │  │ LangGraph    │  │ .py (legacy  │
   │ LLM cannot     │  │ ReAct +      │  │  tool loop)  │
   │ write          │  │ MCP tools    │  │              │
   └───────┬────────┘  └──────┬───────┘  └──────┬───────┘
           │                  │                 │
           ▼                  ▼                 ▼
    PostgreSQL          Zomato / Swiggy    redirect links
    (orders, cart)      live APIs
           │
           ▼
    outbox_events ──► outbox_worker ──► WebSocket ──► staff dashboard
                                                      (Next.js: kitchen,
                                                       waiter, admin, track)
```

One honest note on this diagram: there are **three** paths in the code, not two.
The third one (`orchestrator.py`) is the old hand-rolled tool loop. It is still
wired in at `telegram.py:238` as the fallback when the MCP connection dies. If
they ask "is it really two paths", the answer is "two by design, three in the
repo, and the third is dead weight I should delete".

---

## Chain 1 — The opener

### Q1. Walk me through NourishBot.

NourishBot is a food ordering assistant that lives inside Telegram, plus a
staff dashboard for the restaurant side. Backend is FastAPI on Postgres,
frontend is Next.js.

A customer chats normally — "two chicken biryanis to Kondapur" — and the bot
builds a cart, collects the details it needs, places the order, and takes
payment through Razorpay. On the other side, the kitchen screen, waiter view
and admin dashboard update live over WebSocket.

The one design decision that shaped everything: **there are two execution
paths, and the difference is who is allowed to write to the database.**

- For our own restaurant's orders, a fixed five-stage pipeline runs. The LLM
  plans and the LLM writes the final message, but deterministic Python code is
  the only thing that touches the orders table.
- For Zomato and Swiggy, I use a LangGraph ReAct agent talking to their MCP
  servers. There the model does drive the tool calls, because those flows are
  open-ended and I do not own the data.

That is the trade I made: safety where I own the money, flexibility where I do not.

> Interview tip: stop talking here. That last line is the hook. Let them pull
> the thread they care about.

### Q2. Why did you build it? Was it for a client or a course?

Self-directed. Not a course project, not a client brief.

I wanted to find out what actually breaks when you put an LLM in front of a
real transaction — real money, real Razorpay, real kitchen. Demos of chat
ordering are everywhere and they all work fine on the happy path. I wanted to
see the second half of the problem.

The restaurant domain was deliberate: orders have prices, prices have to be
right, and a wrong number is a visible failure, not a fuzzy one. That makes it
a good test of whether an agent can be trusted with writes.

### Q3. Which part took the longest?

Not the features. The reliability work.

Getting the happy path running took about a week. Making it survive reality
took much longer, and reality meant:

- The MCP connection to Zomato dying halfway through a conversation. I had to
  add a per-platform client cache, a 30 second reconnect cooldown
  (`MCP_RECONNECT_COOLDOWN_S`), and a cache reset when a call fails
  (`mcp_clients.py:195-206`).
- `mcp-remote` running as a Node subprocess and throwing `AbortError` during
  its own cleanup — after the tool call had already succeeded. I spent a
  genuinely embarrassing amount of time treating a success as a failure. The
  fix and the reasoning are in `mcp_agent.py:614-632`.
- The model returning JSON that did not parse. The planner now retries once
  with a smaller prompt, then falls back to a plain `SUPPORT` goal rather than
  crashing (`planner.py:221-263`).
- Replies getting cut off mid-sentence. The responder checks `finish_reason`
  and also checks whether the text ends on a real punctuation mark, and retries
  once with a shorter prompt (`responder.py:73-88`).

None of that is in a demo. All of it is the actual work.

---

## Chain 2 — Architecture and routing

### Q4. Why two separate execution paths instead of one agent?

Because of one rule I did not want to negotiate with: **the LLM must never be
able to commit a write to our own orders table.**

If the model can call `place_order` directly, then every bad output is a bad
order. Wrong price, wrong quantity, duplicate order, an order placed before the
customer said yes. Those are not chat bugs, they are refunds.

So in the in-house path I split it. The model produces a plan — a JSON list of
proposed actions. Python decides whether to run them. Look at
`planner.py:98`: the proposed actions are hard-capped at 6 and every unknown
tool name is dropped. Then `executor.py` is the only module in the codebase
that opens a write session for orders.

For Zomato and Swiggy, the calculation flips. I do not own that data, the flows
are genuinely open-ended (search, pick an address, pick a restaurant, browse a
menu, build a cart), and the worst case is a failed API call rather than a
corrupted row in my own DB. That is where an agent earns its keep, so that path
is `create_react_agent` with the MCP tools bound (`mcp_agent.py:482`).

### Q5. Couldn't you have enforced that with prompts?

No.

A prompt is a request. An architecture is a guarantee.

I can write "never place an order without confirmation" in capital letters
inside a box drawn with Unicode characters — and I did, it is
`mcp_agent.py:316-400` — and the model will still get it wrong some fraction of
the time. Every prompt rule is a probability, not a constraint.

In the in-house pipeline the model does not have a write tool. It has no
database session. It cannot reach one. `executor.py` is the only thing holding
`AsyncSessionLocal`, and it is plain Python: it re-reads every price from the
menu table before it writes, it checks opening hours, it checks the delivery
minimum, it checks kitchen capacity. If the model proposes a `place_order` with
a price it invented, the price never survives contact with the executor —
`executor.py:345-362` throws the model's numbers away and reads the real ones.

That difference — "unlikely" versus "impossible" — is the whole argument. A
prompt lowers the failure rate. An architecture sets it to zero for that class
of failure.

The honest boundary: this protects against the model doing the wrong thing. It
does not protect against my executor having a bug. It moves the risk from a
probabilistic system into a deterministic one, where I can actually test it.
(And right now I do not test it — see Q22.)

### Q6. What's the cost of maintaining two paths?

Real, and I knew it going in.

- **Two places to fix one bug.** Session handling, history trimming and error
  messages all exist twice, in slightly different shapes.
- **Two mental models.** A new developer has to learn that "how the bot thinks"
  depends on which button the user pressed earlier.
- **Two persistence stories.** The pipeline persists structured state
  (`ContextSnapshot` fields into the session row). The MCP path replays a rolling
  message history. They are not the same thing and cannot be inspected the same
  way.
- **Drift.** The clearest proof: `orchestrator.py` is a *third* path that I have
  not deleted. It only runs when MCP is unavailable, so it is rarely exercised
  and quietly rots.

I accepted it because the safety property was worth more than the tidiness at
this size. If the system grew, I would move both onto LangGraph — one graph,
one checkpointer, one trace to look at when something breaks — and keep the
safety property by *not binding a write tool* on the in-house sub-graph, rather
than by having a separate codebase.

---

## Chain 3 — The five-stage pipeline

### Q7. Walk me through Context, Plan, Execute, Verify, Respond.

```
 user message
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CONTEXT   context_builder.build_context()                │
│    One read of the DB. Builds ContextSnapshot:              │
│    customer, cart + total, active orders, current_state,    │
│    pending_slots, kitchen_load, menu, last turns, policy.   │
│    No LLM.                                                  │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PLAN      planner.Planner.plan()          ← LLM #1       │
│    OpenAI in JSON mode. Returns:                            │
│      goal, missing_slots, constraints,                      │
│      proposed_actions[], requires_confirmation              │
│    Capped at 6 actions. Unknown tools dropped.              │
│    Then _guard_order_plan() strips place_order if any       │
│    required slot is still missing.  ← deterministic guard   │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EXECUTE   executor.Executor.run()                        │
│    Plain Python. Only module that writes orders.            │
│    Re-reads prices from menu. Enforces policy: hours,       │
│    delivery minimum, max items, max qty, kitchen capacity.  │
│    Returns committed[] and rejected[].  No LLM.             │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. VERIFY    verifier.Verifier.verify()                     │
│    Re-reads what was just written, from the DB, fresh       │
│    session. Recomputes the order total from OrderItem rows  │
│    and compares. Re-checks availability. Builds the         │
│    VerifiedSummary — the ONLY facts the responder sees.     │
│    No LLM.                                                  │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. RESPOND   responder.Responder.respond()   ← LLM #2       │
│    Turns verified facts into a friendly reply.              │
│    Gets facts only — no DB handle, no tools.                │
│    Falls back to a plain template if the call fails.        │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
              state machine transition + persist turn
```

Two LLM calls per turn, at the two ends. Everything that touches money in the
middle is deterministic.

The bit I am proudest of is small: the responder physically cannot see the
database. It gets a `VerifiedSummary` dataclass. So the bot cannot tell the
customer a price that was not read back out of the orders table.

### Q8. What happens when Verify fails?

**This is the one where my prepared answer was wrong. Read this carefully.**

What I *used* to say: "it routes back to Plan with the failure reason, bounded
by a retry count."

What the code actually does (`pipeline.py:114-133`):

```python
while remaining_actions and iteration < settings.AGENT_MAX_ITERATIONS:
    exec_result = await Executor.run(remaining_actions, ctx)
    remaining_actions = []          # <-- everything consumed in one go
    summary = await Verifier.verify(...)
    if not summary.safe_to_respond:
        logger.error(...)
        break                       # <-- no route back to Plan
    iteration += 1
```

So: the loop is written to look iterative, but `remaining_actions` is cleared
on the first pass, which means **it runs exactly once**. When Verify sets
`safe_to_respond = False` it logs, breaks, and the Responder composes an honest
"something went wrong" reply. There is no re-plan.

There *are* two real correction mechanisms, they just are not a plan loop:

1. **Before the fact** — `_guard_order_plan` (`pipeline.py:311-384`) deletes
   `place_order` from the plan if name, phone, order type or delivery address
   are still missing. This is the self-correction that actually fires, many
   times a day.
2. **After the fact** — if Verify recomputes the order total and it disagrees
   with `order.total_cents`, it **writes the correct total back to the DB**
   (`verifier.py:214-223`) rather than reporting a wrong number to the customer.

If they push on this, I say plainly: the retry loop is vestigial, it looks like
a loop but executes once, and I would either implement the re-plan properly
with the failure reason appended to the planner prompt, or delete the loop and
stop pretending. Claiming a feedback loop I do not have would be worse than
admitting the design is one-shot with a strong pre-check.

### Q9. Give me a concrete example of something Verify caught.

Two real ones, both in the code, both with the handling right there:

**Price drift.** After `place_order` commits, Verify opens a *fresh* session,
re-reads every `OrderItem`, and recomputes `sum(unit_price_cents * qty)`. If it
differs from `order.total_cents` by more than 1 paisa it logs an error and
corrects the row (`verifier.py:214-223`). This catches the window where a menu
price changed between the planner reading the menu and the executor writing the
order.

**Item went unavailable mid-order.** After the order is written, Verify checks
each `MenuItem.available` flag again (`verifier.py:226-238`). The order is
already placed, so it logs a warning rather than blocking — but the signal is
there for the kitchen.

**The class of failure it is really designed for** — and I would say it this way
rather than invent an incident — is *the gap between reading and writing*.
Between the planner deciding and the executor committing, the world can change:
prices, stock, kitchen capacity. Verify's job is to answer "what is actually in
the database right now", so the customer is never told a number the system does
not hold.

The strongest single line in the whole file is the one that sets
`safe_to_respond = False` when the order it just wrote cannot be found on
re-read (`verifier.py:201-206`). That should never happen. If it ever does, the
bot says so instead of confidently making something up.

---

## Chain 4 — LangGraph specifics

### Q10. Show me your state schema.

**Another one where honesty is required.** The LangGraph path does not have a
custom state schema. I use the prebuilt:

```python
agent = create_react_agent(llm, bound_tools)   # mcp_agent.py:482
```

which means the state is LangGraph's built-in `MessagesState` — a single
`messages` key with the `add_messages` reducer. I did not define my own.

The real state in this project lives in two other places, and I should lead with
these because they are mine:

**`ContextSnapshot`** (`context_builder.py:50`) — the dataclass rebuilt from
Postgres at the start of every pipeline turn. Real fields:

```
chat_id, restaurant_id, user_name
customer_id, customer_name, customer_phone, language, user_language
cart[], cart_total_cents, cart_id
active_orders[]
channel_id, session_id
current_state                  # BotState enum
pending_goal, pending_slots[], upsell_count
user_preferences, memory_summary, top_cravings[]
order_type, delivery_address
pending_payment_order_id, pending_order_ref
policy, is_open, kitchen_load, menu_snapshot[]
last_turns[], time_of_day, now
```

**`BotState`** (`state_machine.py:20`) — 11 states, persisted as a string on the
session row: `IDLE`, `BROWSING_MENU`, `BUILDING_CART`, `COLLECTING_DETAILS`,
`AWAITING_CONFIRMATION`, `AWAITING_PAYMENT`, `ORDER_PLACED`,
`ORDER_IN_PROGRESS`, `OUT_FOR_DELIVERY`, `COMPLETED`, `SUPPORT_CASE`.

Transitions are a plain dict keyed by `(state, event)` with 18 events, and
`TOOL_EVENTS` maps each committed tool to the event it emits. Some tools emit
different events depending on the result — `place_order` emits
`PAYMENT_REQUIRED` if Razorpay came back with an intent, `ORDER_PLACED` if it
is cash on delivery (`state_machine.py:212-218`).

If asked why no custom LangGraph state: I did not need extra channels, the
prebuilt covered the ReAct loop, and my real state was already in Postgres. If
I unified the two paths I would need a proper schema, and that is one of the
reasons unifying is not a free change.

### Q11. Why add_messages as a reducer instead of overwriting?

Because a ReAct turn produces several messages and they all have to survive.

One turn is: AI message with tool calls → one ToolMessage per call → possibly
another AI message with more tool calls → final AI message. If the `messages`
channel were last-write-wins, each node returning `{"messages": [...]}` would
throw away everything before it, and the model would lose the tool results it
just asked for.

`add_messages` appends instead of replacing, and it also de-duplicates by
message id, so re-entering a node does not double up history. Scalar fields in
a state schema stay last-write-wins, which is what you want for something like
a step counter — you want the new value, not both values.

I get this behaviour from `MessagesState` through `create_react_agent` rather
than declaring it myself.

### Q12. What's your loop bound, and what happens when it's hit?

Different answer per path. Be precise here.

**In-house pipeline:** `AGENT_MAX_ITERATIONS = 6` (`config.py:49`). It is used
in two places: the planner hard-truncates `proposed_actions` to 6
(`planner.py:98`), and the executor loop is bounded by it. So the real bound is
"at most 6 tool calls per turn", enforced before execution rather than during.
There is no exception path — the executor returns what it committed and the
responder replies with whatever is true.

**MCP / LangGraph path:** I did **not** set `recursion_limit`, so it is
LangGraph's default of 25. The bound that actually fires in practice is the
timeout: the whole agent turn is wrapped in
`asyncio.wait_for(..., timeout=MCP_TOOL_TIMEOUT_S * 2)` = 90 seconds
(`mcp_agent.py:598-601`). On timeout the user gets a graceful message —
"that took too long on the Zomato side, could you try again" — not a stack
trace.

Saying it honestly: the recursion limit is the safety net I inherited from the
framework, not a bound I designed. What I designed is the timeout and the
per-stage caps. If I were doing it again I would set `recursion_limit`
explicitly and route to a final-response node when it trips, so the bound is
visible in my code rather than hidden in a default.

### Q13. Did you use a checkpointer?

No. Plainly: no.

What I do instead — history lives in my own Postgres session row. Each MCP turn
calls `get_session(chat_id)`, replays the last 8 visible user/assistant turns
into the prompt, invokes the agent, then writes the turn back with
`save_session` (`mcp_agent.py:538-547`, `700-712`). `_trim_mcp_history` caps
visible history at 16 turns but always preserves the newest hidden
`mcp_context` turn, which is where I stash the real `address_id` and `res_id`
values from the last tool call so the model reuses IDs instead of inventing
them.

Why it went that way: the NourishBot session table already existed for the in-house
path, and reusing it meant no schema change. That was the right call for
shipping and the wrong call for the long run.

What I would do now — `AsyncPostgresSaver`, with `thread_id` set to the chat id.
It would give me three things I currently hand-roll or do without: real message
persistence including tool calls (not just the text I extract), the ability to
resume a turn after a crash instead of losing it, and time-travel for debugging
a bad conversation. The last one alone would have saved me days on the
`AbortError` bug.

---

## Chain 5 — Reliability and data

### Q14. Explain the transactional outbox and what problem it solved.

**Careful. My prepared answer describes the pattern correctly and the code does
not implement it. Do not claim the clean version.**

**The problem it is meant to solve** — the dual-write problem. Placing an order
needs two things to happen: the row goes into Postgres, and the kitchen screen
finds out. If I write the row and then push to WebSocket, and the process dies
in between, the order exists and nobody cooks it. Two systems, no shared
transaction, and a crash in the gap loses the event.

**The correct pattern** — write the order and the event row in the *same*
transaction. Either both land or neither does. A separate worker reads the
event table and does the delivery. The DB becomes the queue.

```
   ┌───────────── ONE TRANSACTION ─────────────┐
   │  INSERT INTO orders ...                   │
   │  INSERT INTO outbox_events ...            │   ← both, or neither
   └───────────────────────┬───────────────────┘
                           │ commit
                           ▼
              outbox_worker (every 1.0s, 100 rows)
                           │
                           ▼
                  ws_manager.broadcast()
                           │
                           ▼
              kitchen / waiter / admin screens
```

**What my code actually does.** The table exists (`OutboxEvent`:
`id, restaurant_id, event_type, payload, created_at, processed_at`). The worker
exists and runs fine. But:

- The pipeline's `place_order` (`executor.py:302-455`) — the path that actually
  runs for our restaurant — **writes no outbox event at all.**
- The only producer is the legacy `orchestrator.py:471-484`, and it opens a
  **separate session with its own commit**, wrapped in `try/except` that logs a
  warning and moves on. That is a second write, not one transaction. It is
  exactly the dual-write problem the pattern is supposed to remove.

So the correct answer to this question is: "I built the outbox table and the
drain worker, I did not finish wiring the producer side, and the one producer
that exists is not in the same transaction — which means today it does not
actually solve the problem it was designed for. The fix is small and I know what
it is: move the `OutboxEvent` insert inside the `async with AsyncSessionLocal()`
block in `executor._tool_place_order`, before the existing `await db.commit()`
on line 405."

That answer is much stronger than being caught describing code I did not write.

**The tradeoff worth volunteering** (and this part is real): an outbox gives you
at-least-once delivery, never exactly-once. The worker can broadcast and then
crash before setting `processed_at`, so the event goes out twice. Consumers have
to be idempotent. My WebSocket events carry `event_id`, so a dashboard can
de-duplicate on it — though I have not made the frontend do that yet.

### Q15. What happens if the outbox worker crashes mid-dispatch?

For what is written today:

The worker selects unprocessed rows, broadcasts each, sets `processed_at` in
memory, and commits once at the end of the batch (`outbox_worker.py:30-68`). If
it dies after broadcasting but before the commit, the whole batch stays
`processed_at IS NULL` and gets picked up again on restart. So events are
re-delivered — at-least-once, as expected.

Now the parts I have **not** built, which I should say before they ask:

- **No retry count and no backoff.** If one event fails to broadcast
  permanently, it is retried every second, forever. There is no `attempts`
  column and no dead-letter path. A poison event would spin.
- **No row locking.** The query has no `FOR UPDATE SKIP LOCKED`, so two workers
  would grab the same rows and double-broadcast. Today that is invisible because
  there is only ever one worker — it is an `asyncio.create_task` inside the
  FastAPI lifespan (`api/main.py:62`). The moment I run two replicas it breaks.
- **Per-event error handling is soft.** A broadcast failure is caught and logged,
  and the event simply is not marked processed, which is the right default — but
  it means a broken consumer produces an unbounded retry loop rather than an
  alert.

The three fixes in order: add `attempts` + `next_attempt_at` with exponential
backoff; add `FOR UPDATE SKIP LOCKED` so it is safe to scale out; move the
worker out of the API process into its own deployment.

### Q16. Why per-platform locks? What race were you preventing?

**My prepared answer was about the wrong lock.** I had been saying it prevents
two rapid user messages from interleaving. It does not. Here is what is really
in the code.

The lock is `_connect_locks` — one `asyncio.Lock` per platform, keyed "Zomato" /
"Swiggy" (`mcp_clients.py:82-89`), and it guards **connection setup**, not
message handling.

The race it prevents: connecting to an MCP server means spawning a
`mcp-remote` Node subprocess over stdio. If three users message at once and the
tool cache is cold, all three coroutines see an empty cache and all three try to
spawn a subprocess. That is three processes where one is needed, three OAuth
handshakes, and two leaked children — because the client has to be kept
referenced or the subprocess dies (`mcp_clients.py:179-186`).

So the shape is the standard double-checked cache fill:

```python
if platform in _tools_caches:            # fast path, no lock
    return _tools_caches[platform]

async with _get_connect_lock(platform):
    if platform in _tools_caches:        # someone else filled it while we waited
        return _tools_caches[platform]
    ...spawn, connect, cache...
```

It is *per platform* rather than one global lock so that a slow Zomato connect
does not block a Swiggy user. Same pattern again at `mcp_agent.py:455` for
building the agent itself.

**And the race I have not covered:** there is genuinely no per-user lock
anywhere. Two fast messages from the same chat can run two pipeline turns
concurrently, both reading the same `ContextSnapshot` and both writing the
session row — last write wins, and the first turn's state update is lost. I
should say this out loud rather than let them find it. The fix is a per-chat_id
lock, and because of the answer to Q17 it should be a Redis or Postgres advisory
lock from day one, not an in-process one.

### Q17. Does that lock survive across multiple processes?

No. It is `asyncio.Lock` — it is scoped to one event loop inside one Python
process. Two uvicorn workers, or two containers, and each has its own lock
object and neither knows about the other. It is not a distributed lock and I
would not claim it is.

For the MCP connect lock this is *almost* fine, because the cost of losing it is
a duplicate subprocess per process, not corrupted data. For the per-user
serialisation from Q16 it would be useless.

What I would change: `pg_advisory_xact_lock(hashtext(chat_id))` so the lock is
held by the transaction and released automatically if the process dies, or a
Redis lock with a TTL if I wanted it outside the DB. Redis is already a
dependency (`REDIS_URL` in config), so the plumbing exists.

There is a bigger version of this problem worth admitting in the same breath:
this app cannot currently run more than one replica *at all*, and the lock is
not the reason. The Telegram bot uses long polling started inside the FastAPI
lifespan (`api/main.py:71-76`). Two replicas means two pollers fighting over the
same update queue, plus two outbox workers double-broadcasting. Horizontal
scaling needs three changes together: webhook instead of polling, the outbox
worker split into its own process with `SKIP LOCKED`, and shared locks. Naming
all three is the answer; naming only the lock would miss the point.

### Q18. Walk me through your three-model fallback chain.

Again it is per-path, and the paths genuinely differ.

**MCP path** — a real 3-tier chain via LangChain (`mcp_agent.py:267-290`):

```
gpt-4o   ──fails──►  gpt-4o-mini  ──fails──►  gpt-3.5-turbo  ──fails──►  raise
(primary)            (cheap, stable)          (legacy)
```

built with `primary.with_fallbacks([fallback1, fallback2])`, all three
configurable by env var. On top of that sits `ProviderError.classify()`
(`mcp_agent.py:40-91`), which buckets whatever comes out into rate limit, auth
failure, service unavailable, timeout, bad request or unknown, and maps each to
a message a customer can actually read.

**In-house pipeline** — no model chain. The planner and responder use
`OPENAI_PRIMARY` only. What they have instead is two attempts with a *smaller
prompt* on the retry, then a deterministic non-LLM fallback: the planner returns
`goal="SUPPORT"` (`planner.py:109`) and the responder renders a plain template
from the verified facts (`responder.py:208`). So the in-house path degrades to
"correct but plain English", never to an exception.

**The honest weakness**, and I would volunteer it because it is the interesting
part of the question: `with_fallbacks()` retries on *any* exception. It does not
distinguish retryable from non-retryable. A 429 should absolutely fall through
to the next model. A 401 from a bad API key should not — it will fail on all
three, burn three round trips, and give the user a 90-second wait before an
error. I have the classifier that knows the difference
(`ProviderError.classify`), I just do not use it to decide whether to retry. That
is maybe twenty lines of work and it is the first thing I would fix in that file.

---

## Chain 6 — The stress questions

### Q19. What would break first if you had 10,000 concurrent users?

Not the database. In order, what actually gives:

**1. The process model, before anything else.** Everything runs in one process
and one event loop: FastAPI, the Telegram poller, and the outbox worker
(`api/main.py:57-98`). There is no second replica to add — adding one creates
duplicate Telegram polling and duplicate outbox broadcasts. So the honest first
answer is that the system does not scale horizontally at all yet, and that has
to be fixed before any other number matters.

**2. OpenAI rate limits and latency.** Every in-house turn is two sequential
LLM calls. At a couple of seconds each that is a hard floor of roughly 3-5
seconds per reply, and it is not something I can optimise away with better
Python. Under load I hit tokens-per-minute limits and every user queues behind
the same bucket. Mitigations: cache the planner output for repeated intents,
drop the responder call entirely for template-able replies (order confirmed,
cart shown), and batch or shard across keys.

**3. The thread pool.** This one is specific and I like it as an answer because
it is non-obvious. The planner and responder use the *synchronous* OpenAI client
pushed into `loop.run_in_executor(None, ...)` (`planner.py:233`,
`responder.py:62`). `None` means the default executor — bounded, shared, and the
same pool FastAPI uses for every other blocking call. Each in-flight LLM call
occupies a thread for its full duration. A few hundred concurrent turns and the
pool is saturated, at which point *everything* blocks, including health checks.
The fix is `AsyncOpenAI` and awaiting it properly.

**4. Database connections.** `pool_size=10, max_overflow=20` on the main engine,
so 30 connections maximum — and the Telegram loop gets its *own* engine with
another 15 (`session.py:22-28, 49-60`). The executor also opens a fresh session
per tool call rather than sharing one per turn, which multiplies checkouts.
Raising the pool is easy; the real fix is pgbouncer and fewer sessions per turn.

**5. MCP.** Each platform holds one subprocess and one shared client. That was
never designed for concurrency, and 10,000 users through one stdio pipe is not a
tuning problem, it is a redesign.

**6. In-process state.** The tool caches, the agent cache and the connect locks
are all module-level dicts. They are per-process, so with N replicas you get N
copies, N subprocesses, and N first-request cold starts.

### Q20. What's the worst bug you hit in this project?

The MCP `AbortError`. Telling it as a debugging story, not a win:

**What I saw.** Users on the Zomato path would send a message, wait, and get a
generic failure. But when I checked the Zomato side afterwards, the action had
often *worked* — the search ran, the cart updated. Failure reported, side effect
committed. That is the worst possible shape for a bug, because the user retries
and you get it twice.

**What I suspected first, and got wrong.** My first theory was a timeout — 45
seconds felt tight for a chain of remote calls, so I raised it. No change. My
second theory was that the tool schema from `langchain-mcp-adapters` was
malformed and the model was calling something that did not exist. I logged every
tool call and every result to check. That is why `mcp_agent.py:645-700` has
that much logging in it — it is debugging scaffolding I left in on purpose.

**What the logs actually showed.** The tool call went out. The tool result came
back, correct and complete. The AI message after it was fine. And *then* an
exception surfaced, with `AbortError` in the string, from the Node side.

**The cause.** `mcp-remote` runs as a Node subprocess bridging remote HTTP to
local stdio. When it finishes a call and tears down the SSE connection, undici
raises a `DOMException [AbortError]` during cleanup. That propagates up through
the adapter into my `await`. The work was already done. The error was the
sound of the door closing behind it.

**What I changed.** Classify it separately and treat it as success
(`mcp_agent.py:614-632`). Real errors still reset the platform cache and force a
fresh connect. The comment there is long on purpose — including the note that
the Node subprocess still prints the error to stderr and I cannot suppress it
from Python, so the next person is not sent down the same path.

**What I learned.** In a system with subprocesses, an exception reaching your
`except` block does not tell you the work failed. It tells you *something* on
the way back raised. Those are different claims, and I now log the tool result
separately from the invocation outcome so I can tell them apart.

### Q21. What would you do differently if you rebuilt it today?

Five, roughly in order of what I would regret not doing:

1. **Start with LangGraph's Postgres checkpointer.** I rebuilt session
   persistence by hand and got a worse version — text only, no tool calls, no
   resume, no time travel. That is the single biggest "I should have read the
   docs for one more hour" decision in the project.

2. **One graph, two sub-graphs.** Keep the safety property by not binding write
   tools on the in-house branch, rather than by maintaining a parallel codebase.
   One persistence story, one trace, one place to add a node. And delete
   `orchestrator.py` rather than letting a third path linger.

3. **Finish the outbox properly.** Producer inside the order transaction,
   `attempts` + backoff on the table, `FOR UPDATE SKIP LOCKED` on the drain, and
   the worker in its own process. It is half-built today and half-built
   infrastructure is worse than none, because it looks done.

4. **Tests from the first write path, not after.** See Q22. This is the real
   answer; the other four are refinements.

5. **Design for two replicas on day one.** Telegram webhook instead of polling,
   no module-level mutable state, shared locks. Retrofitting that is much harder
   than starting with it.

### Q22. How did you test any of this?

I have to be blunt here, because a two-second look at the repo shows it.

**There are no tests.** `tests/unit`, `tests/smoke` and `tests/evals` exist and
contain nothing but `.gitkeep`. There is no pytest config, and there is no CI
workflow either — `.github/workflows` does not exist, despite a commit message
claiming CI/CD scaffolding. Zero coverage.

What I actually did instead — not a substitute, but it is the truth:

- Manual end-to-end runs through Telegram against a real Postgres, for each
  flow: browse, cart, details, confirm, pay, track.
- Heavy structured logging at every stage boundary, with timing —
  `Pipeline[chat=..] plan — goal=.. missing=.. actions=..` and a duration per
  stage. That is how I debugged, and it is why the logs are as verbose as they
  are.
- Deterministic guards doing work unit tests would normally do: `_guard_order_plan`
  strips unsafe actions, the executor re-reads prices, the verifier recomputes
  totals. The checks exist in production code and are exercised on every order —
  they are just not asserted anywhere.

**What I would write first, in order** — and having a specific list is the only
thing that makes this answer recoverable:

1. `Verifier._verify_order` with a seeded price mismatch. Pure function over the
   DB, highest value per line, catches the money bug.
2. `_guard_order_plan` as a table test — every combination of missing slots,
   assert `place_order` is stripped. No LLM, no DB, fast.
3. `StateMachine.next_from_tool` across all 18 events, including the branching
   ones (`place_order` with and without a payment intent).
4. An executor integration test on a throwaway Postgres: place an order with a
   cart whose prices were tampered with in memory, assert the DB total matches
   the menu.
5. Only then, an eval set of recorded conversations for the planner — because
   that one is probabilistic and needs a different kind of harness.

If they ask why none of it exists: I prioritised making the system correct
structurally over proving it, and structure without proof is a half-finished
argument. I would not make that trade again on the write path.

---

## Corrections — where my prepared script disagreed with the code

Print this. It is the highest-value page in the document.

| # | What I was going to say | What the code does | Say instead |
|---|---|---|---|
| Q8 | "Verify routes back to Plan, bounded by a retry count" | `remaining_actions = []` on the first pass — the loop runs **once**; failure `break`s | "One-shot with a strong pre-execution guard. The retry loop is vestigial and I would either implement it or delete it." |
| Q14 | "Order row and event in one transaction" | Pipeline `place_order` writes **no** outbox event; legacy path writes one in a **separate** session | "Table and worker are built, producer side is not wired, and the one producer is not transactional. Here is the exact fix." |
| Q15 | "Retry counts and backoff" | No `attempts` column, no backoff, no `SKIP LOCKED` | "Retries forever on failure, single-worker-only. Three named fixes." |
| Q16 | "Per-platform locks stop two user messages interleaving" | The locks guard **MCP client construction**, not message ordering; there is **no** per-user lock | "They stop duplicate subprocess spawns. Per-user serialisation is a real gap I have not closed." |
| Q10 | "Here is my state schema" | `create_react_agent` → prebuilt `MessagesState`; no custom schema | "No custom LangGraph schema. My real state is `ContextSnapshot` and `BotState` — here they are." |
| Q12 | "Step counter, forced route to a final node" | `AGENT_MAX_ITERATIONS=6` caps *planned actions*; ReAct uses the **default** recursion limit of 25; the real bound is a 90s timeout | "Caps before execution plus a timeout. The recursion limit is a framework default, not my design." |
| Q18 | "Three-model chain, retryable vs non-retryable" | Chain exists only on the MCP path; `with_fallbacks()` retries on **any** exception; pipeline uses one model | "Real chain on one path only, and it does not distinguish error types yet — I have the classifier, I just do not use it for that." |
| Q22 | "Lighter testing, mostly integration on the write path" | **Zero** tests. Directories hold `.gitkeep`. No CI. | "None. Here is what I did instead, and here are the five tests I would write in order." |

The pattern: in every case the truth is a better interview answer than the
script, because the truth comes with a specific fix and a line number. Fluency
about code I did not write dies on the second follow-up.

---

## Extra questions they are likely to ask

These were not in my list and all of them have an obvious hook in the code.

### E1. Your bot and your API run in the same process. Why?

Deliberate, and the comment says so (`api/main.py:69-72`): so the Telegram
handlers and the FastAPI handlers share one event loop, because asyncpg
connections cannot be used across loops. I hit that bug — it is also why
`session.py` keeps a per-loop engine cache.

The cost is that the bot cannot be scaled or deployed independently of the API,
and one crash takes both down. The correct shape is Telegram webhook → a FastAPI
route → a queue, with workers consuming. Then the bot is just another HTTP
endpoint and the loop problem disappears.

### E2. How do you stop duplicate orders?

Partly. `Order.idempotency_key` is unique at the DB level
(`db/models/cart_orders.py:67`) and is set to `f"{phone}-{order_ref}"`
(`executor.py:385`). But `order_ref` is a fresh random `uuid4` hex on every
call, so the key is unique every time — it guarantees uniqueness, which is not
the same as guaranteeing idempotency. Two identical taps produce two different
keys and two orders.

To make it real, the key has to derive from the *request*: customer + cart
contents + a time bucket. Then a repeat within the window collides and the
insert is rejected. I would say this plainly — half-built idempotency is a good
thing to be able to spot in your own code.

### E3. The user writes in Telugu. What happens?

`ContextSnapshot` carries both `language` and `user_language`, and
`language_persistence.py` keeps the choice across sessions — so it survives a
`/reset`, which was the bug that made me build it. `update_customer` accepts a
`language` argument, so the planner can set it when it notices a switch.

### E4. Walk me through a payment.

`place_order` commits the order, then auto-chains into
`_tool_create_payment_intent` for DELIVERY and PICKUP when Razorpay keys are
configured (`executor.py:430-442`). If they are not configured, it logs and
treats the order as cash on delivery — a deliberate degrade, not a failure.
`create_payment_intent` puts the state machine into `AWAITING_PAYMENT`;
`check_payment_status` returning CAPTURED or AUTHORIZED moves it to
`ORDER_PLACED` and clears `pending_payment_order_id`.

The weak point I would name before they do: the auto-chain happens *after* the
order transaction commits. If intent creation fails, the order exists with no
payment attached, and recovery depends on someone calling
`check_payment_status` later.

### E5. What does the staff dashboard actually get?

`outbox_worker` broadcasts to `ws_manager`, which fans out per `restaurant_id`
to connected clients (`ws_manager.py:41-52`). The Next.js app has four views:
`kitchen`, `waiter`, `admin`, `track`. Events carry `event_id` so a client
could de-duplicate — nothing does yet, which matters given at-least-once
delivery.

Note the multi-tenant seam: every model carries `restaurant_id` via
`RestaurantMixin` and broadcasts are scoped by it. The system was shaped for
multiple restaurants even though it runs one today.

### E6. Why is the planner's OpenAI call synchronous?

It should not be. `planner.py:233` and `responder.py:62` both use the sync
client inside `run_in_executor(None, ...)`. It works, but it consumes a shared
thread pool slot for the whole call. `AsyncOpenAI` would remove that entirely.
I would call this a straightforward mistake rather than defend it — see Q19.

### E7. How does the model avoid inventing a restaurant ID?

Two layers. The prompt forbids it loudly (`RULE 2` and `RULE 3`,
`mcp_agent.py:338-350`). The prompt alone is not enough, so there is a
structural layer too: after each turn I parse real `address_id` and `res_id`
values out of the tool results and store them as a hidden `mcp_context` turn
(`_build_context_from_messages`, `mcp_agent.py:190-224`). On the next turn, if
the user says "the second one" or names a restaurant,
`_normalize_followup_selection` rewrites their message to include the exact ID
before the model ever sees it (`mcp_agent.py:166-188`).

This is a nice one to volunteer, because it is the same philosophy as Q5 applied
where I *cannot* remove the model's power — I could not take the tools away
there, so instead I removed the need to guess.

### E8. What is the single riskiest thing in this repo right now?

No tests on the write path, and I would say that before they say it. Second is
the half-wired outbox, because it looks finished and is not. Third is the
per-user race from Q16.

### E9. How do you know the bot is behaving in production?

Structured logs per stage with durations, and the MCP path logs every tool call,
its arguments and its result. That is enough to debug a specific conversation
after the fact. What is missing: no traces, no metrics, no alerting, no eval
suite. If I had one thing it would be LangSmith on both paths, since it is
almost free to add and would have paid for itself during the `AbortError` hunt.

### E10. If you had one week on this, what would you do?

Day 1-2: the five tests from Q22, plus CI that runs them.
Day 3: finish the outbox — producer in-transaction, `attempts`, `SKIP LOCKED`.
Day 4: `AsyncOpenAI` everywhere, kill the thread pool problem.
Day 5: per-chat advisory lock and the real idempotency key.

Not the checkpointer migration — that is valuable but it is a refactor, and a
week is better spent making what exists trustworthy.

---

## 20 minutes before the call

Read these, in this order, and say the field names out loud:

1. `backend/app/agent/state_machine.py` — all 11 states, 18 events. Short file, read it fully.
2. `backend/app/agent/pipeline.py:100-150` — the loop that runs once. Know why.
3. `backend/app/agent/context_builder.py:50-105` — the `ContextSnapshot` fields.
4. `backend/app/agent/verifier.py:189-250` — the price recompute. This is the best thing in the repo.
5. `backend/app/agent/executor.py:339-410` — the write path. Note where the prices are re-read.
6. `backend/app/agent/mcp_agent.py:267-290` and `:482` — the fallback chain and `create_react_agent`.
7. `backend/app/workers/outbox_worker.py` — 71 lines, read all of it.
8. `backend/app/core/config.py:44-49` — the model names and the iteration cap.

---

## Two things to hold on to

**They will find the edge. That is what the follow-ups are for.** Q4 → Q5 → Q6
is not three questions, it is one question asked three times with the pressure
turned up. The pass condition is not knowing everything. It is saying "I have
not done that — here is what I would look at" and then stopping talking.
Candidates fail on question three of a chain by bluffing, not by admitting a
gap on question one.

**My strongest moment is Q5, and my most dangerous moments are Q14 and Q22.**
Q5 is where I have a real architectural conviction backed by real code, and I
should not soften it. Q14 and Q22 are where my script described a system nicer
than the one I built — and both are checkable in under a minute by anyone with
the repo open. Leading with the honest version turns both into evidence that I
know my own system well, which is the actual thing being tested.
