# trader

An autonomous trading engine for NSE equity/F&O (via Dhan) and OKX spot crypto.

It is built around one idea: **when something breaks, the break must be
visible.** The system this replaces failed quietly — a failed equity read
returned `0` and was read as a 100% drawdown, a failed risk-state read returned
zeros and was read as "no drawdown" (which disarmed the circuit), and a failed
bot-list read returned `[]` and was read as "no bots" while ten were running,
one of them live. Every one of those looked fine on a dashboard.

So the rule everywhere in this codebase is: **a failure must never be
representable as a legitimate value.** When something cannot be established the
code returns `None` or raises. It never returns a number that looks plausible.

---

## Architecture

Dependency direction is strictly one way. Nothing below the gate calls anything
above it.

```
  Strategy ──▶ Intent ──▶ OrderGate ──▶ RiskCircuit
   (pure)                     │
                              ▼
                        BrokerAdapter ──▶ Fill ──▶ PositionStore
```

| File | What it is | The one thing it guarantees |
|---|---|---|
| `core/types.py` | The vocabulary: `Intent`, `Approval`, `OrderRef`, `Fill`, `Position`, `Mode` | **Frozen.** `OrderRef.status` is always terminal — there is no fire-and-forget |
| `core/protocols.py` | The seams: `Strategy`, `BrokerAdapter`, `PositionStoreProto`, … | **Frozen.** `equity()` returns `None`, never `0`, when it cannot be established |
| `core/profiles.py` | Everything that differs between asset classes | **Frozen.** If you write `if asset_class == EQUITY` anywhere else, it belongs here |
| `core/gate.py` | `OrderGate` — the single chokepoint | No order reaches a broker without an `Approval`. Sealed after construction |
| `core/risk.py` | `RiskCircuit` — drawdown and daily-loss ladder | Fails **closed**: unreadable equity or state refuses new entries |
| `core/store.py` | SQLite ledger + `SQLiteGateStateStore` | Every read filters on `mode`. Paper rows can never reach a live decision |
| `strategies/` | Pure signal logic | No I/O, no clock, no broker, no sizing. `propose(ctx) -> [Intent]` |
| `adapters/` | `PaperAdapter`, `DhanAdapter`, `OkxAdapter` | `place()` blocks until the order is terminal |
| `engine.py` | The asyncio supervisor | Every failure ends in an ERROR event, a disabled strategy, or a halted class |
| `api/server.py` | Read-only HTTP + two operator levers | Places no orders. Auth is structural middleware, not a per-route decorator |
| `runner.py` | **The composition root** | The only file that decides which real object goes where |

### Five rules that are load-bearing

1. **Paper vs live is a property of the adapter, never a flag a strategy
   reads.** Strategies are written once and cannot tell the difference.
2. **`Approval.mode` is authoritative.** The gate *coerces* live to paper when
   the asset class is not armed, so a stale live-flagged object cannot place a
   real order on its own. The engine then refuses any approval whose mode is
   not the mode that tick was measured in.
3. **A failed read is never an empty read.** An unreadable position store skips
   the tick (reading "flat" would re-enter positions we already hold). An
   unfetchable quote is *absent* from `ctx.quotes`, never zero.
4. **A strategy that raises is disabled for the session** — logged with its
   traceback and reported forever after. If it held positions when it died,
   those positions are now unmanaged and get their own repeating alarm.
5. **A ledger write that fails halts the class.** If a real fill cannot be
   recorded, every downstream cap is being computed against a book we know is
   wrong. This is deliberately *not* a flatten: liquidating on numbers we just
   admitted we cannot write is its own disaster.

---

## Running it locally in paper mode

Requires Python 3.12+. **Use `python3` — this project has no `python` binary.**

```bash
cd apps/trader
python3 -m pip install -r requirements.txt

export TRADER_DB=./var/trader.db
export TRADER_CLASSES=crypto            # or: crypto,equity
export ENGINE_API_TOKEN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"

python3 -m runner
```

That starts the engine **and** the API in one process on
`http://127.0.0.1:8080`. One process is deliberate: `api/server.py`'s event bus
and strategy registry are in-memory, so an API in a *separate* process would
serve an almost-empty `/api/events` and a 503 from `/api/strategies`. The
durable surfaces (positions, orders, P&L, risk and control state) go through
SQLite and are correct either way.

Engine only, no HTTP:

```bash
python3 -m runner --no-api
```

### Looking at it

```bash
TOKEN=$ENGINE_API_TOKEN
curl -s localhost:8080/health                                     # no auth
curl -s -H "Authorization: Bearer $TOKEN" localhost:8080/ready
curl -s -H "Authorization: Bearer $TOKEN" localhost:8080/api/state
curl -s -H "Authorization: Bearer $TOKEN" localhost:8080/api/positions
curl -s -H "Authorization: Bearer $TOKEN" localhost:8080/api/strategies
curl -s -N -H "Authorization: Bearer $TOKEN" localhost:8080/api/events   # SSE
```

Every path except `/health` requires the bearer token. The app **refuses to
start** without `ENGINE_API_TOKEN` and refuses a CORS config containing `*`.
There is no development mode that grants access — a fallback that only triggers
in dev is a fallback that eventually ships.

### The two operator levers

Both only ever *subtract*. Nothing an operator can POST widens what the engine
may do, and `_ControlBody` forbids extra fields so a body carrying
`{"mode":"live"}` is a 422 rather than a silently ignored field.

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"reason":"fills look wrong","actor":"me"}' localhost:8080/api/control/pause

curl -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"reason":"closing for the day","actor":"me"}' localhost:8080/api/control/flatten
```

`pause` stops new positions being opened and closes nothing. Exits keep working
— refusing to let a strategy close a position because you paused would trap the
book in exactly the state you reached for the button to get out of.

`flatten` **records a request**; the API holds no gate and no order path. The
engine picks it up on its next tick, routes the exits through the gate, and only
then writes `flatten_acked_id` back. Until that lands, the response says
`"executed": false` and `/api/state` shows the request as pending. **A 200 OK
from `/api/control/flatten` does not mean the book is closed.**

### Local market data is fake, and says so

`runner.SyntheticBarFeed` generates a deterministic triangle wave so the stack
can be run end to end without credentials, a network, or a data vendor. It is
**not market data**. It refuses to serve any class the engine resolved to live —
`bind()` raises — so it cannot follow you into production. Replace it with a
real feed before arming anything.

---

## The two arming switches

```
EQUITY_LIVE=1      arms NSE equity / F&O
CRYPTO_LIVE=1      arms OKX spot
```

**Setting these is the owner's action and nobody else's.** Not this README's,
not the runner's, not an agent's, not a deploy script's. No code in this package
sets them, defaults them, or infers them; `core/gate.py` and `engine.py` read
them from the environment and nothing else writes them.

What they actually do:

* **Arming is per asset class**, so a crypto experiment can never arm the equity
  book by accident.
* Live requires **both** `TRADER_MODE=live` *and* the class's flag. `TRADER_MODE`
  alone is a *request*; with no flag set, every class resolves to paper and the
  boot report says so line by line.
* An unrecognised value **raises**. `EQUITY_LIVE=ture` does not quietly become
  `False` — silently disarming on a typo looks identical to a correctly
  disarmed engine, and the operator would never learn which one they had.
* The flags are read **once, at construction**. A gate whose arming can change
  under it mid-session cannot be reasoned about from a log line.
* Constructing a live adapter is **not** arming. `runner.py` builds none at all;
  wiring one is a separate, deliberate edit.

Before you set either one, three things must be true:

1. **Replace `SyntheticBarFeed` with a real market data feed.** It will refuse
   to start otherwise, which is the point.
2. **Wire a `live_adapter` into the `ClassWiring`** for that class. The engine
   refuses to start if a class is armed, live was requested, and no live adapter
   is wired — rather than trading a live-approved order on paper.
3. **Read `adapters/dhan.py` on F&O execution.** There is no bid/ask in the
   quote path; a LIMIT order needs fill confirmation or a short leg can end up
   naked.

Check what you have armed without guessing — the boot report prints one line per
class, and `/api/state` shows `armed_live` per class:

```bash
python3 -c "
import runner
s = runner.build_stack()
print(s.engine.arm_report_text())
"
```

---

## Tests

```bash
cd apps/trader
python3 -m pytest tests/ -q
```

No async plugin is required — the suite drives the engine's event loop with
`asyncio.run` directly.

The suite covers failure paths, not just happy ones. For every safety rule there
is a test that fails if the rule is removed; several were verified by mutation
(breaking the implementation on purpose and confirming the test goes red).

---

## Known sharp edges

These are real and unresolved. They are written down rather than smoothed over.

**The gate CAPS, it does not SIZE.** `strategies/base.py` says "no sizing — the
gate sizes and caps", but `core/gate.py` only ever *rejects* an order that
breaches `max_order_notional_fraction` (10% of equity by default); the only path
that scales a quantity down is the risk circuit's `size_multiplier`. So a crypto
strategy proposing one unit of the base asset needs equity of roughly ten times
that asset's price or every entry is refused with `notional_cap`. The paper
float in `RunnerConfig` is sized for this. Symptom if you get it wrong: an
engine that ticks forever, logs cleanly, and never trades.

**A transient data gap can permanently disable a strategy.** When a strategy
holds a position and its bars go missing or stale, it raises `MissingDataError`
rather than managing the position blind — and `engine.py` disables any strategy
that raises for the rest of the session, on the reasoning that "a strategy that
throws on one tick will throw on the next". That reasoning does not hold for a
feed hiccup. The behaviour is the safe direction (loud, disabled, positions
alarmed as unmanaged rather than silently unmanaged) but it means a five-minute
feed outage ends a strategy's day and needs an operator to restart it.

**`ctx.extra["bars"]` is mandatory.** A missing bar book is treated as a wiring
failure, not an empty market, so if no `extra_provider` is wired *every*
strategy is disabled on tick 1. `runner.py` always wires one.

**The gate holds one risk circuit; circuits are bound to one mode.** With one
class armed and another not, no single circuit is correct for both.
`runner._PerClassCircuit` routes each read to the circuit for that class.
`advance()` raises there — the circuit's only writer belongs to the order path,
and the engine already calls it per class.

**Gate claims are never pruned.** `SQLiteGateStateStore.record_submission` is a
plain `INSERT`, so re-claiming an `idempotency_key` raises and the gate refuses.
That is intentional — an expired dedup window does not un-approve an order that
was already approved once — but it means a key must genuinely rotate per
decision. Strategy keys carry the deciding bar's timestamp and flatten keys
carry a time bucket, so they do.

**The `gate_submissions` table is mode-scoped, not effective-mode-scoped.** With
`TRADER_MODE=live` and only one class armed, the unarmed class's coerced-to-paper
orders are counted against the live day's budget. That over-counts, which
refuses earlier; the alternative — an uncounted order — is the one that cannot
be allowed.
