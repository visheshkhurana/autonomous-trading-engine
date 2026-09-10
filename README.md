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
# Optional: local Dhan scrip-master CSV. No network, no Dhan token.
# Unset = IndexOptionTrendStrategy is not registered.
# export TRADER_SCRIP_MASTER=./var/scrip_master.csv

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
curl -s -H "Authorization: Bearer $TOKEN" localhost:8080/api/orders
curl -s -H "Authorization: Bearer $TOKEN" localhost:8080/api/strategies
curl -s -N -H "Authorization: Bearer $TOKEN" localhost:8080/api/events   # SSE
```

Every path except `/health` requires the bearer token. The app **refuses to
start** without `ENGINE_API_TOKEN` and refuses a CORS config containing `*`.
There is no development mode that grants access — a fallback that only triggers
in dev is a fallback that eventually ships.

---

## Deploying

This repository is the engine. It runs anywhere Python 3.12 does; see
`Dockerfile` for the image. Deployment specifics (hosting, regions, volumes,
secrets) belong to whoever operates it and are deliberately not described here.

---

## Kraken paper smoke (one knob, still disarmed)

One env var names the unused live adapter **and** couples the public
Kraken feed. `KRAKEN_LIVE` stays **unset**. Signal path is the existing
`crypto_mandate` regime+score (same as OKX) — no new scraper and no
scrape-to-order trigger. A later paper/research source may fill
`strategies/mandate/research.py` extras (`funding`, `oi_change`, …);
that stub returns `{}` today and is not on the live path.

```bash
cd apps/trader
export TRADER_DB=./var/trader-kraken-paper.db
export TRADER_CLASSES=crypto
export TRADER_CRYPTO_VENUE=kraken          # one knob; feed becomes kraken
# Do not set KRAKEN_LIVE, CRYPTO_LIVE, or TRADER_MODE=live.
export ENGINE_API_TOKEN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"

python3 -m runner --status
# crypto_venue=kraken  kraken_live=false  feed=kraken
# adapter=PaperAdapter  live_adapter=KrakenAdapter  armed_live=false
# signal=crypto_mandate

python3 -m runner                          # engine + API on :8080
curl -s -H "Authorization: Bearer $ENGINE_API_TOKEN" localhost:8080/ready
# same four fields: crypto_venue, kraken_live, feed, classes.crypto.live_adapter
```

Equivalent explicit recipe (the venue knob already implies the feed):

```bash
TRADER_CRYPTO_VENUE=kraken TRADER_FEED=kraken   # KRAKEN_LIVE off
```

Public ticker (no keys): `curl -s 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD'`

**Today's paper caps stay 50 USDT / 250 daily / max 1 open** on the
existing 1000 USDT `crypto_mandate` sleeve. Consecutive-loss halt is the
shared mandate rule (3 losers / 2% daily), same as the OKX paper path.

Later **Kraken live sleeve A** (Vishesh-locked, document only — do **not**
arm in this PR): **1000 USDT sleeve / 100 USDT per clip / 500 per day /
max 1 open**, same consecutive-loss halt. Apply those caps only when the
owner later sets `KRAKEN_LIVE=1` *and* `TRADER_MODE=live` *and*
`CRYPTO_LIVE=1` *and* Fly secrets `KRAKEN_API_KEY`/`KRAKEN_API_SECRET`.
`CRYPTO_LIVE` still arms **OKX** when the venue is unset; naming kraken
without `KRAKEN_LIVE` refuses to boot. Do not set those flags from this
README.

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
* Constructing a live adapter is **not** arming. `runner.py` wires `OkxAdapter`
  as crypto's `live_adapter` so that when the owner later sets the flags there
  is somewhere for a live-approved order to go. It is never called unless both
  `TRADER_MODE=live` and `CRYPTO_LIVE` are set. Equity's live adapter is still
  not constructed — that would need a Dhan token, and the live Fly engine holds
  the only permitted one.

Crypto, once the owner arms it (and only then):

1. **`OkxFeed` serves the book.** A live crypto class is routed to the public
   OKX feed even if `TRADER_FEED` is left at its paper default (`synthetic`).
   `SyntheticBarFeed` still refuses to serve any live class; that refusal is
   load-bearing and must not be reopened.
2. **The live adapter is already wired.** No second code edit. Missing OKX
   trade secrets (`OKX_API_KEY`, `OKX_API_SECRET`, `OKX_PASSPHRASE`) refuse to
   boot an armed crypto class rather than start an unsigned live book.
3. **Size is capped and sized down.** The gate's crypto `ClassLimits`: **10% of
   equity per order**, **30% of equity for the class**, 5 open positions, 20
   orders/day. An oversized entry is scaled down to the cap (lot-snapped,
   ROUND_DOWN), not only refused. Live is not a reason to uncap. Unknown
   equity, unknown exposure, a scale that prints nothing, and reducing orders
   are still fail-closed.
4. **Risk is bounded.** Crypto `RiskLimits` flatten at **25% drawdown** and
   **8% daily loss** (pause at 15% / 5%, reduce at 8% / 3%). Each wired crypto
   strategy also carries an 8% hard stop. Flatten and pause refuse new entries;
   reducing exits still pass.

Do not set `CRYPTO_LIVE` or `EQUITY_LIVE` in this README, in a deploy script,
or in a test that would ship armed. Paper is the default and must stay the
default.

Check what you have armed without guessing — the boot report prints one line per
class, and `/ready` / `/api/state` show `armed_live` per class:

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

**The gate SIZES down to the cap, then refuses what cannot print.** Strategies
propose a raw unit qty (one lot, one coin). An entry whose notional exceeds
`max_order_notional_fraction` (10% of equity by default) is scaled down through
the profile (ROUND_DOWN) rather than only rejected. A scale that snaps to zero
lots, or under the min notional, is still a refusal. Reducing orders are never
resized — shrinking an exit leaves a tail on. A structure short that would
outrun its (possibly sized) hedge is still refused, not silently shrunk.
Unknown equity still refuses; paper vs live remains an adapter choice.

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
