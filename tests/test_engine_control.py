"""The engine's half of the operator control contract.

`api/server.py` promises an operator three things, and until the engine reads
the control store none of them was true:

  * pause STOPS new positions being opened,
  * `/api/control/flatten` "records a request the engine picks up on its next
    tick and executes through the gate",
  * `flatten_acked_id` is "written by the ENGINE when it has taken the request",
    and until then the request is a wish.

Every test here fails if the corresponding rule is removed from engine.py. The
one that matters most is `test_an_unreadable_control_row_closes_entries`: an
operator who pressed pause was told 200 OK, so "I could not check" must never
be treated as "nobody asked me to stop".
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional

import pytest

import engine as eng
from adapters.paper import InMemoryQuoteSource, PaperAdapter
from core.gate import GateConfig, OrderGate
from core.risk import RiskCircuit
from core.store import SQLiteGateStateStore, SQLitePositionStore
from core.types import (
    AssetClass, Fill, Instrument, Intent, MarketContext, Mode, OrderType, Side,
)

NOW = dt.datetime(2026, 3, 2, 12, 0, tzinfo=dt.timezone.utc)

BTC = Instrument("BTC/USDT", AssetClass.CRYPTO, "BTC/USDT", 1, Decimal("0.1"),
                 qty_precision=6, min_notional=Decimal("5"))


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------

class MemControl:
    """An in-memory ControlStoreProto. Fine HERE — these tests are one process
    and the point is the engine's behaviour, not the store's durability."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        self.read_error: Optional[Exception] = None
        self.write_error: Optional[Exception] = None
        self.reads = 0

    def read(self, scope: str) -> Optional[Mapping[str, str]]:
        self.reads += 1
        if self.read_error is not None:
            raise self.read_error
        row = self.rows.get(scope)
        return dict(row) if row is not None else None

    def mutate(self, scope: str,
               fn: Callable[[Optional[Mapping[str, str]]], Mapping[str, str]],
               ) -> Mapping[str, str]:
        if self.write_error is not None:
            raise self.write_error
        current = self.rows.get(scope)
        new = dict(fn(dict(current) if current is not None else None))
        self.rows[scope] = new
        return new

    # -- helpers the tests read like an operator would --------------------
    def pause(self, scope: str, reason: str = "operator says so") -> None:
        row = self.rows.setdefault(scope, {"paused": "0"})
        row["paused"] = "1"
        row["paused_at"] = NOW.isoformat()
        row["paused_reason"] = reason

    def request_flatten(self, scope: str, request_id: str = "req-1") -> None:
        row = self.rows.setdefault(scope, {"paused": "0"})
        row["paused"] = "1"
        row["paused_at"] = NOW.isoformat()
        row["flatten_request_id"] = request_id
        row["flatten_requested_at"] = NOW.isoformat()
        row["flatten_reason"] = "close it"
        row.pop("flatten_acked_id", None)
        row.pop("flatten_acked_at", None)


class AlwaysBuys:
    """Proposes one BUY every tick. An ENTRY, so anything that closes entries
    must suppress it."""
    id = "buyer"
    asset_class = AssetClass.CRYPTO

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, ctx: MarketContext):
        self.calls += 1
        return [Intent(instrument=BTC, side=Side.BUY, qty=Decimal("1"),
                       order_type=OrderType.MARKET, strategy_id=self.id,
                       reason="always buys",
                       idempotency_key=f"buy-{self.calls}")]


class AlwaysSells:
    """Proposes one SELL every tick. With a long position open this REDUCES,
    so it must survive a pause: refusing to let a strategy close a position
    because the operator paused traps the book."""
    id = "seller"
    asset_class = AssetClass.CRYPTO

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, ctx: MarketContext):
        self.calls += 1
        return [Intent(instrument=BTC, side=Side.SELL, qty=Decimal("1"),
                       order_type=OrderType.MARKET, strategy_id=self.id,
                       reason="always sells",
                       idempotency_key=f"sell-{self.calls}")]


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

class Harness:
    def __init__(self, tmp_path, strategies, control: Optional[MemControl]) -> None:
        path = str(tmp_path / "t.db")
        self.store = SQLitePositionStore(path)
        self.gate_state = SQLiteGateStateStore(path, Mode.PAPER)
        self.circuit = RiskCircuit(self.store, Mode.PAPER, clock=lambda: NOW)
        self.gate = OrderGate(position_store=self.store,
                              gate_state=self.gate_state,
                              risk_circuit=self.circuit,
                              config=GateConfig(), env={})
        self.quotes = InMemoryQuoteSource({"BTC/USDT": Decimal("60000")})
        self.adapter = PaperAdapter(AssetClass.CRYPTO, self.quotes,
                                    Decimal("5000000"), seed=1,
                                    now=lambda: NOW, sleep=lambda s: None)
        self.control = control
        wiring = eng.ClassWiring(
            asset_class=AssetClass.CRYPTO, paper_adapter=self.adapter,
            risk=self.circuit, strategies=tuple(strategies),
            instruments=(BTC,), extra_provider=lambda ac, now: {})
        self.engine = eng.Engine(
            wirings=[wiring], store=self.store, gate=self.gate,
            config=eng.EngineConfig(close_store_on_stop=False),
            clock=lambda: NOW, env={}, control=control)

    def tick(self) -> None:
        """Drive one tick. `asyncio.run` per tick matches tests/test_engine.py:
        this package has no async pytest plugin and does not need one."""
        asyncio.run(self.engine.tick_once(AssetClass.CRYPTO))

    def events(self, kind: eng.EventKind) -> list[eng.Event]:
        return [e for e in self.engine.events.recent() if e.kind is kind]

    def messages(self, kind: eng.EventKind) -> list[str]:
        return [e.message for e in self.events(kind)]

    def open_long(self) -> None:
        """Put a real long in the ledger so exits and flattens have something
        to act on."""
        self.store.apply(
            Fill(order_id="seed", instrument=BTC, side=Side.BUY,
                 qty=Decimal("1"), price=Decimal("60000"), fees=Decimal("0"),
                 at=NOW, mode=Mode.PAPER),
            strategy_id="seller")

    def close(self) -> None:
        self.store.close()
        self.gate_state.close()


@pytest.fixture()
def harness(tmp_path):
    made: list[Harness] = []

    def build(strategies=(), control: Optional[MemControl] = None) -> Harness:
        h = Harness(tmp_path, strategies, control)
        made.append(h)
        return h

    yield build
    for h in made:
        h.close()


def orders_placed(h: Harness) -> int:
    return len(list(h.store.orders(Mode.PAPER)))


# ==========================================================================
# pause
# ==========================================================================

def test_a_paused_class_places_no_new_entries(harness) -> None:
    control = MemControl()
    h = harness([AlwaysBuys()], control)

    h.tick()
    assert orders_placed(h) == 1, "control: unpaused, the entry goes through"

    control.pause("crypto", "hands off")
    h.tick()
    assert orders_placed(h) == 1, "the paused tick must not have placed anything"
    assert any("paused by operator" in e.data.get("reason", "")
               for e in h.events(eng.EventKind.REJECTION))


def test_the_pause_reason_reaches_the_event_stream(harness) -> None:
    """An operator reading the log must be able to see WHY entries stopped, or
    a pause is indistinguishable from a strategy that went quiet."""
    control = MemControl()
    control.pause("crypto", "suspicious fills")
    h = harness([AlwaysBuys()], control)
    h.tick()
    reasons = [e.data.get("reason") for e in h.events(eng.EventKind.REJECTION)]
    assert any("suspicious fills" in str(r) for r in reasons)


def test_a_pause_does_not_stop_a_strategy_closing_a_position(harness) -> None:
    """Pause only ever subtracts. Blocking exits would trap the book in exactly
    the state the operator reached for the button to get out of."""
    control = MemControl()
    h = harness([AlwaysSells()], control)
    h.open_long()
    control.pause("crypto")

    h.tick()
    assert orders_placed(h) == 1, "the reducing SELL must still be placed"


def test_resuming_lets_entries_through_again(harness) -> None:
    control = MemControl()
    control.pause("crypto")
    h = harness([AlwaysBuys()], control)
    h.tick()
    assert orders_placed(h) == 0

    control.rows["crypto"]["paused"] = "0"
    h.tick()
    assert orders_placed(h) == 1


def test_a_pause_is_announced_once_not_every_tick(harness) -> None:
    """A pause is a state. Re-announcing it every 5s pushes the events that
    explain the incident out of the ring buffer."""
    control = MemControl()
    h = harness([AlwaysBuys()], control)
    h.tick()
    control.pause("crypto")
    for _ in range(3):
        h.tick()
    paused = [m for m in h.messages(eng.EventKind.CONTROL) if "PAUSED" in m]
    assert len(paused) == 1


def test_the_opening_observation_does_not_invent_an_operator_action(harness) -> None:
    """The first read of an unpaused class is a STATE, not a transition. Saying
    "resumed by operator" would put an action nobody took into the audit trail."""
    h = harness([AlwaysBuys()], MemControl())
    h.tick()
    assert not any("resumed" in m for m in h.messages(eng.EventKind.CONTROL))


# ==========================================================================
# an unreadable control row fails CLOSED
# ==========================================================================

def test_an_unreadable_control_row_closes_entries(harness) -> None:
    """The rule this whole file exists for.

    A read that FAILED is not an unpaused engine. The operator was told 200 OK
    when they pressed pause; treating "I could not check" as "nobody asked me
    to stop" is how a paused engine keeps trading.
    """
    control = MemControl()
    h = harness([AlwaysBuys()], control)
    control.read_error = RuntimeError("disk gone")

    h.tick()
    assert orders_placed(h) == 0
    assert any("UNKNOWN" in m for m in h.messages(eng.EventKind.CONTROL))


def test_an_unreadable_control_row_still_allows_exits(harness) -> None:
    control = MemControl()
    h = harness([AlwaysSells()], control)
    h.open_long()
    control.read_error = RuntimeError("disk gone")

    h.tick()
    assert orders_placed(h) == 1, "an exit must survive an unreadable control row"


def test_a_corrupt_paused_flag_is_not_read_as_unpaused(harness) -> None:
    """`paused="maybe"` read as False un-pauses a paused engine."""
    control = MemControl()
    control.rows["crypto"] = {"paused": "maybe"}
    h = harness([AlwaysBuys()], control)

    h.tick()
    assert orders_placed(h) == 0
    assert any("corrupt" in str(e.data.get("detail", ""))
               for e in h.events(eng.EventKind.CONTROL))


def test_a_pause_with_no_timestamp_is_treated_as_corrupt(harness) -> None:
    control = MemControl()
    control.rows["crypto"] = {"paused": "1"}          # no paused_at
    h = harness([AlwaysBuys()], control)
    h.tick()
    assert orders_placed(h) == 0


def test_a_row_that_was_never_written_is_a_real_answer(harness) -> None:
    """Distinct from unreadable: nobody has ever paused this class, so entries
    are open. Collapsing the two would stop an engine nobody paused."""
    h = harness([AlwaysBuys()], MemControl())
    h.tick()
    assert orders_placed(h) == 1


def test_one_class_pause_does_not_leak_into_another(harness, tmp_path) -> None:
    control = MemControl()
    control.pause("equity")
    h = harness([AlwaysBuys()], control)
    h.tick()
    assert orders_placed(h) == 1, "crypto is not paused by an equity pause"


# ==========================================================================
# flatten
# ==========================================================================

def test_a_requested_flatten_closes_the_book(harness) -> None:
    control = MemControl()
    h = harness([], control)
    h.open_long()
    control.request_flatten("crypto")

    h.tick()
    refs = list(h.store.orders(Mode.PAPER))
    assert len(refs) == 1
    assert refs[0].intent.side is Side.SELL
    assert refs[0].intent.qty == Decimal("1")
    assert "operator" in refs[0].intent.reason.lower()


def test_the_engine_acknowledges_the_request_it_carried_out(harness) -> None:
    control = MemControl()
    h = harness([], control)
    h.open_long()
    control.request_flatten("crypto", "req-abc")

    h.tick()
    assert control.rows["crypto"]["flatten_acked_id"] == "req-abc"
    assert control.rows["crypto"]["flatten_acked_at"]


def test_an_acknowledged_flatten_is_not_repeated(harness) -> None:
    """Once acked, the request is spent. Re-flattening every tick would keep
    firing exits at a book that is already closed."""
    control = MemControl()
    h = harness([], control)
    h.open_long()
    control.request_flatten("crypto")

    h.tick()
    first = orders_placed(h)
    h.tick()
    assert orders_placed(h) == first


def test_a_new_request_after_an_ack_is_carried_out_again(harness) -> None:
    control = MemControl()
    h = harness([], control)
    h.open_long()
    control.request_flatten("crypto", "req-1")
    h.tick()
    assert control.rows["crypto"]["flatten_acked_id"] == "req-1"

    h.open_long()
    control.request_flatten("crypto", "req-2")
    h.tick()
    assert control.rows["crypto"]["flatten_acked_id"] == "req-2"


def test_a_flatten_request_blocks_new_entries(harness) -> None:
    control = MemControl()
    h = harness([AlwaysBuys()], control)
    control.request_flatten("crypto")

    h.tick()
    reasons = [str(e.data.get("reason")) for e in h.events(eng.EventKind.REJECTION)]
    assert any("operator flatten" in r for r in reasons)
    assert not any(o.intent.side is Side.BUY for o in h.store.orders(Mode.PAPER))


def test_a_pending_request_survives_a_restart(harness, tmp_path) -> None:
    """The engine compares against the id in the ROW, not against anything it
    remembers, so a request made while it was down is still carried out."""
    control = MemControl()
    control.request_flatten("crypto", "req-while-down")

    h = harness([], control)
    h.open_long()
    h.tick()
    assert control.rows["crypto"]["flatten_acked_id"] == "req-while-down"


def test_a_failed_acknowledgement_is_loud_and_leaves_it_pending(harness) -> None:
    """The exits were placed but the ack could not be written. The operator's
    screen will keep saying "pending", and they must be told why rather than
    concluding nothing happened."""
    control = MemControl()
    h = harness([], control)
    h.open_long()
    control.request_flatten("crypto")
    control.write_error = RuntimeError("db locked")

    h.tick()
    assert orders_placed(h) == 1, "the exit was still placed"
    assert "flatten_acked_id" not in control.rows["crypto"]
    assert any("could not be acknowledged" in e.message
               for e in h.events(eng.EventKind.ERROR))


def test_a_halted_class_does_not_claim_it_flattened(harness) -> None:
    """A halted class stops placing orders. Acknowledging anyway would tell the
    operator the book closed when nothing was sent."""
    control = MemControl()
    h = harness([], control)
    h.open_long()
    control.request_flatten("crypto")
    h.engine._runners[AssetClass.CRYPTO].halt("test halt")

    h.tick()
    assert "flatten_acked_id" not in control.rows["crypto"]


def test_operator_and_session_flattens_do_not_share_an_idempotency_key(
        harness) -> None:
    """Two exits landing in the same 60s bucket must not dedup each other away:
    the gate would silently drop the operator's request."""
    control = MemControl()
    h = harness([], control)
    h.open_long()
    ctx = MarketContext(now=NOW, quotes={"BTC/USDT": Decimal("60000")},
                        positions=tuple(h.store.positions(Mode.PAPER)),
                        equity=Decimal("5000000"))
    runner = h.engine._runners[AssetClass.CRYPTO]
    session = runner._square_off(ctx, (), eng.IntentSource.SQUARE_OFF)
    operator = runner._square_off(ctx, (), eng.IntentSource.OPERATOR)
    assert session and operator
    assert session[0][0].idempotency_key != operator[0][0].idempotency_key


# ==========================================================================
# heartbeat
# ==========================================================================

def test_the_engine_stamps_a_heartbeat(harness) -> None:
    control = MemControl()
    h = harness([], control)
    h.tick()
    assert control.rows[eng.CONTROL_ENGINE_SCOPE]["heartbeat_at"]


def test_the_heartbeat_scope_matches_what_the_api_reads(harness) -> None:
    """A mismatch here is invisible until an operator sees a healthy engine
    reported as never having started."""
    from api.server import ENGINE_SCOPE
    assert eng.CONTROL_ENGINE_SCOPE == ENGINE_SCOPE


def test_a_heartbeat_write_failure_does_not_stop_trading(harness) -> None:
    """The heartbeat describes the loop; it must not be able to kill it. The
    API reports the beat as stale, which is the truth."""
    control = MemControl()
    h = harness([AlwaysBuys()], control)
    control.write_error = RuntimeError("db locked")

    h.tick()
    assert orders_placed(h) == 1
    assert any("heartbeat" in e.message for e in h.events(eng.EventKind.ERROR))


def test_the_heartbeat_preserves_an_existing_engine_row(harness) -> None:
    control = MemControl()
    control.rows[eng.CONTROL_ENGINE_SCOPE] = {"paused": "1",
                                              "paused_at": NOW.isoformat()}
    h = harness([], control)
    h.tick()
    row = control.rows[eng.CONTROL_ENGINE_SCOPE]
    assert row["paused"] == "1", "a heartbeat must not clear an engine-wide pause"
    assert row["heartbeat_at"]


# ==========================================================================
# no control store at all
# ==========================================================================

def test_without_a_control_store_the_engine_still_trades(harness) -> None:
    """A backtest or a test harness has no operator to obey. Closing entries
    here would make the engine useless by default."""
    h = harness([AlwaysBuys()], None)
    h.tick()
    assert orders_placed(h) == 1


def test_a_missing_control_store_is_announced_at_boot(harness) -> None:
    """Silence here is how an operator ends up pressing a pause button wired to
    nothing: the API would accept the pause, write the row, and no one reads it."""
    h = harness([AlwaysBuys()], None)
    h.tick()
    assert any("no control store" in m for m in h.messages(eng.EventKind.CONTROL))


def test_a_missing_control_store_writes_no_heartbeat(harness) -> None:
    """Better UNKNOWN than a beat this engine cannot honour a pause against."""
    h = harness([], None)
    h.tick()
    assert h.engine._last_beat is None
