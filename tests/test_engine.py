"""
Tests for engine.py — the supervisor loop.

The bar for this file: for every safety rule engine.py claims, there is a test
here that FAILS if the rule is deleted. The rules that matter most are the ones
about failures being distinguishable from empty results, so most of what
follows is unhappy paths.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Callable, Optional, Sequence

import pytest

from adapters.paper import InMemoryQuoteSource, PaperAdapter
from core.gate import ClassLimits, GateConfig, GateStateUnavailable, OrderGate
from core.profiles import IST, profile_for
from core.risk import CircuitAction, RiskCircuit
from core.store import SQLitePositionStore
from core.types import (
    Approval, AssetClass, Fill, Instrument, Intent, MarketContext, Mode,
    OrderRef, OrderStatus, OrderType, Side,
)
from engine import (
    ClassWiring, Engine, EngineConfig, EngineConfigError, Event, EventBus,
    EventKind, IntentSource, _marketable_limit, _reduces, _session_close,
)
from strategies import CryptoEmaTrendStrategy

UTC = dt.timezone.utc

BTC = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO,
                 broker_id="BTC/USDT", lot_size=1, tick_size=Decimal("0.01"),
                 qty_precision=3, min_notional=Decimal("5"))
ETH = Instrument(symbol="ETH/USDT", asset_class=AssetClass.CRYPTO,
                 broker_id="ETH/USDT", lot_size=1, tick_size=Decimal("0.01"),
                 qty_precision=3, min_notional=Decimal("5"))
NIFTY = Instrument(symbol="NIFTY-EQ", asset_class=AssetClass.EQUITY,
                   broker_id="26000", lot_size=50, tick_size=Decimal("0.05"),
                   qty_precision=0)
BANK = Instrument(symbol="BANKNIFTY-EQ", asset_class=AssetClass.EQUITY,
                  broker_id="26009", lot_size=25, tick_size=Decimal("0.05"),
                  qty_precision=0)

CRYPTO_NOON = dt.datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
# 2026-08-27 is a Thursday; NSE trades 09:15-15:30 IST.
EQ_MIDDAY = dt.datetime(2026, 8, 27, 12, 0, tzinfo=IST)
EQ_NEAR_CLOSE = dt.datetime(2026, 8, 27, 15, 20, tzinfo=IST)
EQ_WEEKEND = dt.datetime(2026, 8, 29, 12, 0, tzinfo=IST)


# ---------------------------------------------------------------- fakes


class FakeClock:
    def __init__(self, at: dt.datetime) -> None:
        self.at = at

    def __call__(self) -> dt.datetime:
        return self.at

    def advance(self, seconds: float) -> None:
        self.at = self.at + dt.timedelta(seconds=seconds)


class FakeGateState:
    """Satisfies GateStateStoreProto. TESTS ONLY — a dict here is exactly the
    per-worker-counters failure core/gate.py refuses to ship."""

    def __init__(self) -> None:
        self.seen: dict[str, dt.datetime] = {}
        self.counts: dict[tuple[AssetClass, str], int] = {}
        self.fail = False

    def seen_recently(self, idempotency_key: str, window_s: float,
                      now: dt.datetime) -> bool:
        if self.fail:
            raise GateStateUnavailable("test")
        ts = self.seen.get(idempotency_key)
        return ts is not None and (now - ts).total_seconds() < window_s

    def order_count(self, asset_class: AssetClass, day_key: str) -> int:
        if self.fail:
            raise GateStateUnavailable("test")
        return self.counts.get((asset_class, day_key), 0)

    def record_submission(self, idempotency_key: str, asset_class: AssetClass,
                          day_key: str, now: dt.datetime) -> None:
        if self.fail:
            raise GateStateUnavailable("test")
        self.seen[idempotency_key] = now
        key = (asset_class, day_key)
        self.counts[key] = self.counts.get(key, 0) + 1


class ScriptedStrategy:
    def __init__(self, sid: str, asset_class: AssetClass,
                 fn: Callable[[MarketContext], Sequence[Intent]]) -> None:
        self.id = sid
        self.asset_class = asset_class
        self._fn = fn
        self.calls = 0

    def propose(self, ctx: MarketContext) -> Sequence[Intent]:
        self.calls += 1
        return self._fn(ctx)


class FakeAdapter:
    """A BrokerAdapter whose every seam can be made to fail."""

    def __init__(self, asset_class: AssetClass, mode: Mode = Mode.PAPER, *,
                 equity: Optional[Decimal] = Decimal("100000"),
                 price: Optional[Decimal] = Decimal("50000"),
                 clock: Optional[Callable[[], dt.datetime]] = None) -> None:
        self.asset_class = asset_class
        self.mode = mode
        self._equity = equity
        self._price = price
        self._clock = clock or (lambda: CRYPTO_NOON)
        self.equity_error: Optional[BaseException] = None
        self.quote_error: Optional[BaseException] = None
        self.place_error: Optional[BaseException] = None
        self.place_result: Optional[Any] = None
        self.placed: list[Intent] = []
        self.fills_per_order = 1

    def equity(self) -> Optional[Decimal]:
        if self.equity_error is not None:
            raise self.equity_error
        return self._equity

    def quote(self, instrument: Instrument) -> Optional[Decimal]:
        if self.quote_error is not None:
            raise self.quote_error
        return self._price

    def positions(self) -> Sequence[Any]:
        return ()

    def cancel(self, order_id: str) -> bool:
        return False

    def place(self, intent: Intent, timeout_s: float = 30.0) -> OrderRef:
        self.placed.append(intent)
        if self.place_error is not None:
            raise self.place_error
        if self.place_result is not None:
            return self.place_result
        n = self.fills_per_order
        each = intent.qty / n
        price = self._price if self._price is not None else Decimal("1")
        fills = tuple(
            Fill(order_id=f"FAKE-{len(self.placed):04d}", instrument=intent.instrument,
                 side=intent.side, qty=each, price=price, fees=Decimal("0"),
                 at=self._clock(), mode=self.mode)
            for _ in range(n)
        )
        return OrderRef(order_id=f"FAKE-{len(self.placed):04d}",
                        status=OrderStatus.FILLED, intent=intent, mode=self.mode,
                        filled_qty=intent.qty, avg_price=price, fills=fills)


class AmbiguousSubmit(RuntimeError):
    ambiguous = True


class BreakableStore:
    """Wraps the real store so one method at a time can be made to fail."""

    def __init__(self, inner: SQLitePositionStore) -> None:
        self._inner = inner
        self.positions_error: Optional[BaseException] = None
        self.apply_error: Optional[BaseException] = None
        self.record_error: Optional[BaseException] = None
        self.closed = False

    def positions(self, mode: Mode, asset_class: Optional[AssetClass] = None):
        if self.positions_error is not None:
            raise self.positions_error
        return self._inner.positions(mode, asset_class)

    def apply(self, fill: Fill, strategy_id: str,
              structure_id: Optional[str] = None,
              fill_index: Optional[int] = None) -> None:
        if self.apply_error is not None:
            raise self.apply_error
        self._inner.apply(fill, strategy_id, structure_id, fill_index)

    def record_order(self, ref: OrderRef) -> None:
        if self.record_error is not None:
            raise self.record_error
        self._inner.record_order(ref)

    def realized_pnl(self, mode: Mode, asset_class: AssetClass,
                     since_iso: str) -> Decimal:
        return self._inner.realized_pnl(mode, asset_class, since_iso)

    def read_risk_state(self, mode: Mode, asset_class: AssetClass):
        return self._inner.read_risk_state(mode, asset_class)

    def write_risk_state(self, mode: Mode, asset_class: AssetClass, state) -> None:
        self._inner.write_risk_state(mode, asset_class, state)

    def close(self) -> None:
        self.closed = True
        self._inner.close()


class CountingCircuit:
    """Wraps a real RiskCircuit and counts which entry point was used."""

    def __init__(self, inner: RiskCircuit) -> None:
        self._inner = inner
        self.advances = 0
        self.reads = 0
        self.advance_error: Optional[BaseException] = None

    @property
    def mode(self) -> Mode:
        return self._inner.mode

    def read(self, asset_class: AssetClass, equity: Optional[Decimal]):
        self.reads += 1
        return self._inner.read(asset_class, equity)

    def advance(self, asset_class: AssetClass, equity: Decimal):
        self.advances += 1
        if self.advance_error is not None:
            raise self.advance_error
        return self._inner.advance(asset_class, equity)


# ---------------------------------------------------------------- harness


class Harness:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def build(*, asset_class: AssetClass = AssetClass.CRYPTO,
          strategies: Sequence[Any] = (),
          instruments: Optional[Sequence[Instrument]] = None,
          prices: Optional[dict[str, Decimal]] = None,
          cash: Decimal = Decimal("100000"),
          clock_at: Optional[dt.datetime] = None,
          env: Optional[dict[str, str]] = None,
          requested_mode: Mode = Mode.PAPER,
          adapter: Optional[Any] = None,
          live_adapter: Optional[Any] = None,
          gate: Optional[Any] = None,
          config: Optional[EngineConfig] = None,
          limits: Optional[ClassLimits] = None,
          extra_provider: Optional[Callable[..., Any]] = None) -> Harness:
    if clock_at is None:
        clock_at = CRYPTO_NOON if asset_class is AssetClass.CRYPTO else EQ_MIDDAY
    clock = FakeClock(clock_at)
    if instruments is None:
        instruments = (BTC,) if asset_class is AssetClass.CRYPTO else (NIFTY,)
    if prices is None:
        prices = ({"BTC/USDT": Decimal("50000"), "ETH/USDT": Decimal("3000")}
                  if asset_class is AssetClass.CRYPTO
                  else {"NIFTY-EQ": Decimal("100.00"),
                        "BANKNIFTY-EQ": Decimal("200.00")})
    quotes = InMemoryQuoteSource(dict(prices))
    inner = SQLitePositionStore(":memory:")
    store = BreakableStore(inner)
    if adapter is None:
        adapter = PaperAdapter(asset_class, quotes, cash, now=clock,
                               sleep=lambda s: None)
    env = {} if env is None else env
    mode = Mode.LIVE if (requested_mode is Mode.LIVE
                         and env.get({AssetClass.CRYPTO: "CRYPTO_LIVE",
                                      AssetClass.EQUITY: "EQUITY_LIVE"}[asset_class],
                                     "") in {"1", "true", "yes", "on"}) else Mode.PAPER
    circuit = CountingCircuit(RiskCircuit(inner, mode, clock=clock))
    gate_state = FakeGateState()
    if gate is None:
        gate = OrderGate(
            position_store=store, gate_state=gate_state, risk_circuit=circuit,
            config=GateConfig(requested_mode=requested_mode,
                              default_limits=limits or ClassLimits()),
            env=env,
        )
    wiring = ClassWiring(
        asset_class=asset_class, paper_adapter=adapter, risk=circuit,
        strategies=tuple(strategies), instruments=tuple(instruments),
        live_adapter=live_adapter, tick_interval_s=0.01,
        extra_provider=extra_provider,
    )
    eng = Engine(wirings=[wiring], store=store, gate=gate,
                 config=config or EngineConfig(requested_mode=requested_mode),
                 clock=clock, env=env)
    return Harness(engine=eng, store=store, inner=inner, gate=gate,
                   adapter=adapter, quotes=quotes, circuit=circuit,
                   clock=clock, gate_state=gate_state, wiring=wiring,
                   asset_class=asset_class, bus=eng.events)


def tick(h: Harness) -> None:
    asyncio.run(h.engine.tick_once(h.asset_class))


def of(h: Harness, kind: EventKind) -> list[Event]:
    return [e for e in h.bus.recent() if e.kind is kind]


def buy(instrument: Instrument = BTC, *, sid: str = "s1",
        qty: Decimal = Decimal("0.1"), key: str = "k1",
        side: Side = Side.BUY,
        order_type: OrderType = OrderType.MARKET,
        limit: Optional[Decimal] = None,
        structure_id: Optional[str] = None) -> Intent:
    return Intent(instrument=instrument, side=side, qty=qty,
                  order_type=order_type, strategy_id=sid, reason="test",
                  limit_price=limit, structure_id=structure_id,
                  idempotency_key=key)


# ================================================================ EventBus


def test_ring_buffer_is_bounded_and_reports_the_gap():
    bus = EventBus(buffer_size=3)
    for i in range(6):
        bus.publish(EventKind.TICK, f"t{i}")
    kept = bus.recent()
    assert len(kept) == 3
    assert [e.message for e in kept] == ["t3", "t4", "t5"]
    # A late client asking for everything after seq 1 must be able to see that
    # events 2 and 3 are gone rather than believing it caught up.
    assert bus.oldest_seq() == 4
    assert bus.last_seq == 6


def test_recent_filters_by_seq_and_kind():
    bus = EventBus(buffer_size=50)
    bus.publish(EventKind.TICK, "a")
    marker = bus.publish(EventKind.ERROR, "b")
    bus.publish(EventKind.TICK, "c")
    assert [e.message for e in bus.recent(after_seq=marker.seq)] == ["c"]
    assert [e.message for e in bus.recent(kinds=[EventKind.ERROR])] == ["b"]


def test_subscriber_receives_events():
    async def go():
        bus = EventBus()
        sub = bus.subscribe()
        bus.publish(EventKind.TICK, "one")
        bus.publish(EventKind.FILL, "two")
        got = [await anext(sub), await anext(sub)]
        bus.unsubscribe(sub)
        return got
    got = asyncio.run(go())
    assert [e.message for e in got] == ["one", "two"]


def test_slow_subscriber_drops_oldest_and_says_so():
    async def go():
        bus = EventBus()
        sub = bus.subscribe(maxsize=2)
        for i in range(5):
            bus.publish(EventKind.TICK, f"t{i}")
        first = await anext(sub)
        return sub.dropped, first.message
    dropped, first = asyncio.run(go())
    # Silently dropping would leave a client believing its stream was whole.
    assert dropped == 3
    assert first == "t3"


def test_publish_from_a_worker_thread_reaches_a_subscriber():
    async def go():
        bus = EventBus()
        sub = bus.subscribe()
        await asyncio.to_thread(bus.publish, EventKind.FILL, "from-thread")
        return await asyncio.wait_for(anext(sub), timeout=2)
    assert asyncio.run(go()).message == "from-thread"


def test_event_as_dict_keeps_decimals_exact():
    bus = EventBus()
    e = bus.publish(EventKind.FILL, "f",
                    data={"price": Decimal("0.1"), "at": CRYPTO_NOON,
                          "mode": Mode.PAPER, "legs": [Decimal("2.5")]})
    d = e.as_dict()
    assert d["data"]["price"] == "0.1"
    assert d["data"]["legs"] == ["2.5"]
    assert d["data"]["mode"] == "paper"
    assert d["data"]["at"] == CRYPTO_NOON.isoformat()


# ========================================================== wiring / boot


def test_paper_slot_refuses_a_live_adapter():
    with pytest.raises(EngineConfigError, match="paper_adapter"):
        ClassWiring(asset_class=AssetClass.CRYPTO,
                    paper_adapter=FakeAdapter(AssetClass.CRYPTO, Mode.LIVE),
                    risk=object())


def test_adapter_for_the_wrong_asset_class_is_refused():
    with pytest.raises(EngineConfigError, match="trades"):
        ClassWiring(asset_class=AssetClass.CRYPTO,
                    paper_adapter=FakeAdapter(AssetClass.EQUITY, Mode.PAPER),
                    risk=object())


def test_strategy_for_the_wrong_asset_class_is_refused():
    strat = ScriptedStrategy("x", AssetClass.EQUITY, lambda ctx: [])
    with pytest.raises(EngineConfigError, match="wired into"):
        ClassWiring(asset_class=AssetClass.CRYPTO,
                    paper_adapter=FakeAdapter(AssetClass.CRYPTO),
                    risk=object(), strategies=(strat,))


def test_duplicate_strategy_ids_across_classes_refuse_to_start():
    # Two strategies sharing an id net into ONE position book in the store.
    crypto = ClassWiring(asset_class=AssetClass.CRYPTO,
                         paper_adapter=FakeAdapter(AssetClass.CRYPTO),
                         risk=object(),
                         strategies=(ScriptedStrategy("dup", AssetClass.CRYPTO,
                                                      lambda c: []),))
    equity = ClassWiring(asset_class=AssetClass.EQUITY,
                         paper_adapter=FakeAdapter(AssetClass.EQUITY),
                         risk=object(),
                         strategies=(ScriptedStrategy("dup", AssetClass.EQUITY,
                                                      lambda c: []),))
    with pytest.raises(EngineConfigError, match="dup"):
        Engine(wirings=[crypto, equity], store=object(), gate=object(), env={})


def test_armed_without_a_live_adapter_refuses_to_start():
    wiring = ClassWiring(asset_class=AssetClass.CRYPTO,
                         paper_adapter=FakeAdapter(AssetClass.CRYPTO),
                         risk=object())
    with pytest.raises(EngineConfigError, match="no live_adapter"):
        Engine(wirings=[wiring], store=object(), gate=object(),
               config=EngineConfig(requested_mode=Mode.LIVE),
               env={"CRYPTO_LIVE": "1"})


def test_circuit_bound_to_the_wrong_mode_refuses_to_start():
    class LiveCircuit:
        mode = Mode.LIVE

        def read(self, a, e): ...
        def advance(self, a, e): ...

    wiring = ClassWiring(asset_class=AssetClass.CRYPTO,
                         paper_adapter=FakeAdapter(AssetClass.CRYPTO),
                         risk=LiveCircuit())
    with pytest.raises(EngineConfigError, match="risk circuit is bound"):
        Engine(wirings=[wiring], store=object(), gate=object(), env={})


def test_requesting_live_without_the_env_flag_stays_on_paper():
    live = FakeAdapter(AssetClass.CRYPTO, Mode.LIVE)
    h = build(requested_mode=Mode.LIVE, env={}, live_adapter=live)
    assert h.engine.runner_modes()[AssetClass.CRYPTO] is Mode.PAPER
    assert h.engine.arm_report()[0].adapter == "PaperAdapter"


def test_env_flag_alone_does_not_arm_anything():
    # The env says live, the engine was asked for paper. Nothing here may
    # promote that: arming takes BOTH.
    live = FakeAdapter(AssetClass.CRYPTO, Mode.LIVE)
    h = build(requested_mode=Mode.PAPER, env={"CRYPTO_LIVE": "1"},
              live_adapter=live)
    assert h.engine.runner_modes()[AssetClass.CRYPTO] is Mode.PAPER
    assert live.placed == []


def test_a_typo_in_the_arm_flag_raises_instead_of_reading_as_disarmed():
    live = FakeAdapter(AssetClass.CRYPTO, Mode.LIVE)
    with pytest.raises(ValueError, match="not a boolean"):
        build(requested_mode=Mode.LIVE, env={"CRYPTO_LIVE": "ture"},
              live_adapter=live)


def test_construction_never_writes_to_the_process_environment():
    import os
    before = dict(os.environ)
    build(requested_mode=Mode.LIVE, env={"CRYPTO_LIVE": "1"},
          live_adapter=FakeAdapter(AssetClass.CRYPTO, Mode.LIVE))
    assert dict(os.environ) == before


def test_arm_report_names_the_flag_and_carries_no_secrets():
    h = build()
    line = h.engine.arm_report_text()
    assert "crypto" in line and "mode=paper" in line
    assert "CRYPTO_LIVE=disarmed" in line
    assert "PaperAdapter" in line


def test_boot_event_is_emitted_once():
    h = build()
    tick(h)
    tick(h)
    boots = of(h, EventKind.BOOT)
    assert len(boots) == 1
    assert boots[0].data["classes"][0]["arm_var"] == "CRYPTO_LIVE"


# ================================================== full tick, happy path


def test_full_tick_places_records_and_nets_a_fill():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    h = build(strategies=[strat])
    tick(h)

    positions = h.inner.positions(Mode.PAPER, AssetClass.CRYPTO)
    assert len(positions) == 1
    assert positions[0].qty == Decimal("0.1")
    assert positions[0].strategy_id == "s1"
    # The order row and the fill row are BOTH written; record_order alone
    # leaves the book flat.
    assert len(h.inner.orders(Mode.PAPER)) == 1
    assert len(h.inner.fills(Mode.PAPER)) == 1

    kinds = [e.kind for e in h.bus.recent()]
    for expected in (EventKind.TICK, EventKind.INTENT, EventKind.APPROVAL,
                     EventKind.ORDER, EventKind.FILL, EventKind.RISK):
        assert expected in kinds, expected


def test_the_gates_snapped_qty_is_what_gets_placed():
    # 0.12345 snaps to 0.123 at qty_precision=3. The assertion is on what the
    # ADAPTER received, not on the resulting position: a well-behaved adapter
    # re-quantizes on the way in and would hide an engine that placed the raw
    # intent, leaving the gate's snap (and its circuit throttle) decorative.
    strat = ScriptedStrategy(
        "s1", AssetClass.CRYPTO,
        lambda ctx: [buy(qty=Decimal("0.12345"))])
    adapter = FakeAdapter(AssetClass.CRYPTO, clock=lambda: CRYPTO_NOON)
    h = build(strategies=[strat], adapter=adapter)
    tick(h)
    assert [i.qty for i in adapter.placed] == [Decimal("0.123")]
    positions = h.inner.positions(Mode.PAPER, AssetClass.CRYPTO)
    assert positions[0].qty == Decimal("0.123")


def test_risk_advances_exactly_once_per_class_per_tick():
    strat = ScriptedStrategy(
        "s1", AssetClass.CRYPTO,
        lambda ctx: [buy(key=f"k{ctx.now.timestamp()}")])
    h = build(strategies=[strat])
    tick(h)
    h.clock.advance(600)
    tick(h)
    assert h.circuit.advances == 2


def test_a_rejected_intent_never_reaches_the_adapter():
    # No idempotency_key: the gate refuses, so nothing is placed.
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO,
                             lambda ctx: [buy(key="")])
    h = build(strategies=[strat])
    tick(h)
    assert of(h, EventKind.REJECTION)
    assert of(h, EventKind.ORDER) == []
    assert h.inner.positions(Mode.PAPER) == []


def test_context_carries_mode_filtered_positions_and_the_provider_extra():
    seen: dict[str, Any] = {}

    def spy(ctx: MarketContext):
        seen["positions"] = ctx.positions
        seen["extra"] = dict(ctx.extra)
        seen["quotes"] = dict(ctx.quotes)
        seen["equity"] = ctx.equity
        return []

    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, spy)
    h = build(strategies=[strat],
              extra_provider=lambda ac, now: {"bars": {"BTC/USDT": []}})
    # A LIVE row must never appear in a PAPER tick.
    h.inner.apply(Fill(order_id="live-1", instrument=BTC, side=Side.BUY,
                       qty=Decimal("1"), price=Decimal("50000"),
                       fees=Decimal("0"), at=CRYPTO_NOON, mode=Mode.LIVE),
                  "other")
    tick(h)
    assert seen["positions"] == ()
    assert seen["extra"] == {"bars": {"BTC/USDT": []}}
    assert seen["quotes"]["BTC/USDT"] == Decimal("50000")
    assert seen["equity"] == Decimal("100000")


# ============================================== equity: unknown vs zero


def test_unknown_equity_reads_the_circuit_instead_of_advancing_it():
    adapter = FakeAdapter(AssetClass.CRYPTO, equity=None)
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    h = build(strategies=[strat], adapter=adapter)
    tick(h)
    # advance() would persist a baseline derived from a number we do not have.
    assert h.circuit.advances == 0
    assert h.circuit.reads >= 1
    risk = of(h, EventKind.RISK)[-1]
    assert risk.data["advanced"] is False
    assert risk.data["equity_known"] is False
    # ...and the gate refuses the entry rather than sizing against a guess.
    assert of(h, EventKind.ORDER) == []
    assert any("equity" in e.data.get("reason", "")
               for e in of(h, EventKind.REJECTION))


def test_equity_that_raises_becomes_unknown_not_a_number():
    adapter = FakeAdapter(AssetClass.CRYPTO)
    adapter.equity_error = RuntimeError("broker down")
    h = build(adapter=adapter)
    tick(h)
    assert h.circuit.advances == 0
    errors = [e for e in of(h, EventKind.ERROR) if e.data.get("where") == "equity"]
    assert errors and "UNKNOWN" in errors[0].message
    risk = of(h, EventKind.RISK)[-1]
    # A substituted 0 would be a KNOWN equity of zero, which the circuit reads
    # as a 100% drawdown and answers with FLATTEN: a broker outage would
    # liquidate the book. None is the only honest answer.
    assert risk.data["equity"] is None
    assert risk.data["equity_known"] is False
    assert risk.data["action"] != CircuitAction.FLATTEN.value


def test_non_decimal_equity_is_rejected_rather_than_coerced():
    adapter = FakeAdapter(AssetClass.CRYPTO, equity=100000.0)  # type: ignore[arg-type]
    h = build(adapter=adapter)
    tick(h)
    assert h.circuit.advances == 0
    assert any(e.data.get("where") == "equity" for e in of(h, EventKind.ERROR))


# ================================================== store read failures


def test_an_unreadable_position_store_skips_the_tick_instead_of_trading_flat():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    h = build(strategies=[strat])
    h.store.positions_error = RuntimeError("db locked")
    with pytest.raises(RuntimeError):
        tick(h)
    # Reading "flat" would re-enter positions we already hold.
    assert strat.calls == 0
    assert of(h, EventKind.ORDER) == []
    assert any(e.data.get("where") == "positions" for e in of(h, EventKind.ERROR))


def test_the_circuit_still_advances_when_the_trading_half_failed():
    h = build()
    h.store.positions_error = RuntimeError("db locked")
    with pytest.raises(RuntimeError):
        tick(h)
    # A high-water mark that freezes during an outage under-reports the
    # drawdown once the outage clears.
    assert h.circuit.advances == 1


def test_a_failing_extra_provider_stops_the_tick_and_is_reported():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    h = build(strategies=[strat],
              extra_provider=lambda ac, now: (_ for _ in ()).throw(
                  RuntimeError("feed down")))
    with pytest.raises(RuntimeError):
        tick(h)
    assert strat.calls == 0
    assert any(e.data.get("where") == "extra_provider"
               for e in of(h, EventKind.ERROR))


# ================================================== quotes never fake a price


def test_a_failed_quote_is_absent_not_zero():
    seen: dict[str, Any] = {}
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO,
                             lambda ctx: seen.update(q=dict(ctx.quotes)) or [])
    adapter = FakeAdapter(AssetClass.CRYPTO)
    adapter.quote_error = RuntimeError("no feed")
    h = build(strategies=[strat], adapter=adapter)
    tick(h)
    assert seen["q"] == {}
    assert any(e.data.get("where") == "quote" for e in of(h, EventKind.ERROR))


def test_a_nonpositive_quote_is_discarded():
    adapter = FakeAdapter(AssetClass.CRYPTO, price=Decimal("0"))
    seen: dict[str, Any] = {}
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO,
                             lambda ctx: seen.update(q=dict(ctx.quotes)) or [])
    h = build(strategies=[strat], adapter=adapter)
    tick(h)
    assert seen["q"] == {}


# ================================================ strategy isolation


def test_a_raising_strategy_is_disabled_for_the_session():
    boom = ScriptedStrategy(
        "boom", AssetClass.CRYPTO,
        lambda ctx: (_ for _ in ()).throw(RuntimeError("bad maths")))
    fine = ScriptedStrategy("fine", AssetClass.CRYPTO, lambda ctx: [])
    h = build(strategies=[boom, fine])
    tick(h)
    tick(h)
    # Called once, never retried; the other strategy keeps running.
    assert boom.calls == 1
    assert fine.calls == 2
    disabled = of(h, EventKind.STRATEGY_DISABLED)
    assert len(disabled) == 1
    assert disabled[0].data["strategy_id"] == "boom"
    status = h.engine.status()
    assert status.classes[0].disabled_strategies == ("boom",)
    assert status.degraded


def test_a_strategy_returning_a_non_intent_is_disabled():
    bad = ScriptedStrategy("bad", AssetClass.CRYPTO, lambda ctx: ["not an intent"])
    h = build(strategies=[bad])
    tick(h)
    assert h.engine.status().classes[0].disabled_strategies == ("bad",)
    assert of(h, EventKind.ORDER) == []


def test_a_strategy_signing_an_intent_with_another_strategys_id_is_disabled():
    # Otherwise it writes into someone else's position book and the store's
    # per-strategy netting stops meaning anything.
    forger = ScriptedStrategy("forger", AssetClass.CRYPTO,
                              lambda ctx: [buy(sid="victim")])
    h = build(strategies=[forger])
    tick(h)
    assert h.engine.status().classes[0].disabled_strategies == ("forger",)
    assert of(h, EventKind.ORDER) == []


def test_a_disabled_strategy_holding_a_position_raises_an_unmanaged_alarm():
    boom = ScriptedStrategy(
        "boom", AssetClass.CRYPTO,
        lambda ctx: (_ for _ in ()).throw(RuntimeError("bad maths")))
    h = build(strategies=[boom])
    h.inner.apply(Fill(order_id="seed", instrument=BTC, side=Side.BUY,
                       qty=Decimal("0.5"), price=Decimal("50000"),
                       fees=Decimal("0"), at=CRYPTO_NOON, mode=Mode.PAPER),
                  "boom")
    tick(h)
    alarms = [e for e in of(h, EventKind.ERROR)
              if e.data.get("where") == "unmanaged_positions"]
    assert alarms, "an unwatched open position must not be silent"
    assert alarms[0].data["symbols"] == ["BTC/USDT"]
    status = h.engine.status()
    assert status.classes[0].unmanaged_positions == ("BTC/USDT",)
    assert status.degraded


def test_a_real_strategy_missing_its_bar_book_is_disabled_loudly():
    # StrategyBase treats an absent bar book as a wiring failure, not an empty
    # market. The engine must surface that, not swallow it.
    strat = CryptoEmaTrendStrategy(instruments=[BTC])
    h = build(strategies=[strat])
    tick(h)
    assert h.engine.status().classes[0].disabled_strategies == ("crypto_trend",)


# ================================================ order routing / modes


def test_an_approval_in_another_mode_is_refused():
    class LyingGate:
        def check(self, intent: Intent, ctx: MarketContext) -> Approval:
            return Approval(True, Mode.LIVE, intent, ())

    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    adapter = FakeAdapter(AssetClass.CRYPTO)
    h = build(strategies=[strat], adapter=adapter, gate=LyingGate())
    tick(h)
    assert adapter.placed == []
    assert any(e.data.get("where") == "mode_mismatch"
               for e in of(h, EventKind.ERROR))


def test_an_adapter_whose_mode_disagrees_with_the_approval_is_refused():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    h = build(strategies=[strat])
    impostor = FakeAdapter(AssetClass.CRYPTO, Mode.LIVE)
    runner = h.engine._runners[AssetClass.CRYPTO]
    runner.adapters = MappingProxyType({Mode.PAPER: impostor})
    runner.adapter = impostor
    tick(h)
    assert impostor.placed == []
    assert any(e.data.get("where") == "routing" for e in of(h, EventKind.ERROR))


def test_an_armed_class_routes_to_the_live_adapter_and_records_live():
    live = FakeAdapter(AssetClass.CRYPTO, Mode.LIVE, clock=lambda: CRYPTO_NOON)
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    h = build(strategies=[strat], requested_mode=Mode.LIVE,
              env={"CRYPTO_LIVE": "1"}, live_adapter=live)
    assert h.engine.runner_modes()[AssetClass.CRYPTO] is Mode.LIVE
    tick(h)
    assert len(live.placed) == 1
    assert h.inner.positions(Mode.LIVE, AssetClass.CRYPTO)
    # ...and the paper book stays untouched.
    assert h.inner.positions(Mode.PAPER, AssetClass.CRYPTO) == []


# ================================================ execution failures


def test_an_ambiguous_submit_halts_the_class():
    strat = ScriptedStrategy(
        "s1", AssetClass.CRYPTO,
        lambda ctx: [buy(key="a"), buy(instrument=ETH, key="b")])
    adapter = FakeAdapter(AssetClass.CRYPTO)
    adapter.place_error = AmbiguousSubmit("submit died in flight")
    h = build(strategies=[strat], adapter=adapter,
              instruments=(BTC, ETH))
    tick(h)
    status = h.engine.status().classes[0]
    assert status.halted
    assert "untracked order" in (status.halt_reason or "")
    # The second intent must not be sent after we lost track of the first.
    assert len(adapter.placed) == 1


def test_a_plain_place_failure_reports_but_does_not_halt():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    adapter = FakeAdapter(AssetClass.CRYPTO)
    adapter.place_error = RuntimeError("rejected by risk desk")
    h = build(strategies=[strat], adapter=adapter)
    tick(h)
    assert not h.engine.status().classes[0].halted
    assert any(e.data.get("where") == "place" for e in of(h, EventKind.ERROR))


def test_a_failed_fill_write_halts_the_class():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    h = build(strategies=[strat])
    h.store.apply_error = RuntimeError("disk full")
    tick(h)
    status = h.engine.status().classes[0]
    assert status.halted
    errors = [e for e in of(h, EventKind.ERROR)
              if e.data.get("where") == "apply_fill"]
    assert errors
    # The event must carry enough to reconstruct the fill by hand.
    assert errors[0].data["qty"] == Decimal("0.1")
    assert errors[0].data["symbol"] == "BTC/USDT"


def test_a_failed_order_write_halts_the_class():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    h = build(strategies=[strat])
    h.store.record_error = RuntimeError("disk full")
    tick(h)
    assert h.engine.status().classes[0].halted


def test_a_halted_class_places_nothing_on_later_ticks():
    strat = ScriptedStrategy(
        "s1", AssetClass.CRYPTO,
        lambda ctx: [buy(key=f"k{ctx.now.timestamp()}")])
    adapter = FakeAdapter(AssetClass.CRYPTO)
    h = build(strategies=[strat], adapter=adapter)
    h.engine._runners[AssetClass.CRYPTO].halt("test")
    tick(h)
    assert adapter.placed == []
    assert strat.calls == 0


def test_identical_fills_on_one_order_are_both_netted():
    # The store's default idempotency is content-addressed, so two fills with
    # the same qty/price/fees AND timestamp look like a replay. Passing an
    # explicit fill_index is what keeps the second one.
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    adapter = FakeAdapter(AssetClass.CRYPTO, clock=lambda: CRYPTO_NOON)
    adapter.fills_per_order = 2
    h = build(strategies=[strat], adapter=adapter)
    tick(h)
    positions = h.inner.positions(Mode.PAPER, AssetClass.CRYPTO)
    assert positions[0].qty == Decimal("0.1")
    assert len(h.inner.fills(Mode.PAPER)) == 2


def test_an_adapter_returning_a_non_orderref_halts():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    adapter = FakeAdapter(AssetClass.CRYPTO)
    adapter.place_result = "filled, promise"
    h = build(strategies=[strat], adapter=adapter)
    tick(h)
    assert h.engine.status().classes[0].halted


def test_a_gate_that_raises_blocks_the_order():
    class ExplodingGate:
        def check(self, intent, ctx):
            raise RuntimeError("gate store down")

    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [buy()])
    adapter = FakeAdapter(AssetClass.CRYPTO)
    h = build(strategies=[strat], adapter=adapter, gate=ExplodingGate())
    tick(h)
    assert adapter.placed == []
    assert any(e.data.get("where") == "gate" for e in of(h, EventKind.ERROR))


def test_a_failing_risk_circuit_is_reported_and_clears_the_last_action():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [])
    h = build(strategies=[strat])
    h.circuit.advance_error = RuntimeError("state unreadable")
    tick(h)
    assert any(e.data.get("where") == "risk" for e in of(h, EventKind.ERROR))
    assert h.engine.status().classes[0].last_risk_action is None


# =============================================== session-close arithmetic


def test_session_close_finds_todays_close_not_tomorrows():
    profile = profile_for(AssetClass.EQUITY)
    morning = dt.datetime(2026, 8, 27, 9, 20, tzinfo=IST)
    close = _session_close(profile, morning)
    assert close is not None
    # A plain binary search over a multi-day horizon converges on TOMORROW's
    # close, and the square-off window would then never open today.
    assert close.astimezone(IST).date() == morning.date()
    assert close.astimezone(IST).hour == 15
    assert close.astimezone(IST).minute == 30


def test_session_close_is_none_when_the_market_is_shut():
    assert _session_close(profile_for(AssetClass.EQUITY), EQ_WEEKEND) is None


def test_session_close_is_none_for_a_market_that_never_closes():
    assert _session_close(profile_for(AssetClass.CRYPTO), CRYPTO_NOON) is None


def test_marketable_limit_never_rounds_back_inside_the_spread():
    profile = profile_for(AssetClass.EQUITY)
    # HALF_UP takes a 100.03 sell UP to 100.05, above the mark, where it rests.
    assert profile.quantize_price(NIFTY, Decimal("100.03")) > Decimal("100.03")
    assert _marketable_limit(NIFTY, Side.SELL, Decimal("100.03"), 0) <= Decimal("100.03")
    # ...and a 100.02 buy DOWN to 100.00, below the mark, where it rests.
    assert profile.quantize_price(NIFTY, Decimal("100.02")) < Decimal("100.02")
    assert _marketable_limit(NIFTY, Side.BUY, Decimal("100.02"), 0) >= Decimal("100.02")


def test_marketable_limit_crosses_by_the_requested_ticks_and_stays_on_grid():
    assert _marketable_limit(NIFTY, Side.BUY, Decimal("100.00"), 2) == Decimal("100.10")
    assert _marketable_limit(NIFTY, Side.SELL, Decimal("100.00"), 2) == Decimal("99.90")


def test_marketable_limit_refuses_to_invent_a_nonpositive_price():
    penny = Instrument(symbol="P", asset_class=AssetClass.EQUITY, broker_id="1",
                       tick_size=Decimal("0.05"))
    assert _marketable_limit(penny, Side.SELL, Decimal("0.05"), 2) is None


def test_reduces_only_counts_orders_that_shrink_a_position():
    from core.types import Position
    long_50 = Position(instrument=NIFTY, qty=Decimal("50"),
                       avg_price=Decimal("100"), mode=Mode.PAPER,
                       strategy_id="s")
    assert _reduces(buy(NIFTY, side=Side.SELL, qty=Decimal("50")), [long_50])
    assert _reduces(buy(NIFTY, side=Side.SELL, qty=Decimal("25")), [long_50])
    # Bigger than the position flips through zero: that is a new entry.
    assert not _reduces(buy(NIFTY, side=Side.SELL, qty=Decimal("75")), [long_50])
    assert not _reduces(buy(NIFTY, side=Side.BUY, qty=Decimal("50")), [long_50])
    assert not _reduces(buy(BANK, side=Side.SELL, qty=Decimal("25")), [long_50])


# ==================================================== mandatory square-off


class RejectingAdapter(FakeAdapter):
    def place(self, intent: Intent, timeout_s: float = 30.0) -> OrderRef:
        self.placed.append(intent)
        return OrderRef(order_id=f"REJ-{len(self.placed):04d}",
                        status=OrderStatus.REJECTED, intent=intent,
                        mode=self.mode, error="declined")


def equity_harness(**kw: Any) -> Harness:
    kw.setdefault("asset_class", AssetClass.EQUITY)
    kw.setdefault("clock_at", EQ_NEAR_CLOSE)
    kw.setdefault("instruments", (NIFTY, BANK))
    return build(**kw)


def seed(h: Harness, instrument: Instrument, side: Side, qty: Decimal,
         price: Decimal, sid: str = "seed", at: Optional[dt.datetime] = None,
         structure_id: Optional[str] = None) -> None:
    h.inner.apply(Fill(order_id=f"seed-{instrument.symbol}-{side.value}",
                       instrument=instrument, side=side, qty=qty, price=price,
                       fees=Decimal("0"), at=at or EQ_MIDDAY, mode=Mode.PAPER),
                  sid, structure_id)


def test_square_off_exits_an_open_equity_position_before_the_close():
    adapter = FakeAdapter(AssetClass.EQUITY, price=Decimal("100.00"),
                          clock=lambda: EQ_NEAR_CLOSE)
    h = equity_harness(adapter=adapter)
    seed(h, NIFTY, Side.BUY, Decimal("50"), Decimal("100.00"))
    tick(h)

    intents = of(h, EventKind.INTENT)
    assert len(intents) == 1
    data = intents[0].data
    assert data["source"] == IntentSource.SQUARE_OFF.value
    assert data["side"] == "sell"
    assert data["qty"] == Decimal("50")
    # Equity venues reject MARKET, so the exit must be a limit that crosses.
    assert data["order_type"] == "limit"
    assert data["limit_price"] == Decimal("99.90")
    assert data["strategy_id"] == "seed"
    assert h.inner.positions(Mode.PAPER, AssetClass.EQUITY) == []


def test_square_off_does_not_run_before_the_window_opens():
    adapter = FakeAdapter(AssetClass.EQUITY, price=Decimal("100.00"))
    h = equity_harness(adapter=adapter, clock_at=EQ_MIDDAY)
    seed(h, NIFTY, Side.BUY, Decimal("50"), Decimal("100.00"))
    tick(h)
    assert of(h, EventKind.INTENT) == []
    assert adapter.placed == []


def test_square_off_covers_shorts_before_it_closes_longs():
    # Closing the long leg of a spread first leaves a NAKED SHORT in the window
    # with the least time left to fix it.
    adapter = FakeAdapter(AssetClass.EQUITY, price=Decimal("100.00"),
                          clock=lambda: EQ_NEAR_CLOSE)
    h = equity_harness(adapter=adapter)
    seed(h, NIFTY, Side.BUY, Decimal("50"), Decimal("100.00"),
         sid="spread", structure_id="st1")
    seed(h, BANK, Side.SELL, Decimal("25"), Decimal("100.00"),
         sid="spread", structure_id="st1")
    tick(h)
    sides = [e.data["side"] for e in of(h, EventKind.INTENT)]
    assert sides == ["buy", "sell"]
    assert [i.side for i in adapter.placed] == [Side.BUY, Side.SELL]
    assert of(h, EventKind.INTENT)[0].data["structure_id"] == "st1"


def test_the_square_off_window_refuses_new_entries():
    entry = buy(BANK, sid="s1", qty=Decimal("25"), key="entry",
                order_type=OrderType.LIMIT, limit=Decimal("100.00"))
    strat = ScriptedStrategy("s1", AssetClass.EQUITY, lambda ctx: [entry])
    adapter = FakeAdapter(AssetClass.EQUITY, price=Decimal("100.00"),
                          clock=lambda: EQ_NEAR_CLOSE)
    h = equity_harness(adapter=adapter, strategies=[strat])
    tick(h)
    assert adapter.placed == []
    rejections = of(h, EventKind.REJECTION)
    assert rejections and "square-off window" in rejections[0].data["reason"]


def test_the_square_off_window_still_allows_exits():
    exit_intent = buy(NIFTY, sid="s1", qty=Decimal("50"), key="exit",
                      side=Side.SELL, order_type=OrderType.LIMIT,
                      limit=Decimal("99.90"))
    strat = ScriptedStrategy("s1", AssetClass.EQUITY, lambda ctx: [exit_intent])
    adapter = FakeAdapter(AssetClass.EQUITY, price=Decimal("100.00"),
                          clock=lambda: EQ_NEAR_CLOSE)
    h = equity_harness(adapter=adapter, strategies=[strat])
    seed(h, NIFTY, Side.BUY, Decimal("50"), Decimal("100.00"), sid="s1")
    tick(h)
    # One exit, not two: the engine must not stack its square-off on top of the
    # strategy's own exit and sell the position twice.
    assert len(adapter.placed) == 1
    assert of(h, EventKind.INTENT)[0].data["source"] == IntentSource.STRATEGY.value


def test_square_off_refuses_to_invent_a_price_when_the_quote_is_gone():
    adapter = FakeAdapter(AssetClass.EQUITY, price=None,
                          clock=lambda: EQ_NEAR_CLOSE)
    h = equity_harness(adapter=adapter)
    seed(h, NIFTY, Side.BUY, Decimal("50"), Decimal("100.00"))
    tick(h)
    assert adapter.placed == []
    errors = [e for e in of(h, EventKind.ERROR)
              if e.data.get("where") == "square_off"]
    assert errors and "no quote" in errors[0].message
    # The position is still there, and the operator can see why.
    assert len(h.inner.positions(Mode.PAPER, AssetClass.EQUITY)) == 1


def test_square_off_retries_on_the_next_bucket_not_on_every_tick():
    adapter = RejectingAdapter(AssetClass.EQUITY, price=Decimal("100.00"),
                               clock=lambda: EQ_NEAR_CLOSE)
    h = equity_harness(adapter=adapter)
    seed(h, NIFTY, Side.BUY, Decimal("50"), Decimal("100.00"))

    tick(h)
    assert len(adapter.placed) == 1
    keys = [e.data["idempotency_key"] for e in of(h, EventKind.INTENT)]

    h.clock.advance(5)          # same 60s bucket: the gate dedups it
    tick(h)
    assert len(adapter.placed) == 1
    assert any("already submitted" in e.data.get("reason", "")
               for e in of(h, EventKind.REJECTION))

    h.clock.advance(60)         # new bucket: retried
    tick(h)
    assert len(adapter.placed) == 2
    keys = [e.data["idempotency_key"] for e in of(h, EventKind.INTENT)]
    assert keys[0] != keys[-1]


def test_crypto_never_squares_off():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [])
    adapter = FakeAdapter(AssetClass.CRYPTO)
    h = build(strategies=[strat], adapter=adapter)
    h.inner.apply(Fill(order_id="seed", instrument=BTC, side=Side.BUY,
                       qty=Decimal("0.5"), price=Decimal("50000"),
                       fees=Decimal("0"), at=CRYPTO_NOON, mode=Mode.PAPER),
                  "s1")
    tick(h)
    assert adapter.placed == []
    assert of(h, EventKind.INTENT) == []


# ================================================== market hours / cadence


def test_a_closed_equity_market_does_not_tick_the_strategies():
    strat = ScriptedStrategy("s1", AssetClass.EQUITY, lambda ctx: [buy(NIFTY)])
    adapter = FakeAdapter(AssetClass.EQUITY)
    h = equity_harness(strategies=[strat], adapter=adapter,
                       clock_at=EQ_WEEKEND)
    tick(h)
    tick(h)
    assert strat.calls == 0
    assert adapter.placed == []
    # The transition is announced once, not on every tick all weekend.
    sessions = of(h, EventKind.SESSION)
    assert len(sessions) == 1
    assert sessions[0].data["open"] is False
    assert h.engine.status().classes[0].ticks == 0


def test_the_session_transition_is_announced_when_the_market_opens():
    strat = ScriptedStrategy("s1", AssetClass.EQUITY, lambda ctx: [])
    h = equity_harness(strategies=[strat], adapter=FakeAdapter(AssetClass.EQUITY),
                       clock_at=EQ_WEEKEND)
    tick(h)
    h.clock.at = EQ_MIDDAY
    tick(h)
    assert [e.data["open"] for e in of(h, EventKind.SESSION)] == [False, True]


def test_crypto_ticks_regardless_of_the_hour():
    strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [])
    h = build(strategies=[strat],
              clock_at=dt.datetime(2026, 8, 30, 3, 0, tzinfo=UTC))  # a Sunday
    tick(h)
    assert strat.calls == 1


# ======================================================= loop / shutdown


def test_the_loop_ticks_and_stops_cleanly():
    async def go():
        strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [])
        h = build(strategies=[strat])
        runner = asyncio.create_task(h.engine.run())
        for _ in range(200):
            if strat.calls >= 2:
                break
            await asyncio.sleep(0.01)
        await h.engine.stop(drain_timeout_s=5)
        await asyncio.wait_for(runner, timeout=5)
        return h, strat
    h, strat = asyncio.run(go())
    assert strat.calls >= 2
    assert not h.engine.running
    # The ledger is checkpointed on the way out.
    assert h.store.closed
    shutdowns = of(h, EventKind.SHUTDOWN)
    assert [e.message for e in shutdowns][-1] == "engine stopped"


def test_stop_can_leave_a_shared_store_open():
    async def go():
        h = build(config=EngineConfig(close_store_on_stop=False))
        await h.engine.start()
        await h.engine.stop(drain_timeout_s=5)
        return h
    h = asyncio.run(go())
    assert not h.store.closed


def test_shutdown_stops_issuing_new_orders_mid_tick():
    placed_before_stop: list[int] = []

    class StoppingGate:
        def __init__(self, inner: OrderGate, engine_box: dict[str, Any]) -> None:
            self._inner = inner
            self._box = engine_box

        def check(self, intent: Intent, ctx: MarketContext) -> Approval:
            approval = self._inner.check(intent, ctx)
            # Simulate stop() landing while the tick is already running.
            self._box["engine"]._stopping = True
            return approval

    adapter = FakeAdapter(AssetClass.CRYPTO)
    strat = ScriptedStrategy(
        "s1", AssetClass.CRYPTO,
        lambda ctx: [buy(key="a"), buy(instrument=ETH, key="b")])
    box: dict[str, Any] = {}
    h = build(strategies=[strat], adapter=adapter, instruments=(BTC, ETH))
    inner_gate = h.gate
    h.engine.gate = StoppingGate(inner_gate, box)
    box["engine"] = h.engine
    tick(h)
    placed_before_stop.append(len(adapter.placed))
    assert placed_before_stop == [1]
    assert any("shutting down" in e.message for e in of(h, EventKind.REJECTION))


def test_repeated_tick_failures_halt_the_class():
    async def go():
        h = build(config=EngineConfig(max_consecutive_tick_errors=2))
        h.store.positions_error = RuntimeError("db gone")
        await asyncio.wait_for(h.engine.run(), timeout=5)
        return h
    h = asyncio.run(go())
    status = h.engine.status().classes[0]
    # Retrying forever every 5s looks identical to working, from outside.
    assert status.halted
    assert "consecutive tick failures" in (status.halt_reason or "")
    assert of(h, EventKind.HALT)


def test_a_recovered_tick_clears_the_error_counter():
    async def go():
        strat = ScriptedStrategy("s1", AssetClass.CRYPTO, lambda ctx: [])
        h = build(strategies=[strat],
                  config=EngineConfig(max_consecutive_tick_errors=50))
        h.store.positions_error = RuntimeError("blip")
        task = asyncio.create_task(h.engine.run())
        for _ in range(200):
            if h.engine.status().classes[0].consecutive_errors:
                break
            await asyncio.sleep(0.01)
        h.store.positions_error = None
        for _ in range(200):
            if strat.calls:
                break
            await asyncio.sleep(0.01)
        await h.engine.stop(drain_timeout_s=5)
        await asyncio.wait_for(task, timeout=5)
        return h
    h = asyncio.run(go())
    assert h.engine.status().classes[0].consecutive_errors == 0
    assert not h.engine.status().classes[0].halted


def test_starting_twice_is_refused():
    async def go():
        h = build()
        await h.engine.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await h.engine.start()
        finally:
            await h.engine.stop(drain_timeout_s=5)
    asyncio.run(go())


def test_status_is_clean_before_anything_happens():
    h = build()
    status = h.engine.status()
    assert not status.running and not status.degraded
    assert status.classes[0].mode is Mode.PAPER
    assert status.classes[0].ticks == 0


def test_two_classes_run_side_by_side():
    clock = FakeClock(CRYPTO_NOON)
    quotes_c = InMemoryQuoteSource({"BTC/USDT": Decimal("50000")})
    quotes_e = InMemoryQuoteSource({"NIFTY-EQ": Decimal("100.00")})
    inner = SQLitePositionStore(":memory:")
    store = BreakableStore(inner)
    crypto_adapter = PaperAdapter(AssetClass.CRYPTO, quotes_c, Decimal("100000"),
                                  now=clock, sleep=lambda s: None)
    equity_adapter = PaperAdapter(AssetClass.EQUITY, quotes_e, Decimal("100000"),
                                  now=clock, sleep=lambda s: None)
    circuit = CountingCircuit(RiskCircuit(inner, Mode.PAPER, clock=clock))
    gate = OrderGate(position_store=store, gate_state=FakeGateState(),
                     risk_circuit=circuit, config=GateConfig(), env={})
    cs = ScriptedStrategy("c1", AssetClass.CRYPTO, lambda ctx: [buy(sid="c1")])
    es = ScriptedStrategy("e1", AssetClass.EQUITY, lambda ctx: [])
    eng = Engine(
        wirings=[
            ClassWiring(asset_class=AssetClass.CRYPTO, paper_adapter=crypto_adapter,
                        risk=circuit, strategies=(cs,), instruments=(BTC,)),
            ClassWiring(asset_class=AssetClass.EQUITY, paper_adapter=equity_adapter,
                        risk=circuit, strategies=(es,), instruments=(NIFTY,)),
        ],
        store=store, gate=gate, clock=clock, env={})
    asyncio.run(eng.tick_once(AssetClass.CRYPTO))
    # 12:00 UTC is 17:30 IST: the equity market is shut, so its tick is a no-op
    # while crypto keeps trading.
    asyncio.run(eng.tick_once(AssetClass.EQUITY))
    assert cs.calls == 1 and es.calls == 0
    assert len(inner.positions(Mode.PAPER, AssetClass.CRYPTO)) == 1
    assert eng.arm_report_text().count("mode=paper") == 2


def test_the_arming_vocabulary_still_lives_in_core_gate():
    # engine.py imports these private names deliberately so "what arms a class"
    # has ONE definition. If core.gate renames them, this test says so before
    # the two copies can disagree in production.
    import core.gate as gate_mod
    assert hasattr(gate_mod, "_ARM_ENV")
    assert hasattr(gate_mod, "_parse_arm_flag")
    assert set(gate_mod._ARM_ENV) == {AssetClass.EQUITY, AssetClass.CRYPTO}


def test_unmanaged_positions_keep_alarming_after_the_tick_that_killed_them():
    # The strategy dies while flat, so the disable-time alarm has nothing to
    # report. The position appears afterwards — and nothing is watching it.
    boom = ScriptedStrategy(
        "boom", AssetClass.CRYPTO,
        lambda ctx: (_ for _ in ()).throw(RuntimeError("bad maths")))
    h = build(strategies=[boom],
              config=EngineConfig(unmanaged_alarm_interval_s=1))

    def alarms() -> list[Event]:
        return [e for e in of(h, EventKind.ERROR)
                if e.data.get("where") == "unmanaged_positions"]

    tick(h)
    assert alarms() == []

    h.inner.apply(Fill(order_id="later", instrument=BTC, side=Side.BUY,
                       qty=Decimal("0.2"), price=Decimal("50000"),
                       fees=Decimal("0"), at=CRYPTO_NOON, mode=Mode.PAPER),
                  "boom")
    h.clock.advance(10)
    tick(h)
    assert len(alarms()) == 1
    assert alarms()[0].data["symbols"] == ["BTC/USDT"]

    # Loud, but rate-limited: an alarm on every 5s tick buries the day's other
    # events in the ring buffer.
    tick(h)
    assert len(alarms()) == 1
    h.clock.advance(10)
    tick(h)
    assert len(alarms()) == 2
    assert h.engine.status().classes[0].unmanaged_positions == ("BTC/USDT",)


def test_a_halted_class_stops_calling_the_venue_for_marks():
    # It keeps ticking so the circuit still sees the account and the operator
    # still gets a heartbeat, but a halted class must not poll a live venue for
    # prices nothing will act on.
    adapter = FakeAdapter(AssetClass.CRYPTO)
    calls: list[str] = []
    real_quote = adapter.quote
    adapter.quote = lambda inst: (calls.append(inst.symbol), real_quote(inst))[1]
    h = build(adapter=adapter)
    h.engine._runners[AssetClass.CRYPTO].halt("test")
    tick(h)
    assert calls == []
    ticks = of(h, EventKind.TICK)
    assert ticks and ticks[-1].data["halted"] is True
    # The circuit is still advanced: a halted book still moves in value.
    assert h.circuit.advances == 1
