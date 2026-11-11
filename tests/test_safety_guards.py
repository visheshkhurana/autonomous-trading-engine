"""Safety gates the risk officer asked for: native stops, heartbeat, blackout, key freeze.

Adapters are fakes. Nothing here opens a socket or signs a Kraken request.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

import engine as eng
from adapters.kraken import KrakenAdapter
from api.server import ApiSettings, create_app
from core.gate import ClassLimits, GateConfig, OrderGate
from core.risk import RiskCircuit
from core.safety import (
    blackout_active, blackout_causes, fetch_status_windows, key_freeze_active,
    load_configured_windows, overlap_stops_allowed, status_max_window,
    stop_qty_for_ledger, windows_from_status_payload,
)
from core.store import SQLitePositionStore
from core.types import (
    AssetClass, Fill, Instrument, Intent, MarketContext, Mode, OrderRef,
    OrderStatus, OrderType, Position, Side,
)
from tests.test_api_server import (
    AUTH, NOW as API_NOW, TOKEN, MemoryControlStore, build_handle,
)
from tests.test_okx_adapter import FILLED_ORDER, FakeClock, FakeExchange, spot

UTC = dt.timezone.utc
D = Decimal
WHEN = dt.datetime(2026, 8, 27, 12, 0, tzinfo=UTC)

BTC = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO,
                 broker_id="BTC/USDT", lot_size=1, tick_size=D("0.01"),
                 qty_precision=8, min_notional=D("5"))
ETH = Instrument(symbol="ETH/USDT", asset_class=AssetClass.CRYPTO,
                 broker_id="ETH/USDT", lot_size=1, tick_size=D("0.01"),
                 qty_precision=8, min_notional=D("5"))


class Clock:
    def __init__(self, at: dt.datetime = WHEN) -> None:
        self.at = at

    def __call__(self) -> dt.datetime:
        return self.at

    def advance(self, seconds: float) -> None:
        self.at = self.at + dt.timedelta(seconds=seconds)


class StopBook:
    """Paper-shaped adapter that records stop lifecycle without a wallet size."""

    manages_native_stops = True
    asset_class = AssetClass.CRYPTO
    # This book does not lock stop funds inside `free`. Tests that model
    # Kraken set the flag on the instance.
    stops_lock_base = False

    def __init__(self, mode: Mode = Mode.PAPER, *, free: D = D("0.098"),
                 wallet: D = D("0.088")) -> None:
        self.mode = mode
        self.free: Optional[D] = free
        self.wallet = wallet  # must never become an order qty
        self.fill_at: Optional[dt.datetime] = None
        self.price = D("50000")
        self.fail_stop = False
        self.stops: dict[str, tuple[D, D]] = {}
        self.log: list[tuple] = []
        self._n = 0
        self.timeouts: list[float] = []
        self.filled_on_cancel = False
        self.fail_next = False
        self.unconfirmed_next = False
        self.sticky_unconfirmed = False
        self.sticky_ids: set[str] = set()
        self.filled_reports: dict[str, Fill] = {}
        # Sum of qtys if a new stop is placed while another is still open.
        self.concurrent: list[D] = []

    def equity(self) -> D:
        return D("100000")

    def quote(self, instrument: Instrument) -> D:
        return self.price

    def positions(self):
        return ()

    def cancel(self, order_id: str) -> bool:
        return False

    def free_base(self, instrument: Instrument) -> Optional[D]:
        return self.free

    def place(self, intent: Intent, timeout_s: float = 30.0) -> OrderRef:
        self.log.append(("place", intent.side.value, intent.qty, intent.instrument.symbol))
        fill = Fill(
            order_id=f"ORD-{len(self.log)}", instrument=intent.instrument,
            side=intent.side, qty=intent.qty, price=self.price, fees=D("0"),
            at=self.fill_at or WHEN, mode=self.mode,
        )
        return OrderRef(
            order_id=fill.order_id, status=OrderStatus.FILLED, intent=intent,
            mode=self.mode, filled_qty=intent.qty, avg_price=self.price,
            fills=(fill,),
        )

    def place_native_stop(self, instrument, qty, stop_price, timeout_s=10.0,
                          fill_at=None, use_cached_balance=False):
        from core.safety import StopPlaceResult
        self.timeouts.append(timeout_s)
        self.log.append(("stop", qty, stop_price, instrument.symbol))
        if self.stops:
            self.concurrent.append(sum(q for q, _ in self.stops.values()) + qty)
        if self.fail_next:
            self.fail_next = False
            return StopPlaceResult(False, None, qty, stop_price, "rejected")
        if self.unconfirmed_next:
            self.unconfirmed_next = False
            self._n += 1
            txid = f"STOP-{self._n}"
            self.stops[txid] = (qty, stop_price)
            if self.sticky_unconfirmed:
                self.sticky_ids.add(txid)
            return StopPlaceResult(False, txid, qty, stop_price, "not confirmed")
        if self.fail_stop:
            return StopPlaceResult(False, None, qty, stop_price, "not on open orders")
        self._n += 1
        txid = f"STOP-{self._n}"
        self.stops[txid] = (qty, stop_price)
        return StopPlaceResult(True, txid, qty, stop_price, "open")

    def native_stop_open(self, txid, instrument):
        return txid in self.stops

    def describe_native_stop(self, txid, instrument):
        from core.safety import StopCancelResult
        if txid in self.stops:
            return StopCancelResult("still_open", reason="resting")
        fill = self.filled_reports.get(txid)
        if fill is not None:
            return StopCancelResult("filled", fill=fill, reason="filled away")
        return StopCancelResult("unknown", reason="not on the book")

    def cancel_native_stop(self, txid, instrument):
        from core.safety import StopCancelResult
        self.log.append(("cancel", txid))
        if txid in self.sticky_ids:
            return StopCancelResult("unknown", reason="cancel unconfirmed")
        qty, price = self.stops.pop(txid, (None, None))
        if qty is None:
            return StopCancelResult("unknown", reason="gone")
        if self.filled_on_cancel:
            fill = Fill(
                order_id=txid, instrument=instrument, side=Side.SELL,
                qty=qty, price=price, fees=D("0"), at=WHEN, mode=self.mode,
            )
            return StopCancelResult("filled", fill=fill, reason="already filled")
        return StopCancelResult("cancelled", reason="cancelled")


class NamingStrategy:
    def __init__(self, intents=None, stop: Optional[D] = D("48000")) -> None:
        self.id = "s1"
        self.asset_class = AssetClass.CRYPTO
        self.intents = list(intents or [])
        self.stop = stop
        self.calls = 0

    def propose(self, ctx: MarketContext):
        self.calls += 1
        return list(self.intents)

    def protective_stop(self, ctx, position) -> Optional[D]:
        return self.stop


def _buy(qty="0.01", stop="48000", key="k1", instrument=BTC) -> Intent:
    return Intent(
        instrument=instrument, side=Side.BUY, qty=D(qty),
        order_type=OrderType.MARKET, strategy_id="s1", reason="test entry",
        limit_price=D("50000"), idempotency_key=key, stop_price=D(stop),
    )


def _sell(qty="0.01", key="x1", instrument=BTC) -> Intent:
    return Intent(
        instrument=instrument, side=Side.SELL, qty=D(qty),
        order_type=OrderType.MARKET, strategy_id="s1", reason="test exit",
        limit_price=D("50000"), idempotency_key=key,
    )


def _engine(tmp_path, adapter, strategy, *, env=None, live=False, clock=None,
            blackout_status_fetch=False):
    clock = clock or Clock()
    env = {} if env is None else dict(env)
    store = SQLitePositionStore(str(tmp_path / "book.db"))
    mode = Mode.LIVE if live else Mode.PAPER
    if live:
        env["CRYPTO_LIVE"] = "1"
    circuit = RiskCircuit(store, mode, clock=clock)
    gate = OrderGate(
        position_store=store, gate_state=_Gate(), risk_circuit=circuit,
        config=GateConfig(
            requested_mode=mode,
            default_limits=ClassLimits(
                max_order_notional=D("100000"),
                max_daily_notional=D("100000"),
            ),
        ),
        env=env,
    )
    paper = adapter if not live else StopBook(Mode.PAPER)
    live_adapter = adapter if live else None
    if live:
        adapter.mode = Mode.LIVE
    engine = eng.Engine(
        wirings=[eng.ClassWiring(
            asset_class=AssetClass.CRYPTO,
            paper_adapter=paper,
            live_adapter=live_adapter,
            risk=circuit,
            strategies=(strategy,),
            instruments=(BTC, ETH),
        )],
        store=store, gate=gate,
        config=eng.EngineConfig(
            requested_mode=mode,
            close_store_on_stop=False,
            watchdog_enabled=False,
            blackout_status_fetch=blackout_status_fetch,
        ),
        clock=clock, env=env,
    )
    return engine, store, clock


class _Gate:
    def __init__(self) -> None:
        self.seen: dict[str, dt.datetime] = {}

    def seen_recently(self, key, window_s, now):
        ts = self.seen.get(key)
        return ts is not None and (now - ts).total_seconds() < window_s

    def order_count(self, asset_class, day_key):
        return 0

    def daily_notional(self, asset_class, day_key):
        return D("0")

    def record_submission(self, key, asset_class, day_key, now, notional=D("0")):
        self.seen[key] = now


def _tick(engine):
    asyncio.run(engine.tick_once(AssetClass.CRYPTO))


def _rejections(engine):
    return [e for e in engine.events.recent() if e.kind is eng.EventKind.REJECTION]


# ---------------------------------------------------------------- native stop


def test_fill_is_followed_by_a_stop_sized_to_the_ledger_within_the_confirm_window(tmp_path):
    book = StopBook(free=D("0.098"), wallet=D("0.088"))
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        stops = [row for row in book.log if row[0] == "stop"]
        assert stops, "entry fill must be followed by a native stop"
        _kind, qty, price, symbol = stops[0]
        assert qty == D("0.01")
        assert qty != book.wallet
        assert qty != book.free
        assert price == D("48000")
        assert symbol == "BTC/USDT"
        assert book.timeouts[0] <= 10
        position = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
        assert position.stop_txid
        assert position.stop_qty == D("0.01")
        assert position.stop_status == "open"
        assert position.qty == D("0.01")
    finally:
        store.close()


def test_stop_that_is_not_confirmed_flattens_the_ledger_qty_and_halts_entries(tmp_path):
    book = StopBook()
    book.fail_stop = True
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells, "an unconfirmed stop must market-flatten"
        assert sells[0][2] == D("0.01")
        assert sells[0][2] != book.wallet
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO) == []
        strategy.intents = [_buy(key="k2")]
        _tick(engine)
        reasons = [e.data.get("reason") for e in _rejections(engine)]
        assert "native_stop_unconfirmed" in reasons
        assert not any(row[0] == "place" and row[1] == "buy" and row[3] == "BTC/USDT"
                       for row in book.log[2:])
    finally:
        store.close()


def test_engine_exit_cancels_the_native_stop_before_selling(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        assert any(row[0] == "stop" for row in book.log)
        strategy.intents = [_sell()]
        _tick(engine)
        cancel_at = max(i for i, row in enumerate(book.log) if row[0] == "cancel")
        exit_places = [i for i, row in enumerate(book.log)
                       if row[0] == "place" and row[1] == "sell"]
        assert exit_places and exit_places[0] > cancel_at
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO) == []
    finally:
        store.close()


def test_a_stop_that_already_filled_is_not_sold_again(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        book.filled_on_cancel = True
        strategy.intents = [_sell()]
        _tick(engine)
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells == []
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO) == []
    finally:
        store.close()


def test_a_full_size_replace_cancels_before_placing_so_two_stops_never_rest(tmp_path):
    """Two ledger-sized stops would be 2x the book. That is not allowed,
    even when raw free balance looks large enough to hold both."""
    book = StopBook(free=D("0.098"), wallet=D("0.088"))
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        strategy.intents = []
        strategy.stop = D("49000")
        _tick(engine)
        tail = book.log[book.log.index(("stop", D("0.01"), D("48000"), "BTC/USDT")) + 1:]
        assert tail[0][0] == "cancel"
        assert tail[1] == ("stop", D("0.01"), D("49000"), "BTC/USDT")
        assert book.concurrent == []
        position = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
        assert position.stop_price == D("49000")
        assert position.qty == D("0.01")
        assert len(book.stops) == 1
    finally:
        store.close()


def test_startup_without_a_native_stop_flattens_a_live_position(tmp_path):
    book = StopBook(mode=Mode.LIVE)
    strategy = NamingStrategy([], stop=None)
    engine, store, _clock = _engine(tmp_path, book, strategy, live=True)
    try:
        store.apply(Fill(
            order_id="seed", instrument=BTC, side=Side.BUY, qty=D("0.01"),
            price=D("50000"), fees=D("0"), at=WHEN, mode=Mode.LIVE,
        ), "s1")
        _tick(engine)
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells and sells[0][2] == D("0.01")
        assert sells[0][2] != book.wallet
        assert store.positions(Mode.LIVE, AssetClass.CRYPTO) == []
    finally:
        store.close()


def test_manual_reserve_blocks_a_second_stop_funded_by_that_balance(tmp_path):
    """free = reserve + ledger. Non-reserved free equals the ledger, so two
    simultaneous stops (2x ledger) cannot be funded without the manual BTC.
    """
    reserved = D("0.0880257350")
    ledger = D("0.01")
    book = StopBook(free=reserved + ledger, wallet=reserved)
    strategy = NamingStrategy([_buy(qty=str(ledger))], stop=D("48000"))
    engine, store, _clock = _engine(
        tmp_path, book, strategy,
        env={"MANUAL_RESERVED_BASE": f"BTC:{reserved}"},
    )
    try:
        _tick(engine)
        assert book.stops
        strategy.intents = []
        strategy.stop = D("49000")
        _tick(engine)
        assert book.concurrent == [], (
            "a second stop must not rest while the first still claims the ledger"
        )
        kinds = [row[0] for row in book.log]
        cancel_at = kinds.index("cancel")
        second_stop = max(i for i, row in enumerate(book.log) if row[0] == "stop")
        assert cancel_at < second_stop
        open_qty = sum(q for q, _ in book.stops.values())
        non_reserved = book.free - reserved
        assert open_qty <= ledger
        assert open_qty <= non_reserved
        assert open_qty != reserved
        assert len(book.stops) == 1
    finally:
        store.close()


def test_a_fee_sized_shortfall_clips_the_stop_and_syncs_the_ledger(tmp_path):
    reserved = D("0.0880257350")
    ledger = D("0.01")
    # 1% of the ledger is gone to a base-asset fee. Non-reserved free is short
    # by less than 2%, so the stop shrinks instead of flattening.
    book = StopBook(free=reserved + ledger * D("0.99"), wallet=reserved)
    strategy = NamingStrategy([_buy(qty=str(ledger))], stop=D("48000"))
    engine, store, _clock = _engine(
        tmp_path, book, strategy,
        env={"MANUAL_RESERVED_BASE": f"BTC:{reserved}"},
    )
    try:
        _tick(engine)
        clipped = ledger * D("0.99")
        stops = [row for row in book.log if row[0] == "stop"]
        assert stops[-1][1] == clipped
        assert not any(row[0] == "place" and row[1] == "sell" for row in book.log)
        position = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
        assert position.qty == clipped
        assert position.stop_qty == clipped
        assert position.stop_status == "open"
    finally:
        store.close()


def test_a_shortfall_over_two_percent_flattens_and_halts(tmp_path):
    reserved = D("0.0880257350")
    ledger = D("0.01")
    book = StopBook(free=reserved + ledger * D("0.97"), wallet=reserved)
    strategy = NamingStrategy([_buy(qty=str(ledger))], stop=D("48000"))
    engine, store, _clock = _engine(
        tmp_path, book, strategy,
        env={"MANUAL_RESERVED_BASE": f"BTC:{reserved}"},
    )
    try:
        _tick(engine)
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells and sells[0][2] == ledger
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO) == []
        strategy.intents = [_buy(qty=str(ledger), key="k2")]
        _tick(engine)
        reasons = [e.data.get("reason") for e in _rejections(engine)]
        assert "native_stop_unconfirmed" in reasons
    finally:
        store.close()


def test_replace_that_cannot_place_the_new_stop_flattens_and_halts(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        book.fail_next = True
        strategy.intents = []
        strategy.stop = D("49000")
        _tick(engine)
        kinds = [row[0] for row in book.log]
        assert kinds.index("cancel") < len(kinds) - 1
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells and sells[-1][2] == D("0.01")
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO) == []
    finally:
        store.close()


# ---------------------------------------------------------------- heartbeat


def test_stale_heartbeat_blocks_entries_but_not_exits(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()])
    clock = Clock()
    engine, store, clock = _engine(tmp_path, book, strategy, clock=clock)
    try:
        _tick(engine)
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO)
        clock.advance(90)
        strategy.intents = [_sell(), _buy(key="k-eth", instrument=ETH, qty="0.02")]
        _tick(engine)
        reasons = [e.data.get("reason") for e in _rejections(engine)]
        assert "stale_heartbeat" in reasons
        eth_buys = [row for row in book.log
                    if row[0] == "place" and row[1] == "buy" and row[3] == "ETH/USDT"]
        assert eth_buys == []
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells, "a stale loop must still allow the exit"
    finally:
        store.close()


# ---------------------------------------------------------------- blackout / freeze


def test_blackout_blocks_entries_and_leaves_the_native_stop(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()])
    clock = Clock()
    window = json.dumps([{
        "start": "2026-08-27T00:00:00+00:00",
        "end": "2026-08-28T00:00:00+00:00",
    }])
    engine, store, _clock = _engine(
        tmp_path, book, strategy, clock=clock,
        env={"BLACKOUT_WINDOWS": window},
    )
    try:
        _tick(engine)
        assert not any(row[0] == "place" for row in book.log)
        reasons = [e.data.get("reason") for e in _rejections(engine)]
        assert "blackout" in reasons
    finally:
        store.close()


def test_blackout_keeps_the_native_stop_and_still_allows_an_exit(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()])
    engine, store, clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        assert book.stops
        txid = next(iter(book.stops))
        clock.advance(1)
        strategy.intents = []
        engine.safety_env = {
            "BLACKOUT_WINDOWS": json.dumps([{
                "start": "2026-08-27T00:00:00+00:00",
                "end": "2026-08-28T00:00:00+00:00",
            }]),
        }
        _tick(engine)
        assert txid in book.stops, "blackout must leave the resting stop in place"
        assert not any(row[0] == "cancel" for row in book.log)
        strategy.intents = [_sell()]
        clock.advance(1)
        _tick(engine)
        assert any(row[0] == "place" and row[1] == "sell" for row in book.log)
    finally:
        store.close()


def test_key_freeze_blocks_new_orders(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()])
    engine, store, clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        assert book.stops
        clock.advance(1)
        engine.safety_env = {"KEY_FREEZE": "1"}
        strategy.intents = [_sell(), _buy(key="k2", instrument=ETH, qty="0.02")]
        before = len(book.log)
        _tick(engine)
        assert book.log[before:] == [], "key freeze must not place, cancel, or replace"
        assert book.stops, "a stop already on the book stays there"
        reasons = [e.data.get("reason") for e in _rejections(engine)]
        assert reasons
        assert all(r == "key_freeze" for r in reasons)
    finally:
        store.close()


def test_key_freeze_file_blocks_kraken_private_order_calls(tmp_path):
    freeze = tmp_path / "KEY_FREEZE"
    freeze.write_text("1")
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT")},
        tickers={"BTC/USDT": {"last": 100}},
        balance={"free": {"BTC": "0.098"}, "total": {"BTC": "0.098"}},
        order_states=[FILLED_ORDER],
    )
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE,
        key_freeze=lambda: key_freeze_active({"KEY_FREEZE_PATH": str(freeze)}),
        monotonic=FakeClock().monotonic, sleep=FakeClock().sleep,
        now=lambda: WHEN,
    )
    intent = Intent(
        instrument=BTC, side=Side.BUY, qty=D("0.01"), order_type=OrderType.MARKET,
        strategy_id="s1", reason="frozen", limit_price=D("100"),
        idempotency_key="frozen-buy",
    )
    ref = adapter.place(intent)
    assert ref.status is OrderStatus.REJECTED
    assert "key_freeze" in (ref.error or "")
    assert ex.create_calls == []
    stopped = adapter.place_native_stop(BTC, D("0.01"), D("90"))
    assert stopped.ok is False
    assert ex.create_calls == []
    assert adapter.cancel("OID-1") is False
    assert ex.cancel_calls == []


def test_kraken_stop_uses_ledger_qty_not_the_manual_btc_balance():
    engine_qty = D("0.001")
    manual = D("0.088")
    free = manual + engine_qty
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT", amount_step="0.00000001", price_step="0.1",
                          min_amount="0.00001")},
        tickers={"BTC/USDT": {"last": 100000}},
        balance={"free": {"BTC": format(free, "f"), "USDT": "10000"},
                 "total": {"BTC": format(free, "f"), "USDT": "10000"}},
    )
    ex.open_orders = []

    def fetch_open_orders(symbol=None):
        return list(ex.open_orders)

    def create_order(symbol, type_, side, amount, price=None, params=None):
        ex.create_calls.append((symbol, type_, side, amount, price, params))
        ex.open_orders.append({
            "id": "STOP-1", "symbol": symbol, "status": "open",
            "amount": amount, "type": type_,
        })
        return {"id": "STOP-1"}

    ex.fetch_open_orders = fetch_open_orders
    ex.create_order = create_order
    clock = FakeClock()
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, poll_interval_s=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep, now=lambda: WHEN,
        key_freeze=lambda: False,
    )
    result = adapter.place_native_stop(BTC, engine_qty, D("90000"), timeout_s=10)
    assert result.ok is True
    assert result.txid == "STOP-1"
    _symbol, typ, side, amount, price, _params = ex.create_calls[0]
    assert typ == "stop-loss"
    assert side == "sell"
    assert D(str(amount)) == engine_qty
    assert D(str(amount)) != manual
    assert D(str(amount)) != free
    assert clock.t < 10


def test_kraken_stop_that_never_appears_on_open_orders_is_not_ok():
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT")},
        tickers={"BTC/USDT": {"last": 100}},
        balance={"free": {"BTC": "1"}, "total": {"BTC": "1"}},
    )
    ex.open_orders = []
    ex.fetch_open_orders = lambda symbol=None: []
    ex.fetch_order = lambda order_id, symbol=None: {
        "id": order_id, "status": "open", "amount": "0.001", "filled": "0",
    }
    # Visible via fetch_order counts as confirmed. A book that stays empty
    # AND an order that is not open must time out.
    ex.fetch_order = lambda order_id, symbol=None: {
        "id": order_id, "status": "canceled", "amount": "0.001", "filled": "0",
    }
    clock = FakeClock()
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, poll_interval_s=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep, now=lambda: WHEN,
        key_freeze=lambda: False,
    )
    result = adapter.place_native_stop(BTC, D("0.001"), D("90"), timeout_s=10)
    assert result.ok is False
    assert clock.t >= 10


# ---------------------------------------------------------------- pure policy


def test_configured_windows_apply_and_a_bad_file_does_not_fail_closed(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{
        "start": "2026-11-12T00:00:00Z",
        "end": "2026-11-12T02:00:00Z",
        "cancel_before": "15m",
    }]))
    windows, err = load_configured_windows({
        "BLACKOUT_WINDOWS_FILE": str(good),
        "BLACKOUT_WINDOWS": "{not json",
    })
    assert err
    assert len(windows) == 1
    assert windows[0].start == dt.datetime(2026, 11, 11, 23, 45, tzinfo=UTC)
    missing, miss_err = load_configured_windows({
        "BLACKOUT_WINDOWS_FILE": str(tmp_path / "nope.json"),
    })
    assert missing == []
    assert miss_err


def test_status_page_windows_are_padded_and_a_failed_fetch_adds_none():
    payload = {"scheduled_maintenances": [{
        "name": "REST API maintenance",
        "scheduled_for": "2026-11-12T12:00:00.000Z",
        "scheduled_until": "2026-11-12T13:00:00.000Z",
        "components": [{"name": "REST"}],
    }]}
    windows = windows_from_status_payload(payload)
    assert windows[0].start == dt.datetime(2026, 11, 12, 11, 45, tzinfo=UTC)
    assert windows[0].end == dt.datetime(2026, 11, 12, 13, 30, tzinfo=UTC)
    fetched, error = fetch_status_windows(opener=lambda url: (_ for _ in ()).throw(RuntimeError("down")))
    assert fetched == []
    assert error


STATUS_FIXTURE = Path(__file__).parent / "fixtures" / "kraken_status_2026_11_11.json"


def _status_fixture() -> dict:
    return json.loads(STATUS_FIXTURE.read_text(encoding="utf-8"))


def _windows_from_fixture(doc: dict, **kw) -> list:
    windows = []
    for key in ("active", "upcoming"):
        windows.extend(windows_from_status_payload(doc[key], **kw))
    return windows


def test_today_status_page_ignores_the_delisting_and_ends_stale_api_maintenance():
    """The 11 Nov 2026 status page, as published.

    The delisting notice is scheduled_for 24 Sep 14:25Z through scheduled_until
    10 Dec 15:05Z. The previous gate padded that to 14:10Z–15:35Z and blocked
    every new entry inside it. It must not be a blackout.

    The WebSocket/FIX/REST maintenance is still ``in_progress`` after 14:00Z.
    It ends at scheduled_until plus the 30 minute pad (14:30Z), not "until
    Kraken edits the page".
    """
    windows = _windows_from_fixture(_status_fixture())
    names = {w.name for w in windows}
    assert names == {
        "WebSocket, FIX + REST API Maintenance",
        "Derivatives Platform Maintenance",
        "Kraken Website and API Maintenance",
    }
    api = next(w for w in windows if w.name.startswith("WebSocket"))
    assert api.end == dt.datetime(2026, 11, 11, 14, 30, tzinfo=UTC)
    assert api.start == dt.datetime(2026, 11, 11, 6, 45, tzinfo=UTC)
    assert api.contains(dt.datetime(2026, 11, 11, 13, 0, tzinfo=UTC))
    # Still inside the 30 minute pad, even though scheduled_until has passed.
    assert api.contains(dt.datetime(2026, 11, 11, 14, 15, tzinfo=UTC))
    assert not api.contains(dt.datetime(2026, 11, 11, 14, 30, tzinfo=UTC))
    stale = dt.datetime(2026, 11, 11, 14, 36, tzinfo=UTC)
    assert not api.contains(stale)
    assert blackout_active(stale, windows) is False
    # Dual-investments (15:00Z) and the banking window are not trading.
    assert blackout_active(dt.datetime(2026, 11, 11, 15, 5, tzinfo=UTC), windows) is False
    assert blackout_active(dt.datetime(2026, 11, 21, 12, 0, tzinfo=UTC), windows) is False
    # The old padded delisting bounds themselves are not a blackout.
    assert blackout_active(dt.datetime(2026, 9, 24, 14, 10, tzinfo=UTC), windows) is False
    assert blackout_active(dt.datetime(2026, 12, 10, 15, 35, tzinfo=UTC), windows) is False

    website = next(w for w in windows if "Website and API" in w.name)
    assert website.start == dt.datetime(2026, 11, 12, 7, 46, tzinfo=UTC)
    assert website.end == dt.datetime(2026, 11, 12, 8, 46, tzinfo=UTC)
    assert website.contains(dt.datetime(2026, 11, 12, 7, 50, tzinfo=UTC))
    assert website.contains(dt.datetime(2026, 11, 12, 8, 10, tzinfo=UTC))
    assert website.contains(dt.datetime(2026, 11, 12, 8, 40, tzinfo=UTC))
    assert not website.contains(dt.datetime(2026, 11, 12, 7, 45, tzinfo=UTC))
    assert not website.contains(dt.datetime(2026, 11, 12, 8, 46, tzinfo=UTC))


def test_a_short_delisting_and_asset_rails_are_not_blackouts():
    """The 12 hour cap is not what drops a delisting. The subject is."""
    short = {"scheduled_maintenances": [{
        "name": "Notice of scheduled asset delistings - August 2026",
        "status": "scheduled",
        "scheduled_for": "2026-11-11T14:00:00.000Z",
        "scheduled_until": "2026-11-11T14:30:00.000Z",
        "components": [{"name": "REST"}],
    }, {
        "name": "API3 (API3) - Ethereum funding",
        "status": "scheduled",
        "scheduled_for": "2026-11-11T14:00:00.000Z",
        "scheduled_until": "2026-11-11T14:30:00.000Z",
        "components": [{"name": "API3 (API3) - Ethereum"}],
    }, {
        "name": "Website maintenance",
        "status": "scheduled",
        "scheduled_for": "2026-11-11T14:00:00.000Z",
        "scheduled_until": "2026-11-11T14:15:00.000Z",
        "components": [{"name": "Website"}],
    }, {
        "name": "US derivatives maintenance",
        "status": "scheduled",
        "scheduled_for": "2026-11-11T14:00:00.000Z",
        "scheduled_until": "2026-11-11T14:15:00.000Z",
        "components": [{"name": "Order Placement"}],
    }]}
    windows = windows_from_status_payload(short)
    assert [w.name for w in windows] == ["US derivatives maintenance"]
    assert windows[0].start == dt.datetime(2026, 11, 11, 13, 45, tzinfo=UTC)
    assert windows[0].end == dt.datetime(2026, 11, 11, 14, 45, tzinfo=UTC)


def test_status_windows_longer_than_the_cap_are_ignored_and_the_cap_is_configurable():
    twelve = {
        "name": "REST API maintenance",
        "status": "scheduled",
        "scheduled_for": "2026-11-11T07:00:00.000Z",
        "scheduled_until": "2026-11-11T19:00:00.000Z",
        "components": [{"name": "REST"}],
    }
    just_over = dict(twelve, scheduled_until="2026-11-11T19:00:01.000Z")
    kept = windows_from_status_payload({"scheduled_maintenances": [twelve]})
    assert len(kept) == 1
    assert kept[0].end == dt.datetime(2026, 11, 11, 19, 30, tzinfo=UTC)
    assert windows_from_status_payload({"scheduled_maintenances": [just_over]}) == []
    assert status_max_window({}) == dt.timedelta(hours=12)
    wider = status_max_window({"KRAKEN_STATUS_MAX_WINDOW_H": "13"})
    assert windows_from_status_payload(
        {"scheduled_maintenances": [just_over]}, max_window=wider)
    doc = _status_fixture()

    def opener(url: str) -> dict:
        if url.endswith("/active.json"):
            return doc["active"]
        return doc["upcoming"]

    fetched, error = fetch_status_windows(
        opener, env={"KRAKEN_STATUS_MAX_WINDOW_H": "6"})
    assert error is None
    assert "WebSocket, FIX + REST API Maintenance" not in {w.name for w in fetched}
    assert "Kraken Website and API Maintenance" in {w.name for w in fetched}


def test_configured_blackout_applies_when_the_status_page_does_not():
    doc = _status_fixture()
    status = _windows_from_fixture(doc)
    now = dt.datetime(2026, 11, 11, 14, 36, tzinfo=UTC)
    assert blackout_active(now, status) is False
    configured, err = load_configured_windows({
        "BLACKOUT_WINDOWS": json.dumps([{
            "start": "2026-09-24T14:10:00Z",
            "end": "2026-12-10T15:35:00Z",
            "name": "owner halt",
        }]),
    })
    assert err is None
    assert len(configured) == 1
    combined = list(configured) + status
    assert blackout_active(now, combined) is True
    assert blackout_causes(now, combined) == [
        {"source": "configured", "name": "owner halt"},
    ]
    failed, fetch_err = fetch_status_windows(
        opener=lambda url: (_ for _ in ()).throw(RuntimeError("down")))
    assert failed == []
    assert fetch_err
    assert blackout_active(now, list(configured) + failed) is True


def test_engine_refresh_drops_the_delisting_and_honours_the_window_cap(tmp_path):
    doc = _status_fixture()

    def opener(url: str) -> dict:
        if url.endswith("/active.json"):
            return doc["active"]
        if url.endswith("/upcoming.json"):
            return doc["upcoming"]
        raise AssertionError(url)

    engine, store, _clock = _engine(
        tmp_path, StopBook(), NamingStrategy([]),
        env={"KRAKEN_STATUS_MAX_WINDOW_H": "6"},
        blackout_status_fetch=True,
    )
    engine._status_fetcher = opener
    try:
        engine.refresh_blackout()
        names = {w.name for w in engine._status_windows}
        assert "WebSocket, FIX + REST API Maintenance" not in names
        assert "Notice of scheduled asset delistings - August 2026" not in names
        assert "Kraken Website and API Maintenance" in names
        assert engine.blackout_is_active(dt.datetime(2026, 11, 11, 14, 36, tzinfo=UTC)) is False
        assert engine.blackout_is_active(dt.datetime(2026, 11, 12, 8, 0, tzinfo=UTC)) is True
        engine._status_fetcher = lambda url: (_ for _ in ()).throw(RuntimeError("down"))
        engine._status_fetched_mono = None
        engine.refresh_blackout()
        assert {w.name for w in engine._status_windows} == names
        assert engine._blackout_status_error
    finally:
        store.close()


def test_two_stops_cannot_exceed_ledger_qty_from_non_reserved_balance():
    reserved = D("0.0880257350")
    ledger = D("0.001")
    # Wallet is the manual stack plus the engine position. Non-reserved
    # free is exactly the ledger, which cannot fund two ledger-sized stops.
    free = reserved + ledger
    assert overlap_stops_allowed(
        free=free, reserved=reserved, old_qty=ledger, new_qty=ledger,
        ledger_qty=ledger,
    ) is False
    # Even a fat non-reserved balance cannot justify two stops whose sum
    # is larger than the ledger. The extra coins are not this position.
    assert overlap_stops_allowed(
        free=reserved + ledger * 5, reserved=reserved,
        old_qty=ledger, new_qty=ledger, ledger_qty=ledger,
    ) is False
    # A pair whose sum is the ledger, fully covered by non-reserved free, may overlap.
    assert overlap_stops_allowed(
        free=reserved + ledger, reserved=reserved,
        old_qty=D("0.0004"), new_qty=D("0.0006"), ledger_qty=ledger,
    ) is True
    clip, decision = stop_qty_for_ledger(ledger, ledger * D("0.99"))
    assert decision == "clip"
    assert clip == ledger * D("0.99")
    _qty, halt = stop_qty_for_ledger(ledger, ledger * D("0.97"))
    assert halt == "halt"


def test_kraken_sell_and_stop_cannot_spend_the_manual_reserve():
    reserved = D("0.0880257350")
    ledger = D("0.001")
    free = reserved + ledger
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT", amount_step="0.00000001", price_step="0.1",
                          min_amount="0.00000001")},
        tickers={"BTC/USDT": {"last": 100000}},
        balance={"free": {"BTC": format(free, "f"), "USDT": "10000"},
                 "total": {"BTC": format(free, "f"), "USDT": "10000"}},
        order_states=[dict(FILLED_ORDER, filled=float(ledger), amount=float(ledger),
                           average=100000, cost=100000 * float(ledger))],
    )
    clock = FakeClock()
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, poll_interval_s=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep, now=lambda: WHEN,
        key_freeze=lambda: False,
        manual_reserved=f"BTC:{reserved}",
    )
    # An intent for the whole wallet must be clipped to the non-reserved ledger.
    sell = Intent(
        instrument=BTC, side=Side.SELL, qty=free, order_type=OrderType.MARKET,
        strategy_id="s1", reason="sell the wallet", limit_price=D("100000"),
        idempotency_key="sell-wallet",
    )
    ref = adapter.place(sell)
    assert ref.status is OrderStatus.FILLED
    sent = D(str(ex.create_calls[0][3]))
    assert sent == ledger
    assert sent != free
    assert sent != reserved
    # A stop 3% above non-reserved free is not sent.
    ex.create_calls.clear()
    refused = adapter.place_native_stop(BTC, ledger / D("0.97"), D("90000"))
    assert refused.ok is False
    assert ex.create_calls == []
    # A 1% fee gap is clipped onto the wire, not flattened by the adapter.
    ex.open_orders = []
    ex.fetch_open_orders = lambda symbol=None: list(ex.open_orders)

    def create_order(symbol, type_, side, amount, price=None, params=None):
        ex.create_calls.append((symbol, type_, side, amount, price, params))
        ex.open_orders.append({"id": "STOP-CLIP", "symbol": symbol, "status": "open",
                               "amount": amount})
        return {"id": "STOP-CLIP"}

    ex.create_order = create_order
    clipped = adapter.place_native_stop(BTC, ledger / D("0.99"), D("90000"))
    assert clipped.ok is True
    assert D(str(ex.create_calls[-1][3])) == ledger


def test_a_stop_that_filled_off_book_is_booked_before_any_new_sell(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        txid = next(iter(book.stops))
        qty, price = book.stops.pop(txid)
        book.filled_reports[txid] = Fill(
            order_id=txid, instrument=BTC, side=Side.SELL, qty=qty, price=price,
            fees=D("1.25"), at=WHEN, mode=Mode.PAPER,
        )
        strategy.intents = []
        stops_before = [row for row in book.log if row[0] == "stop"]
        _tick(engine)
        assert [row for row in book.log if row[0] == "stop"] == stops_before
        assert not any(row[0] == "place" and row[1] == "sell" for row in book.log)
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO) == []
        assert any(f.order_id == txid and f.fees == D("1.25")
                   for f in store.fills(Mode.PAPER))
    finally:
        store.close()


def test_an_unreadable_missing_stop_is_not_replaced(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        txid = next(iter(book.stops))
        book.stops.pop(txid)
        strategy.intents = []
        stops_before = len([row for row in book.log if row[0] == "stop"])
        _tick(engine)
        assert len([row for row in book.log if row[0] == "stop"]) == stops_before
        position = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
        assert position.stop_txid == txid
        assert position.qty == D("0.01")
    finally:
        store.close()


def test_an_unconfirmed_overlap_cancel_keeps_the_old_stop_id(tmp_path, monkeypatch):
    """If the old stop cannot be confirmed cancelled, its txid stays put
    and the replacement is rolled back instead of becoming the only record."""
    book = StopBook(free=D("1"))
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    monkeypatch.setattr(eng, "overlap_stops_allowed", lambda **_kw: True)
    try:
        _tick(engine)
        position = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
        old = position.stop_txid
        book.sticky_ids.add(old)
        strategy.intents = []
        strategy.stop = D("49000")
        _tick(engine)
        kept = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
        assert kept.stop_txid == old
        assert old in book.stops
        # The replacement was cancelled; it must not be the stored id.
        assert not (kept.stop_status or "").startswith("overlap:")
        assert any(row[0] == "cancel" and row[1] != old for row in book.log)
    finally:
        store.close()


def test_kraken_stop_fill_keeps_the_venue_fee():
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT")},
        tickers={"BTC/USDT": {"last": 90000}},
    )
    ex.fetch_order = lambda order_id, symbol=None: {
        "id": order_id, "status": "closed", "filled": "0.001", "amount": "0.001",
        "average": "90000", "symbol": "BTC/USDT",
        "fee": {"cost": "1.25", "currency": "USDT"},
    }
    clock = FakeClock()
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, poll_interval_s=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep, now=lambda: WHEN,
        key_freeze=lambda: False,
    )
    described = adapter.describe_native_stop("STOP-FEE", BTC)
    assert described.outcome == "filled"
    assert described.fill is not None
    assert described.fill.fees == D("1.25")
    assert described.fill.qty == D("0.001")


def test_key_freeze_env_and_file(tmp_path):
    assert key_freeze_active({"KEY_FREEZE": "1"}) is True
    assert key_freeze_active({"KEY_FREEZE": "0"}) is False
    path = tmp_path / "KEY_FREEZE"
    assert key_freeze_active({"KEY_FREEZE_PATH": str(path)}) is False
    path.write_text("")
    assert key_freeze_active({"KEY_FREEZE_PATH": str(path)}) is True


def test_store_keeps_the_stop_txid_on_the_position_row(tmp_path):
    store = SQLitePositionStore(str(tmp_path / "stops.db"))
    try:
        store.apply(Fill(
            order_id="o1", instrument=BTC, side=Side.BUY, qty=D("0.01"),
            price=D("50000"), fees=D("0"), at=WHEN, mode=Mode.PAPER,
        ), "s1")
        store.set_native_stop(
            Mode.PAPER, "BTC/USDT", "s1", txid="STOP-9",
            stop_price=D("48000"), stop_qty=D("0.01"), status="open",
        )
        store.apply(Fill(
            order_id="o2", instrument=BTC, side=Side.BUY, qty=D("0.01"),
            price=D("50000"), fees=D("0"), at=WHEN, mode=Mode.PAPER,
        ), "s1")
        position = store.positions(Mode.PAPER)[0]
        assert position.qty == D("0.02")
        assert position.stop_txid == "STOP-9"
    finally:
        store.close()


# ---------------------------------------------------------------- api surface


def test_state_reports_heartbeat_blackout_freeze_and_native_stop(tmp_path):
    beat = (API_NOW - dt.timedelta(seconds=90)).astimezone(UTC).isoformat()
    control = MemoryControlStore()
    control.rows["__engine__"] = {"heartbeat_at": beat, "paused": "0"}
    position = Position(
        instrument=BTC, qty=D("0.01"), avg_price=D("50000"),
        mode=Mode.PAPER, strategy_id="s1",
        stop_txid="STOP-1", stop_price=D("48000"), stop_qty=D("0.01"),
        stop_status="open",
    )
    handle = build_handle(
        ledger=_Ledger([position]), control=control, clock=lambda: API_NOW,
    )
    window = json.dumps([{
        "start": "2026-08-27T00:00:00+00:00",
        "end": "2026-08-28T00:00:00+00:00",
    }])
    app = create_app(handle, ApiSettings(token=TOKEN), env={
        "BLACKOUT_WINDOWS": window,
        "KEY_FREEZE": "1",
    })
    body = TestClient(app).get("/api/state", headers=AUTH).json()
    assert body["key_freeze"] is True
    assert body["entries_allowed"] is False
    assert body["entries_block_reason"] == "key_freeze"
    assert body["heartbeat_age_s"] == pytest.approx(90, abs=1)
    assert body["blackout_active"] is True
    assert body["blackout_causes"] == [{"source": "configured", "name": ""}]
    assert body["next_window"]["source"] == "config"
    native = body["classes"]["crypto"]["positions"][0]["native_stop"]
    assert native["status"] == "open"
    assert native["txid"] == "STOP-1"
    assert native["qty"] == "0.01"
    ready = TestClient(app).get("/ready", headers=AUTH).json()
    assert ready["entries_allowed"] is False
    assert ready["entries_block_reason"] == "key_freeze"


def test_state_names_the_status_page_incident_and_drops_a_cached_delisting():
    """A cache written before the filter still names today's delisting.

    The old gate stored it as 24 Sep 14:10Z–10 Dec 15:35Z (the published
    span plus padding). /api/state must not report that as the cause.
    """
    moment = dt.datetime(2026, 11, 11, 14, 36, tzinfo=UTC)
    control = MemoryControlStore()
    control.rows["__engine__"] = {
        "heartbeat_at": moment.isoformat(),
        "paused": "0",
        "blackout_status_json": json.dumps([
            {
                "start": "2026-09-24T14:10:00+00:00",
                "end": "2026-12-10T15:35:00+00:00",
                "source": "kraken_status",
                "name": "Notice of scheduled asset delistings - August 2026",
            },
            {
                "start": "2026-11-11T00:00:00+00:00",
                "end": "2026-11-11T16:00:00+00:00",
                "source": "kraken_status",
                "name": "REST API maintenance",
            },
            {
                "start": "2026-11-12T07:46:00+00:00",
                "end": "2026-11-12T08:46:00+00:00",
                "source": "kraken_status",
                "name": "Kraken Website and API Maintenance",
            },
        ]),
    }

    class _Clock:
        def __init__(self, at: dt.datetime) -> None:
            self.at = at

        def __call__(self) -> dt.datetime:
            return self.at

    clock = _Clock(moment)
    handle = build_handle(control=control, clock=clock)
    app = create_app(handle, ApiSettings(token=TOKEN), env={})
    client = TestClient(app)
    body = client.get("/api/state", headers=AUTH).json()
    assert body["blackout_active"] is False
    assert body["blackout_causes"] == []
    assert body["entries_allowed"] is True
    assert body["next_window"]["name"] == "Kraken Website and API Maintenance"
    assert body["next_window"]["source"] == "kraken_status"

    clock.at = dt.datetime(2026, 11, 12, 8, 0, tzinfo=UTC)
    control.rows["__engine__"]["heartbeat_at"] = clock.at.isoformat()
    during = client.get("/api/state", headers=AUTH).json()
    assert during["blackout_active"] is True
    assert during["blackout_causes"] == [{
        "source": "status_page",
        "name": "Kraken Website and API Maintenance",
    }]
    assert during["entries_block_reason"] == "blackout"

    both = create_app(handle, ApiSettings(token=TOKEN), env={
        "BLACKOUT_WINDOWS": json.dumps([{
            "start": "2026-11-12T08:00:00Z",
            "end": "2026-11-12T09:00:00Z",
            "name": "owner halt",
        }]),
    })
    stacked = TestClient(both).get("/api/state", headers=AUTH).json()
    assert stacked["blackout_causes"] == [
        {"source": "status_page", "name": "Kraken Website and API Maintenance"},
        {"source": "configured", "name": "owner halt"},
    ]


def test_health_and_live_fail_when_the_heartbeat_is_past_the_dead_threshold():
    control = MemoryControlStore()
    control.rows["__engine__"] = {
        "heartbeat_at": (API_NOW - dt.timedelta(seconds=181)).astimezone(UTC).isoformat(),
    }
    handle = build_handle(control=control, clock=lambda: API_NOW)
    app = create_app(handle, ApiSettings(token=TOKEN), env={})
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    assert client.get("/health").json()["status"] == "stale_heartbeat"
    assert client.get("/live").status_code == 503
    fresh = MemoryControlStore()
    fresh.rows["__engine__"] = {"heartbeat_at": API_NOW.astimezone(UTC).isoformat()}
    alive = create_app(
        build_handle(control=fresh, clock=lambda: API_NOW),
        ApiSettings(token=TOKEN), env={},
    )
    assert TestClient(alive).get("/health").json() == {"status": "ok"}


class _Ledger:
    def __init__(self, positions) -> None:
        self._positions = positions

    def positions(self, mode, asset_class=None):
        rows = [p for p in self._positions if p.mode is mode]
        if asset_class is not None:
            rows = [p for p in rows if p.instrument.asset_class is asset_class]
        return rows

    def orders(self, mode, asset_class=None, since_iso=None, limit=None):
        return []

    def realized_pnl(self, mode, asset_class, since_iso):
        return D("0")


# ---------------------------------------------------------------- fill-to-rest


def test_fill_to_rest_timestamps_and_log_line(tmp_path, caplog):
    book = StopBook()
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        with caplog.at_level(logging.INFO, logger="trader.engine"):
            _tick(engine)
        position = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
        assert position.stop_fill_at == WHEN
        assert position.stop_placed_at == WHEN
        assert position.stop_confirmed_at == WHEN
        assert position.origin == "engine_fill"
        rested = [e for e in engine.events.recent() if "native_stop rested" in e.message]
        assert rested, "a confirmed stop must emit the rested line"
        assert "txid=" in rested[0].message
        assert "fill_to_rest_ms=" in rested[0].message
        assert "native_stop rested BTC/USDT" in caplog.text
        native = {
            "fill_at": position.stop_fill_at.astimezone(UTC).isoformat(),
            "placed_at": position.stop_placed_at.astimezone(UTC).isoformat(),
            "confirmed_at": position.stop_confirmed_at.astimezone(UTC).isoformat(),
        }
        assert native["fill_at"].startswith("2026-08-27T12:00:00")
        assert "fill_to_rest_ms=0" in rested[0].message
    finally:
        store.close()


def test_stop_deadline_is_measured_from_the_fill_not_submission(tmp_path):
    book = StopBook()
    book.fill_at = WHEN - dt.timedelta(seconds=12)
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        assert book.timeouts == [], "a fill already outside the 10s budget must not start a fresh window"
        assert book.stops == {}
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells and sells[0][2] == D("0.01")
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO) == []
    finally:
        store.close()


def test_kraken_stop_confirm_budget_shrinks_by_the_fill_age():
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT", amount_step="0.00000001", price_step="0.1",
                          min_amount="0.00000001")},
        tickers={"BTC/USDT": {"last": 100000}},
        balance={"free": {"BTC": "1"}, "total": {"BTC": "1"}},
    )
    ex.open_orders = []
    ex.fetch_open_orders = lambda symbol=None: []
    ex.fetch_order = lambda order_id, symbol=None: {
        "id": order_id, "status": "canceled", "amount": "0.001", "filled": "0",
    }

    def create_order(symbol, type_, side, amount, price=None, params=None):
        ex.create_calls.append((symbol, type_, side, amount, price, params))
        return {"id": "STOP-LATE"}

    ex.create_order = create_order
    clock = FakeClock()
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, poll_interval_s=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep, now=lambda: WHEN,
        key_freeze=lambda: False,
    )
    fill_at = WHEN - dt.timedelta(seconds=9)
    result = adapter.place_native_stop(
        BTC, D("0.001"), D("90000"), timeout_s=10, fill_at=fill_at)
    assert result.ok is False
    assert ex.create_calls, "budget remaining must still allow the order"
    assert clock.t < 10
    assert clock.t >= 1


def test_unreadable_balance_never_sends_an_unchecked_stop():
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT", amount_step="0.00000001", price_step="0.1",
                          min_amount="0.00000001")},
        tickers={"BTC/USDT": {"last": 100000}},
        balance={"free": {"BTC": "1"}, "total": {"BTC": "1"}},
    )
    ex.balance_error = RuntimeError("balance down")
    clock = FakeClock()
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, poll_interval_s=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep, now=lambda: WHEN,
        key_freeze=lambda: False,
    )
    result = adapter.place_native_stop(BTC, D("0.001"), D("90000"))
    assert result.ok is False
    assert ex.create_calls == []
    assert "reserve" in result.reason or "unreadable" in result.reason


def test_engine_flattens_when_free_balance_cannot_be_read(tmp_path):
    book = StopBook()
    book.free = None
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        assert not any(row[0] == "stop" for row in book.log)
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells and sells[0][2] == D("0.01")
    finally:
        store.close()


def test_resting_stop_above_spendable_is_clipped_or_flattened(tmp_path):
    book = StopBook(free=D("0.02"))
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        book.free = D("0.004")
        strategy.intents = []
        strategy.stop = D("48000")
        _tick(engine)
        stops = [row for row in book.log if row[0] == "stop"]
        assert stops[-1][1] == D("0.004")
        assert not any(row[0] == "place" and row[1] == "sell" for row in book.log)
        position = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
        assert position.qty == D("0.004")
        assert position.stop_qty == D("0.004")
        check = json.loads(position.reserve_check)
        assert check["action"] == "clipped"
        book.fail_next = True
        book.free = D("0.001")
        _tick(engine)
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells and sells[-1][2] == D("0.001")
    finally:
        store.close()


class _ReconBook(StopBook):
    def free_by_asset(self):
        return {"BTC": self.free if self.free is not None else D("0")}


def test_ledger_qty_mismatch_blocks_entries(tmp_path):
    book = _ReconBook(free=D("0.01"))
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        assert store.positions(Mode.PAPER, AssetClass.CRYPTO)
        book.free = D("0.05")
        strategy.intents = [_buy(key="k2")]
        _tick(engine)
        reasons = [e.data.get("reason") for e in _rejections(engine)]
        assert "ledger_mismatch" in reasons
        runner = engine._runners[AssetClass.CRYPTO]
        assert runner._ledger_mismatch
        assert any(row.get("reason") == "qty"
                   for row in runner._ledger_mismatch["mismatches"])
    finally:
        store.close()


def test_foreign_position_origin_blocks_entries(tmp_path):
    book = StopBook()
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        store._conn.execute(
            "UPDATE positions SET origin=? WHERE symbol=?",
            ("import", "BTC/USDT"),
        )
        strategy.intents = [_buy(key="k-eth", instrument=ETH, qty="0.02")]
        _tick(engine)
        reasons = [e.data.get("reason") for e in _rejections(engine)]
        assert "ledger_mismatch" in reasons
        eth_buys = [row for row in book.log
                    if row[0] == "place" and row[1] == "buy" and row[3] == "ETH/USDT"]
        assert eth_buys == []
    finally:
        store.close()


def test_state_exposes_stop_times_reserve_check_and_ledger_mismatch(tmp_path):
    position = Position(
        instrument=BTC, qty=D("0.01"), avg_price=D("50000"),
        mode=Mode.PAPER, strategy_id="s1",
        stop_txid="STOP-1", stop_price=D("48000"), stop_qty=D("0.01"),
        stop_status="open",
        stop_fill_at=WHEN, stop_placed_at=WHEN, stop_confirmed_at=WHEN,
        reserve_check=json.dumps({
            "action": "clipped", "spendable": "0.004", "stop_qty": "0.01",
        }),
    )
    control = MemoryControlStore()
    control.rows["__engine__"] = {
        "heartbeat_at": API_NOW.astimezone(UTC).isoformat(),
        "paused": "0",
        "entry_halt_reason": "ledger_mismatch",
        "ledger_mismatch_json": json.dumps({
            "mismatches": [{"asset": "BTC", "reason": "qty", "diff": "0.04"}],
        }),
    }
    handle = build_handle(
        ledger=_Ledger([position]), control=control, clock=lambda: API_NOW,
    )
    app = create_app(handle, ApiSettings(token=TOKEN), env={})
    body = TestClient(app).get("/api/state", headers=AUTH).json()
    assert body["entries_block_reason"] == "ledger_mismatch"
    assert body["ledger_mismatch"]["mismatches"][0]["asset"] == "BTC"
    native = body["classes"]["crypto"]["positions"][0]["native_stop"]
    assert native["fill_at"].startswith("2026-08-27T12:00:00")
    assert native["placed_at"].startswith("2026-08-27T12:00:00")
    assert native["confirmed_at"].startswith("2026-08-27T12:00:00")
    assert native["reserve_check"]["action"] == "clipped"


def test_reserved_base_is_excluded_from_drawdown_only_when_opted_in(tmp_path):
    from core.risk import DEFAULT_LIMITS, equity_for_drawdown
    markets = {"BTC/USDT": spot("BTC/USDT"), "ETH/USDT": spot("ETH/USDT")}
    tickers = {"BTC/USDT": {"last": 60000.0}, "ETH/USDT": {"last": 3000.0}}
    ex = FakeExchange(
        markets, tickers=tickers,
        balance={"total": {"USDT": 1000.0, "BTC": 0.5},
                 "free": {"USDT": 1000.0, "BTC": 0.5}},
    )
    plain = KrakenAdapter(
        ex, mode=Mode.LIVE, now=lambda: WHEN, key_freeze=lambda: False,
        manual_reserved="BTC:0.1", risk_exclude_reserved=False,
    )
    opted = KrakenAdapter(
        ex, mode=Mode.LIVE, now=lambda: WHEN, key_freeze=lambda: False,
        manual_reserved="BTC:0.1", risk_exclude_reserved=True,
    )
    # 0.5 BTC * 60000 + 1000 USDT. The 0.1 BTC reserve is 6000.
    assert plain.equity() == D("31000")
    assert opted.equity() == D("25000")
    limits = DEFAULT_LIMITS[AssetClass.CRYPTO]
    store = SQLitePositionStore(str(tmp_path / "risk.db"))
    try:
        circuit = RiskCircuit(store, Mode.PAPER, clock=lambda: WHEN)
        seeded = circuit.advance(AssetClass.CRYPTO, D("31000"))
        assert seeded.action.value == "ok"
        net = equity_for_drawdown(D("31000"), D("6000"), exclude_reserved=True)
        decision = circuit.advance(AssetClass.CRYPTO, net)
        assert net == D("25000")
        assert decision.drawdown_pct == (D("31000") - D("25000")) / D("31000")
        # Ladder thresholds and the stored high-water are unchanged.
        assert circuit._limits[AssetClass.CRYPTO] == limits
        assert circuit.read(AssetClass.CRYPTO, D("31000")).drawdown_pct == D("0")
        untouched = equity_for_drawdown(D("31000"), D("6000"), exclude_reserved=False)
        assert untouched == D("31000")
    finally:
        store.close()


def test_enabling_reserved_exclusion_rebases_the_high_water_once(tmp_path, caplog):
    """A 62% reserve must not look like a drawdown the moment the flag turns on."""
    from core.risk import CircuitAction
    store = SQLitePositionStore(str(tmp_path / "rebase.db"))
    try:
        circuit = RiskCircuit(store, Mode.PAPER, clock=lambda: WHEN)
        full = D("100000")
        engine_only = D("38000")  # 62% of the account is the manual reserve
        seeded = circuit.advance(AssetClass.CRYPTO, full)
        assert seeded.drawdown_pct == D("0")
        with caplog.at_level(logging.INFO, logger="trader.risk"):
            rebased = circuit.advance(
                AssetClass.CRYPTO, engine_only, "engine_only")
        assert rebased.action is CircuitAction.OK
        assert rebased.drawdown_pct == D("0")
        assert rebased.daily_loss_pct == D("0")
        row = store.read_risk_state(Mode.PAPER, AssetClass.CRYPTO)
        assert row is not None
        assert row["high_water"] == "38000"
        assert row["equity_basis"] == "engine_only"
        audit = json.loads(row["hwm_rebase_json"])
        assert audit["old_high_water"] == "100000"
        assert audit["new_high_water"] == "38000"
        assert "rebased once" in caplog.text
        # A later dip uses the rebased mark. It does not rebase again.
        later = circuit.advance(AssetClass.CRYPTO, D("37000"), "engine_only")
        kept = store.read_risk_state(Mode.PAPER, AssetClass.CRYPTO)
        assert kept is not None and kept["high_water"] == "38000"
        assert later.drawdown_pct == (D("38000") - D("37000")) / D("38000")
        # Flag off never moves an existing mark down.
        plain = RiskCircuit(store, Mode.LIVE, clock=lambda: WHEN)
        plain.advance(AssetClass.CRYPTO, full)
        plain.advance(AssetClass.CRYPTO, engine_only)
        live = store.read_risk_state(Mode.LIVE, AssetClass.CRYPTO)
        assert live is not None and live["high_water"] == "100000"
        assert live["equity_basis"] == "full"
    finally:
        store.close()


def test_cancel_that_is_not_confirmed_does_not_sell(tmp_path):
    book = StopBook(free=D("0.02"))
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        txid = next(iter(book.stops))
        book.sticky_ids.add(txid)
        book.free = D("0.004")
        strategy.intents = []
        strategy.stop = D("48000")
        _tick(engine)
        assert not any(row[0] == "place" and row[1] == "sell" for row in book.log)
        assert txid in book.stops
        assert engine._runners[AssetClass.CRYPTO]._entry_halt_reason == "native_stop_unconfirmed"
    finally:
        store.close()


def test_unconfirmed_replacement_is_cancelled_before_any_flatten(tmp_path):
    book = StopBook(free=D("0.02"))
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        book.free = D("0.004")
        book.unconfirmed_next = True
        strategy.intents = []
        strategy.stop = D("48000")
        _tick(engine)
        cancels = [row[1] for row in book.log if row[0] == "cancel"]
        assert len(cancels) >= 2
        sells = [row for row in book.log if row[0] == "place" and row[1] == "sell"]
        assert sells, "a confirmed cancel of the replacement may then flatten"
        assert book.stops == {}
        # The replacement cancel is the last cancel and it precedes the sell.
        last_cancel = max(i for i, row in enumerate(book.log) if row[0] == "cancel")
        first_sell = min(i for i, row in enumerate(book.log)
                         if row[0] == "place" and row[1] == "sell")
        assert last_cancel < first_sell
    finally:
        store.close()


def test_unconfirmed_replacement_that_cannot_be_cancelled_does_not_sell(tmp_path):
    book = StopBook(free=D("0.02"))
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        book.free = D("0.004")
        book.unconfirmed_next = True
        book.sticky_unconfirmed = True
        strategy.intents = []
        strategy.stop = D("48000")
        _tick(engine)
        assert not any(row[0] == "place" and row[1] == "sell" for row in book.log)
        assert book.stops, "the unconfirmed replacement is still on the book"
        assert engine._runners[AssetClass.CRYPTO]._entry_halt_reason == "native_stop_unconfirmed"
    finally:
        store.close()


def test_reconcile_and_recheck_match_whether_or_not_stops_lock_funds(tmp_path):
    """Held: free excludes the resting stop. Not held: free already includes it."""
    # Not held. free == ledger. Adding the stop qty would false-mismatch.
    plain = _ReconBook(free=D("0.01"))
    assert plain.stops_lock_base is False
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, plain, strategy)
    try:
        _tick(engine)
        assert engine._runners[AssetClass.CRYPTO]._entry_halt_reason != "ledger_mismatch"
        assert len(plain.stops) == 1
        strategy.intents = []
        _tick(engine)
        assert engine._runners[AssetClass.CRYPTO]._ledger_mismatch is None
        assert not any(row[0] == "cancel" for row in plain.log)
    finally:
        store.close()


def test_held_stop_funds_are_added_back_for_reconcile_and_recheck(tmp_path):
    book = _ReconBook(free=D("0.01"))
    book.stops_lock_base = True
    strategy = NamingStrategy([_buy()], stop=D("48000"))
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        # Venue moved the position into the stop. free no longer includes it.
        book.free = D("0")
        strategy.intents = []
        strategy.stop = D("48000")
        _tick(engine)
        runner = engine._runners[AssetClass.CRYPTO]
        assert runner._ledger_mismatch is None
        assert runner._entry_halt_reason != "ledger_mismatch"
        assert len(book.stops) == 1
        assert not any(row[0] == "place" and row[1] == "sell" for row in book.log)
        # A foreign-sized free shortfall that the stop does not explain still clips.
        # free 0 and stop 0.01 is explained. Drop the stop qty by shrinking free
        # below zero is impossible; raise the reserve instead via env on a new tick.
    finally:
        store.close()


def test_trade_level_base_fee_shrinks_the_cached_balance_once():
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT")},
        tickers={"BTC/USDT": {"last": 100000}},
        balance={"free": {"BTC": "0.088"}, "total": {"BTC": "0.088"}},
    )
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, now=lambda: WHEN, key_freeze=lambda: False,
        monotonic=FakeClock().monotonic, sleep=FakeClock().sleep,
    )
    adapter._free_cache = {"BTC": D("0.088")}
    adapter._free_cache_mono = 0.0
    adapter._free_adjust = {}
    trade_only = {
        "filled": "1",
        "trades": [{
            "amount": "1",
            "fee": {"cost": "0.004", "currency": "BTC"},
        }],
    }
    adapter._apply_fill_to_cache(trade_only, "BTC", Side.BUY, D("1"))
    assert adapter._cached_amount("BTC") == D("0.088") + D("1") - D("0.004")
    adapter._free_adjust = {}
    adapter._free_cache = {"BTC": D("0.088")}
    both = {
        "filled": "1",
        "fee": {"cost": "0.004", "currency": "BTC"},
        "trades": [{
            "amount": "1",
            "fee": {"cost": "0.004", "currency": "BTC"},
        }],
    }
    adapter._apply_fill_to_cache(both, "BTC", Side.BUY, D("1"))
    assert adapter._cached_amount("BTC") == D("0.088") + D("1") - D("0.004")


def test_sell_cap_adds_only_our_open_stop(tmp_path):
    reserved = D("0.088")
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT", amount_step="0.00000001", price_step="0.1",
                          min_amount="0.00000001")},
        tickers={"BTC/USDT": {"last": 100000}},
        balance={"free": {"BTC": format(reserved, "f"), "USDT": "10000"},
                 "total": {"BTC": format(reserved + D("0.02"), "f"), "USDT": "10000"}},
        order_states=[dict(FILLED_ORDER, filled=0.01, amount=0.01, average=100000,
                           cost=1000)],
    )
    ex.open_orders = [
        {"id": "STOP-OURS", "symbol": "BTC/USDT", "status": "open", "amount": "0.01"},
        {"id": "STOP-THEIRS", "symbol": "BTC/USDT", "status": "open", "amount": "0.05"},
    ]
    ex.fetch_open_orders = lambda symbol=None: list(ex.open_orders)

    def create_order(symbol, type_, side, amount, price=None, params=None):
        ex.create_calls.append((symbol, type_, side, amount, price, params))
        return {"id": "SELL-1", "status": "closed", "filled": amount, "average": "100000",
                "amount": amount}

    ex.create_order = create_order
    ex.fetch_order = lambda order_id, symbol=None: {
        "id": order_id, "status": "closed", "filled": "0.01", "amount": "0.01",
        "average": "100000", "symbol": "BTC/USDT",
    }
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, now=lambda: WHEN, key_freeze=lambda: False,
        manual_reserved=f"BTC:{reserved}",
        monotonic=FakeClock().monotonic, sleep=FakeClock().sleep,
    )
    adapter.note_engine_stops([("BTC/USDT", "STOP-OURS", D("0.01"))])
    sell = Intent(
        instrument=BTC, side=Side.SELL, qty=D("0.02"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="sell",
        limit_price=D("100000"), idempotency_key="sell-cap",
    )
    ref = adapter.place(sell)
    assert ref.status is OrderStatus.FILLED
    sent = D(str(ex.create_calls[0][3]))
    # free is only the reserve. Our open stop of 0.01 is added back. Theirs is not.
    assert sent == D("0.01")
    assert sent != D("0.06")


def test_fill_age_is_subtracted_from_the_stop_budget_once(tmp_path):
    book = StopBook()
    book.fill_at = WHEN - dt.timedelta(seconds=6)
    strategy = NamingStrategy([_buy()])
    engine, store, _clock = _engine(tmp_path, book, strategy)
    try:
        _tick(engine)
        assert book.timeouts == [10]
        assert book.stops
        assert not any(row[0] == "place" and row[1] == "sell" for row in book.log)
    finally:
        store.close()
