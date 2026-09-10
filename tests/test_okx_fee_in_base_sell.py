"""
2 Sep 2026 incident: live crypto_trend bought ETH/USDT qty 0.02077 @ 2406.98.
OKX took the fee in ETH, so venue free was 0.02064538 while the engine book
stayed 0.02077. The strategy exit (and flatten) sent book qty. OKX 51008
InsufficientFunds. Position stayed open.

These tests reproduce that on a fake exchange — no network, no credentials,
no CRYPTO_LIVE / EQUITY_LIVE / TRADER_MODE. After the fix, a spot SELL is
min(book, venue free) and the leftover dust is written off so flatten cannot
stick on an unsellable stub.

Buys are not shrunk this way. Caps stay 50 USDT / clip, 250 / day, max 1 open.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from typing import Any, Mapping, Optional

import pytest

import engine as eng
import runner
from adapters.okx import OkxAdapter, OrderSubmitFailed, clip_spot_sell_qty
from core.gate import GateConfig, OrderGate
from core.risk import RiskCircuit
from core.store import SQLiteGateStateStore, SQLitePositionStore
from core.types import (
    AssetClass, Fill, Instrument, Intent, Mode, OrderStatus, OrderType, Side,
)
from strategies.base import BARS_KEY
from strategies.crypto_trend import CryptoEmaTrendStrategy
from tests import test_crypto_trend as ct
from tests._strategy_support import context, position
from tests.test_okx_adapter import FakeClock, FakeExchange, spot

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 2, 17, 0, 7, tzinfo=UTC)

BOOK = Decimal("0.02077")
FREE = Decimal("0.02064538")
MARK = Decimal("2406.98")
ORDER_ID = "3887485836056760320"

# Runner ETH: qty_precision=5, min_notional=5 USDT. Leftover after selling
# free is 0.00012462 ETH ≈ 0.30 USDT — below the 5 USDT floor, so it is dust.
# OKX ETH-USDT minSz is typically 0.00001; the leftover clears that but not
# min notional, which is what makes it an unstoppable stub if left on the book.
ETH = runner.default_instruments(AssetClass.CRYPTO)[1]


class InsufficientFunds(Exception):
    """ccxt-shaped OKX 51008. The venue raises this; the adapter wraps it."""


def _eth_market() -> dict[str, Any]:
    return spot(
        "ETH/USDT",
        amount_step="0.00000001",
        price_step="0.01",
        min_amount="0.00001",
        min_cost="5",
    )


class IncidentExchange(FakeExchange):
    """OKX as of the incident: selling more than free ETH is 51008.

    Amounts stay strings so a binary float cannot invent a size the venue
    would reject, or a fill the ledger would over-apply.
    """

    def __init__(self, *, free_eth: Decimal = FREE, mark: Decimal = MARK):
        super().__init__(
            {"ETH/USDT": _eth_market()},
            tickers={"ETH/USDT": {"last": format(mark, "f")}},
            balance={
                "total": {"USDT": "10000", "ETH": format(free_eth, "f")},
                "free": {"USDT": "10000", "ETH": format(free_eth, "f")},
            },
        )
        self.free_eth = free_eth
        self.mark = mark
        self.last_amount: Optional[Decimal] = None

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        amt = Decimal(str(amount))
        self.create_calls.append((symbol, type_, side, amount, price, params))
        if side == "sell" and amt > self.free_eth:
            raise InsufficientFunds(
                'okx {"code":"51008","msg":"InsufficientFunds. '
                'Order failed because you don’t have enough ETH."}'
            )
        if self.create_error:
            raise self.create_error
        self.last_amount = amt
        filled = format(amt, "f")
        mark = format(self.mark, "f")
        self.order_states = [{
            "id": ORDER_ID, "symbol": symbol, "status": "closed",
            "amount": filled, "filled": filled,
            "average": mark, "cost": format(amt * self.mark, "f"),
            "timestamp": 1756832407000,
            "fee": {"cost": "0", "currency": "USDT"},
        }]
        return {"id": ORDER_ID}


def _adapter(exchange: FakeExchange, **kw) -> OkxAdapter:
    clock = FakeClock()
    kw.setdefault("mode", Mode.PAPER)
    kw.setdefault("sandbox", True)
    kw.setdefault("poll_interval_s", 1.0)
    return OkxAdapter(
        exchange, monotonic=clock.monotonic, sleep=clock.sleep,
        now=lambda: NOW, **kw,
    )


def _sell_intent(qty: Decimal = BOOK, *, key: str = "exit-eth") -> Intent:
    return Intent(
        instrument=ETH, side=Side.SELL, qty=qty,
        order_type=OrderType.MARKET, strategy_id="crypto_trend",
        reason="incident exit", idempotency_key=key,
        limit_price=MARK,
    )


def _buy_intent(qty: Decimal = BOOK, *, key: str = "entry-eth") -> Intent:
    return Intent(
        instrument=ETH, side=Side.BUY, qty=qty,
        order_type=OrderType.MARKET, strategy_id="crypto_trend",
        reason="incident entry", idempotency_key=key,
        limit_price=MARK,
    )


def _sent_amount(ex: FakeExchange) -> Decimal:
    assert ex.create_calls, "expected a venue create_order"
    return Decimal(str(ex.create_calls[0][3]))


# --------------------------------------------------------------------------
# the venue rule, without the adapter
# --------------------------------------------------------------------------


def test_unclipped_book_qty_is_51008_at_the_venue() -> None:
    """Reproduce the incident: sending the engine book is InsufficientFunds."""
    ex = IncidentExchange()
    with pytest.raises(InsufficientFunds, match="51008"):
        ex.create_order("ETH/USDT", "market", "sell", format(BOOK, "f"), None, {})


def test_clip_spot_sell_qty_is_min_of_book_and_free() -> None:
    assert clip_spot_sell_qty(BOOK, FREE) == FREE
    assert clip_spot_sell_qty(FREE, FREE) == FREE
    assert clip_spot_sell_qty(Decimal("0.01"), Decimal("0.02")) == Decimal("0.01")
    assert clip_spot_sell_qty(BOOK, Decimal("0")) == Decimal("0")


# --------------------------------------------------------------------------
# adapter.place
# --------------------------------------------------------------------------


def test_spot_sell_is_clipped_to_free_eth_and_does_not_51008() -> None:
    ex = IncidentExchange()
    ref = _adapter(ex).place(_sell_intent())
    assert ref.status is OrderStatus.FILLED
    sent = _sent_amount(ex)
    assert sent == FREE
    assert sent <= FREE
    assert sent < BOOK
    assert ref.filled_qty == FREE
    assert "clipped" in (ref.error or "")


def test_spot_buy_is_not_shrunk_to_free_base() -> None:
    """Buys keep their sized qty. Free ETH is the coin we are about to acquire,
    not a cap on the order. Existing clip caps (50 USDT) are the only shrink."""
    ex = IncidentExchange(free_eth=Decimal("0"))
    ref = _adapter(ex).place(_buy_intent())
    assert ref.status is OrderStatus.FILLED
    assert _sent_amount(ex) == BOOK
    assert ex.create_calls[0][5].get("tgtCcy") == "base_ccy"


def test_unreadable_free_refuses_a_sell_rather_than_sending_book_qty() -> None:
    ex = IncidentExchange()
    ex.balance = {"total": {"ETH": format(FREE, "f")}}   # no `free` map
    ref = _adapter(ex).place(_sell_intent())
    assert ref.status is OrderStatus.REJECTED
    assert not ex.create_calls
    assert "unreadable" in (ref.error or "")


def test_free_balance_fetch_failure_refuses_the_sell() -> None:
    ex = IncidentExchange()
    ex.balance_error = RuntimeError("fetch_balance 500")
    ref = _adapter(ex).place(_sell_intent())
    assert ref.status is OrderStatus.REJECTED
    assert not ex.create_calls


def test_dust_leftover_after_a_clipped_sell_is_written_off_on_the_ref() -> None:
    leftover = BOOK - FREE
    assert leftover * MARK < ETH.min_notional

    ex = IncidentExchange()
    ref = _adapter(ex).place(_sell_intent())
    assert ref.filled_qty == FREE
    dust = [f for f in ref.fills if f.qty == leftover]
    assert len(dust) == 1
    assert dust[0].side is Side.SELL
    assert dust[0].fees == Decimal("0")
    assert "dust leftover" in (ref.error or "")


def test_a_sellable_remainder_is_not_written_off() -> None:
    """Half the coins locked, half free: leftover notional is well above the
    5 USDT floor. That is not dust; do not invent a close."""
    book = Decimal("1.0")
    free = Decimal("0.5")
    ex = IncidentExchange(free_eth=free)
    ref = _adapter(ex).place(_sell_intent(book, key="locked-half"))
    assert _sent_amount(ex) == free
    assert ref.filled_qty == free
    assert sum((f.qty for f in ref.fills), Decimal("0")) == free
    assert "dust leftover" not in (ref.error or "")


def test_place_still_raises_order_submit_failed_when_the_venue_rejects() -> None:
    """A clipped send that still dies in flight must not look like a rejection."""
    ex = IncidentExchange()
    ex.create_error = InsufficientFunds("okx 51008 after clip")
    # Force free >= book so the clip does not save us.
    ex.free_eth = BOOK
    ex.balance["free"]["ETH"] = format(BOOK, "f")
    with pytest.raises(OrderSubmitFailed):
        _adapter(ex).place(_sell_intent(key="still-fails"))


# --------------------------------------------------------------------------
# engine flatten + strategy exit
# --------------------------------------------------------------------------


class MemControl:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}

    def read(self, scope: str) -> Optional[Mapping[str, str]]:
        row = self.rows.get(scope)
        return dict(row) if row is not None else None

    def mutate(self, scope: str, fn) -> Mapping[str, str]:
        current = self.rows.get(scope)
        new = dict(fn(dict(current) if current is not None else None))
        self.rows[scope] = new
        return new

    def request_flatten(self, scope: str = "crypto",
                        request_id: str = "flatten-eth") -> None:
        row = self.rows.setdefault(scope, {"paused": "0"})
        row["paused"] = "1"
        row["paused_at"] = NOW.isoformat()
        row["flatten_request_id"] = request_id
        row["flatten_requested_at"] = NOW.isoformat()
        row["flatten_reason"] = "incident flatten"
        row.pop("flatten_acked_id", None)
        row.pop("flatten_acked_at", None)


class IncidentHarness:
    def __init__(self, tmp_path, *, strategies=(), extra_provider=None,
                 control: Optional[MemControl] = None,
                 clock_at: dt.datetime = NOW) -> None:
        path = str(tmp_path / "incident.db")
        self.at = clock_at
        self.store = SQLitePositionStore(path)
        self.gate_state = SQLiteGateStateStore(path, Mode.PAPER)
        self.circuit = RiskCircuit(self.store, Mode.PAPER, clock=lambda: self.at)
        self.gate = OrderGate(
            position_store=self.store, gate_state=self.gate_state,
            risk_circuit=self.circuit, config=GateConfig(), env={},
        )
        self.ex = IncidentExchange()
        self.adapter = _adapter(self.ex)
        self.control = control
        wiring = eng.ClassWiring(
            asset_class=AssetClass.CRYPTO, paper_adapter=self.adapter,
            risk=self.circuit, strategies=tuple(strategies),
            instruments=(ETH,), extra_provider=extra_provider,
        )
        self.engine = eng.Engine(
            wirings=[wiring], store=self.store, gate=self.gate,
            config=eng.EngineConfig(close_store_on_stop=False),
            clock=lambda: self.at, env={}, control=control,
        )

    def seed_book(self, strategy_id: str = "crypto_trend") -> None:
        self.store.apply(
            Fill(order_id="seed-eth", instrument=ETH, side=Side.BUY,
                 qty=BOOK, price=MARK, fees=Decimal("0"),
                 at=self.at, mode=Mode.PAPER),
            strategy_id=strategy_id,
        )

    def tick(self) -> None:
        asyncio.run(self.engine.tick_once(AssetClass.CRYPTO))

    def book(self):
        return list(self.store.positions(Mode.PAPER, AssetClass.CRYPTO))

    def close(self) -> None:
        self.store.close()
        self.gate_state.close()


def test_flatten_sends_no_more_than_free_eth_and_the_ledger_goes_flat(
        tmp_path) -> None:
    control = MemControl()
    h = IncidentHarness(tmp_path, control=control)
    try:
        h.seed_book()
        assert h.book()[0].qty == BOOK
        control.request_flatten()
        h.tick()
        assert _sent_amount(h.ex) == FREE
        assert _sent_amount(h.ex) <= FREE
        assert h.book() == [], (
            "fee-in-base leftover must be written off; an unsellable stub "
            "is how flatten got stuck on 2 Sep"
        )
    finally:
        h.close()


def test_strategy_exit_proposes_book_qty_and_the_adapter_clips_the_send(
        tmp_path) -> None:
    """crypto_trend._exit stays pure and still names the book. The clip is
    the adapter's: sending book qty is the 51008."""
    bars = ct.make_bars(ct.CROSS_DOWN, ct.utc(2026, 8, 24, 0, 0), ct.INTERVAL)
    strategy = CryptoEmaTrendStrategy(
        instruments=[ETH], fast_period=3, slow_period=5, atr_period=3,
        bar_interval=ct.INTERVAL,
    )
    proposed = list(strategy.propose(context(
        bars[-1].ts, bars={"ETH/USDT": bars},
        quotes={"ETH/USDT": MARK},
        positions=[position(ETH, BOOK, MARK, strategy_id="crypto_trend")],
    )))
    assert len(proposed) == 1
    assert proposed[0].side is Side.SELL
    assert proposed[0].qty == BOOK, "strategy must not silently shrink the exit"

    def extra(_ac, _now):
        return {BARS_KEY: {"ETH/USDT": bars}}

    h = IncidentHarness(
        tmp_path, strategies=[strategy], extra_provider=extra,
        clock_at=bars[-1].ts,
    )
    try:
        h.seed_book()
        h.tick()
        assert h.ex.create_calls, "the exit must reach the venue"
        assert _sent_amount(h.ex) == FREE
        assert _sent_amount(h.ex) <= FREE
        assert h.book() == []
    finally:
        h.close()


def test_caps_are_still_fifty_two_fifty_and_one_open() -> None:
    assert runner.CRYPTO_CLIP_USDT == Decimal("50")
    assert runner.CRYPTO_DAILY_USDT == Decimal("250")
    assert runner.CRYPTO_CLASS_LIMITS.max_open_positions == 1
    assert runner.CRYPTO_CLASS_LIMITS.max_order_notional == Decimal("50")
    assert runner.CRYPTO_CLASS_LIMITS.max_daily_notional == Decimal("250")
