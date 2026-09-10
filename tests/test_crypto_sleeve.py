"""
Vishesh's OKX sleeve — the proofs the rest of the book is not enough on
its own.

Caps (50 / 250 / max-1) live in the gate. Strategies are pure. Paper is
an adapter, not a flag. These tests stitch those pieces together and
fail if any one of them is decorative:

  * a paper fill happens without constructing a live venue
  * a missing or zero quote never becomes an order
  * a second open symbol is refused
  * the 50 / 250 USDT ceilings still bind through a real strategy intent
  * nothing in the OKX adapter can call withdraw / transfer / earn

No test here sets CRYPTO_LIVE, EQUITY_LIVE, or TRADER_MODE=live.
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
import sys
from decimal import Decimal
from pathlib import Path

import pytest

import runner
from adapters.okx import FundsMovementForbidden, OkxAdapter, is_funds_movement
from adapters.paper import InMemoryQuoteSource, PaperAdapter
from core.store import SQLitePositionStore
from core.types import (
    AssetClass, Fill, Intent, MarketContext, Mode, OrderStatus, OrderType, Side,
)
from strategies.base import BARS_KEY
from strategies.crypto_breakout import CryptoBreakoutStrategy
from strategies.crypto_trend import CryptoEmaTrendStrategy
from tests import test_crypto_breakout as bo
from tests import test_crypto_trend as ct
from tests import test_engine as eng_tests
from tests._strategy_support import coin, context
from tests.test_okx_adapter import FILLED_ORDER, FakeExchange, adapter, spot

CLIP = Decimal("50")
DAILY = Decimal("250")
BTC_PX = Decimal("78000")
ETH_PX = Decimal("2460")
NOW = dt.datetime(2026, 3, 2, 12, 0, tzinfo=dt.timezone.utc)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _btc():
    return runner.default_instruments(AssetClass.CRYPTO)[0]


def _eth():
    return runner.default_instruments(AssetClass.CRYPTO)[1]


def _intent(*, instrument=None, qty="1", key="sleeve", side=Side.BUY,
            strategy_id="crypto_trend") -> Intent:
    return Intent(
        instrument=instrument or _btc(),
        side=side,
        qty=Decimal(qty),
        order_type=OrderType.MARKET,
        strategy_id=strategy_id,
        reason="sleeve probe",
        idempotency_key=key,
    )


def _ctx(*, quotes=None, equity="2000000") -> MarketContext:
    return MarketContext(
        now=NOW,
        quotes=quotes if quotes is not None else {
            "BTC/USDT": BTC_PX, "ETH/USDT": ETH_PX,
        },
        equity=Decimal(equity),
    )


# --------------------------------------------------------------------------
# watchlist + wired strategies (extend, do not duplicate)
# --------------------------------------------------------------------------


def test_the_crypto_watchlist_is_btc_eth_and_sol_usdt() -> None:
    symbols = [i.symbol for i in runner.default_instruments(AssetClass.CRYPTO)]
    assert symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_the_wired_crypto_strategy_is_the_paper_mandate() -> None:
    """15m EMA-spray is no longer the default path. crypto_trend /
    crypto_breakout stay importable but are gated behind CRYPTO_EMA_SPRAY."""
    from strategies.crypto_mandate import CryptoMandateStrategy
    instruments = runner.default_instruments(AssetClass.CRYPTO)
    wired = runner.default_strategies(AssetClass.CRYPTO, instruments)
    assert [type(s) for s in wired] == [CryptoMandateStrategy]
    assert [s.id for s in wired] == ["crypto_mandate"]
    assert all(s.asset_class is AssetClass.CRYPTO for s in wired)


# --------------------------------------------------------------------------
# unknown quote is never a trade
# --------------------------------------------------------------------------


def test_both_crypto_strategies_skip_when_the_quote_is_absent() -> None:
    trend_ctx, _ = ct.ctx_from(ct.CROSS_UP, quotes={})
    break_ctx, _ = bo.make_ctx(bo.RANGE_ROWS + [bo.BREAKOUT_ROW], quotes={})
    assert list(ct.strat().propose(trend_ctx)) == []
    assert list(bo.strat().propose(break_ctx)) == []


def test_a_zero_quote_is_not_a_price_for_either_strategy() -> None:
    trend_ctx, _ = ct.ctx_from(ct.CROSS_UP, quotes={"BTC/USDT": Decimal("0")})
    break_ctx, _ = bo.make_ctx(
        bo.RANGE_ROWS + [bo.BREAKOUT_ROW], quotes={"ETH/USDT": Decimal("0")})
    assert list(ct.strat().propose(trend_ctx)) == []
    assert list(bo.strat().propose(break_ctx)) == []


def test_the_gate_refuses_a_market_order_when_the_quote_is_unknown(tmp_path) -> None:
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        approval = s.gate.check(_intent(), _ctx(quotes={}))
        assert not approval.approved
        assert any("price" in c.detail.lower() or "quote" in c.detail.lower()
                   for c in approval.rejections)
    finally:
        s.close()
        s.store.close()


def test_the_gate_refuses_a_market_order_when_the_quote_is_zero(tmp_path) -> None:
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        approval = s.gate.check(
            _intent(key="zero-quote"),
            _ctx(quotes={"BTC/USDT": Decimal("0"), "ETH/USDT": ETH_PX}),
        )
        assert not approval.approved
        assert any("price" in c.detail.lower() or "quote" in c.detail.lower()
                   for c in approval.rejections)
    finally:
        s.close()
        s.store.close()


def test_the_gate_stamps_the_last_known_quote_onto_a_naked_market_intent(
        tmp_path) -> None:
    """A raw MARKET (what crypto_trend used to emit) must leave the gate
    carrying the quote the drift guard needs. Sizing already used it."""
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        approval = s.gate.check(_intent(qty="1"), _ctx())
        assert approval.approved, approval.reason
        assert approval.intent.order_type is OrderType.MARKET
        assert approval.intent.limit_price == BTC_PX
        assert approval.intent.qty * BTC_PX <= CLIP
    finally:
        s.close()
        s.store.close()


# --------------------------------------------------------------------------
# 2 Sep live reject: MARKET + known quote must pass the drift guard
# --------------------------------------------------------------------------


def test_a_trend_entry_with_a_known_quote_is_not_rejected_for_missing_reference(
        tmp_path) -> None:
    """Reproduce the 2 Sep live book: EMA-cross-up, clip ~50 USDT
    (qty 0.000643), then OkxAdapter refused with
    'MARKET intent carries no reference price'. A known last quote must
    let the drift guard run; the order must be sent, not fail-closed for
    a missing reference."""
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        ctx, bars = ct.ctx_from(ct.CROSS_UP, quotes={"BTC/USDT": BTC_PX})
        # Bar close and last quote can diverge on a 15-min bar. The
        # drift guard must be bound to the quote, not the stale close.
        quote = BTC_PX
        assert bars[-1].close != quote
        intents = list(CryptoEmaTrendStrategy(
            instruments=[_btc()], fast_period=3, slow_period=5, atr_period=3,
            bar_interval=ct.INTERVAL,
        ).propose(context(
            ctx.now, bars=ctx.extra[BARS_KEY], quotes={"BTC/USDT": quote},
        )))
        assert len(intents) == 1
        assert intents[0].order_type is OrderType.MARKET
        assert intents[0].limit_price == quote

        approval = s.gate.check(intents[0], _ctx())
        assert approval.approved, approval.reason
        assert approval.mode is Mode.PAPER
        assert approval.intent.order_type is OrderType.MARKET
        assert approval.intent.limit_price == quote
        assert approval.intent.qty * quote <= CLIP
        assert approval.intent.qty < Decimal(1)
        # Live clip was 0.000643; 50 / 78000 = 0.000641 at 6dp.
        assert approval.intent.qty == Decimal("0.000641")

        ex = FakeExchange(
            {"BTC/USDT": spot("BTC/USDT")},
            tickers={"BTC/USDT": {"last": float(quote)}},
            order_states=[FILLED_ORDER],
        )
        ref = adapter(ex).place(approval.intent)
        assert "reference price" not in (ref.error or "").lower()
        assert "unbounded" not in (ref.error or "").lower()
        assert ex.create_calls, "a MARKET with a known quote must reach OKX"
        assert ex.create_calls[0][1] == "market"
    finally:
        s.close()
        s.store.close()


def test_okx_still_refuses_a_market_order_that_carries_no_reference() -> None:
    """Do not weaken the unbounded-order refuse. A MARKET that somehow
    still has no reference must not be sent, even if the ticker is live."""
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT")},
        tickers={"BTC/USDT": {"last": float(BTC_PX)}},
        order_states=[FILLED_ORDER],
    )
    ref = adapter(ex).place(_intent(qty="0.000641", key="unbounded"))
    assert ref.status is OrderStatus.REJECTED
    assert "drift guard" in (ref.error or "")
    assert not ex.create_calls


# --------------------------------------------------------------------------
# paper fills without a live venue
# --------------------------------------------------------------------------


def test_paper_path_fills_a_trend_entry_without_a_live_venue() -> None:
    """Strategy → gate → PaperAdapter. No OkxAdapter, no ccxt, no network."""
    bars = ct.make_bars(ct.CROSS_UP, ct.utc(2026, 8, 24, 0, 0), ct.INTERVAL)
    last = bars[-1]
    instrument = coin("BTC/USDT")
    strategy = CryptoEmaTrendStrategy(
        instruments=[instrument],
        fast_period=3, slow_period=5, atr_period=3,
        bar_interval=ct.INTERVAL,
    )

    def extra(_ac, _now):
        return {BARS_KEY: {"BTC/USDT": bars}}

    h = eng_tests.build(
        strategies=[strategy],
        instruments=(instrument,),
        prices={"BTC/USDT": last.close},
        cash=Decimal("2000000"),
        clock_at=last.ts,
        extra_provider=extra,
        limits=runner.CRYPTO_CLASS_LIMITS,
    )
    assert isinstance(h.adapter, PaperAdapter)
    assert h.adapter.mode is Mode.PAPER
    assert not isinstance(h.adapter, OkxAdapter)

    eng_tests.tick(h)

    held = h.inner.positions(Mode.PAPER, AssetClass.CRYPTO)
    assert len(held) == 1
    assert held[0].instrument.symbol == "BTC/USDT"
    assert held[0].qty > 0
    assert held[0].qty * last.close <= CLIP
    assert held[0].mode is Mode.PAPER
    filled = h.adapter.order("PAPER-crypto-000001")
    assert filled is not None and filled.status is OrderStatus.FILLED


def test_paper_adapter_fills_without_constructing_okx() -> None:
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    paper = PaperAdapter(AssetClass.CRYPTO, quotes, Decimal("100000"),
                         now=lambda: NOW, sleep=lambda _s: None, seed=1)
    instrument = _btc()
    ref = paper.place(Intent(
        instrument=instrument, side=Side.BUY, qty=Decimal("0.001"),
        order_type=OrderType.MARKET, strategy_id="crypto_trend",
        reason="paper-only fill", idempotency_key="paper-only",
    ))
    assert ref.status is OrderStatus.FILLED
    assert ref.mode is Mode.PAPER
    assert ref.filled_qty > 0
    # A live venue object was never in this path.
    assert not isinstance(paper, OkxAdapter)


# --------------------------------------------------------------------------
# cap enforcement through a real strategy intent
# --------------------------------------------------------------------------


def test_a_unit_btc_intent_is_clipped_to_fifty_usdt(tmp_path) -> None:
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        ctx, _ = ct.ctx_from(ct.CROSS_UP, quotes={"BTC/USDT": BTC_PX})
        # Re-stamp the instrument to the runner's BTC (qty_precision=6).
        intents = list(CryptoEmaTrendStrategy(
            instruments=[_btc()], fast_period=3, slow_period=5, atr_period=3,
            bar_interval=ct.INTERVAL,
        ).propose(context(
            ctx.now, bars=ctx.extra[BARS_KEY], quotes={"BTC/USDT": BTC_PX},
        )))
        assert len(intents) == 1
        assert intents[0].qty == Decimal(1)   # strategies still propose one unit
        approval = s.gate.check(intents[0], _ctx())
        assert approval.approved, approval.reason
        assert approval.mode is Mode.PAPER
        assert approval.intent.qty * BTC_PX <= CLIP
        assert approval.intent.qty < Decimal(1)
    finally:
        s.close()
        s.store.close()


def test_a_second_open_symbol_is_refused_while_eth_is_held(tmp_path) -> None:
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        s.store.apply(
            Fill(order_id="seed-eth", instrument=_eth(), side=Side.BUY,
                 qty=Decimal("0.02"), price=ETH_PX, fees=Decimal("0"),
                 at=NOW, mode=Mode.PAPER),
            strategy_id="crypto_trend",
        )
        approval = s.gate.check(_intent(key="btc-while-eth"), _ctx())
        assert not approval.approved
        assert approval.rejections[0].name == "max_open_positions"
    finally:
        s.close()
        s.store.close()


def test_daily_notional_stops_a_sixth_clip(tmp_path) -> None:
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        for i in range(5):
            s.gate_state.record_submission(
                f"spent-{i}", AssetClass.CRYPTO, "2026-03-02", NOW,
                notional=CLIP)
        approval = s.gate.check(_intent(key="sixth"), _ctx())
        assert not approval.approved
        assert approval.rejections[0].name == "daily_notional_cap"
    finally:
        s.close()
        s.store.close()


def test_wired_crypto_caps_are_still_fifty_two_fifty_and_one(tmp_path) -> None:
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        limits = s.gate._OrderGate__config.limits_for(AssetClass.CRYPTO)
        assert limits.max_order_notional == CLIP
        assert limits.max_daily_notional == DAILY
        assert limits.max_open_positions == 1
        assert runner.CRYPTO_CLIP_USDT == CLIP
        assert runner.CRYPTO_DAILY_USDT == DAILY
    finally:
        s.close()
        s.store.close()


# --------------------------------------------------------------------------
# leftover paper ETH is flattened on boot
# --------------------------------------------------------------------------


def test_oversized_paper_eth_is_flat_before_the_first_tick(tmp_path) -> None:
    db = str(tmp_path / "trader.db")
    with SQLitePositionStore(db) as seed:
        seed.apply(
            Fill(order_id="PAPER-crypto-000011", instrument=_eth(),
                 side=Side.BUY, qty=Decimal("1"), price=ETH_PX,
                 fees=Decimal("0"), at=NOW, mode=Mode.PAPER),
            strategy_id="crypto_trend",
        )
    s = runner.build_stack(
        env={"TRADER_DB": db, "ENGINE_API_TOKEN": "t" * 32,
             "TRADER_CLASSES": "crypto"},
        clock=lambda: NOW,
    )
    try:
        assert s.store.positions(Mode.PAPER, AssetClass.CRYPTO) == []
    finally:
        s.close()
        s.store.close()


# --------------------------------------------------------------------------
# withdraw / transfer / earn never leave this process
# --------------------------------------------------------------------------


_FUNDS_MOVE_NAMES = (
    "withdraw", "transfer", "earn",
    "private_post_asset_withdrawal", "withdraw_ccy",
    "transfer_out", "staking",
)


def test_is_funds_movement_names_the_dangerous_surface() -> None:
    for name in _FUNDS_MOVE_NAMES:
        assert is_funds_movement(name), name
    for name in ("create_order", "fetch_order", "fetch_ticker",
                 "fetch_balance", "load_markets", "cancel_order"):
        assert not is_funds_movement(name), name


def test_okx_adapter_blocks_withdraw_on_the_injected_exchange() -> None:
    class FundMover(FakeExchange):
        def __init__(self) -> None:
            super().__init__({"BTC/USDT": spot("BTC/USDT")})
            self.moved = []

        def withdraw(self, *args, **kwargs):
            self.moved.append(("withdraw", args, kwargs))
            return {"id": "W1"}

        def transfer(self, *args, **kwargs):
            self.moved.append(("transfer", args, kwargs))
            return {"id": "T1"}

    inner = FundMover()
    a = adapter(inner)
    with pytest.raises(FundsMovementForbidden, match="withdraw"):
        a._ex.withdraw("BTC", 1, "addr")
    with pytest.raises(FundsMovementForbidden, match="transfer"):
        a._ex.transfer("USDT", 10, "spot", "funding")
    assert hasattr(a._ex, "withdraw") is False
    assert hasattr(a._ex, "create_order") is True
    assert inner.moved == []


def test_okx_adapter_source_never_calls_withdraw_transfer_or_earn() -> None:
    """Token strings in the classifier are data. `self._ex.withdraw` is not."""
    root = Path(inspect.getfile(OkxAdapter)).resolve()
    tree = ast.parse(root.read_text())
    forbidden = {"withdraw", "transfer", "earn"}
    called = [
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr.lower() in forbidden
    ]
    assert called == [], f"OKX adapter references funds-move attrs: {called}"


def test_okx_adapter_has_no_withdraw_api() -> None:
    for name in ("withdraw", "transfer", "earn"):
        assert not hasattr(OkxAdapter, name)
        assert name not in dir(OkxAdapter)
