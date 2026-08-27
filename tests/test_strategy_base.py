"""How StrategyBase behaves when its inputs are wrong.

The rules pinned here are the ones that keep a data failure from looking like
a quiet, uneventful tick.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from core.protocols import Strategy
from core.types import AssetClass, MarketContext, OrderType, Side
from strategies.base import (
    MissingDataError, StrategyBase, StrategyInputError,
)
from tests._strategy_support import (
    cash_equity, coin, context, d, make_bars, position, utc,
)

NOW = utc(2026, 8, 24, 12, 0)
INTERVAL = dt.timedelta(minutes=15)


class Probe(StrategyBase):
    """Minimal concrete strategy — the base is what is under test."""

    def __init__(self, asset_class=AssetClass.CRYPTO, **kw):
        super().__init__(strategy_id="probe", asset_class=asset_class,
                         bar_interval=INTERVAL, **kw)

    def propose(self, ctx):
        return []


def bars_ending_at(end: dt.datetime, n: int = 5):
    return make_bars([100 + i for i in range(n)], end - (n - 1) * INTERVAL, INTERVAL)


# ------------------------------------------------------------------ wiring

def test_strategy_satisfies_the_frozen_protocol():
    assert isinstance(Probe(), Strategy)


def test_constructor_rejects_nonsense():
    with pytest.raises(StrategyInputError):
        StrategyBase(strategy_id="  ", asset_class=AssetClass.CRYPTO, bar_interval=INTERVAL)
    with pytest.raises(StrategyInputError):
        StrategyBase(strategy_id="x", asset_class=AssetClass.CRYPTO,
                     bar_interval=dt.timedelta(0))
    with pytest.raises(StrategyInputError):
        StrategyBase(strategy_id="x", asset_class=AssetClass.CRYPTO,
                     bar_interval=INTERVAL, stale_bar_multiple=0)


# --------------------------------------------------------------- bar book

def test_missing_bar_book_is_a_wiring_error_not_an_empty_market():
    """`ctx.extra['bars']` absent means the runner did not publish. Returning
    an empty book here would read as 'the market printed nothing'."""
    ctx = MarketContext(now=NOW)
    with pytest.raises(StrategyInputError):
        Probe().bar_book(ctx)


def test_bar_book_of_the_wrong_shape_raises():
    ctx = MarketContext(now=NOW, extra={"bars": [1, 2, 3]})
    with pytest.raises(StrategyInputError):
        Probe().bar_book(ctx)


# ------------------------------------------- absent data: flat vs holding

def test_no_bars_for_a_symbol_we_do_not_hold_is_simply_no_signal():
    p = Probe()
    ctx = context(NOW, bars={"ETH/USDT": bars_ending_at(NOW)})
    assert p.bars_for(ctx, "BTC/USDT", min_bars=3, must_have=False) is None


def test_no_bars_for_a_symbol_we_HOLD_raises():
    """Silence here is indistinguishable from 'nothing to do' while an open
    position runs without a stop."""
    p = Probe()
    ctx = context(NOW, bars={"ETH/USDT": bars_ending_at(NOW)})
    with pytest.raises(MissingDataError):
        p.bars_for(ctx, "BTC/USDT", min_bars=3, must_have=True)


def test_stale_feed_blocks_entries_and_raises_on_an_open_position():
    p = Probe()
    stale_end = NOW - 4 * INTERVAL          # stale_bar_multiple is 3
    ctx = context(NOW, bars={"BTC/USDT": bars_ending_at(stale_end)})
    assert p.bars_for(ctx, "BTC/USDT", min_bars=3, must_have=False) is None
    with pytest.raises(MissingDataError):
        p.bars_for(ctx, "BTC/USDT", min_bars=3, must_have=True)


def test_a_fresh_feed_is_accepted():
    p = Probe()
    ctx = context(NOW, bars={"BTC/USDT": bars_ending_at(NOW)})
    assert p.bars_for(ctx, "BTC/USDT", min_bars=3, must_have=True) is not None


def test_bars_from_the_future_are_a_wiring_error():
    p = Probe()
    ctx = context(NOW, bars={"BTC/USDT": bars_ending_at(NOW + INTERVAL)})
    with pytest.raises(StrategyInputError):
        p.bars_for(ctx, "BTC/USDT", min_bars=3, must_have=False)


def test_short_history_blocks_entries_but_raises_for_an_open_position():
    p = Probe()
    ctx = context(NOW, bars={"BTC/USDT": bars_ending_at(NOW, n=2)})
    assert p.bars_for(ctx, "BTC/USDT", min_bars=10, must_have=False) is None
    with pytest.raises(MissingDataError):
        p.bars_for(ctx, "BTC/USDT", min_bars=10, must_have=True)


# ------------------------------------------------------------- positions

def test_own_positions_ignores_other_strategies_and_closed_rows():
    p = Probe()
    inst = coin()
    ctx = context(NOW, positions=[
        position(inst, 1, 100, strategy_id="probe"),
        position(coin("ETH/USDT"), 2, 50, strategy_id="someone_else"),
        position(coin("SOL/USDT"), 0, 10, strategy_id="probe"),   # flat row
    ])
    assert [x.instrument.symbol for x in p.own_positions(ctx)] == ["BTC/USDT"]
    assert p.own_position(ctx, "ETH/USDT") is None


def test_entry_is_blocked_by_another_strategys_position_in_the_same_symbol():
    """Two strategies stacking one instrument doubles the real exposure while
    each believes it is at 1x; the gate caps per intent, not per instrument."""
    p = Probe()
    ctx = context(NOW, positions=[
        position(coin("ETH/USDT"), 2, 50, strategy_id="someone_else"),
    ])
    assert p.held_by_anyone(ctx, "ETH/USDT") is True
    assert p.own_position(ctx, "ETH/USDT") is None


# ---------------------------------------------------------------- intents

def test_crypto_intents_are_market_orders_with_no_limit_price():
    p = Probe()
    intent = p.make_intent(instrument=coin(), side=Side.BUY, qty=d(1),
                           reason="test", decision_ts=NOW, tag="entry",
                           ref_price=d(100))
    assert intent.order_type is OrderType.MARKET
    assert intent.limit_price is None
    assert intent.strategy_id == "probe"
    assert intent.idempotency_key


def test_equity_intents_are_marketable_limits_priced_off_the_reference():
    p = Probe(asset_class=AssetClass.EQUITY)
    buy = p.make_intent(instrument=cash_equity(), side=Side.BUY, qty=d(1),
                        reason="test", decision_ts=NOW, tag="entry", ref_price=d(1000))
    sell = p.make_intent(instrument=cash_equity(), side=Side.SELL, qty=d(1),
                         reason="test", decision_ts=NOW, tag="exit", ref_price=d(1000))
    assert buy.order_type is OrderType.LIMIT
    # 10 bps of 1000 = 1.00, quantized to the 0.05 tick
    assert buy.limit_price == Decimal("1001.00")
    assert sell.limit_price == Decimal("999.00")
    assert buy.limit_price % Decimal("0.05") == 0


def test_a_limit_intent_without_a_reference_price_raises():
    """NFO rejects MARKET orders, so an equity intent with no price is not
    placeable. Inventing one would be a fabricated number on a real order."""
    p = Probe(asset_class=AssetClass.EQUITY)
    with pytest.raises(StrategyInputError):
        p.make_intent(instrument=cash_equity(), side=Side.BUY, qty=d(1), reason="t",
                      decision_ts=NOW, tag="entry", ref_price=None)
    with pytest.raises(StrategyInputError):
        p.make_intent(instrument=cash_equity(), side=Side.BUY, qty=d(1), reason="t",
                      decision_ts=NOW, tag="entry", ref_price=d(0))


def test_entry_unit_qty_is_one_lot_or_one_coin_and_never_zero():
    assert Probe().entry_unit_qty(coin()) == Decimal(1)
    assert Probe(asset_class=AssetClass.EQUITY).entry_unit_qty(
        cash_equity("NIFTYOPT", lot_size=75)) == Decimal(75)


def test_finalize_refuses_two_intents_on_one_instrument():
    """Different tags mean different idempotency keys, so the gate's dedup
    would let BOTH through — selling a position twice is going short."""
    p = Probe()
    inst = coin()
    a = p.make_intent(instrument=inst, side=Side.SELL, qty=d(1), reason="stop",
                      decision_ts=NOW, tag="exit", ref_price=d(100))
    b = p.make_intent(instrument=inst, side=Side.SELL, qty=d(1), reason="signal",
                      decision_ts=NOW, tag="signal_exit", ref_price=d(100))
    assert p.finalize([a]) == [a]
    with pytest.raises(StrategyInputError):
        p.finalize([a, b])
