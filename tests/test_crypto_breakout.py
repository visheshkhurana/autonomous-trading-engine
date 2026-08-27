"""Donchian breakout with a two-sided volatility filter, on OKX spot."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from core.types import OrderType, Side
from strategies.base import donchian
from strategies.crypto_breakout import CryptoBreakoutStrategy
from strategies.base import MissingDataError, StrategyInputError
from tests._strategy_support import (
    cash_equity, coin, context, d, ohlc_bars, position, utc,
)

INTERVAL = dt.timedelta(hours=1)
ETH = coin("ETH/USDT")
START = utc(2026, 8, 24, 0, 0)

# Six quiet bars ranging 100-102 (ATR 2, i.e. ~2% of price), then a bar that
# closes through the range high.
RANGE_ROWS = [(101, 102, 100, 101)] * 6
BREAKOUT_ROW = (101, 105, 101, 105)


def strat(**kw) -> CryptoBreakoutStrategy:
    params = dict(instruments=[ETH], entry_period=5, exit_period=3, atr_period=3,
                  min_atr_pct=Decimal("0.005"), max_atr_pct=Decimal("0.06"),
                  max_loss_pct=Decimal("0.08"), bar_interval=INTERVAL)
    params.update(kw)
    return CryptoBreakoutStrategy(**params)


def make_ctx(rows, positions=(), now=None):
    bars = ohlc_bars(rows, START, INTERVAL)
    return context(now or bars[-1].ts, bars={"ETH/USDT": bars}, positions=positions), bars


# ------------------------------------------------------------------ entry

def test_a_close_through_the_range_high_opens_a_long():
    ctx, bars = make_ctx(RANGE_ROWS + [BREAKOUT_ROW])
    intents = strat().propose(ctx)
    assert len(intents) == 1
    i = intents[0]
    assert i.side is Side.BUY
    assert i.order_type is OrderType.MARKET and i.limit_price is None
    assert "breakout" in i.reason and "5-bar high 102" in i.reason


def test_the_breaking_bar_is_not_part_of_its_own_channel():
    """If the channel included the current bar its high would be the channel
    high and close > upper could never be true — the strategy would be a
    no-op that looks healthy."""
    ctx, bars = make_ctx(RANGE_ROWS + [BREAKOUT_ROW])
    including = donchian(bars, 5, exclude_last=False)
    assert bars[-1].close <= including.upper        # would never fire
    assert len(strat().propose(ctx)) == 1           # excluding it, it does


def test_touching_the_high_is_not_a_breakout():
    ctx, _ = make_ctx(RANGE_ROWS + [(101, 102, 101, 102)])
    assert list(strat().propose(ctx)) == []


def test_a_dead_range_is_filtered_out():
    """ATR at 0.1% of price is noise; the 'breakout' is a spread-paying
    accident."""
    quiet = [("101", "101.05", "100.95", "101")] * 6
    ctx, _ = make_ctx(quiet + [("101", "101.2", "101", "101.2")],)
    assert list(strat().propose(ctx)) == []


def test_a_volatility_blowoff_is_filtered_out():
    """Above the band the move has already happened and the channel stop sits
    an unacceptable distance below."""
    wild = [(100, 130, 70, 100)] * 6
    ctx, _ = make_ctx(wild + [(100, 140, 100, 140)])
    assert list(strat().propose(ctx)) == []


def test_no_entry_while_already_holding_or_when_another_strategy_holds():
    rows = RANGE_ROWS + [BREAKOUT_ROW]
    ctx, _ = make_ctx(rows, positions=[position(ETH, 1, 101, strategy_id="crypto_breakout")])
    assert all(i.side is Side.SELL for i in strat().propose(ctx))
    ctx, _ = make_ctx(rows, positions=[position(ETH, 1, 101, strategy_id="crypto_trend")])
    assert list(strat().propose(ctx)) == []


def test_short_history_is_quiet():
    ctx, _ = make_ctx([(101, 102, 100, 101)] * 3)
    assert list(strat().propose(ctx)) == []


# ------------------------------------------------------------------- exit

def test_losing_the_exit_channel_closes_the_position():
    rows = RANGE_ROWS + [BREAKOUT_ROW] + [(105, 105, 99, 99)]
    ctx, _ = make_ctx(rows, positions=[position(ETH, 1, 105, strategy_id="crypto_breakout")])
    intents = strat().propose(ctx)
    assert len(intents) == 1
    assert intents[0].side is Side.SELL and intents[0].qty == Decimal(1)
    assert "channel exit" in intents[0].reason


def test_holding_inside_the_channel_does_nothing():
    rows = RANGE_ROWS + [BREAKOUT_ROW] + [(105, 106, 104, 105)]
    ctx, _ = make_ctx(rows, positions=[position(ETH, 1, 105, strategy_id="crypto_breakout")])
    assert list(strat().propose(ctx)) == []


def test_the_hard_stop_takes_precedence_and_only_one_exit_is_emitted():
    """Both exits trigger on this bar. Two SELLs with different tags carry
    different idempotency keys, so the gate would place both and we would end
    up short."""
    rows = RANGE_ROWS + [BREAKOUT_ROW] + [(105, 105, 80, 80)]
    ctx, _ = make_ctx(rows, positions=[position(ETH, 1, 105, strategy_id="crypto_breakout")])
    intents = strat().propose(ctx)
    assert len(intents) == 1
    assert "hard stop" in intents[0].reason


def test_exit_sells_exactly_what_is_held():
    rows = RANGE_ROWS + [BREAKOUT_ROW] + [(105, 105, 99, 99)]
    held = Decimal("3.1234567")
    ctx, _ = make_ctx(rows, positions=[position(ETH, held, 105, strategy_id="crypto_breakout")])
    assert strat().propose(ctx)[0].qty == held


# ------------------------------------------------------- data and wiring

def test_a_dead_feed_raises_while_the_position_is_open():
    rows = RANGE_ROWS + [BREAKOUT_ROW]
    bars = ohlc_bars(rows, START, INTERVAL)
    ctx = context(bars[-1].ts + 4 * INTERVAL, bars={"ETH/USDT": bars},
                  positions=[position(ETH, 1, 105, strategy_id="crypto_breakout")])
    with pytest.raises(MissingDataError):
        strat().propose(ctx)


def test_a_short_spot_position_raises():
    rows = RANGE_ROWS + [BREAKOUT_ROW]
    ctx, _ = make_ctx(rows, positions=[position(ETH, -1, 105, strategy_id="crypto_breakout")])
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_repeating_the_tick_repeats_the_idempotency_key():
    rows = RANGE_ROWS + [BREAKOUT_ROW]
    bars = ohlc_bars(rows, START, INTERVAL)
    a = strat().propose(context(bars[-1].ts, bars={"ETH/USDT": bars}))
    b = strat().propose(context(bars[-1].ts + dt.timedelta(minutes=31),
                                bars={"ETH/USDT": bars}))
    assert a[0].idempotency_key == b[0].idempotency_key


def test_construction_rejects_bad_parameters():
    with pytest.raises(StrategyInputError):
        strat(instruments=[cash_equity()])
    with pytest.raises(StrategyInputError):
        strat(exit_period=5, entry_period=5)     # exit must be shorter
    with pytest.raises(StrategyInputError):
        strat(min_atr_pct=Decimal("0.1"), max_atr_pct=Decimal("0.05"))
    with pytest.raises(StrategyInputError):
        strat(entry_period=1)
