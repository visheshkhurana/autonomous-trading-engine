"""Intraday mean reversion on liquid cash equities."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from core.types import OrderType, Side
from strategies.base import MissingDataError, StrategyInputError
from strategies.equity_meanrev import EquityMeanReversionStrategy
from tests._strategy_support import (
    cash_equity, coin, context, d, ist, make_bars, position,
)

INTERVAL = dt.timedelta(minutes=5)
REL = cash_equity("RELIANCE")

# SMA(5) 98.80, ATR(3) 3.00 -> entry band 94.30; last close 94 is under it and
# RSI(3) is 0. Bars end 10:30 IST on a Monday.
SETUP = [100, 100, 100, 100, 94]
FIRST_BAR = ist(2026, 8, 24, 10, 10)


def strat(**kw) -> EquityMeanReversionStrategy:
    params = dict(instruments=[REL], rsi_period=3, sma_period=5, atr_period=3,
                  rsi_oversold=Decimal("30"), entry_atr_mult=Decimal("1.5"),
                  stop_pct=Decimal("0.015"), bar_interval=INTERVAL)
    params.update(kw)
    return EquityMeanReversionStrategy(**params)


def make_ctx(values=SETUP, *, now=None, quote="94", positions=(), start=FIRST_BAR,
             spread="0.5"):
    bars = make_bars(values, start, INTERVAL, spread=spread)
    return context(now or bars[-1].ts, bars={"RELIANCE": bars},
                   quotes={"RELIANCE": quote}, positions=positions)


# ------------------------------------------------------------------ entry

def test_oversold_and_below_the_band_opens_a_long():
    intents = strat().propose(make_ctx())
    assert len(intents) == 1
    i = intents[0]
    assert i.side is Side.BUY
    # F&O and NSE cash go out as marketable LIMITs, never MARKET.
    assert i.order_type is OrderType.LIMIT and i.limit_price is not None
    assert i.limit_price > Decimal("94")
    assert "RSI(3)" in i.reason and "SMA(5)" in i.reason and "hard stop" in i.reason


def test_no_entry_when_price_is_not_stretched():
    assert list(strat().propose(make_ctx([100, 100, 100, 100, 99], quote="99"))) == []


def test_no_entry_when_rsi_is_not_oversold():
    # Falling far enough for the band, but rising into the last bar.
    assert list(strat().propose(make_ctx([100, 101, 102, 103, 104], quote="104"))) == []


def test_no_entry_without_a_live_quote():
    """A LIMIT order needs a price. Falling back to the last bar close would
    be inventing one."""
    bars = make_bars(SETUP, FIRST_BAR, INTERVAL, spread="0.5")
    ctx = context(bars[-1].ts, bars={"RELIANCE": bars})
    assert list(strat().propose(ctx)) == []


def test_no_entry_after_the_cutoff_so_the_position_can_still_be_closed():
    # `start` is the first of five 5-minute bars, so the decision lands 20
    # minutes later: 14:40, then 14:45.
    assert len(strat().propose(make_ctx(start=ist(2026, 8, 24, 14, 20)))) == 1
    assert list(strat().propose(make_ctx(start=ist(2026, 8, 24, 14, 25)))) == []


def test_nothing_is_proposed_outside_the_session():
    sunday = ist(2026, 8, 23, 10, 10)
    assert list(strat().propose(make_ctx(start=sunday))) == []
    after_close = ist(2026, 8, 24, 15, 35)
    assert list(strat().propose(make_ctx(start=after_close))) == []


def test_no_entry_when_another_strategy_holds_the_name():
    ctx = make_ctx(positions=[position(REL, 10, 95, strategy_id="someone_else")])
    assert list(strat().propose(ctx)) == []


# ------------------------------------------------------------------- exit

def held(qty="10", avg="100"):
    return [position(REL, qty, avg, strategy_id="equity_meanrev")]


def test_the_hard_stop_fires_at_a_fixed_percentage_of_the_entry_price():
    # 1.5% under 100 is 98.50.
    ctx = make_ctx(quote="98.40", positions=held())
    intents = strat().propose(ctx)
    assert len(intents) == 1
    assert intents[0].side is Side.SELL and intents[0].qty == Decimal("10")
    assert "hard stop" in intents[0].reason


def test_the_stop_does_not_widen_when_volatility_expands():
    """An ATR-scaled stop recomputed each tick moves away from price exactly
    during the move it exists to cap. This stop is anchored to the entry."""
    ctx = make_ctx(quote="98.40", positions=held(), spread="8")   # ATR ~ 16
    intents = strat().propose(ctx)
    assert len(intents) == 1 and "hard stop" in intents[0].reason


def test_just_above_the_stop_nothing_happens():
    ctx = make_ctx(quote="98.60", positions=held())
    assert list(strat().propose(ctx)) == []


def test_reaching_the_mean_takes_the_profit():
    ctx = make_ctx(quote="99.00", positions=held())     # SMA(5) is 98.80
    intents = strat().propose(ctx)
    assert len(intents) == 1
    assert "target reached" in intents[0].reason


def test_the_position_is_flattened_before_the_close():
    """EquityProfile.must_flatten_intraday is True; an intraday position
    carried overnight is not the trade that was sized."""
    ctx = make_ctx(quote="98.60", positions=held(), start=ist(2026, 8, 24, 14, 55))
    assert ctx.now == ist(2026, 8, 24, 15, 15)
    intents = strat().propose(ctx)
    assert len(intents) == 1
    assert "session flatten" in intents[0].reason


def test_only_one_exit_when_the_stop_and_the_flatten_both_apply():
    ctx = make_ctx(quote="90", positions=held(), start=ist(2026, 8, 24, 14, 55))
    intents = strat().propose(ctx)
    assert len(intents) == 1
    assert "hard stop" in intents[0].reason


def test_exit_sells_exactly_what_is_held():
    ctx = make_ctx(quote="98.40", positions=held(qty="7"))
    assert strat().propose(ctx)[0].qty == Decimal("7")


# ------------------------------------------------------- data and wiring

def test_no_quote_for_an_open_position_raises():
    bars = make_bars(SETUP, FIRST_BAR, INTERVAL, spread="0.5")
    ctx = context(bars[-1].ts, bars={"RELIANCE": bars}, positions=held())
    with pytest.raises(MissingDataError):
        strat().propose(ctx)


def test_a_dead_feed_raises_while_the_position_is_open():
    bars = make_bars(SETUP, FIRST_BAR, INTERVAL, spread="0.5")
    ctx = context(bars[-1].ts + 4 * INTERVAL, bars={"RELIANCE": bars},
                  quotes={"RELIANCE": d("94")}, positions=held())
    with pytest.raises(MissingDataError):
        strat().propose(ctx)


def test_a_short_position_contradicts_a_long_only_strategy_and_raises():
    ctx = make_ctx(quote="98.40", positions=[position(REL, -10, 100,
                                                      strategy_id="equity_meanrev")])
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_a_non_positive_entry_price_raises():
    ctx = make_ctx(quote="98.40", positions=held(avg="0"))
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_repeating_the_tick_repeats_the_idempotency_key():
    bars = make_bars(SETUP, FIRST_BAR, INTERVAL, spread="0.5")
    a = strat().propose(context(bars[-1].ts, bars={"RELIANCE": bars},
                                quotes={"RELIANCE": d("94")}))
    b = strat().propose(context(bars[-1].ts + dt.timedelta(minutes=2),
                                bars={"RELIANCE": bars}, quotes={"RELIANCE": d("94")}))
    assert a[0].idempotency_key == b[0].idempotency_key


def test_construction_rejects_bad_parameters():
    with pytest.raises(StrategyInputError):
        strat(instruments=[])
    with pytest.raises(StrategyInputError):
        strat(instruments=[coin()])
    with pytest.raises(StrategyInputError):
        strat(stop_pct=Decimal("0"))
    with pytest.raises(StrategyInputError):
        strat(rsi_oversold=Decimal("120"))
    with pytest.raises(StrategyInputError):
        # A flatten at or after the 15:30 close never runs, and the position is
        # carried overnight by accident.
        strat(flatten_after=dt.time(15, 30))
    with pytest.raises(StrategyInputError):
        strat(no_new_entries_after=dt.time(15, 10), flatten_after=dt.time(15, 5))
