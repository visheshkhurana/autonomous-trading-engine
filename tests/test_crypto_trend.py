"""EMA-cross trend follower on OKX spot."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from core.types import OrderType, Side
from strategies.base import MissingDataError, StrategyInputError
from strategies.crypto_trend import CryptoEmaTrendStrategy
from tests._strategy_support import (
    cash_equity, coin, context, d, make_bars, ohlc_bars, position, utc,
)

INTERVAL = dt.timedelta(minutes=15)
BTC = coin("BTC/USDT")

# Cross UP lands on the LAST bar: EMA(3) 95.5 -> 97.25 crossing EMA(5) 96.0 -> 97.0
CROSS_UP = [100, 99, 98, 97, 96, 95, 94, 96, 99]
# Same series one bar later: fast is already above slow, so there is no cross.
NO_CROSS = CROSS_UP + [104]
CROSS_DOWN = [100, 101, 102, 103, 104, 105, 106, 104, 101]


def strat(**kw) -> CryptoEmaTrendStrategy:
    params = dict(instruments=[BTC], fast_period=3, slow_period=5, atr_period=3,
                  atr_stop_mult=Decimal("2"), max_loss_pct=Decimal("0.08"),
                  bar_interval=INTERVAL)
    params.update(kw)
    return CryptoEmaTrendStrategy(**params)


def ctx_from(values, now=None, positions=(), spread=1, quotes=None):
    bars = make_bars(values, utc(2026, 8, 24, 0, 0), INTERVAL, spread=spread)
    now = now or bars[-1].ts
    if quotes is None:
        quotes = {"BTC/USDT": bars[-1].close}
    return context(now, bars={"BTC/USDT": bars}, positions=positions,
                   quotes=quotes), bars


def ctx_bars(bars, now=None, positions=(), quotes=None):
    if quotes is None:
        quotes = {"BTC/USDT": bars[-1].close}
    return context(now or bars[-1].ts, bars={"BTC/USDT": bars},
                   positions=positions, quotes=quotes)


# ------------------------------------------------------------------ entry

def test_cross_up_opens_a_long_with_a_market_order():
    ctx, bars = ctx_from(CROSS_UP)
    intents = strat().propose(ctx)
    assert len(intents) == 1
    i = intents[0]
    assert i.side is Side.BUY
    assert i.order_type is OrderType.MARKET
    assert i.limit_price == bars[-1].close
    assert i.limit_price > 0
    assert i.qty == Decimal(1)
    assert "EMA cross up" in i.reason and "Stop starts at" in i.reason
    assert i.structure_id is None


def test_an_established_trend_does_not_re_enter_every_bar():
    """Without the previous-bar condition this fires on every bar that has
    fast above slow, which is most of a trend."""
    ctx, _ = ctx_from(NO_CROSS)
    assert list(strat().propose(ctx)) == []


def test_no_entry_while_the_position_is_already_open():
    ctx, bars = ctx_from(CROSS_UP)
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 96, strategy_id="crypto_trend")])
    assert [i.side for i in strat().propose(ctx)] == []


def test_no_entry_when_another_strategy_already_holds_the_coin():
    ctx, bars = ctx_from(CROSS_UP)
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 96, strategy_id="crypto_breakout")])
    assert list(strat().propose(ctx)) == []


def test_short_history_is_quiet_rather_than_wrong():
    ctx, _ = ctx_from([100, 99, 98])
    assert list(strat().propose(ctx)) == []


def test_a_cross_without_a_quote_is_not_an_entry():
    """Unknown is never a free price. Bars can fire and the quote can still
    be absent; treating that as 'trade anyway' is how a 0 mark becomes an
    order. The engine documents that strategies skip a missing name."""
    ctx, _ = ctx_from(CROSS_UP, quotes={})
    assert list(strat().propose(ctx)) == []


def test_a_zero_quote_is_unknown_and_does_not_enter():
    ctx, _ = ctx_from(CROSS_UP, quotes={"BTC/USDT": Decimal("0")})
    assert list(strat().propose(ctx)) == []


def test_an_open_position_with_no_quote_raises_rather_than_going_unmanaged():
    ctx, bars = ctx_from(CROSS_DOWN, quotes={})
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 100, strategy_id="crypto_trend")],
                   quotes={})
    with pytest.raises(MissingDataError, match="quote"):
        strat().propose(ctx)


# ------------------------------------------------------------------- exit

def test_cross_down_closes_the_position():
    ctx, bars = ctx_from(CROSS_DOWN)
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 100, strategy_id="crypto_trend")])
    intents = strat().propose(ctx)
    assert len(intents) == 1
    assert intents[0].side is Side.SELL
    assert intents[0].qty == Decimal(1)
    assert "EMA cross down" in intents[0].reason


def test_exit_sells_exactly_what_is_held_and_is_not_rounded_down():
    """A quantized exit leaves a residue of the position open at the venue."""
    ctx, bars = ctx_from(CROSS_DOWN)
    held = Decimal("0.4237891")           # finer than the 6dp quantization step
    ctx = ctx_bars(bars, positions=[position(BTC, held, 100, strategy_id="crypto_trend")])
    assert strat().propose(ctx)[0].qty == held


# Every bar ranges exactly 2 wide with no gaps, so ATR is exactly 2 and the
# 2x ATR stop sits at 100 - 4 = 96, well above the 92 hard floor.
DRIFT_DOWN = [(100, 101, 99, 100)] * 4 + [
    ("100", "100.5", "98.5", "99"),
    ("99", "99.5", "97.5", "98"),
    ("98", "98.5", "96.5", "97"),
    ("97", "97.5", "95.5", "96"),
    ("96", "96.5", "94.5", "95"),
]


def test_atr_stop_fires_when_price_falls_through_it():
    bars = ohlc_bars(DRIFT_DOWN, utc(2026, 8, 24, 0, 0), INTERVAL)
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 100, strategy_id="crypto_trend")])
    intents = strat().propose(ctx)
    assert len(intents) == 1 and intents[0].side is Side.SELL
    assert "stop hit" in intents[0].reason


def test_price_above_the_stop_is_left_alone():
    # Same drift, stopped one bar earlier: close 97 is still above the 96 stop.
    bars = ohlc_bars(DRIFT_DOWN[:-2], utc(2026, 8, 24, 0, 0), INTERVAL)
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 100, strategy_id="crypto_trend")])
    assert list(strat().propose(ctx)) == []


def test_the_hard_percentage_floor_caps_a_stop_that_atr_has_widened():
    """An ATR stop recomputed each bar runs away from price exactly when
    volatility spikes. With ATR at 20 the ATR stop would sit at 60 and this
    position would keep bleeding; the 8% floor puts it at 92."""
    rows = [(100, 110, 90, 100)] * 6 + [(100, 101, 91, 91)]
    bars = ohlc_bars(rows, utc(2026, 8, 24, 0, 0), INTERVAL)
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 100, strategy_id="crypto_trend")])
    intents = strat().propose(ctx)
    assert len(intents) == 1 and intents[0].side is Side.SELL
    assert "hard floor" in intents[0].reason


def test_only_one_exit_intent_even_when_stop_and_signal_both_trigger():
    rows = [(100 + i, 101 + i, 99 + i, 100 + i) for i in range(8)] + [(107, 108, 80, 80)]
    bars = ohlc_bars(rows, utc(2026, 8, 24, 0, 0), INTERVAL)
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 100, strategy_id="crypto_trend")])
    intents = strat().propose(ctx)
    assert len(intents) == 1


# ------------------------------------------------------- data and wiring

def test_a_dead_feed_raises_while_the_position_is_open():
    _, bars = ctx_from(CROSS_UP)
    late = bars[-1].ts + 4 * INTERVAL
    ctx = ctx_bars(bars, now=late,
                   positions=[position(BTC, 1, 100, strategy_id="crypto_trend")])
    with pytest.raises(MissingDataError):
        strat().propose(ctx)


def test_a_short_spot_position_is_a_contradiction_and_raises():
    """Spot cannot be short. A short row means the store and the strategy
    disagree about the book, and guessing which is right places orders."""
    ctx, bars = ctx_from(CROSS_DOWN)
    ctx = ctx_bars(bars, positions=[position(BTC, -1, 100, strategy_id="crypto_trend")])
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_a_non_positive_entry_price_raises_rather_than_producing_a_free_stop():
    ctx, bars = ctx_from(CROSS_DOWN)
    ctx = ctx_bars(bars, positions=[position(BTC, 1, 0, strategy_id="crypto_trend")])
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_repeating_the_tick_repeats_the_idempotency_key():
    """A repeated tick on the same bar must not be able to double-fire."""
    _, bars = ctx_from(CROSS_UP)
    first = strat().propose(ctx_bars(bars, now=bars[-1].ts))
    later = strat().propose(ctx_bars(bars, now=bars[-1].ts + dt.timedelta(minutes=7)))
    assert first[0].idempotency_key == later[0].idempotency_key


def test_a_new_bar_is_a_new_decision():
    _, bars = ctx_from(CROSS_UP)
    first = strat().propose(ctx_bars(bars))
    shifted = make_bars(CROSS_UP, utc(2026, 8, 24, 0, 15), INTERVAL)
    later = strat().propose(ctx_bars(shifted))
    assert first[0].idempotency_key != later[0].idempotency_key


def test_construction_rejects_bad_parameters():
    with pytest.raises(StrategyInputError):
        strat(instruments=[])
    with pytest.raises(StrategyInputError):
        strat(instruments=[cash_equity()])          # wrong asset class
    with pytest.raises(StrategyInputError):
        strat(instruments=[BTC, coin("BTC/USDT")])  # duplicate
    with pytest.raises(StrategyInputError):
        strat(fast_period=9, slow_period=5)
    with pytest.raises(StrategyInputError):
        strat(max_loss_pct=Decimal("1.5"))
    with pytest.raises(StrategyInputError):
        strat(atr_stop_mult=Decimal("0"))
