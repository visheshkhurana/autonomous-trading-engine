"""Indicator maths against hand-computed values.

Every expected number here was worked out by hand and is written down in the
test as arithmetic, not copied from a run of the code.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from strategies.base import (
    Bar, DonchianChannel, StrategyInputError, atr, closes, donchian, ema,
    ema_series, floor_ts, idempotency_key, rsi, sma, structure_key, true_range,
    true_ranges, validate_bars,
)
from core.types import Side
from tests._strategy_support import close, d, make_bars, ohlc_bars, utc

START = utc(2026, 8, 24, 0, 0)


# ---------------------------------------------------------------- SMA / EMA

def test_sma_is_the_mean_of_the_last_period_values():
    assert sma([d(1), d(2), d(3), d(4)], 4) == Decimal("2.5")
    # only the last 2 count: (3 + 4) / 2
    assert sma([d(1), d(2), d(3), d(4)], 2) == Decimal("3.5")


def test_ema_series_matches_hand_computation():
    # period 2 -> k = 2/3, seed = (1+2)/2 = 1.5
    # 1.5 + 2/3*(3-1.5) = 2.5 ; 2.5 + 2/3*(4-2.5) = 3.5
    series = ema_series([d(1), d(2), d(3), d(4)], 2)
    assert len(series) == 3
    assert close(series[0], "1.5")
    assert close(series[1], "2.5")
    assert close(series[2], "3.5")
    assert close(ema([d(1), d(2), d(3), d(4)], 2), "3.5")


def test_ema_series_period_three_matches_hand_computation():
    # seed = (2+4+6)/3 = 4, k = 0.5 ; 4 + .5*(8-4) = 6 ; 6 + .5*(10-6) = 8
    series = ema_series([d(2), d(4), d(6), d(8), d(10)], 3)
    assert close(series[0], "4")
    assert close(series[1], "6")
    assert close(series[2], "8")


def test_indicators_return_none_not_zero_when_history_is_short():
    """A zero here would read as a real, extreme indicator value and produce a
    real order. None cannot be mistaken for a number."""
    assert sma([d(1)], 5) is None
    assert ema([d(1)], 5) is None
    assert ema_series([d(1)], 5) == []
    assert rsi([d(1), d(2)], 14) is None
    assert atr(make_bars([10, 11], START), 14) is None
    assert donchian(make_bars([10, 11], START), 20) is None


def test_indicator_periods_must_be_positive():
    for fn in (sma, ema, rsi):
        with pytest.raises(StrategyInputError):
            fn([d(1), d(2), d(3)], 0)
    with pytest.raises(StrategyInputError):
        atr(make_bars([1, 2, 3], START), 0)
    with pytest.raises(StrategyInputError):
        donchian(make_bars([1, 2, 3], START), 0)


# ---------------------------------------------------------------------- ATR

def test_true_range_uses_the_gap_terms():
    # prev close 9, bar 19-20: high-low = 1, |high-prev| = 11, |low-prev| = 10
    bar = ohlc_bars([(19, 20, 19, 20)], START)[0]
    assert true_range(d(9), bar) == Decimal("11")


def test_atr_matches_hand_computation():
    bars = ohlc_bars([
        (9, 10, 8, 9),     # no TR (no previous close)
        (9, 11, 9, 10),    # TR = max(2, 2, 0) = 2
        (10, 12, 10, 11),  # TR = max(2, 2, 0) = 2
        (11, 15, 11, 14),  # TR = max(4, 4, 0) = 4
    ], START)
    assert [str(t) for t in true_ranges(bars)] == ["2", "2", "4"]
    # period 2: seed = (2+2)/2 = 2 ; then (2*1 + 4)/2 = 3
    assert close(atr(bars, 2), "3")
    # period 3: seed = (2+2+4)/3
    assert close(atr(bars, 3), Decimal(8) / Decimal(3))


def test_atr_needs_period_plus_one_bars():
    bars = ohlc_bars([(9, 10, 8, 9), (9, 11, 9, 10)], START)
    assert atr(bars, 2) is None
    assert atr(bars, 1) is not None


# ---------------------------------------------------------------------- RSI

def test_rsi_matches_hand_computation():
    # closes 10,11,10,11,12 -> deltas +1,-1,+1,+1 ; period 2
    # seed avg_gain = (1+0)/2 = .5, avg_loss = (0+1)/2 = .5
    # step: gain (.5*1+1)/2 = .75, loss (.5*1+0)/2 = .25
    # step: gain (.75*1+1)/2 = .875, loss (.25*1+0)/2 = .125
    # RS = 7 -> 100 - 100/8 = 87.5
    assert close(rsi([d(10), d(11), d(10), d(11), d(12)], 2), "87.5")


def test_rsi_saturates_at_100_and_0_without_dividing_by_zero():
    assert rsi([d(x) for x in range(1, 20)], 14) == Decimal(100)
    assert rsi([d(x) for x in range(20, 1, -1)], 14) == Decimal(0)


# ----------------------------------------------------------------- Donchian

def test_donchian_excludes_the_current_bar_by_default():
    bars = ohlc_bars([
        (9, 10, 8, 9),
        (9, 12, 9, 11),
        (11, 11, 7, 8),
        (11, 13, 10, 13),  # current bar, excluded
    ], START)
    ch = donchian(bars, 3)
    assert ch == DonchianChannel(upper=Decimal("12"), lower=Decimal("7"))
    assert ch.mid == Decimal("9.5")


def test_a_channel_that_includes_the_current_bar_can_never_break_out():
    """close <= high always, so including the current bar makes
    `close > upper` unreachable. This pins the exclude_last default."""
    bars = ohlc_bars([(9, 10, 8, 9), (9, 12, 9, 11), (11, 20, 10, 20)], START)
    included = donchian(bars, 3, exclude_last=False)
    assert included is not None and bars[-1].close <= included.upper
    excluded = donchian(bars, 2)
    assert excluded is not None and bars[-1].close > excluded.upper


# --------------------------------------------------------------------- Bars

def test_bar_rejects_incoherent_prices():
    with pytest.raises(StrategyInputError):
        Bar.of(START, 10, 8, 9, 9)          # high < low
    with pytest.raises(StrategyInputError):
        Bar.of(START, 10, 10, 9, 12)        # close outside high/low
    with pytest.raises(StrategyInputError):
        Bar.of(START, 10, 11, 0, 10)        # non-positive low
    with pytest.raises(StrategyInputError):
        Bar.of(dt.datetime(2026, 8, 24, 9, 0), 10, 11, 9, 10)  # naive ts


def test_bar_rejects_floats():
    """Decimal(0.1) is not 0.1; a price that is silently 5e-18 off becomes a
    tick mismatch the broker rejects after the other leg has filled."""
    with pytest.raises(StrategyInputError):
        Bar.of(START, 10.0, 11.0, 9.0, 10.0)


def test_validate_bars_rejects_out_of_order_and_foreign_objects():
    good = make_bars([10, 11, 12], START)
    assert validate_bars("X", good) == good
    with pytest.raises(StrategyInputError):
        validate_bars("X", list(reversed(good)))
    with pytest.raises(StrategyInputError):
        validate_bars("X", [good[0], good[0]])       # duplicate timestamp
    with pytest.raises(StrategyInputError):
        validate_bars("X", [{"close": 10}])
    with pytest.raises(StrategyInputError):
        validate_bars("X", "not-a-sequence-of-bars")
    assert validate_bars("X", None) == []
    assert closes(good) == [d(10), d(11), d(12)]


# --------------------------------------------------------------------- Keys

def test_idempotency_key_is_stable_for_the_same_decision():
    a = idempotency_key("s1", "BTC/USDT", START, Side.BUY, "entry")
    b = idempotency_key("s1", "BTC/USDT", START, Side.BUY, "entry")
    assert a == b


def test_idempotency_key_changes_with_every_component():
    base = idempotency_key("s1", "BTC/USDT", START, Side.BUY, "entry")
    assert base != idempotency_key("s2", "BTC/USDT", START, Side.BUY, "entry")
    assert base != idempotency_key("s1", "ETH/USDT", START, Side.BUY, "entry")
    assert base != idempotency_key("s1", "BTC/USDT", START + dt.timedelta(minutes=15),
                                   Side.BUY, "entry")
    assert base != idempotency_key("s1", "BTC/USDT", START, Side.SELL, "entry")
    assert base != idempotency_key("s1", "BTC/USDT", START, Side.BUY, "exit")


def test_keys_reject_naive_timestamps():
    naive = dt.datetime(2026, 8, 24, 0, 0)
    with pytest.raises(StrategyInputError):
        idempotency_key("s1", "X", naive, Side.BUY)
    with pytest.raises(StrategyInputError):
        structure_key("s1", "NIFTY", naive)


def test_same_instant_in_another_timezone_is_the_same_decision():
    from core.profiles import IST
    assert idempotency_key("s1", "X", START, Side.BUY) == \
        idempotency_key("s1", "X", START.astimezone(IST), Side.BUY)


def test_floor_ts_buckets_time_driven_decisions():
    ts = utc(2026, 8, 24, 10, 7)
    assert floor_ts(ts, dt.timedelta(minutes=5)) == utc(2026, 8, 24, 10, 5)
    assert floor_ts(ts, dt.timedelta(minutes=5)) == \
        floor_ts(ts + dt.timedelta(seconds=90), dt.timedelta(minutes=5))
    with pytest.raises(StrategyInputError):
        floor_ts(dt.datetime(2026, 8, 24), dt.timedelta(minutes=5))
    with pytest.raises(StrategyInputError):
        floor_ts(ts, dt.timedelta(0))
