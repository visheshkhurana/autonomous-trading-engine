"""0-100 entry score. <65 is a hard no-trade."""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping, Optional, Sequence

from strategies.base import Bar, atr, closes, rsi

from .constants import ATR_PERIOD, MIN_RR, PREFER_RR
from .regime import _stack
from .types import ScoreBreakdown, TradePlan


def score_components(
    *,
    trend: int,
    structure: int,
    momentum: int,
    volume: int,
    reward_risk: int,
    funding: int,
    oi_liq: int,
    cross_market: int,
    options: int,
) -> ScoreBreakdown:
    return ScoreBreakdown(
        trend=trend, structure=structure, momentum=momentum, volume=volume,
        reward_risk=reward_risk, funding=funding, oi_liq=oi_liq,
        cross_market=cross_market, options=options,
    )


def _trend_points(bars: Sequence[Bar]) -> int:
    px = closes(bars)
    close = px[-1]
    e20, e50, e200, e20_prev = _stack(px)
    if e20 is not None and e50 is not None and e200 is not None:
        if close > e20 > e50 > e200 and (e20_prev is None or e20 > e20_prev):
            return 20
        if close > e50 > e200:
            return 14
        if close > e200:
            return 8
        if close < e50 and e20 < e50:
            return 0
        return 4
    if e50 is not None:
        return 10 if close > e50 else 3
    return 4


def _structure_points(bars: Sequence[Bar], entry: Decimal) -> int:
    if len(bars) < 12:
        return 4
    prior = bars[-21:-1] if len(bars) > 21 else bars[:-1]
    if not prior:
        return 4
    prior_high = max(b.high for b in prior[-20:])
    prior_low = min(b.low for b in prior[-10:])
    last = bars[-1]
    if last.close > prior_high and last.close >= entry:
        return 15
    if last.low > prior_low and last.high >= prior_high * Decimal("0.99"):
        return 10
    if last.close < prior_low:
        return 0
    return 4


def _momentum_points(bars: Sequence[Bar]) -> int:
    value = rsi(closes(bars), 14)
    if value is None:
        return 4
    if Decimal("55") <= value <= Decimal("70"):
        return 10
    if Decimal("50") <= value < Decimal("55") or Decimal("70") < value <= Decimal("75"):
        return 7
    if Decimal("45") <= value < Decimal("50") or Decimal("75") < value <= Decimal("80"):
        return 4
    return 1


def _volume_points(bars: Sequence[Bar]) -> int:
    if len(bars) < 21:
        return 4
    window = [b.volume for b in bars[-21:-1]]
    avg = sum(window, Decimal(0)) / len(window)
    last = bars[-1].volume
    if avg <= 0:
        return 4
    if last >= avg * Decimal("1.5"):
        return 10
    if last >= avg:
        return 6
    return 3


def _rr_points(rr: Decimal) -> int:
    if rr >= PREFER_RR:
        return 15
    if rr >= MIN_RR:
        return 10
    return 0


def _funding_points(extras: Mapping[str, object], side_long: bool) -> tuple[int, str]:
    raw = extras.get("funding")
    if not isinstance(raw, Decimal):
        return 5, "unknown"
    if side_long:
        if raw <= Decimal("0"):
            return 10, f"pays longs ({raw})"
        if raw <= Decimal("0.0001"):
            return 6, f"quiet ({raw})"
        return 2, f"crowded longs ({raw})"
    if raw >= Decimal("0"):
        return 10, f"pays shorts ({raw})"
    return 2, f"crowded shorts ({raw})"


def _oi_points(extras: Mapping[str, object], side_long: bool) -> tuple[int, str]:
    change = extras.get("oi_change")
    liq = extras.get("liquidations")
    if change is None and liq is None:
        return 5, "unknown"
    note = f"oi={change or 'n/a'} liq={liq or 'n/a'}"
    if side_long and change == "up" and liq == "short_squeeze":
        return 10, note
    if side_long and change == "down":
        return 4, note
    if not side_long and change == "up" and liq == "long_squeeze":
        return 10, note
    if change == "up":
        return 7, note
    return 5, note


def _cross_points(extras: Mapping[str, object], eth_agrees: Optional[bool]) -> int:
    if extras.get("cross_agree") is True or eth_agrees is True:
        return 5
    if extras.get("cross_agree") is False or eth_agrees is False:
        return 0
    return 2


def _options_points(extras: Mapping[str, object], side_long: bool) -> int:
    skew = extras.get("skew")
    iv = extras.get("iv")
    if skew is None and iv is None:
        return 2
    if side_long and extras.get("skew") == "calls_bid":
        return 5
    if not side_long and extras.get("skew") == "puts_bid":
        return 5
    return 3


def score_plan(
    bars: Sequence[Bar],
    plan_rr: Decimal,
    entry: Decimal,
    extras: Optional[Mapping[str, object]] = None,
    *,
    side_long: bool = True,
    eth_agrees: Optional[bool] = None,
) -> ScoreBreakdown:
    extras = extras or {}
    funding_pts, _ = _funding_points(extras, side_long)
    oi_pts, _ = _oi_points(extras, side_long)
    return score_components(
        trend=_trend_points(bars),
        structure=_structure_points(bars, entry),
        momentum=_momentum_points(bars),
        volume=_volume_points(bars),
        reward_risk=_rr_points(plan_rr),
        funding=funding_pts,
        oi_liq=oi_pts,
        cross_market=_cross_points(extras, eth_agrees),
        options=_options_points(extras, side_long),
    )


def vol_condition(bars: Sequence[Bar]) -> str:
    value = atr(bars, ATR_PERIOD)
    if value is None or len(bars) < ATR_PERIOD + 20:
        return "unknown"
    prior = atr(bars[:-20], ATR_PERIOD)
    if prior is None or prior <= 0:
        return f"atr={value}"
    ratio = value / prior
    if ratio >= Decimal("2"):
        return f"expanded ({ratio:.2f}x)"
    if ratio <= Decimal("0.6"):
        return f"compressed ({ratio:.2f}x)"
    return f"normal ({ratio:.2f}x)"


def annotate_from_extras(
    extras: Mapping[str, object], side_long: bool,
) -> tuple[str, str]:
    _, funding = _funding_points(extras, side_long)
    _, oi = _oi_points(extras, side_long)
    return funding, oi


# Re-export for tests that score a finished plan.
def score_of(plan: TradePlan, extras: Mapping[str, object] | None = None) -> int:
    return plan.score.total
