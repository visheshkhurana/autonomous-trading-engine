"""Regime first. No trade is considered before this returns."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Optional, Sequence

from strategies.base import Bar, atr, closes, ema, ema_series

from .constants import ATR_PERIOD, EMA_FAST, EMA_MID, EMA_SLOW, MARKET_KEY
from .types import Regime


@dataclass(frozen=True)
class RegimeRead:
    regime: Regime
    detail: str
    btc_ema20: Optional[Decimal] = None
    btc_ema50: Optional[Decimal] = None
    btc_ema200: Optional[Decimal] = None


def _higher_highs_higher_lows(bars: Sequence[Bar], lookback: int = 20) -> Optional[bool]:
    window = bars[-lookback:] if len(bars) >= lookback else bars
    if len(window) < 6:
        return None
    mid = len(window) // 2
    first, second = window[:mid], window[mid:]
    hh = max(b.high for b in second) > max(b.high for b in first)
    hl = min(b.low for b in second) > min(b.low for b in first)
    lh = max(b.high for b in second) < max(b.high for b in first)
    ll = min(b.low for b in second) < min(b.low for b in first)
    if hh and hl:
        return True
    if lh and ll:
        return False
    return None


def _stack(px: Sequence[Decimal]) -> tuple[
        Optional[Decimal], Optional[Decimal], Optional[Decimal],
        Optional[Decimal]]:
    e20 = ema(px, EMA_FAST)
    e50 = ema(px, EMA_MID)
    e200 = ema(px, EMA_SLOW)
    e20_prev: Optional[Decimal] = None
    series20 = ema_series(px, EMA_FAST)
    if len(series20) >= 2:
        e20_prev = series20[-2]
    return e20, e50, e200, e20_prev


def _bear_stack(close: Decimal, e20: Optional[Decimal],
                e50: Optional[Decimal], e200: Optional[Decimal]) -> bool:
    """A 20/50 dip without a 200-EMA is a range, not a bear call."""
    if e20 is not None and e50 is not None and e200 is not None:
        return close < e50 and e20 < e50 < e200
    return False


def _strong_stack(close: Decimal, e20: Optional[Decimal],
                  e50: Optional[Decimal], e200: Optional[Decimal],
                  e20_prev: Optional[Decimal]) -> bool:
    if e20 is None or e50 is None or e200 is None:
        return False
    rising = e20_prev is None or e20 > e20_prev
    return close > e20 > e50 > e200 and rising


def _weak_bull(close: Decimal, e50: Optional[Decimal],
               e200: Optional[Decimal]) -> bool:
    if e200 is not None and close > e200:
        if e50 is None or close > e50:
            return True
    if e50 is not None and close > e50 and e200 is None:
        return True
    return False


def _vol_crash(bars: Sequence[Bar]) -> bool:
    if len(bars) < ATR_PERIOD + 20:
        return False
    now = atr(bars, ATR_PERIOD)
    then = atr(bars[:-20], ATR_PERIOD)
    if now is None or then is None or then <= 0:
        return False
    return now / then >= Decimal("3")


def extras_from_ctx(extra: Mapping[str, object]) -> Mapping[str, object]:
    raw = extra.get(MARKET_KEY)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{MARKET_KEY} must be a mapping, got {type(raw).__name__}")
    return raw


def classify_regime(
    btc_bars: Sequence[Bar],
    eth_bars: Optional[Sequence[Bar]] = None,
    extras: Optional[Mapping[str, object]] = None,
) -> RegimeRead:
    """A / B / C / D from BTC structure first, then ETH and tape extras.

    Missing 200-EMA history cannot produce STRONG_BULL — that would be a
    fabricated regime. Missing extras are ignored, never invented as bullish.
    """
    extras = extras or {}
    if len(btc_bars) < EMA_MID:
        return RegimeRead(
            Regime.RANGE_NEUTRAL,
            f"insufficient BTC history ({len(btc_bars)} bars); "
            "holding USDT until regime can be read",
        )

    px = closes(btc_bars)
    close = px[-1]
    e20, e50, e200, e20_prev = _stack(px)
    structure = _higher_highs_higher_lows(btc_bars)
    crash = _vol_crash(btc_bars)

    eth_bear = False
    eth_strong = False
    if eth_bars and len(eth_bars) >= EMA_MID:
        epx = closes(eth_bars)
        ee20, ee50, ee200, ee20p = _stack(epx)
        eth_bear = _bear_stack(epx[-1], ee20, ee50, ee200)
        eth_strong = _strong_stack(epx[-1], ee20, ee50, ee200, ee20p)

    funding = extras.get("funding")
    crowded = (
        isinstance(funding, Decimal) and funding > Decimal("0.0005")
        and extras.get("oi_change") == "up"
    )
    btc_d = extras.get("btc_dominance")
    basis = extras.get("basis")

    if crash and (structure is False or _bear_stack(close, e20, e50, e200)):
        return RegimeRead(
            Regime.BEAR,
            f"BTC crash vol with down structure; close {close} "
            f"EMA20/50/200={e20}/{e50}/{e200}",
            e20, e50, e200,
        )

    if _bear_stack(close, e20, e50, e200) or (
            structure is False and e50 is not None and close < e50
            and (e200 is None or close < e200)):
        return RegimeRead(
            Regime.BEAR,
            f"BTC bear stack/structure; close {close} "
            f"EMA20/50/200={e20}/{e50}/{e200}",
            e20, e50, e200,
        )

    if _strong_stack(close, e20, e50, e200, e20_prev) and structure is not False:
        if eth_bear or crowded:
            return RegimeRead(
                Regime.WEAK_BULL,
                "BTC stacked but ETH/funding contradicts; downgraded from "
                f"STRONG_BULL. close {close} EMA20/50/200={e20}/{e50}/{e200}",
                e20, e50, e200,
            )
        note = f"BTC strong stack close {close} > EMA{EMA_FAST} {e20} > EMA{EMA_MID} {e50} > EMA{EMA_SLOW} {e200}"
        if eth_strong:
            note += "; ETH agrees"
        if btc_d == "up":
            note += "; BTC.D rising"
        if isinstance(basis, Decimal) and basis > 0:
            note += f"; spot/perp basis {basis}"
        return RegimeRead(Regime.STRONG_BULL, note, e20, e50, e200)

    if _weak_bull(close, e50, e200) and not eth_bear:
        return RegimeRead(
            Regime.WEAK_BULL,
            f"BTC above longer EMA (close {close}, EMA50={e50}, EMA200={e200}) "
            "without a full 20/50/200 stack",
            e20, e50, e200,
        )

    return RegimeRead(
        Regime.RANGE_NEUTRAL,
        f"BTC range/neutral; close {close} EMA20/50/200={e20}/{e50}/{e200}. "
        "Holding USDT is valid.",
        e20, e50, e200,
    )
