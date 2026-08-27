"""
OKX spot EMA-cross trend follower with an ATR-scaled stop.

Long only — this is spot, there is nothing to borrow.

The stop is `entry - atr_mult * ATR`, floored by a hard percentage of the entry
price:

    stop = max(entry - atr_mult * ATR, entry * (1 - max_loss_pct))

`max()` takes the TIGHTER of the two. That floor is the point: an ATR stop
recomputed each bar widens exactly when volatility expands, so on the one move
it exists to cap it slides further away from price. The percentage floor bounds
the loss regardless of what ATR does.

Anchoring the stop to the position's own avg_price rather than to bar history
also means the exit does not depend on `opened_at` being populated, or on the
bar window still reaching back to the entry. There is no path here where a
missing timestamp leaves a live position without a stop.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Optional, Sequence

from core.types import AssetClass, Instrument, Intent, MarketContext, Side

from .base import (
    Bar, StrategyBase, StrategyInputError, atr, closes, ema_series,
)


class CryptoEmaTrendStrategy(StrategyBase):

    def __init__(self, *, instruments: Sequence[Instrument],
                 strategy_id: str = "crypto_trend",
                 fast_period: int = 12,
                 slow_period: int = 26,
                 atr_period: int = 14,
                 atr_stop_mult: Decimal = Decimal("2.0"),
                 max_loss_pct: Decimal = Decimal("0.08"),
                 bar_interval: dt.timedelta = dt.timedelta(minutes=15)) -> None:
        if not instruments:
            raise StrategyInputError("crypto_trend needs at least one instrument")
        for i in instruments:
            if i.asset_class is not AssetClass.CRYPTO:
                raise StrategyInputError(f"{i.symbol} is not a CRYPTO instrument")
        symbols = [i.symbol for i in instruments]
        if len(set(symbols)) != len(symbols):
            raise StrategyInputError(f"duplicate instruments in universe: {symbols}")
        if fast_period >= slow_period:
            raise StrategyInputError("fast_period must be shorter than slow_period")
        if atr_stop_mult <= 0:
            raise StrategyInputError("atr_stop_mult must be > 0")
        if not (Decimal(0) < max_loss_pct < Decimal(1)):
            raise StrategyInputError("max_loss_pct must be in (0,1)")
        super().__init__(
            strategy_id=strategy_id, asset_class=AssetClass.CRYPTO,
            bar_interval=bar_interval,
        )
        self.instruments = tuple(instruments)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_loss_pct = max_loss_pct

    @property
    def min_bars(self) -> int:
        # slow_period + 1 closes so the slow EMA exists on the previous bar too
        # — a cross is a statement about two bars, not one.
        return max(self.slow_period + 1, self.atr_period + 1)

    def propose(self, ctx: MarketContext) -> Sequence[Intent]:
        intents: list[Intent] = []
        for instrument in self.instruments:
            pos = self.own_position(ctx, instrument.symbol)
            if pos is not None and pos.qty < 0:
                raise StrategyInputError(
                    f"{self.id}: holds a SHORT {instrument.symbol} ({pos.qty}) on spot, which "
                    "this strategy never opens; the position store and the strategy disagree"
                )
            bars = self.bars_for(ctx, instrument.symbol,
                                 min_bars=self.min_bars, must_have=pos is not None)
            if bars is None:
                continue
            if pos is not None:
                exit_intent = self._exit(ctx, instrument, pos.avg_price, pos.qty, bars)
                if exit_intent is not None:
                    intents.append(exit_intent)
                continue
            if self.held_by_anyone(ctx, instrument.symbol):
                continue
            entry = self._entry(instrument, bars)
            if entry is not None:
                intents.append(entry)
        return self.finalize(intents)

    # -- signal ------------------------------------------------------------
    def _emas(self, bars: Sequence[Bar]) -> Optional[tuple[Decimal, Decimal, Decimal, Decimal]]:
        """(fast_now, slow_now, fast_prev, slow_prev) or None."""
        px = closes(bars)
        fast = ema_series(px, self.fast_period)
        slow = ema_series(px, self.slow_period)
        if len(fast) < 2 or len(slow) < 2:
            return None
        return fast[-1], slow[-1], fast[-2], slow[-2]

    def _entry(self, instrument: Instrument, bars: Sequence[Bar]) -> Optional[Intent]:
        emas = self._emas(bars)
        if emas is None:
            return None
        fast, slow, fast_prev, slow_prev = emas
        # A cross, not merely "fast above slow": without the previous-bar
        # condition this re-enters on every bar of an established trend.
        if not (fast > slow and fast_prev <= slow_prev):
            return None
        vol = atr(bars, self.atr_period)
        if vol is None or vol <= 0:
            return None
        last = bars[-1]
        stop = self._stop_level(last.close, vol)
        reason = (
            f"{instrument.symbol} EMA cross up: EMA({self.fast_period}) "
            f"{fast.quantize(Decimal('0.01'))} crossed above EMA({self.slow_period}) "
            f"{slow.quantize(Decimal('0.01'))} on the bar closing {last.ts.isoformat()} at "
            f"{last.close}. Stop starts at {stop.quantize(Decimal('0.01'))} "
            f"({self.atr_stop_mult}x ATR({self.atr_period}) "
            f"{vol.quantize(Decimal('0.01'))}, floored at "
            f"{self.max_loss_pct * 100}% below entry)."
        )
        return self.make_intent(
            instrument=instrument, side=Side.BUY,
            qty=self.entry_unit_qty(instrument),
            reason=reason, decision_ts=last.ts, tag="entry", ref_price=last.close,
        )

    def _stop_level(self, entry_price: Decimal, vol: Decimal) -> Decimal:
        atr_stop = entry_price - self.atr_stop_mult * vol
        hard_floor = entry_price * (Decimal(1) - self.max_loss_pct)
        return max(atr_stop, hard_floor)

    def _exit(self, ctx: MarketContext, instrument: Instrument, entry_price: Decimal,
              qty: Decimal, bars: Sequence[Bar]) -> Optional[Intent]:
        if entry_price <= 0:
            raise StrategyInputError(
                f"{self.id}: {instrument.symbol} position carries avg_price {entry_price}; a "
                "non-positive entry price makes the stop meaningless"
            )
        vol = atr(bars, self.atr_period)
        if vol is None:
            # bars_for already guaranteed enough history for an open position,
            # so this is unreachable unless atr_period changed under us.
            raise StrategyInputError(
                f"{self.id}: cannot compute ATR({self.atr_period}) for open {instrument.symbol}"
            )
        last = bars[-1]
        stop = self._stop_level(entry_price, vol)
        if last.close <= stop:
            which = ("hard floor" if stop == entry_price * (Decimal(1) - self.max_loss_pct)
                     else f"{self.atr_stop_mult}x ATR")
            reason = (
                f"{instrument.symbol} stop hit: close {last.close} at or below "
                f"{stop.quantize(Decimal('0.01'))} ({which} below entry {entry_price}, "
                f"ATR({self.atr_period}) {vol.quantize(Decimal('0.01'))}). Cutting the trade."
            )
        else:
            emas = self._emas(bars)
            if emas is None:
                return None
            fast, slow, fast_prev, slow_prev = emas
            if not (fast < slow and fast_prev >= slow_prev):
                return None
            reason = (
                f"{instrument.symbol} EMA cross down: EMA({self.fast_period}) "
                f"{fast.quantize(Decimal('0.01'))} crossed below EMA({self.slow_period}) "
                f"{slow.quantize(Decimal('0.01'))} at {last.close}; the trend this position was "
                f"opened for has ended. Entry {entry_price}."
            )
        return self.make_intent(
            instrument=instrument, side=Side.SELL,
            qty=abs(qty),
            reason=reason, decision_ts=last.ts, tag="exit", ref_price=last.close,
        )
