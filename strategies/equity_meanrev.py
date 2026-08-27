"""
Intraday mean reversion on liquid cash equities.

Long only. Not because shorting is impossible on NSE intraday, but because a
short in cash equity has an unbounded loss and no structural cap, and this book
already carries directional risk through the option spread. The asymmetry is a
deliberate choice, not an oversight.

Three exits, checked in this order, and at most ONE is emitted per position per
tick:

  1. hard stop      — a fixed percentage below the entry price;
  2. session flatten — EquityProfile.must_flatten_intraday is True, and an
                       intraday position carried overnight is a different trade
                       than the one that was sized;
  3. target         — price has reverted to the mean.

The stop is a fixed percentage of the ENTRY price and not a multiple of the
current ATR, deliberately. An ATR stop recomputed each tick widens exactly when
volatility spikes, i.e. it runs away from price during the move it exists to
cap.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Optional, Sequence

from core.profiles import IST
from core.types import AssetClass, Instrument, Intent, MarketContext, Side

from .base import (
    Bar, MissingDataError, StrategyBase, StrategyInputError, atr, closes, rsi, sma,
)


class EquityMeanReversionStrategy(StrategyBase):

    def __init__(self, *, instruments: Sequence[Instrument],
                 strategy_id: str = "equity_meanrev",
                 rsi_period: int = 14,
                 sma_period: int = 20,
                 atr_period: int = 14,
                 rsi_oversold: Decimal = Decimal("30"),
                 entry_atr_mult: Decimal = Decimal("1.5"),
                 stop_pct: Decimal = Decimal("0.015"),
                 no_new_entries_after: dt.time = dt.time(14, 45),
                 flatten_after: dt.time = dt.time(15, 15),
                 bar_interval: dt.timedelta = dt.timedelta(minutes=5),
                 slippage_bps: Decimal = Decimal("10")) -> None:
        if not instruments:
            raise StrategyInputError("equity_meanrev needs at least one instrument")
        for i in instruments:
            if i.asset_class is not AssetClass.EQUITY:
                raise StrategyInputError(f"{i.symbol} is not an EQUITY instrument")
        symbols = [i.symbol for i in instruments]
        if len(set(symbols)) != len(symbols):
            raise StrategyInputError(f"duplicate instruments in universe: {symbols}")
        if not (Decimal(0) < stop_pct < Decimal(1)):
            raise StrategyInputError("stop_pct must be in (0,1)")
        if entry_atr_mult <= 0:
            raise StrategyInputError("entry_atr_mult must be > 0")
        if not (Decimal(0) < rsi_oversold < Decimal(100)):
            raise StrategyInputError("rsi_oversold must be in (0,100)")
        if flatten_after <= no_new_entries_after:
            raise StrategyInputError("flatten_after must be later than no_new_entries_after")
        if flatten_after >= dt.time(15, 30):
            # 15:30 IST is the close. A flatten scheduled at or after it never
            # runs, and the position is carried overnight by accident.
            raise StrategyInputError("flatten_after must be before the 15:30 IST close")
        super().__init__(
            strategy_id=strategy_id, asset_class=AssetClass.EQUITY,
            bar_interval=bar_interval, slippage_bps=slippage_bps,
        )
        self.instruments = tuple(instruments)
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.rsi_oversold = rsi_oversold
        self.entry_atr_mult = entry_atr_mult
        self.stop_pct = stop_pct
        self.no_new_entries_after = no_new_entries_after
        self.flatten_after = flatten_after

    @property
    def min_bars(self) -> int:
        return max(self.sma_period, self.rsi_period + 1, self.atr_period + 1)

    def propose(self, ctx: MarketContext) -> Sequence[Intent]:
        if not self.profile.is_open(ctx.now):
            # Nothing is placeable outside the session, and an intent that the
            # gate will certainly reject is noise in the approval log.
            return []

        local = ctx.now.astimezone(IST).time()
        intents: list[Intent] = []
        for instrument in self.instruments:
            pos = self.own_position(ctx, instrument.symbol)
            if pos is not None and pos.qty < 0:
                raise StrategyInputError(
                    f"{self.id}: holds a SHORT {instrument.symbol} ({pos.qty}) but this strategy "
                    "only ever buys; the position store and the strategy disagree"
                )
            bars = self.bars_for(ctx, instrument.symbol,
                                 min_bars=self.min_bars, must_have=pos is not None)
            if bars is None:
                continue
            if pos is not None:
                exit_intent = self._exit(ctx, instrument, pos.avg_price, pos.qty, bars, local)
                if exit_intent is not None:
                    intents.append(exit_intent)
                continue
            if self.held_by_anyone(ctx, instrument.symbol):
                continue
            if local >= self.no_new_entries_after:
                continue
            entry = self._entry(ctx, instrument, bars)
            if entry is not None:
                intents.append(entry)
        return self.finalize(intents)

    # -- entry -------------------------------------------------------------
    def _entry(self, ctx: MarketContext, instrument: Instrument,
               bars: Sequence[Bar]) -> Optional[Intent]:
        px = closes(bars)
        mean = sma(px, self.sma_period)
        strength = rsi(px, self.rsi_period)
        vol = atr(bars, self.atr_period)
        if mean is None or strength is None or vol is None or vol <= 0:
            return None
        last = bars[-1]
        band = mean - self.entry_atr_mult * vol
        if not (strength < self.rsi_oversold and last.close < band):
            return None

        quote = ctx.quote(instrument.symbol)
        if quote is None:
            # NFO/NSE entries are marketable LIMITs; without a live price there
            # is nothing honest to put in the limit field.
            return None
        stop = quote * (Decimal(1) - self.stop_pct)
        distance = (mean - last.close) / vol
        reason = (
            f"{instrument.symbol} mean-reversion entry: RSI({self.rsi_period}) "
            f"{strength.quantize(Decimal('0.01'))} below {self.rsi_oversold}, close "
            f"{last.close} is {distance.quantize(Decimal('0.01'))} ATR below SMA"
            f"({self.sma_period}) {mean.quantize(Decimal('0.01'))} "
            f"(ATR {vol.quantize(Decimal('0.01'))}). Target the mean; hard stop "
            f"{self.stop_pct * 100}% below entry (~{stop.quantize(Decimal('0.01'))}); "
            f"flat by {self.flatten_after.strftime('%H:%M')} IST."
        )
        return self.make_intent(
            instrument=instrument, side=Side.BUY,
            qty=self.entry_unit_qty(instrument),
            reason=reason, decision_ts=last.ts, tag="entry", ref_price=quote,
        )

    # -- exit --------------------------------------------------------------
    def _exit(self, ctx: MarketContext, instrument: Instrument, entry_price: Decimal,
              qty: Decimal, bars: Sequence[Bar], local: dt.time) -> Optional[Intent]:
        last = bars[-1]
        quote = ctx.quote(instrument.symbol)
        if quote is None:
            raise MissingDataError(
                f"{self.id}: no quote for {instrument.symbol} while long {qty}; cannot price an "
                "exit and will not send an unpriced order for a live position"
            )
        if entry_price <= 0:
            raise StrategyInputError(
                f"{self.id}: {instrument.symbol} position carries avg_price {entry_price}; a "
                "non-positive entry price makes the stop meaningless"
            )
        stop = entry_price * (Decimal(1) - self.stop_pct)
        mean = sma(closes(bars), self.sma_period)

        if quote <= stop:
            reason = (
                f"{instrument.symbol} hard stop: last {quote} at or below stop "
                f"{stop.quantize(Decimal('0.01'))} "
                f"({self.stop_pct * 100}% under entry {entry_price}). Cutting the trade."
            )
        elif local >= self.flatten_after:
            reason = (
                f"{instrument.symbol} session flatten: {local.strftime('%H:%M')} IST is past the "
                f"{self.flatten_after.strftime('%H:%M')} cutoff — intraday positions are not "
                f"carried overnight. Last {quote}, entry {entry_price}."
            )
        elif mean is not None and quote >= mean:
            reason = (
                f"{instrument.symbol} target reached: last {quote} back at SMA"
                f"({self.sma_period}) {mean.quantize(Decimal('0.01'))}; the reversion this trade "
                f"was opened for has happened. Entry {entry_price}."
            )
        else:
            return None

        return self.make_intent(
            instrument=instrument, side=Side.SELL,
            # Exactly what is held; quantizing an exit down leaves a residue.
            qty=abs(qty),
            reason=reason, decision_ts=last.ts, tag="exit", ref_price=quote,
        )
