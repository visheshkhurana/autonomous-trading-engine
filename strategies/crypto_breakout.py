"""
OKX spot Donchian range breakout with a volatility filter.

Long only (spot). Entry: the close takes out the highest high of the previous
`entry_period` bars. Exit: the close loses the lowest low of the previous
`exit_period` bars, or a hard percentage stop, whichever comes first.

Two details that decide whether this works at all:

  * The channel EXCLUDES the current bar. A bar's own high is part of its own
    channel and close <= high always, so a breakout measured against a channel
    that includes the current bar can never trigger. `donchian(exclude_last=True)`
    is the default for that reason.
  * The volatility filter is two-sided. Below `min_atr_pct` the "range" is
    noise and the breakout is a spread-paying accident; above `max_atr_pct` the
    move has already happened and the channel stop sits an unacceptable
    distance below. Both ends are refusals to trade, never a resize — sizing
    belongs to the gate.

The exit channel is deliberately shorter than the entry channel; a symmetric
channel gives back most of a trend before admitting it ended.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Optional, Sequence

from core.types import AssetClass, Instrument, Intent, MarketContext, Side

from .base import Bar, StrategyBase, StrategyInputError, atr, donchian


class CryptoBreakoutStrategy(StrategyBase):

    def __init__(self, *, instruments: Sequence[Instrument],
                 strategy_id: str = "crypto_breakout",
                 entry_period: int = 20,
                 exit_period: int = 10,
                 atr_period: int = 14,
                 min_atr_pct: Decimal = Decimal("0.004"),
                 max_atr_pct: Decimal = Decimal("0.06"),
                 max_loss_pct: Decimal = Decimal("0.08"),
                 bar_interval: dt.timedelta = dt.timedelta(hours=1)) -> None:
        if not instruments:
            raise StrategyInputError("crypto_breakout needs at least one instrument")
        for i in instruments:
            if i.asset_class is not AssetClass.CRYPTO:
                raise StrategyInputError(f"{i.symbol} is not a CRYPTO instrument")
        symbols = [i.symbol for i in instruments]
        if len(set(symbols)) != len(symbols):
            raise StrategyInputError(f"duplicate instruments in universe: {symbols}")
        if entry_period < 2:
            raise StrategyInputError("entry_period must be >= 2")
        if not (1 <= exit_period < entry_period):
            raise StrategyInputError("exit_period must be >= 1 and shorter than entry_period")
        if not (Decimal(0) <= min_atr_pct < max_atr_pct):
            raise StrategyInputError("need 0 <= min_atr_pct < max_atr_pct")
        if not (Decimal(0) < max_loss_pct < Decimal(1)):
            raise StrategyInputError("max_loss_pct must be in (0,1)")
        super().__init__(
            strategy_id=strategy_id, asset_class=AssetClass.CRYPTO,
            bar_interval=bar_interval,
        )
        self.instruments = tuple(instruments)
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.max_loss_pct = max_loss_pct

    @property
    def min_bars(self) -> int:
        # +1 everywhere because the channel excludes the current bar.
        return max(self.entry_period + 1, self.exit_period + 1, self.atr_period + 1)

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
            quote = self.known_quote(ctx, instrument.symbol,
                                     must_have=pos is not None)
            if quote is None:
                continue
            if pos is not None:
                exit_intent = self._exit(instrument, pos.avg_price, pos.qty,
                                         bars, quote)
                if exit_intent is not None:
                    intents.append(exit_intent)
                continue
            if self.held_by_anyone(ctx, instrument.symbol):
                continue
            entry = self._entry(instrument, bars, quote)
            if entry is not None:
                intents.append(entry)
        return self.finalize(intents)

    def _atr_pct(self, bars: Sequence[Bar]) -> Optional[tuple[Decimal, Decimal]]:
        vol = atr(bars, self.atr_period)
        if vol is None:
            return None
        close = bars[-1].close
        # close > 0 is enforced by Bar, so this division is safe.
        return vol, vol / close

    def _entry(self, instrument: Instrument, bars: Sequence[Bar],
               quote: Decimal) -> Optional[Intent]:
        channel = donchian(bars, self.entry_period)
        if channel is None:
            return None
        last = bars[-1]
        if last.close <= channel.upper:
            return None
        measured = self._atr_pct(bars)
        if measured is None:
            return None
        vol, vol_pct = measured
        if vol_pct < self.min_atr_pct:
            return None
        if vol_pct > self.max_atr_pct:
            return None
        stop = last.close * (Decimal(1) - self.max_loss_pct)
        exit_channel = donchian(bars, self.exit_period)
        exit_txt = ("n/a" if exit_channel is None
                    else str(exit_channel.lower.quantize(Decimal('0.01'))))
        reason = (
            f"{instrument.symbol} breakout: close {last.close} took out the "
            f"{self.entry_period}-bar high {channel.upper} on the bar closing "
            f"{last.ts.isoformat()}. ATR({self.atr_period}) "
            f"{vol.quantize(Decimal('0.01'))} = "
            f"{(vol_pct * 100).quantize(Decimal('0.01'))}% of price, inside the "
            f"{(self.min_atr_pct * 100)}%-{(self.max_atr_pct * 100)}% band. Exit on a close "
            f"below the {self.exit_period}-bar low ({exit_txt}) or the hard stop "
            f"{stop.quantize(Decimal('0.01'))}."
        )
        return self.make_intent(
            instrument=instrument, side=Side.BUY,
            qty=self.entry_unit_qty(instrument),
            reason=reason, decision_ts=last.ts, tag="entry", ref_price=quote,
            stop_price=stop,
        )

    def protective_stop(self, ctx, position):
        """Hard percentage floor under the fill. The channel exit is the
        strategy's own sell, not a second exchange price."""
        if position.avg_price <= 0:
            return None
        return position.avg_price * (Decimal(1) - self.max_loss_pct)

    def _exit(self, instrument: Instrument, entry_price: Decimal, qty: Decimal,
              bars: Sequence[Bar], quote: Decimal) -> Optional[Intent]:
        if entry_price <= 0:
            raise StrategyInputError(
                f"{self.id}: {instrument.symbol} position carries avg_price {entry_price}; a "
                "non-positive entry price makes the stop meaningless"
            )
        last = bars[-1]
        hard_stop = entry_price * (Decimal(1) - self.max_loss_pct)
        channel = donchian(bars, self.exit_period)
        if channel is None:
            raise StrategyInputError(
                f"{self.id}: cannot build the {self.exit_period}-bar exit channel for open "
                f"{instrument.symbol}"
            )
        # Hard stop first: when both trigger it is the tighter statement about
        # what went wrong, and only ONE exit may be emitted per position.
        if last.close <= hard_stop:
            reason = (
                f"{instrument.symbol} hard stop: close {last.close} at or below "
                f"{hard_stop.quantize(Decimal('0.01'))} ({self.max_loss_pct * 100}% under entry "
                f"{entry_price}) before the channel exit "
                f"{channel.lower.quantize(Decimal('0.01'))} was reached."
            )
        elif last.close < channel.lower:
            reason = (
                f"{instrument.symbol} channel exit: close {last.close} lost the "
                f"{self.exit_period}-bar low {channel.lower}; the breakout has failed. "
                f"Entry {entry_price}."
            )
        else:
            return None
        return self.make_intent(
            instrument=instrument, side=Side.SELL,
            qty=abs(qty),
            reason=reason, decision_ts=last.ts, tag="exit", ref_price=quote,
        )
