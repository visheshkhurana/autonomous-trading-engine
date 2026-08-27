"""
Shared machinery for strategies. Everything here is pure: given the same
MarketContext it returns the same Intents, on any machine, at any time. No
clock, no env, no socket, no file. `ctx.now` is the only notion of time, which
is what lets a backtest and the live loop run identical code.

How this module treats bad input matters more than the happy path:

  * MALFORMED input (a bar with high < low, a naive timestamp, a float where a
    Decimal belongs) is a wiring error and RAISES. A strategy that quietly
    reinterprets corrupt input is how a book ends up positioned on numbers
    nobody checked.
  * ABSENT or INSUFFICIENT input (fewer bars than the indicator needs, no
    regime published yet) is a legitimate state: no entries, no exception.
  * ABSENT or STALE input for an instrument we currently HOLD raises
    MissingDataError. This is the case worth being loud about — returning "no
    intents" there is indistinguishable from "nothing to do" while an open
    position goes unmanaged.

Sizing is deliberately NOT done here. A strategy proposes ONE UNIT (one lot for
equity, one unit of the base asset for crypto) and the gate scales and caps it.
For a multi-leg structure the gate must scale every leg sharing a structure_id
by the SAME factor; scaling legs independently turns a defined-risk spread into
a naked short.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Mapping, Optional, Sequence

from core.profiles import profile_for
from core.types import (
    AssetClass, Instrument, Intent, MarketContext, OrderType, Position, Side,
)

# The runner publishes closed bars here: Mapping[symbol, Sequence[Bar]].
BARS_KEY = "bars"

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


class StrategyInputError(ValueError):
    """The strategy was handed something it cannot interpret. Raised rather
    than guessed at: every guess here becomes a real order."""


class MissingDataError(RuntimeError):
    """Something required to manage an OPEN position could not be established.
    Never downgraded to an empty result — silence reads as 'nothing to do'."""


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

def to_decimal(value: object, *, field: str) -> Decimal:
    """Floats are rejected outright. Decimal(0.1) is 0.1000000000000000055...,
    and a price that is silently 5e-18 off compounds into a tick mismatch the
    broker rejects — after the other leg has already filled."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or isinstance(value, float):
        raise StrategyInputError(f"{field}: pass Decimal, int or str — not {type(value).__name__}")
    if isinstance(value, (int, str)):
        return Decimal(value)
    raise StrategyInputError(f"{field}: cannot read {type(value).__name__} as a Decimal")


def floor_ts(ts: dt.datetime, interval: dt.timedelta) -> dt.datetime:
    """Floor an instant onto an interval grid.

    Time-driven decisions (a session-close flatten, an expiry unwind) are keyed
    to a floored `now` so that repeated ticks inside one bucket produce the SAME
    idempotency key. Keying them to the raw tick would emit a fresh, undeduped
    exit order every few seconds.
    """
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise StrategyInputError("floor_ts needs a timezone-aware datetime")
    if interval <= dt.timedelta(0):
        raise StrategyInputError("floor_ts needs a positive interval")
    return _EPOCH + ((ts - _EPOCH) // interval) * interval


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Bar:
    """One CLOSED bar. The runner must not publish the in-progress bar: a bar
    that is still forming changes under a strategy that has already acted on
    it, and the same tick then produces a different decision."""
    ts: dt.datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None or self.ts.utcoffset() is None:
            raise StrategyInputError("Bar.ts must be timezone-aware")
        for name in ("open", "high", "low", "close", "volume"):
            v = getattr(self, name)
            if not isinstance(v, Decimal):
                raise StrategyInputError(f"Bar.{name} must be Decimal, got {type(v).__name__}")
        for name in ("open", "high", "low", "close"):
            if getattr(self, name) <= 0:
                raise StrategyInputError(f"Bar.{name} must be > 0 (got {getattr(self, name)})")
        if self.volume < 0:
            raise StrategyInputError("Bar.volume must be >= 0")
        if self.high < self.low:
            raise StrategyInputError(f"Bar at {self.ts}: high {self.high} < low {self.low}")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise StrategyInputError(
                f"Bar at {self.ts}: high/low {self.high}/{self.low} do not bracket "
                f"open/close {self.open}/{self.close}"
            )

    @classmethod
    def of(cls, ts: dt.datetime, open: object, high: object, low: object,
           close: object, volume: object = 0) -> "Bar":
        return cls(
            ts=ts,
            open=to_decimal(open, field="open"),
            high=to_decimal(high, field="high"),
            low=to_decimal(low, field="low"),
            close=to_decimal(close, field="close"),
            volume=to_decimal(volume, field="volume"),
        )


def validate_bars(symbol: str, raw: object) -> list[Bar]:
    """Reject a malformed series loudly. Out-of-order or duplicated timestamps
    mean the feed merged two sources or replayed a gap; every indicator below
    would return a confident number computed on nonsense."""
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, SequenceABC):
        raise StrategyInputError(f"bars[{symbol}] must be a sequence of Bar, got {type(raw).__name__}")
    bars: list[Bar] = []
    prev: Optional[dt.datetime] = None
    for i, b in enumerate(raw):
        if not isinstance(b, Bar):
            raise StrategyInputError(f"bars[{symbol}][{i}] is {type(b).__name__}, expected Bar")
        if prev is not None and b.ts <= prev:
            raise StrategyInputError(
                f"bars[{symbol}] not strictly increasing: {b.ts} follows {prev}"
            )
        prev = b.ts
        bars.append(b)
    return bars


def closes(bars: Sequence[Bar]) -> list[Decimal]:
    return [b.close for b in bars]


# --------------------------------------------------------------------------
# Indicators — pure functions. All of them return None when there is not
# enough history. Never 0: a 0 EMA or a 0 ATR reads as a real, extreme number
# and produces a real, extreme order.
# --------------------------------------------------------------------------

def sma(values: Sequence[Decimal], period: int) -> Optional[Decimal]:
    if period < 1:
        raise StrategyInputError("sma period must be >= 1")
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window, Decimal(0)) / period


def ema_series(values: Sequence[Decimal], period: int) -> list[Decimal]:
    """Wilder-style seeding: the first EMA value is the SMA of the first
    `period` observations. Result[i] corresponds to values[period - 1 + i]."""
    if period < 1:
        raise StrategyInputError("ema period must be >= 1")
    if len(values) < period:
        return []
    k = Decimal(2) / Decimal(period + 1)
    seed = sum(values[:period], Decimal(0)) / period
    out = [seed]
    for v in values[period:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out


def ema(values: Sequence[Decimal], period: int) -> Optional[Decimal]:
    series = ema_series(values, period)
    return series[-1] if series else None


def true_range(prev_close: Decimal, bar: Bar) -> Decimal:
    """The gap terms are the point of TR: on a gap open the high-low range
    understates the move that actually happened to a held position."""
    return max(
        bar.high - bar.low,
        abs(bar.high - prev_close),
        abs(bar.low - prev_close),
    )


def true_ranges(bars: Sequence[Bar]) -> list[Decimal]:
    return [true_range(bars[i - 1].close, bars[i]) for i in range(1, len(bars))]


def atr(bars: Sequence[Bar], period: int) -> Optional[Decimal]:
    """Wilder's ATR. Needs period + 1 bars because the first true range needs a
    previous close."""
    if period < 1:
        raise StrategyInputError("atr period must be >= 1")
    trs = true_ranges(bars)
    if len(trs) < period:
        return None
    value = sum(trs[:period], Decimal(0)) / period
    for tr in trs[period:]:
        value = (value * (period - 1) + tr) / period
    return value


def rsi(values: Sequence[Decimal], period: int) -> Optional[Decimal]:
    """Wilder's RSI. A zero average loss returns 100 rather than dividing by
    zero — that is the definition, not a swallowed error."""
    if period < 1:
        raise StrategyInputError("rsi period must be >= 1")
    if len(values) < period + 1:
        return None
    gains: list[Decimal] = []
    losses: list[Decimal] = []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(delta if delta > 0 else Decimal(0))
        losses.append(-delta if delta < 0 else Decimal(0))
    avg_gain = sum(gains[:period], Decimal(0)) / period
    avg_loss = sum(losses[:period], Decimal(0)) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return Decimal(100)
    rs = avg_gain / avg_loss
    return Decimal(100) - (Decimal(100) / (Decimal(1) + rs))


@dataclass(frozen=True)
class DonchianChannel:
    upper: Decimal
    lower: Decimal

    @property
    def mid(self) -> Decimal:
        return (self.upper + self.lower) / 2


def donchian(bars: Sequence[Bar], period: int, *, exclude_last: bool = True) -> Optional[DonchianChannel]:
    """Channel over `period` bars.

    exclude_last defaults to True and that default is load-bearing: a bar's own
    high is part of its own channel, and close <= high always, so a breakout
    measured against a channel that INCLUDES the current bar can never trigger.
    """
    if period < 1:
        raise StrategyInputError("donchian period must be >= 1")
    window = bars[:-1] if exclude_last else bars
    if len(window) < period:
        return None
    window = window[-period:]
    return DonchianChannel(
        upper=max(b.high for b in window),
        lower=min(b.low for b in window),
    )


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------

def _digest(parts: Iterable[str]) -> str:
    return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=12).hexdigest()


def idempotency_key(strategy_id: str, symbol: str, decision_ts: dt.datetime,
                    side: Side, tag: str = "") -> str:
    """Stable across retries of the SAME decision, different for a new one.

    `decision_ts` is the timestamp of the INPUT that drove the decision (the
    closed bar, the regime read, a floored `now` for time-driven exits) — never
    the tick time. Keying on the tick would make every repeated tick a fresh
    order, which is precisely the double-fire the gate's dedup exists to stop.
    """
    if decision_ts.tzinfo is None or decision_ts.utcoffset() is None:
        raise StrategyInputError("idempotency_key needs a timezone-aware decision_ts")
    stamp = decision_ts.astimezone(dt.timezone.utc).isoformat()
    return "{}:{}:{}:{}:{}".format(
        strategy_id, symbol, tag or "-", side.value,
        _digest([strategy_id, symbol, tag, side.value, stamp]),
    )


def structure_key(strategy_id: str, root: str, decision_ts: dt.datetime, tag: str = "") -> str:
    """Deterministic, so a repeated tick regroups the same legs under the same
    structure instead of opening a second 'structure' of orphans."""
    if decision_ts.tzinfo is None or decision_ts.utcoffset() is None:
        raise StrategyInputError("structure_key needs a timezone-aware decision_ts")
    stamp = decision_ts.astimezone(dt.timezone.utc).isoformat()
    return "{}:{}:{}".format(strategy_id, root, _digest([strategy_id, root, tag, stamp]))


# --------------------------------------------------------------------------
# Base strategy
# --------------------------------------------------------------------------

class StrategyBase:
    """Common plumbing. Subclasses implement `propose` and nothing else needs
    to know which asset class they are."""

    def __init__(self, *, strategy_id: str, asset_class: AssetClass,
                 bar_interval: dt.timedelta, stale_bar_multiple: int = 3,
                 slippage_bps: Decimal = Decimal("10")) -> None:
        if not strategy_id.strip():
            raise StrategyInputError("strategy_id must be non-empty")
        if bar_interval <= dt.timedelta(0):
            raise StrategyInputError("bar_interval must be positive")
        if stale_bar_multiple < 1:
            raise StrategyInputError("stale_bar_multiple must be >= 1")
        if slippage_bps < 0:
            raise StrategyInputError("slippage_bps must be >= 0")
        self.id = strategy_id
        self.asset_class = asset_class
        self.profile = profile_for(asset_class)
        self.bar_interval = bar_interval
        self.stale_bar_multiple = stale_bar_multiple
        self.slippage_bps = slippage_bps

    # -- Strategy protocol -------------------------------------------------
    def propose(self, ctx: MarketContext) -> Sequence[Intent]:
        raise NotImplementedError

    # -- positions ---------------------------------------------------------
    def own_positions(self, ctx: MarketContext) -> tuple[Position, ...]:
        """Only ours. Unwinding a leg another strategy opened is how one
        strategy's exit becomes another's naked position."""
        return tuple(p for p in ctx.positions if p.strategy_id == self.id and p.qty != 0)

    def own_position(self, ctx: MarketContext, symbol: str) -> Optional[Position]:
        for p in self.own_positions(ctx):
            if p.instrument.symbol == symbol:
                return p
        return None

    def held_by_anyone(self, ctx: MarketContext, symbol: str) -> bool:
        """Entry block. Two strategies stacking the same instrument doubles the
        real exposure while each believes it is at 1x, and the gate caps per
        intent, not per instrument."""
        return any(p.instrument.symbol == symbol and p.qty != 0 for p in ctx.positions)

    # -- bars --------------------------------------------------------------
    def bar_book(self, ctx: MarketContext) -> Mapping[str, object]:
        book = ctx.extra.get(BARS_KEY)
        if book is None:
            raise StrategyInputError(
                f"{self.id}: ctx.extra['{BARS_KEY}'] is missing. An absent bar book is a "
                "wiring failure, not an empty market."
            )
        if not isinstance(book, MappingABC):
            raise StrategyInputError(
                f"{self.id}: ctx.extra['{BARS_KEY}'] must be a mapping of symbol -> bars, "
                f"got {type(book).__name__}"
            )
        return book

    def bars_for(self, ctx: MarketContext, symbol: str, *, min_bars: int,
                 must_have: bool) -> Optional[list[Bar]]:
        """Bars for `symbol`, or None when there is legitimately nothing to act
        on. `must_have=True` (we hold this instrument) converts every one of
        those None paths into a raise: an open position we cannot see is the
        one situation that must never look like an idle tick.
        """
        raw = self.bar_book(ctx).get(symbol)
        bars = validate_bars(symbol, raw)
        if not bars:
            if must_have:
                raise MissingDataError(
                    f"{self.id}: no bars for {symbol} but a position is open in it — "
                    "refusing to manage an open position blind"
                )
            return None
        last = bars[-1].ts
        if last > ctx.now:
            raise StrategyInputError(
                f"{self.id}: last {symbol} bar is stamped {last}, ahead of ctx.now {ctx.now}"
            )
        age = ctx.now - last
        if age > self.bar_interval * self.stale_bar_multiple:
            if must_have:
                raise MissingDataError(
                    f"{self.id}: {symbol} feed is stale (last bar {last}, {age} old) while a "
                    "position is open — cannot evaluate its stop"
                )
            return None
        if len(bars) < min_bars:
            if must_have:
                raise MissingDataError(
                    f"{self.id}: only {len(bars)} bars for {symbol}, need {min_bars} to compute "
                    "the exit levels for an open position"
                )
            return None
        return bars

    # -- intents -----------------------------------------------------------
    def entry_unit_qty(self, instrument: Instrument) -> Decimal:
        """ONE unit. Not a sizing decision — the gate sizes. One lot for
        equity, one unit of the base asset for crypto."""
        raw = (Decimal(instrument.lot_size) if self.asset_class is AssetClass.EQUITY
               else Decimal(1))
        qty = self.profile.quantize_qty(instrument, raw)
        if qty <= 0:
            raise StrategyInputError(
                f"{self.id}: {instrument.symbol} unit qty quantizes to {qty}; instrument "
                "definition (lot_size/qty_precision) is unusable"
            )
        return qty

    def make_intent(self, *, instrument: Instrument, side: Side, qty: Decimal,
                    reason: str, decision_ts: dt.datetime, tag: str,
                    ref_price: Optional[Decimal],
                    structure_id: Optional[str] = None) -> Intent:
        """Order type comes from the asset-class profile, never from a literal
        here — NFO rejects MARKET outright, so equity intents are marketable
        LIMITs and crypto spot is MARKET."""
        if qty <= 0:
            raise StrategyInputError(f"{self.id}: {instrument.symbol} qty must be > 0, got {qty}")
        order_type = self.profile.default_order_type
        limit_price: Optional[Decimal] = None
        if order_type is OrderType.LIMIT:
            if ref_price is None or ref_price <= 0:
                raise StrategyInputError(
                    f"{self.id}: {instrument.symbol} needs a positive reference price to build a "
                    f"LIMIT order, got {ref_price}"
                )
            edge = ref_price * self.slippage_bps / Decimal(10000)
            raw = ref_price + edge if side is Side.BUY else ref_price - edge
            limit_price = self.profile.quantize_price(instrument, raw)
            if limit_price <= 0:
                raise StrategyInputError(
                    f"{self.id}: {instrument.symbol} limit price quantized to {limit_price}"
                )
        return Intent(
            instrument=instrument,
            side=side,
            qty=qty,
            order_type=order_type,
            strategy_id=self.id,
            reason=reason,
            limit_price=limit_price,
            structure_id=structure_id,
            idempotency_key=idempotency_key(self.id, instrument.symbol, decision_ts, side, tag),
        )

    def finalize(self, intents: Sequence[Intent]) -> list[Intent]:
        """Self-check before anything leaves the strategy. Two intents on one
        instrument in a single tick is always a bug here — with different tags
        they carry different idempotency keys, so the gate's dedup would let
        BOTH through and we would sell a position twice, i.e. go short."""
        seen: dict[str, Intent] = {}
        for i in intents:
            sym = i.instrument.symbol
            if sym in seen:
                raise StrategyInputError(
                    f"{self.id}: produced two intents for {sym} in one tick "
                    f"({seen[sym].reason!r} and {i.reason!r})"
                )
            seen[sym] = i
        return list(intents)
