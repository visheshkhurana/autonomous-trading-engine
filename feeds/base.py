"""
Shared contract for real market feeds.

Two rules live here because both feeds need them and getting either wrong is
silent and expensive.

1. NEVER publish the in-progress bar. Every candle API — ccxt and Dhan alike —
   returns the currently-forming candle as its last element. Its close is just
   the current price, so it changes under a strategy that has already acted on
   it, and the same tick then produces a different decision. Verified against
   live OKX data: the last 15m candle's close was identical to the ticker's
   last price.

2. A STALE feed must fail loudly, not serve old prices. A live book trading on
   a frozen quote is barely better than one trading on generated data. The
   engine already disables a strategy that raises, which is the correct
   direction: no data means no decision, not a decision on bad data.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

from core.types import AssetClass, Instrument, Mode
from strategies.base import Bar, MissingDataError


class FeedError(RuntimeError):
    """Base for feed problems. Distinct from a strategy error so the engine can
    tell 'the data is bad' apart from 'the strategy is bad'."""


class FeedStaleError(MissingDataError):
    """The newest closed bar is too old to act on.

    Subclasses MissingDataError so the engine's existing handling applies: the
    strategy stands down rather than trading on a frozen price.
    """


class FeedUnavailableError(FeedError):
    """The upstream could not be reached at all."""


@runtime_checkable
class MarketFeed(Protocol):
    """What the runner requires of any feed, real or synthetic."""

    def bind(self, resolved_modes: Mapping[AssetClass, Mode]) -> "MarketFeed":
        """Called once with the modes the engine actually resolved. A feed that
        is not fit to price a live book must raise here rather than document
        the limitation in a comment."""
        ...

    def bars(self, symbol: str, now: dt.datetime) -> list[Bar]: ...
    def refresh_quotes(self, quotes, asset_class: AssetClass, now: dt.datetime) -> None: ...
    def extra_provider(self, asset_class: AssetClass, now: dt.datetime) -> dict: ...


def drop_forming_bar(
    candles: Sequence[Sequence],
    interval: dt.timedelta,
    now: dt.datetime,
) -> list[Sequence]:
    """Drop the trailing candle if its period has not closed yet.

    `candles` are [ts_ms, o, h, l, c, v] rows, oldest first — the shape both
    ccxt and normalised Dhan candles use. A row is closed once
    `ts + interval <= now`.
    """
    if not candles:
        return []
    out = list(candles)
    last_ts = dt.datetime.fromtimestamp(out[-1][0] / 1000, dt.timezone.utc)
    if last_ts + interval > now:
        out.pop()
    return out


def to_bars(candles: Sequence[Sequence]) -> list[Bar]:
    """[ts_ms, o, h, l, c, v] -> Bar, with Decimal built from str.

    Decimal(float) would carry the float's representation error into a price;
    Decimal(str(float)) does not.
    """
    bars: list[Bar] = []
    for row in candles:
        ts = dt.datetime.fromtimestamp(row[0] / 1000, dt.timezone.utc)
        bars.append(Bar(
            ts=ts,
            open=Decimal(str(row[1])), high=Decimal(str(row[2])),
            low=Decimal(str(row[3])), close=Decimal(str(row[4])),
            volume=Decimal(str(row[5] if len(row) > 5 and row[5] is not None else 0)),
        ))
    return bars


def assert_fresh(
    bars: Sequence[Bar],
    symbol: str,
    interval: dt.timedelta,
    now: dt.datetime,
    max_age_intervals: int = 3,
) -> None:
    """Raise if the newest closed bar is older than the tolerance.

    The tolerance is in INTERVALS, not seconds, so one threshold is correct for
    a 1m crypto feed and a 15m equity feed alike. Callers that legitimately
    expect a gap — a closed market — must not call this.
    """
    if not bars:
        raise FeedStaleError(f"{symbol}: no closed bars available")
    age = now - bars[-1].ts
    limit = interval * max_age_intervals
    if age > limit:
        raise FeedStaleError(
            f"{symbol}: newest closed bar is {age.total_seconds() / 60:.1f}m old "
            f"(tolerance {limit.total_seconds() / 60:.0f}m) — refusing to trade "
            f"on a stale price"
        )
