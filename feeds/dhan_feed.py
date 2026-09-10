"""
Real Indian equity / F&O market data from Dhan.

This module NEVER constructs a Dhan client, never reads a token, and never
authenticates. Dhan permits exactly ONE active access token per account and the
owner's live engine holds it; minting a second would break a book trading real
money. The client is INJECTED and duck-typed against `DhanMarketDataClient`
below, so this file can be exercised end to end with a fake.

Three things in here are load-bearing and are the reason the file exists rather
than a few inline calls:

1. THE FORMING BAR IS NEVER PUBLISHED. Dhan's candle endpoints return the
   currently-forming candle as the last element, and its close is just the
   current price. A strategy that acted on it would see the same bar change
   under it on the next tick. `drop_forming_bar()` from feeds.base does this;
   it is not reimplemented here.

2. MARKET HOURS ARE NOT STALENESS. Outside 09:15-15:30 IST the newest closed
   bar is legitimately hours old, and that is not an error. `assert_fresh()` is
   therefore called ONLY when `EquityProfile().is_open(now)` is true. An alarm
   that fires every evening and every weekend is an alarm the operator learns
   to ignore, and then it is worth less than no alarm at all.

3. NOTHING FAILS QUIETLY. A fetch that errors, returns nothing, returns garbage
   timestamps, or returns a symbol we never asked for raises
   `FeedUnavailableError`. It never returns `[]`, never returns a last known
   price, and never fills a missing quote from a bar close. Empty means empty.

Dhan speaks `security_id`, not symbol. That mapping lives in exactly one place
here — `_Listing`, built once in the constructor from `Instrument.broker_id` —
and a symbol that is not in it is a loud error at the call, never a skip.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from core.profiles import IST, EquityProfile
from core.types import AssetClass, Instrument, Mode
from strategies.base import BARS_KEY, Bar
from strategies.equity_trend import REGIME_KEY, RegimeRead, RegimeState

from .base import (
    FeedError,
    FeedStaleError,  # re-exported: callers catch it from either module
    FeedUnavailableError,
    assert_fresh,
    drop_forming_bar,
    to_bars,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "DhanFeed",
    "DhanMarketDataClient",
    "UnmappedSymbolError",
    "FeedStaleError",
    "FeedUnavailableError",
]


# ---------------------------------------------------------------------------
# the injected client
# ---------------------------------------------------------------------------

@runtime_checkable
class DhanMarketDataClient(Protocol):
    """The read-only slice of the Dhan client this feed uses.

    Deliberately narrow: no order methods are named here, so nothing in this
    file can reach the order path even by accident. The shapes mirror
    `apps/agent/core/dhan_adapter.py`, which is the proven implementation.
    """

    def get_ohlc(self, symbol: str, interval: str, from_date: str,
                 to_date: str, exchange: str = "NSE") -> Any:
        """Candles. Either a list of row dicts or Dhan's column-oriented
        `{open: [...], high: [...], ..., timestamp: [...]}`."""
        ...

    def get_fno_quotes(self, security_ids: list,
                       exchange_segment: str = "NSE_FNO") -> Mapping[str, Mapping[str, Any]]:
        """Batch full quotes keyed by security_id. `{}` when the Dhan Data API
        is not subscribed — which this feed treats as a failure, not as a
        legitimately empty book."""
        ...

    def get_fno_ltps(self, security_ids: list,
                     exchange_segment: str = "NSE_FNO") -> Mapping[str, Any]:
        """Batch last prices keyed by security_id. Missing ids are omitted."""
        ...


_CLIENT_METHODS = ("get_ohlc", "get_fno_quotes", "get_fno_ltps")


# ---------------------------------------------------------------------------
# Dhan vocabulary, in one place
# ---------------------------------------------------------------------------

# Bar interval -> the interval string dhan_adapter.get_ohlc understands.
# An interval that is not in here raises at construction rather than silently
# resolving to Dhan's 5-minute default, which would stamp 5m bars with a 15m
# strategy's expectations.
DHAN_INTERVALS: dict[dt.timedelta, str] = {
    dt.timedelta(minutes=1): "minute",
    dt.timedelta(minutes=3): "3minute",
    dt.timedelta(minutes=5): "5minute",
    dt.timedelta(minutes=15): "15minute",
    dt.timedelta(minutes=30): "30minute",
    dt.timedelta(minutes=60): "60minute",
    dt.timedelta(days=1): "day",
}

# Cash vs derivatives. The only thing that distinguishes them on an Instrument
# is an expiry, so that is the rule, and it lives here.
SEGMENT_CASH = "NSE_EQ"
SEGMENT_FNO = "NSE_FNO"
EXCHANGE_CASH = "NSE"
EXCHANGE_FNO = "NSE_FNO"

# NSE trades 09:15-15:30 => 375 minutes a session. Used only to size the
# lookback window wide enough; over-fetching is free, under-fetching is not.
SESSION_MINUTES = 375

# A 5s engine tick against an uncached 15m candle fetch is ~48 req/min and
# Dhan rate-limit-halts. Closed bars do not change until the next bar closes,
# so they are cached against that close. Serving a cache that cannot vouch
# the last closed bar is refused — a frozen tape is not a quiet market.
_QUOTE_MAX_AGE = dt.timedelta(seconds=60)


@dataclass(frozen=True)
class _Mark:
    """One last price, with enough metadata to tell a live print from a freeze.

    Dhan's thin LTP endpoint carries no timestamp. A number with no age is
    stored as `live=False` so a spread cannot be silently priced off it.
    """
    price: Decimal
    as_of: Optional[dt.datetime]
    live: bool

# Anything outside 2000-2100 is not a market timestamp. Dhan's epoch handling
# has historically been the source of quietly-wrong candle dates, and a bar
# stamped 1985 would sail through every downstream check as "very stale" or, on
# a closed market, as fine.
_MIN_TS_MS = 946_684_800_000     # 2000-01-01T00:00:00Z
_MAX_TS_MS = 4_102_444_800_000   # 2100-01-01T00:00:00Z


class UnmappedSymbolError(FeedError):
    """A symbol was requested that this feed has no Dhan security_id for.

    Loud on purpose. Returning no bars for an unknown symbol reads to every
    caller as "the market had nothing to say", which is how a typo becomes an
    unmanaged position.
    """


@dataclass(frozen=True)
class _Listing:
    """The one and only symbol -> Dhan identity record."""
    instrument: Instrument
    security_id: str
    exchange: str
    segment: str

    @property
    def symbol(self) -> str:
        return self.instrument.symbol

    @property
    def is_derivative(self) -> bool:
        return self.segment == SEGMENT_FNO


# ---------------------------------------------------------------------------
# coercion helpers — every one of these raises rather than guessing
# ---------------------------------------------------------------------------

def _decimal(value: object, what: str) -> Decimal:
    """Coerce a Dhan number to Decimal via str, never via float().

    Decimal(float) carries the float's representation error into a price.
    Decimal(str(float)) does not.
    """
    if value is None:
        raise FeedUnavailableError(f"{what}: missing from the Dhan response")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise FeedUnavailableError(
            f"{what}: cannot read {type(value).__name__} as a number")
    try:
        out = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise FeedUnavailableError(f"{what}: {value!r} is not a number") from None
    if not out.is_finite():
        raise FeedUnavailableError(f"{what}: {value!r} is not finite")
    return out


def _timestamp_ms(value: object, what: str) -> int:
    """Dhan candle timestamp -> epoch milliseconds (UTC).

    Handles every shape the adapter has been seen to produce:
      * epoch seconds / milliseconds (int or float),
      * ISO-8601 strings, with or without an offset,
      * `datetime` / `date` objects.

    A NAIVE string or datetime is read as IST, because that is what an Indian
    broker returns and there is no other defensible reading. It is a documented
    convention rather than a guess buried in a branch. Anything else raises:
    a timestamp this code cannot interpret is not a bar it may invent a time
    for.
    """
    if value is None:
        raise FeedUnavailableError(f"{what}: candle has no timestamp")

    moment: Optional[dt.datetime] = None

    if isinstance(value, bool):
        raise FeedUnavailableError(f"{what}: bool is not a timestamp")
    elif isinstance(value, dt.datetime):
        moment = value
    elif isinstance(value, dt.date):
        moment = dt.datetime.combine(value, dt.time(0, 0), tzinfo=IST)
    elif isinstance(value, (int, float)):
        ms = _epoch_to_ms(value, what)
        return _checked_ms(ms, what, value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise FeedUnavailableError(f"{what}: empty timestamp")
        try:
            # numeric strings are epochs, not ISO
            ms = _epoch_to_ms(float(text), what)
            return _checked_ms(ms, what, value)
        except ValueError:
            pass
        try:
            moment = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            raise FeedUnavailableError(
                f"{what}: {value!r} is not a timestamp this feed can read"
            ) from None
    else:
        raise FeedUnavailableError(
            f"{what}: cannot read {type(value).__name__} as a timestamp")

    if moment.tzinfo is None or moment.utcoffset() is None:
        moment = moment.replace(tzinfo=IST)
    ms = int(moment.timestamp() * 1000)
    return _checked_ms(ms, what, value)


def _epoch_to_ms(value: float, what: str) -> int:
    """Seconds or milliseconds -> milliseconds, decided by magnitude."""
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        raise FeedUnavailableError(f"{what}: non-finite epoch {value!r}")
    magnitude = abs(value)
    if magnitude >= 1e11:      # already milliseconds
        return int(value)
    return int(value * 1000)


def _checked_ms(ms: int, what: str, original: object) -> int:
    if not (_MIN_TS_MS <= ms <= _MAX_TS_MS):
        stamped = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
        raise FeedUnavailableError(
            f"{what}: timestamp {original!r} resolves to {stamped.isoformat()}, "
            "outside 2000-2100 — refusing to build a bar from it"
        )
    return ms


def _floor_utc(now: dt.datetime, step: dt.timedelta) -> dt.datetime:
    seconds = int(step.total_seconds())
    epoch = int(now.timestamp())
    return dt.datetime.fromtimestamp(epoch - (epoch % seconds), tz=dt.timezone.utc)


def _last_closed_start(now: dt.datetime, interval: dt.timedelta) -> dt.datetime:
    """Open time of the newest bar that has already closed."""
    return _floor_utc(now, interval) - interval


def _drop_unclosed(rows: Sequence[Sequence], interval: dt.timedelta,
                   now: dt.datetime, symbol: str) -> list[Sequence]:
    """Every trailing candle whose period has not closed — not just the first.

    `drop_forming_bar()` removes ONE row, which is the right answer while our
    clock agrees with the venue's. When it does not — a venue running fast, a
    replayed `now`, a page stamped in another timezone — the rows BEHIND the
    forming one have not closed either, and `assert_fresh()` cannot catch them:
    its age is then NEGATIVE, and `age > limit` is False for every negative
    age. The louder the timestamp error, the fresher the feed would look.

    Trimming rather than raising is deliberate. A three-second clock skew makes
    the trailing TWO rows unclosed on a 15m grid, and a feed that refuses every
    read over three seconds is worth no more than one that never works. What is
    NOT tolerated is publishing an unclosed bar: if the trim empties the series
    `bars()` raises, which is the right answer for a response we cannot age.
    """
    if not rows:
        return []

    # How far past `now` the newest candle's period runs. A genuine forming
    # candle overshoots by less than one interval; one full interval of slack
    # on top absorbs any plausible clock skew. Beyond that Dhan's clock and
    # ours disagree by at least a whole bar, which means every age computed
    # from these timestamps — including the one `assert_fresh` is about to
    # check while the session is open — is measuring against a lie. Trimming
    # alone would make that case look TIDIER: drop the future rows and what is
    # left reads as perfectly fresh while carrying prices from hours ago.
    overshoot = (dt.datetime.fromtimestamp(rows[-1][0] / 1000, dt.timezone.utc)
                 + interval) - now
    if overshoot > 2 * interval:
        raise FeedUnavailableError(
            f"{symbol}: the newest candle's period ends "
            f"{overshoot.total_seconds() / 60:.1f}m after now ({now.isoformat()}); "
            f"Dhan's clock and ours disagree by more than a {interval} bar, so "
            "no age computed from these timestamps can be trusted")

    out = list(rows)
    while out:
        trimmed = list(drop_forming_bar(out, interval, now))
        if len(trimmed) == len(out):
            break
        out = trimmed
    dropped = len(rows) - len(out)
    if dropped > 1:
        LOG.error(
            "%s: %d trailing candles had not closed at %s; only closed bars "
            "are published", symbol, dropped, now.isoformat())
    return out


def _price_or_none(value: object, what: str) -> Optional[Decimal]:
    """A quote price, or None when the venue simply had nothing to say.

    Absent (None / missing) is None. Zero and negative are ALSO None: zero is
    not a price, and a zero that reads as valid is exactly the class of bug
    that turns an empty read into a confident wrong number. Garbage still
    raises — it means the response shape changed under us.
    """
    if value is None:
        return None
    price = _decimal(value, what)
    if price <= 0:
        return None
    return price


def _quote_timestamp(record: MappingABC, now: dt.datetime, sid: str
                     ) -> Optional[dt.datetime]:
    """Venue trade time if the record carries one, else None.

    None is the honest answer for Dhan's thin LTP. It is not `now`: stamping
    an unageable print with the fetch clock is how a frozen option mark looks
    live.
    """
    del now  # used only to make the contract obvious to callers
    for key in ("last_trade_time", "ltt", "timestamp", "update_time", "last_time"):
        raw = record.get(key)
        if raw is None or raw == "" or raw == 0:
            continue
        try:
            if isinstance(raw, dt.datetime):
                moment = raw
            elif isinstance(raw, bool):
                continue
            elif isinstance(raw, (int, float)):
                ms = _epoch_to_ms(float(raw), f"quote {sid} {key}")
                moment = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
            elif isinstance(raw, str):
                text = raw.strip()
                if not text:
                    continue
                try:
                    ms = _epoch_to_ms(float(text), f"quote {sid} {key}")
                    moment = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
                except ValueError:
                    moment = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
            else:
                continue
            if moment.tzinfo is None or moment.utcoffset() is None:
                moment = moment.replace(tzinfo=IST)
            return moment
        except (FeedUnavailableError, ValueError, OverflowError, OSError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# the feed
# ---------------------------------------------------------------------------

class DhanFeed:
    """Real NSE cash / NFO market data, over an injected Dhan client.

    Construct it with the instruments it is allowed to price and a client
    object. Nothing here authenticates; nothing here places, modifies or
    cancels an order.
    """

    def __init__(
        self,
        instruments: Sequence[Instrument],
        client: Optional[DhanMarketDataClient] = None,
        *,
        bar_interval: dt.timedelta = dt.timedelta(minutes=15),
        history_bars: int = 120,
        max_age_intervals: int = 3,
        regime_aliases: Optional[Mapping[str, str]] = None,
        regime_lookback: int = 20,
        regime_fast: int = 5,
        regime_trend_k: Decimal = Decimal("0.5"),
    ) -> None:
        if bar_interval not in DHAN_INTERVALS:
            supported = ", ".join(
                str(int(k.total_seconds() // 60)) + "m" for k in DHAN_INTERVALS)
            raise ValueError(
                f"DhanFeed cannot request a {bar_interval} bar from Dhan. "
                f"Supported: {supported}. Falling back to another interval "
                "would stamp the wrong period on every bar."
            )
        if history_bars < 2:
            raise ValueError("history_bars must be >= 2")
        if max_age_intervals < 1:
            raise ValueError("max_age_intervals must be >= 1")
        if regime_lookback < 2:
            raise ValueError("regime_lookback must be >= 2")
        if not (1 <= regime_fast < regime_lookback):
            raise ValueError("regime_fast must be >= 1 and < regime_lookback")
        if regime_trend_k < 0:
            raise ValueError("regime_trend_k must be >= 0")

        if client is not None:
            missing = [m for m in _CLIENT_METHODS if not callable(getattr(client, m, None))]
            if missing:
                raise TypeError(
                    f"DhanFeed client is missing {missing}. It must provide "
                    f"{list(_CLIENT_METHODS)} — a client that cannot serve market "
                    "data must not be accepted and then fail at the first tick."
                )

        self._client = client
        self._interval = bar_interval
        self._dhan_interval = DHAN_INTERVALS[bar_interval]
        self._daily = bar_interval >= dt.timedelta(days=1)
        self._history = history_bars
        self._max_age_intervals = max_age_intervals
        self._profile = EquityProfile()
        self._registry: dict[str, _Listing] = {}
        self._bound = False

        for instrument in instruments:
            self._register(instrument)

        self._regime_lookback = regime_lookback
        self._regime_fast = regime_fast
        self._regime_k = regime_trend_k
        self._regime_aliases: dict[str, str] = {}
        for alias, target in (regime_aliases or {}).items():
            if target not in self._registry:
                raise ValueError(
                    f"regime alias {alias!r} points at {target!r}, which is not "
                    "in this feed's universe — an alias that resolves to nothing "
                    "would publish no regime and look like a quiet market"
                )
            self._regime_aliases[alias] = target
        # symbol -> (closed bars, last-closed bar start). See bars().
        self._bar_cache: dict[str, tuple[tuple[Bar, ...], dt.datetime]] = {}

    # -- the one symbol -> security_id map --------------------------------

    def _register(self, instrument: Instrument) -> None:
        if instrument.asset_class is not AssetClass.EQUITY:
            raise ValueError(
                f"DhanFeed cannot price {instrument.symbol} "
                f"({instrument.asset_class.value}); it serves NSE equity and NFO only"
            )
        if instrument.symbol in self._registry:
            raise ValueError(
                f"duplicate symbol {instrument.symbol!r} in the DhanFeed universe; "
                "the second listing would silently shadow the first"
            )
        security_id = str(instrument.broker_id).strip()
        if not security_id.isdigit():
            # Catches the paper placeholders ("PAPER-RELIANCE") and any symbol
            # accidentally left in broker_id. A non-numeric id sent to Dhan
            # would be rejected or, worse, resolved to something else.
            raise ValueError(
                f"{instrument.symbol}: broker_id {instrument.broker_id!r} is not a "
                "Dhan security_id (numeric). Wire the real scrip-master id; a "
                "placeholder must never reach a real feed."
            )
        derivative = instrument.expiry is not None
        self._registry[instrument.symbol] = _Listing(
            instrument=instrument,
            security_id=security_id,
            exchange=EXCHANGE_FNO if derivative else EXCHANGE_CASH,
            segment=SEGMENT_FNO if derivative else SEGMENT_CASH,
        )

    def _listing(self, symbol: str) -> _Listing:
        listing = self._registry.get(symbol)
        if listing is None:
            known = ", ".join(sorted(self._registry)) or "<none>"
            raise UnmappedSymbolError(
                f"{symbol!r} has no Dhan security_id in this feed. Known: {known}. "
                "Refusing to return an empty series for an unknown symbol."
            )
        return listing

    @property
    def instruments(self) -> tuple[Instrument, ...]:
        return tuple(l.instrument for l in self._registry.values())

    def security_id(self, symbol: str) -> str:
        """The Dhan security_id for a symbol. Raises if unmapped."""
        return self._listing(symbol).security_id

    # -- lifecycle ---------------------------------------------------------

    def bind(self, resolved_modes: Mapping[AssetClass, Mode]) -> "DhanFeed":
        """Vouch for the modes the engine actually resolved.

        A live equity book on a feed with no client would price off nothing at
        all, so that combination raises here rather than at the first fetch,
        when a strategy is already mid-decision.

        The refusal below is an identity check, and it has to be: `Mode` is a
        str Enum, so a bare `"live"` from a config file or a JSON round trip
        compares EQUAL to `Mode.LIVE` while failing `is Mode.LIVE`, and would
        walk past this guard while every log line still read `equity->live`.
        A map that is not made of `Mode` members is rejected outright.
        """
        wrong = sorted(f"{a.value}={m!r}" for a, m in resolved_modes.items()
                       if not isinstance(m, Mode))
        if wrong:
            raise RuntimeError(
                f"DhanFeed.bind: {', '.join(wrong)} is not a Mode. A bare "
                "string compares equal to Mode.LIVE and fails the identity "
                "check below; pass the enum member.")
        equity_mode = resolved_modes.get(AssetClass.EQUITY)
        if equity_mode is Mode.LIVE:
            if self._client is None:
                raise RuntimeError(
                    "DhanFeed refuses to bind equity LIVE: no market-data client "
                    "was supplied. A live book must not run on an unconfigured "
                    "feed — inject a Dhan client or do not arm equity."
                )
            if not self._registry:
                raise RuntimeError(
                    "DhanFeed refuses to bind equity LIVE with an empty universe: "
                    "there is nothing it could price."
                )
        self._bound = True
        return self

    def _require_bound(self, what: str) -> None:
        if not self._bound:
            raise RuntimeError(
                f"DhanFeed.{what}() called before bind(); the live-mode check "
                "has not run"
            )

    def _require_client(self) -> DhanMarketDataClient:
        if self._client is None:
            raise FeedUnavailableError(
                "DhanFeed has no market-data client; there is no data to serve "
                "and no last known price to fall back on"
            )
        return self._client

    @staticmethod
    def _checked_now(now: dt.datetime, what: str) -> dt.datetime:
        if not isinstance(now, dt.datetime):
            raise FeedError(f"DhanFeed.{what}() needs a datetime, got {type(now).__name__}")
        if now.tzinfo is None or now.utcoffset() is None:
            # A naive `now` would be read as local time by is_open() and as UTC
            # by the bar maths — two different instants, silently.
            raise FeedError(
                f"DhanFeed.{what}() needs a timezone-aware `now`; a naive one "
                "would put market hours and bar ages in different timezones"
            )
        return now

    # -- bars --------------------------------------------------------------

    def bars(self, symbol: str, now: dt.datetime) -> list[Bar]:
        """Closed bars for one symbol, oldest first.

        The forming bar is dropped. Freshness is asserted only while the NSE
        session is open — outside it, an hours-old newest bar is the correct
        answer, not an alarm.

        Closed bars are cached until the next bar closes. A 5s tick must not
        refetch a 15m candle that cannot have changed. If the cache cannot
        vouch that it holds the last closed bar, it is not served.
        """
        self._require_bound("bars")
        now = self._checked_now(now, "bars")
        listing = self._listing(symbol)
        last_closed = _last_closed_start(now, self._interval)
        cached = self._bar_cache.get(symbol)
        if self._cache_vouches(cached, last_closed, now, symbol):
            assert cached is not None
            return list(cached[0])

        rows = self._normalise_candles(self._fetch_candles(listing, now), symbol)
        self._assert_ordered(rows, symbol)

        # Not `drop_forming_bar` alone: it removes exactly one row, and a
        # response that runs ahead of `now` has more than one unclosed row.
        closed = _drop_unclosed(rows, self._interval, now, symbol)
        bars = to_bars(closed)

        if not bars:
            raise FeedUnavailableError(
                f"{symbol}: Dhan returned {len(rows)} candle(s), none of them "
                "closed — no bar this feed is allowed to publish"
            )

        if self._profile.is_open(now):
            # Session is live: a gap here is a real data failure.
            assert_fresh(bars, symbol, self._interval, now,
                         max_age_intervals=self._max_age_intervals)
        self._bar_cache[symbol] = (tuple(bars), bars[-1].ts)
        return bars

    def _cache_vouches(
        self,
        cached: Optional[tuple[tuple[Bar, ...], dt.datetime]],
        last_closed: dt.datetime,
        now: dt.datetime,
        symbol: str,
    ) -> bool:
        """True only when the cache holds the last closed bar AND can still
        be freshness-checked (or the session is closed, where old bars are
        the correct answer)."""
        if cached is None:
            return False
        bars, vouched = cached
        if not bars or vouched != last_closed:
            return False
        if self._profile.is_open(now):
            try:
                assert_fresh(bars, symbol, self._interval, now,
                             max_age_intervals=self._max_age_intervals)
            except Exception:
                LOG.error(
                    "%s: bar cache cannot vouch freshness for last closed "
                    "%s; refusing to serve it",
                    symbol, last_closed.isoformat())
                return False
        return True

    def _fetch_candles(self, listing: _Listing, now: dt.datetime) -> Any:
        client = self._require_client()
        start, end = self._window(now)
        try:
            raw = client.get_ohlc(
                symbol=listing.symbol,
                interval=self._dhan_interval,
                from_date=self._fmt(start),
                to_date=self._fmt(end),
                exchange=listing.exchange,
            )
        except Exception as exc:
            # The proven adapter logs and re-raises; anything that reaches here
            # is a failure to get data, and a failure to get data is never [].
            raise FeedUnavailableError(
                f"{listing.symbol}: Dhan candle fetch failed: {exc}") from exc
        if raw is None:
            raise FeedUnavailableError(
                f"{listing.symbol}: Dhan candle fetch returned None")
        return raw

    def _window(self, now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
        """A lookback wide enough for `history_bars`, in IST.

        Padded for weekends and holidays. Over-fetching costs a little
        bandwidth; under-fetching costs bars an indicator needed and turns into
        "insufficient history" that looks like a quiet strategy.
        """
        end = now.astimezone(IST)
        if self._daily:
            span = dt.timedelta(days=self._history * 2 + 10)
        else:
            per_session = max(1, SESSION_MINUTES // max(1, int(self._interval.total_seconds() // 60)))
            sessions = -(-self._history // per_session) + 2
            span = dt.timedelta(days=int(sessions * 7 / 5) + 4)
        return end - span, end

    def _fmt(self, moment: dt.datetime) -> str:
        return moment.astimezone(IST).strftime(
            "%Y-%m-%d" if self._daily else "%Y-%m-%d %H:%M:%S")

    def _normalise_candles(self, raw: Any, symbol: str) -> list[list]:
        """Dhan candles -> [[ts_ms, o, h, l, c, v], ...], oldest first.

        Mirrors `dhan_adapter._normalize_candles`: the response is either a list
        of row dicts or the column-oriented `{open: [...], ...}` form. Unlike
        that helper, an unrecognised shape raises instead of returning [] —
        this one feeds a live book.
        """
        rows: list[list] = []

        if isinstance(raw, MappingABC):
            opens = raw.get("open")
            if opens is None:
                raise FeedUnavailableError(
                    f"{symbol}: Dhan candle response has no 'open' column "
                    f"(keys: {sorted(raw.keys())})")
            count = len(opens)
            columns = {}
            for name in ("open", "high", "low", "close"):
                column = raw.get(name)
                if column is None:
                    raise FeedUnavailableError(
                        f"{symbol}: Dhan candle response is missing the {name!r} column")
                if len(column) != count:
                    raise FeedUnavailableError(
                        f"{symbol}: Dhan candle column {name!r} has {len(column)} "
                        f"values against {count} opens — the response is not aligned "
                        "and zipping it would mispair prices across bars")
                columns[name] = column
            stamps = raw.get("timestamp")
            if stamps is None:
                stamps = raw.get("start_Time")
            if stamps is None or len(stamps) != count:
                raise FeedUnavailableError(
                    f"{symbol}: Dhan candle response has no aligned timestamp column")
            volumes = raw.get("volume")
            if volumes is not None and len(volumes) != count:
                raise FeedUnavailableError(
                    f"{symbol}: Dhan candle volume column is not aligned with opens")
            for i in range(count):
                rows.append(self._row(
                    stamps[i], columns["open"][i], columns["high"][i],
                    columns["low"][i], columns["close"][i],
                    None if volumes is None else volumes[i],
                    symbol, i,
                ))
        elif isinstance(raw, (list, tuple)):
            for i, record in enumerate(raw):
                if isinstance(record, MappingABC):
                    stamp = record.get("date")
                    if stamp is None:
                        stamp = record.get("timestamp")
                    if stamp is None:
                        stamp = record.get("start_Time")
                    rows.append(self._row(
                        stamp, record.get("open"), record.get("high"),
                        record.get("low"), record.get("close"),
                        record.get("volume"), symbol, i,
                    ))
                elif isinstance(record, (list, tuple)) and len(record) >= 5:
                    rows.append(self._row(
                        record[0], record[1], record[2], record[3], record[4],
                        record[5] if len(record) > 5 else None, symbol, i,
                    ))
                else:
                    raise FeedUnavailableError(
                        f"{symbol}: candle {i} is a {type(record).__name__} this "
                        "feed cannot read")
        else:
            raise FeedUnavailableError(
                f"{symbol}: Dhan candle response is a {type(raw).__name__}, "
                "not a candle series")

        if not rows:
            # No candles at all is a failed read, not a quiet market. NSE
            # publishes history for every listed scrip; an empty series means
            # the request, the symbol or the subscription is wrong.
            raise FeedUnavailableError(
                f"{symbol}: Dhan returned no candles for the requested window")
        return rows

    def _row(self, stamp: object, o: object, h: object, l: object, c: object,
             v: object, symbol: str, index: int) -> list:
        where = f"{symbol} candle {index}"
        volume = Decimal(0) if v is None else _decimal(v, f"{where} volume")
        # A missing volume is real on index candles and volume is not a price.
        # A missing PRICE is never defaulted.
        return [
            _timestamp_ms(stamp, where),
            _decimal(o, f"{where} open"),
            _decimal(h, f"{where} high"),
            _decimal(l, f"{where} low"),
            _decimal(c, f"{where} close"),
            volume,
        ]

    @staticmethod
    def _assert_ordered(rows: Sequence[Sequence], symbol: str) -> None:
        """Strictly increasing timestamps, oldest first.

        `drop_forming_bar` trims the LAST row, so an out-of-order series would
        have it drop the wrong bar and publish the forming one. Re-sorting
        silently would hide a feed that merged two sources.
        """
        for i in range(1, len(rows)):
            if rows[i][0] <= rows[i - 1][0]:
                previous = dt.datetime.fromtimestamp(rows[i - 1][0] / 1000, dt.timezone.utc)
                current = dt.datetime.fromtimestamp(rows[i][0] / 1000, dt.timezone.utc)
                raise FeedUnavailableError(
                    f"{symbol}: candles are not in ascending time order "
                    f"({previous.isoformat()} then {current.isoformat()}); "
                    "refusing to guess which one is the forming bar"
                )

    # -- quotes ------------------------------------------------------------

    def refresh_quotes(self, quotes: Any, asset_class: AssetClass,
                       now: dt.datetime) -> None:
        """Push last prices for this class into the quote source.

        Batched: one call per exchange segment, never one per instrument, so a
        universe of option legs does not walk into Dhan's rate limit.

        An instrument whose quote did not come back is REMOVED from the quote
        source. It is never filled from a bar close, and — the part that is
        easy to get wrong — it is not merely left out of the update either.
        Omitting it leaves the PREVIOUS tick's mark standing, which is the same
        stale price this module exists to refuse, just arrived by a slower
        route. Downstream, a missing quote rejects the order; a frozen one
        places it.

        Dhan's LTP carries no timestamp, so quote freshness cannot be checked
        here, and `extra_provider()` fetches no bars for derivative legs — so
        for an option leg there is no freshness check anywhere. Dropping the
        mark the moment we cannot re-establish it is the only thing standing
        between a defined-risk structure and a frozen number.
        """
        self._require_bound("refresh_quotes")
        now = self._checked_now(now, "refresh_quotes")
        if asset_class is not AssetClass.EQUITY:
            return  # this feed has nothing of that class; not an error
        listings = list(self._registry.values())
        if not listings:
            return

        by_segment: dict[str, list[_Listing]] = {}
        for listing in listings:
            by_segment.setdefault(listing.segment, []).append(listing)

        prices: dict[str, _Mark] = {}
        for segment, group in by_segment.items():
            prices.update(self._batch_prices([l.security_id for l in group],
                                             segment, now, group))

        unpriced: list[str] = []
        frozen: list[str] = []
        for listing in listings:
            mark = prices.get(listing.security_id)
            if mark is None:
                unpriced.append(listing.symbol)
                self._drop_quote(quotes, listing)
                continue
            quotes.set(listing.symbol, mark.price, as_of=mark.as_of, live=mark.live)
            if not mark.live:
                frozen.append(listing.symbol)

        if unpriced:
            LOG.error(
                "DhanFeed could not price %s; their marks have been REMOVED "
                "rather than left at the previous tick's value",
                ", ".join(sorted(unpriced)))
        if frozen:
            LOG.error(
                "DhanFeed could not age the LTP for %s; stored as frozen "
                "(live=False) so a stale print cannot silently price a spread",
                ", ".join(sorted(frozen)))

        # The blackout raise comes AFTER the drop: a failure that left the last
        # tick's prices sitting in the book would be loud and wrong at the same
        # time.
        if not prices:
            raise FeedUnavailableError(
                f"Dhan returned no usable quote for any of the {len(listings)} "
                "requested instrument(s). Refusing to leave the book priced off "
                "whatever was there before."
            )

    @staticmethod
    def _drop_quote(quotes: Any, listing: _Listing) -> None:
        """Remove a mark we can no longer vouch for.

        Not a swallowed failure: the end state is "this symbol has no price",
        which the paper adapter turns into a REJECTED order and a live adapter
        into a refusal to size. Leaving the old number there is the only
        outcome that would be silent.
        """
        remove = getattr(quotes, "remove", None)
        if remove is None:
            raise FeedUnavailableError(
                f"cannot clear the stale mark for {listing.symbol}: this quote "
                f"source ({type(quotes).__name__}) has no remove()")
        if quotes.quote(listing.instrument) is not None:
            remove(listing.symbol)

    def _batch_prices(self, security_ids: Sequence[str], segment: str,
                      now: dt.datetime, group: Sequence[_Listing]
                      ) -> dict[str, _Mark]:
        client = self._require_client()
        out: dict[str, _Mark] = {}
        if not security_ids:
            return out
        by_id = {l.security_id: l for l in group}

        try:
            full = client.get_fno_quotes(list(security_ids), segment) or {}
        except Exception as exc:
            raise FeedUnavailableError(
                f"Dhan quote fetch failed for {segment}: {exc}") from exc
        if not isinstance(full, MappingABC):
            raise FeedUnavailableError(
                f"Dhan quote fetch for {segment} returned a "
                f"{type(full).__name__}, not a mapping")
        for sid, record in full.items():
            if not isinstance(record, MappingABC):
                continue
            mark = self._mark_from_record(record, by_id.get(str(sid)), now, str(sid))
            if mark is not None:
                out[str(sid)] = mark

        missing = [sid for sid in security_ids if sid not in out]
        if missing:
            # quote_data needs a Data API subscription and returns {} without
            # one; ticker_data is the thinner fallback. Both empty means we
            # genuinely have nothing, which the caller turns into a raise.
            try:
                ltps = client.get_fno_ltps(list(missing), segment) or {}
            except Exception as exc:
                raise FeedUnavailableError(
                    f"Dhan LTP fetch failed for {segment}: {exc}") from exc
            if not isinstance(ltps, MappingABC):
                raise FeedUnavailableError(
                    f"Dhan LTP fetch for {segment} returned a "
                    f"{type(ltps).__name__}, not a mapping")
            for sid, value in ltps.items():
                listing = by_id.get(str(sid))
                # A bare LTP has no timestamp. For a derivative that is a
                # frozen mark; for cash, bars carry freshness and we treat
                # the observation time as as_of.
                mark = self._mark_from_ltp(value, listing, now, str(sid))
                if mark is not None:
                    out[str(sid)] = mark
        return out

    def _mark_from_record(self, record: MappingABC, listing: Optional[_Listing],
                          now: dt.datetime, sid: str) -> Optional[_Mark]:
        price = _price_or_none(record.get("ltp"), f"quote {sid} ltp")
        if price is None:
            price = _price_or_none(record.get("last_price"), f"quote {sid} last_price")
        if price is None:
            return None
        as_of = _quote_timestamp(record, now, sid)
        return self._age_mark(price, as_of, listing, now)

    def _mark_from_ltp(self, value: object, listing: Optional[_Listing],
                       now: dt.datetime, sid: str) -> Optional[_Mark]:
        price = _price_or_none(value, f"ltp {sid}")
        if price is None:
            return None
        return self._age_mark(price, None, listing, now)

    @staticmethod
    def _age_mark(price: Decimal, as_of: Optional[dt.datetime],
                  listing: Optional[_Listing], now: dt.datetime) -> _Mark:
        derivative = listing is not None and listing.is_derivative
        if as_of is None:
            # No venue timestamp. Cash can lean on bar freshness; an option
            # leg cannot. Store the mark as frozen so a spread is not priced
            # off a number nobody can age.
            if derivative:
                return _Mark(price=price, as_of=None, live=False)
            return _Mark(price=price, as_of=now, live=True)
        if as_of > now + dt.timedelta(seconds=5):
            # A future-dated print is not a price we can age.
            return _Mark(price=price, as_of=as_of, live=False)
        if now - as_of > _QUOTE_MAX_AGE:
            return _Mark(price=price, as_of=as_of, live=False)
        return _Mark(price=price, as_of=as_of, live=True)

    # -- ctx.extra ---------------------------------------------------------

    def extra_provider(self, asset_class: AssetClass, now: dt.datetime) -> dict:
        """`ctx.extra` for one asset class: closed bars plus a regime read.

        Bars are fetched for CASH listings only. Option legs are priced through
        `refresh_quotes`; asking Dhan for an option's candle history through
        the equity scrip map would fail, and failing on a leg that never needed
        bars would take the whole tick down.

        A universe with no cash listing at all — an option book with no
        underlying, or an empty registry — therefore has no bars to publish,
        and it RAISES rather than publishing `{bars: {}}`. That empty map is
        not None and it does carry BARS_KEY, so `CompositeFeed`'s guard passes
        it straight through, and downstream it is indistinguishable from a
        market with nothing to say: entries stop, bar-driven exits stop, and
        `refresh_quotes` keeps the legs priced so the class goes on ticking as
        if all were well. Absence of a bar book is a wiring fault, not a quiet
        tape.
        """
        self._require_bound("extra_provider")
        now = self._checked_now(now, "extra_provider")
        if asset_class is not AssetClass.EQUITY:
            return {}

        bars_by_symbol: dict[str, list[Bar]] = {}
        for symbol, listing in self._registry.items():
            if listing.is_derivative:
                continue
            bars_by_symbol[symbol] = self.bars(symbol, now)

        if not bars_by_symbol:
            legs = sorted(s for s, l in self._registry.items() if l.is_derivative)
            what = (f"derivative legs only: {', '.join(legs)}" if legs
                    else "the universe is empty")
            raise FeedUnavailableError(
                f"DhanFeed has no cash listing to draw bars from ({what}); "
                "refusing to publish an empty bar book that reads downstream "
                "as a quiet market. Add the underlying to this feed's universe."
            )

        return {BARS_KEY: bars_by_symbol, REGIME_KEY: self._regimes(bars_by_symbol)}

    def _regimes(self, bars_by_symbol: Mapping[str, Sequence[Bar]]) -> dict[str, RegimeRead]:
        out: dict[str, RegimeRead] = {}
        for symbol, bars in bars_by_symbol.items():
            read = self.regime_read(bars)
            if read is not None:
                out[symbol] = read
        for alias, target in self._regime_aliases.items():
            read = out.get(target)
            if read is not None:
                out[alias] = read
        return out

    def regime_read(self, bars: Sequence[Bar]) -> Optional[RegimeRead]:
        """Trend vs range, from CLOSED bars only.

        A fast mean pulling away from a slow mean by more than the recent bar
        range is a trend; anything inside that band is noise and reads as
        RANGE. The band is what stops a quiet tape from being called a trend
        just because the last few closes drifted a rupee.

        `as_of` is the LAST CLOSED BAR's timestamp, not `now`. Stamping it
        `now` would make an hours-old read look fresh to the strategy's own
        max-age check, which is the one guard standing between a stale regime
        and a flipped live structure.

        Returns None when there is not enough history. That is a legitimate
        absent state — the strategy simply does not enter — and it is very
        deliberately not a fabricated RANGE.
        """
        if len(bars) < self._regime_lookback:
            return None
        window = list(bars[-self._regime_lookback:])
        closes = [b.close for b in window]

        slow = sum(closes) / Decimal(len(closes))
        fast = sum(closes[-self._regime_fast:]) / Decimal(self._regime_fast)
        if slow <= 0:
            return None
        separation = (fast - slow) / slow

        band = sum((b.high - b.low) / b.close for b in window) / Decimal(len(window))
        threshold = band * self._regime_k

        if threshold <= 0:
            # A perfectly flat tape has no band to clear. Any movement at all
            # is then directional; none at all is a range.
            if separation > 0:
                state, confidence = RegimeState.TREND_UP, Decimal(1)
            elif separation < 0:
                state, confidence = RegimeState.TREND_DOWN, Decimal(1)
            else:
                state, confidence = RegimeState.RANGE, Decimal(1)
        else:
            ratio = abs(separation) / threshold
            if separation > threshold:
                state = RegimeState.TREND_UP
                confidence = min(Decimal(1), ratio / Decimal(2))
            elif separation < -threshold:
                state = RegimeState.TREND_DOWN
                confidence = min(Decimal(1), ratio / Decimal(2))
            else:
                state = RegimeState.RANGE
                confidence = Decimal(1) - min(Decimal(1), ratio)

        confidence = min(Decimal(1), max(Decimal(0), confidence))
        return RegimeRead(
            state=state,
            confidence=confidence.quantize(Decimal("0.0001")),
            as_of=window[-1].ts,
        )
