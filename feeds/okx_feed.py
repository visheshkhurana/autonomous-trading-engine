"""
Real OKX spot market data, via ccxt public endpoints.

OKX market data is public: no key, no signature, no account. Nothing in this
file reads a credential, and the exchange object is INJECTED so there is no
code path that could construct an authenticated one by accident.

Five things here exist because their absence is silent and expensive:

1. The forming candle is dropped, always, through `drop_forming_bar()`. OKX
   returns the currently-forming candle as the last row of /candles and its
   close is simply the current price. Verified live on 2026-08-27T19:40Z:
   BTC/USDT's trailing 15m candle closed at 79985.8 and `fetch_ticker` reported
   last=79985.8 in the same second; ETH/USDT, 2508.52 and 2508.52. A strategy
   that acts on that bar acts on a number that changes under it.

2. A stale read FAILS. `assert_fresh()` runs on every `bars()` return —
   including a cache hit, which is the case that actually bites: a cache is
   exactly the mechanism by which a dead feed keeps serving a price that looks
   alive.

3. A failed fetch RAISES `FeedUnavailableError`. It never returns `[]` and it
   never falls back to the cached bars. "I could not reach the venue" and
   "the venue has nothing" are different answers and only one of them is
   survivable.

4. Requested symbols are filtered against the loaded markets before any batch
   call. ccxt raises BadSymbol on an unknown pair, and one delisted name would
   otherwise take down the whole batch — every other symbol included.

5. `bind()` loads markets and resolves every configured instrument before it
   returns. A feed that cannot price its own universe is unusable, and it says
   so at wiring time rather than at the first tick, when a strategy is already
   depending on it.

The vol-regime read published in `ctx.extra` is realised volatility of closed
bars and nothing else. It is not a signal, it is a measurement, and when there
is not enough history to measure it the symbol is ABSENT from the map rather
than present with a zero.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Sequence

from core.types import AssetClass, Instrument, Mode
from feeds.base import (
    FeedError,
    FeedStaleError,
    FeedUnavailableError,
    assert_fresh,
    drop_forming_bar,
    to_bars,
)
from strategies.base import BARS_KEY, Bar

UTC = dt.timezone.utc
LOG = logging.getLogger(__name__)

__all__ = [
    "OkxFeed",
    "VOL_REGIME_KEY",
    "BARS_KEY",
    "FEED_HEALTH_KEY",
    "OKX_TIMEFRAMES",
]

# ctx.extra keys this feed publishes. BARS_KEY is IMPORTED from strategies.base
# rather than restated, so the producer and the consumer cannot drift apart.
VOL_REGIME_KEY = "vol_regime"
FEED_HEALTH_KEY = "feed_health"

# Interval -> ccxt timeframe. Only intervals OKX actually publishes appear
# here; an interval that is not a real venue timeframe would otherwise be
# silently rounded by the exchange into bars of a different length than the
# strategy's stop and staleness maths assume.
OKX_TIMEFRAMES: Mapping[int, str] = {
    60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m",
    3600: "1h", 7200: "2h", 14400: "4h", 21600: "6h", 43200: "12h",
    86400: "1d", 259200: "3d", 604800: "1w",
}

# OKX's /market/candles caps a page at 300.
MAX_HISTORY = 300


class OkxFeedError(FeedError):
    """Base for this feed's own wiring problems."""


class MarketsUnavailable(FeedUnavailableError):
    """The market table could not be read. Distinct from 'this pair is not
    listed': with no table we know nothing about any pair, and treating that as
    'nothing is listed' would stand every symbol down for the wrong reason."""


class SymbolNotListed(OkxFeedError):
    """A configured instrument has no active OKX spot market. A wiring error,
    raised at bind()."""


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------

def _dec_opt(value: Any) -> Optional[Decimal]:
    """Decimal, or None for a genuinely absent field.

    A present-but-unparseable value raises. Reading it as 0 would fabricate a
    price, and a fabricated price is indistinguishable downstream from a real
    one.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("refusing to read a bool as a price")
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"cannot read a number from {value!r}") from exc
    raise TypeError(f"cannot read a number from {type(value).__name__}")


def _ms_to_dt(value: Any) -> Optional[dt.datetime]:
    number = _dec_opt(value)
    if number is None:
        return None
    return dt.datetime.fromtimestamp(float(number) / 1000.0, UTC)


def _drop_unclosed(
    rows: Sequence[Sequence],
    interval: dt.timedelta,
    now: dt.datetime,
    market: str,
) -> list[Sequence]:
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
    NOT tolerated is publishing the unclosed bar: if the trim empties the
    series, `bars()` raises through `assert_fresh`, which is the correct answer
    for a response we cannot age.
    """
    if not rows:
        return []

    # How far past `now` the newest candle's period runs. A genuine forming
    # candle overshoots by less than one interval; one full interval of slack
    # on top absorbs any plausible clock skew. Beyond that the venue's clock
    # and ours disagree by at least a whole bar, which means every age computed
    # from these timestamps — including the one `assert_fresh` is about to
    # check — is measuring against a lie. Trimming alone would make that case
    # look TIDIER: drop the future rows and what is left reads as perfectly
    # fresh while carrying prices from hours ago.
    overshoot = (dt.datetime.fromtimestamp(rows[-1][0] / 1000, UTC)
                 + interval) - now
    if overshoot > 2 * interval:
        raise FeedUnavailableError(
            f"OKX {market}: the newest candle's period ends "
            f"{overshoot.total_seconds() / 60:.1f}m after now ({now.isoformat()}); "
            f"the venue's clock and ours disagree by more than a {interval} bar, "
            "so no age computed from these timestamps can be trusted")

    out = list(rows)
    while out:
        trimmed = list(drop_forming_bar(out, interval, now))
        if len(trimmed) == len(out):
            break
        out = trimmed
    dropped = len(rows) - len(out)
    if dropped > 1:
        LOG.error(
            "OKX %s: %d trailing candles had not closed at %s; only closed "
            "bars are published", market, dropped, now.isoformat())
    return out


def _timeframe_for(interval: dt.timedelta) -> str:
    seconds = int(interval.total_seconds())
    if seconds <= 0 or interval != dt.timedelta(seconds=seconds):
        raise ValueError(f"bar_interval must be a positive whole number of seconds, got {interval}")
    timeframe = OKX_TIMEFRAMES.get(seconds)
    if timeframe is None:
        raise ValueError(
            f"OKX publishes no {interval} candle. Supported: "
            + ", ".join(OKX_TIMEFRAMES[k] for k in sorted(OKX_TIMEFRAMES))
        )
    return timeframe


# ---------------------------------------------------------------------------
# the feed
# ---------------------------------------------------------------------------

class OkxFeed:
    """`MarketFeed` over OKX spot, reading only public endpoints.

    `exchange` is any object with the ccxt surface used here — `load_markets`,
    `fetch_ohlcv`, `fetch_tickers`. It is injected so tests drive a fake and so
    this class has no way to reach a credentialed client.
    """

    def __init__(
        self,
        instruments: Sequence[Instrument],
        *,
        exchange: Any = None,
        bar_interval: dt.timedelta = dt.timedelta(minutes=15),
        history: int = 200,
        max_age_intervals: int = 3,
        quote_max_age: dt.timedelta = dt.timedelta(minutes=2),
        vol_window: int = 20,
        vol_baseline_window: int = 96,
        vol_quiet_ratio: Decimal = Decimal("0.7"),
        vol_elevated_ratio: Decimal = Decimal("1.4"),
    ) -> None:
        instruments = tuple(instruments)
        if not instruments:
            # An empty universe is a wiring mistake. Accepting it would produce
            # a feed that serves an empty bar book forever and looks healthy.
            raise ValueError("OkxFeed needs at least one instrument")
        wrong = [i.symbol for i in instruments if i.asset_class is not AssetClass.CRYPTO]
        if wrong:
            raise ValueError(
                f"OkxFeed prices OKX spot crypto only; refusing {', '.join(wrong)}"
            )
        if history < 2:
            raise ValueError("history must be >= 2 bars")
        if history > MAX_HISTORY:
            raise ValueError(f"OKX serves at most {MAX_HISTORY} candles per page, asked for {history}")
        if max_age_intervals < 2:
            # Not arbitrary. Dropping the forming bar already puts the newest
            # CLOSED bar between one and two intervals behind `now` — at
            # 19:44:59 on a 15m feed the newest closed bar is 19:15, 30 minutes
            # old and perfectly healthy. A tolerance of 1 would therefore refuse
            # every read from a working venue, and a feed that always fails is
            # indistinguishable from one that never works.
            raise ValueError(
                "max_age_intervals must be >= 2: after the forming bar is "
                "dropped a healthy newest closed bar is already up to 2 "
                "intervals old"
            )
        if quote_max_age <= dt.timedelta(0):
            raise ValueError("quote_max_age must be positive")
        if vol_window < 2:
            raise ValueError("vol_window must be >= 2 returns")
        if vol_baseline_window < vol_window:
            raise ValueError("vol_baseline_window must be >= vol_window")
        if not (Decimal(0) < vol_quiet_ratio < vol_elevated_ratio):
            raise ValueError("need 0 < vol_quiet_ratio < vol_elevated_ratio")

        self._instruments = instruments
        self._interval = bar_interval
        # Raises here, at construction, for an interval OKX does not publish.
        self._timeframe = _timeframe_for(bar_interval)
        self._history = history
        self._max_age_intervals = max_age_intervals
        self._quote_max_age = quote_max_age
        self._vol_window = vol_window
        self._vol_baseline_window = vol_baseline_window
        self._vol_quiet = vol_quiet_ratio
        self._vol_elevated = vol_elevated_ratio

        if exchange is None:
            exchange = self._default_exchange()
        self._ex = exchange

        self._lock = threading.RLock()
        self._bound = False
        self._modes: dict[AssetClass, Mode] = {}
        self._markets: Optional[dict[str, Any]] = None
        # instrument symbol -> ccxt market symbol, filled at bind()
        self._market_symbol: dict[str, str] = {}
        self._bars_cache: dict[tuple[str, str], list[Bar]] = {}
        # What we could NOT serve on the last pass, kept so the omission travels
        # with ctx.extra instead of vanishing.
        self._stale_bars: dict[str, str] = {}
        self._unpriced: dict[str, str] = {}

    # -- construction ------------------------------------------------------

    @staticmethod
    def _default_exchange() -> Any:
        """A PUBLIC ccxt.okx. No apiKey, no secret, no password: this object
        cannot sign a request, so it cannot touch an account even by mistake."""
        try:
            import ccxt                      # local import: ccxt is optional for tests
        except ImportError as exc:           # pragma: no cover - environment problem
            raise OkxFeedError(
                "ccxt is not installed and no exchange was injected"
            ) from exc
        return ccxt.okx({"enableRateLimit": True})

    # -- properties --------------------------------------------------------

    @property
    def bar_interval(self) -> dt.timedelta:
        return self._interval

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def instruments(self) -> tuple[Instrument, ...]:
        return self._instruments

    def market_symbol(self, symbol: str) -> str:
        """The ccxt market this feed prices `symbol` from. Raises for a symbol
        this feed was never given — answering with a guess is how a strategy
        ends up priced off a market it is not trading."""
        try:
            return self._market_symbol[symbol]
        except KeyError:
            raise OkxFeedError(
                f"{symbol} is not in this feed's universe "
                f"({', '.join(sorted(self._market_symbol)) or 'unbound'})"
            ) from None

    # -- bind --------------------------------------------------------------

    def bind(self, resolved_modes: Mapping[AssetClass, Mode]) -> "OkxFeed":
        """Vouch for the feed against the modes the engine actually resolved.

        Crypto trades 24/7 on OKX, so `CRYPTO: LIVE` is legitimate here and is
        accepted — that is the whole point of this file existing.

        What is NOT accepted is a feed that cannot do its job. Markets are
        loaded and every configured instrument is resolved to an active spot
        market before this returns. An unusable feed fails now, while an
        operator is watching, rather than at the first tick when a strategy is
        already depending on it.

        Other asset classes are not policed here: this feed is one of possibly
        several, and a composite may bind it with the full mode map while
        routing equity elsewhere. It refuses per call instead — `bars()`,
        `refresh_quotes()` and `extra_provider()` all raise for anything
        outside this feed's universe, and never return an empty result.
        """
        with self._lock:
            markets = self._load_markets()
            resolved: dict[str, str] = {}
            unlisted: list[str] = []
            for instrument in self._instruments:
                market = self._resolve(instrument, markets)
                if market is None:
                    unlisted.append(instrument.symbol)
                else:
                    resolved[instrument.symbol] = market
            if unlisted:
                raise SymbolNotListed(
                    "OKX lists no active spot market for "
                    f"{', '.join(sorted(unlisted))}; refusing to bind a feed "
                    "that cannot price its own universe"
                )
            self._market_symbol = resolved
            self._modes = dict(resolved_modes)
            self._bars_cache.clear()
            self._stale_bars.clear()
            self._unpriced.clear()
            self._bound = True
        LOG.info("OkxFeed bound: %d instruments on %s (%s), modes=%s",
                 len(resolved), self._timeframe,
                 ", ".join(f"{k}->{v}" for k, v in sorted(resolved.items())),
                 {k.value: v.value for k, v in self._modes.items()})
        return self

    def _load_markets(self) -> dict[str, Any]:
        try:
            markets = self._ex.load_markets()
        except Exception as exc:
            raise MarketsUnavailable(f"OKX load_markets failed: {exc!r}") from exc
        if not isinstance(markets, dict) or not markets:
            markets = getattr(self._ex, "markets", None)
        # Zero markets is a failed read, not an exchange that lists nothing.
        # Accepting {} would make every symbol resolve to "not listed" and
        # stand the whole book down for a reason that is not true.
        if not isinstance(markets, dict) or not markets:
            raise MarketsUnavailable(
                "OKX load_markets returned no markets; treating this as a failed "
                "read rather than an exchange that lists nothing"
            )
        self._markets = markets
        return markets

    def _resolve(self, instrument: Instrument, markets: Mapping[str, Any]) -> Optional[str]:
        """The exact market to price this instrument from, or None.

        Deliberately narrower than the trading adapter's quote-currency
        fallback chain: a feed that answers a BTC/USDT question with BTC/USDC
        bars prices a book against a market it is not trading. Only the names
        the caller actually gave us are considered.
        """
        for candidate in (
            (instrument.broker_id or "").strip(),
            instrument.symbol.strip(),
        ):
            if candidate and self._usable(candidate, markets):
                return candidate
        return None

    @staticmethod
    def _usable(symbol: str, markets: Mapping[str, Any]) -> bool:
        market = markets.get(symbol)
        if not isinstance(market, Mapping):
            return False
        # `spot is not True` also excludes OKX swaps such as BTC/USDT:USDT.
        # Pricing a spot book off a perpetual is a basis error nobody sees.
        if market.get("spot") is not True:
            return False
        if market.get("active") is False:       # delisted; absent means unknown
            return False
        return True

    def _require_bound(self) -> None:
        if not self._bound:
            raise OkxFeedError(
                "OkxFeed used before bind(); markets have not been loaded and "
                "the universe has not been checked against the venue"
            )

    def _class_instruments(self, asset_class: AssetClass) -> tuple[Instrument, ...]:
        found = tuple(i for i in self._instruments if i.asset_class is asset_class)
        if not found:
            # NOT an empty result. This feed was asked to serve a class it holds
            # nothing for, which is a routing error; an empty bar book here
            # would read downstream as a quiet market.
            raise FeedUnavailableError(
                f"OkxFeed holds no instruments for {asset_class.value}; it prices "
                f"OKX spot crypto only. Route {asset_class.value} to its own feed."
            )
        return found

    # -- bars --------------------------------------------------------------

    def bars(self, symbol: str, now: dt.datetime) -> list[Bar]:
        """Closed bars for `symbol`, newest last, guaranteed fresh.

        Never returns an empty list: no closed bars is `FeedStaleError`, an
        unreachable venue is `FeedUnavailableError`. Both are loud, and the
        difference between them is preserved.
        """
        self._require_bound()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("OkxFeed.bars needs a timezone-aware `now`")
        market = self.market_symbol(symbol)
        key = (market, self._timeframe)

        with self._lock:
            bars = self._bars_cache.get(key)
            if bars is None or self._could_have_closed(bars, now):
                bars = self._fetch_bars(market, now)
                # Only a non-empty result is cached. Caching [] would make the
                # next call's "has a new bar closed?" question unanswerable and
                # tempt a fallback.
                if bars:
                    self._bars_cache[key] = bars
                elif key in self._bars_cache:
                    del self._bars_cache[key]

        # Runs on a cache hit as well as a fresh fetch. A cache is precisely
        # the mechanism by which a dead feed keeps serving a live-looking price,
        # so the freshness question is asked of what we are about to RETURN,
        # never of what we happened to fetch.
        assert_fresh(bars, symbol, self._interval, now, self._max_age_intervals)
        return list(bars)

    def _could_have_closed(self, bars: Sequence[Bar], now: dt.datetime) -> bool:
        """True when another bar could have closed since the cached newest one.

        The bar after `bars[-1]` opens at `bars[-1].ts + interval` and closes at
        `bars[-1].ts + 2 * interval`. Before that instant a refetch cannot
        return anything new, so we do not make the call. `now` moving backwards
        (a replay, a clock step) also refetches rather than trusting a cache
        built for a different instant.
        """
        if not bars:
            return True
        newest = bars[-1].ts
        return now >= newest + 2 * self._interval or now < newest

    def _fetch_bars(self, market: str, now: dt.datetime) -> list[Bar]:
        try:
            raw = self._ex.fetch_ohlcv(market, self._timeframe, None, self._history)
        except Exception as exc:
            # No fallback to the cache. Serving the last known bars because the
            # network failed is how a book trades a frozen price while every
            # dashboard reads green.
            raise FeedUnavailableError(
                f"OKX fetch_ohlcv({market}, {self._timeframe}) failed: {exc!r}"
            ) from exc
        if raw is None or not isinstance(raw, (list, tuple)):
            raise FeedUnavailableError(
                f"OKX fetch_ohlcv({market}) returned {type(raw).__name__}, not a list of candles"
            )
        rows = sorted((r for r in raw), key=lambda r: r[0])
        # Not `drop_forming_bar` alone: it removes exactly one row, and a
        # response that runs ahead of `now` has more than one unclosed row.
        closed = _drop_unclosed(rows, self._interval, now, market)
        bars = to_bars(closed)
        # to_bars validates each bar; ordering across bars is ours to check.
        # A repeated or out-of-order timestamp means two pages were merged or a
        # gap was replayed, and every indicator below would return a confident
        # number computed on nonsense.
        for i in range(1, len(bars)):
            if bars[i].ts <= bars[i - 1].ts:
                raise FeedUnavailableError(
                    f"OKX {market} candles are not strictly increasing: "
                    f"{bars[i].ts} follows {bars[i - 1].ts}"
                )
        return bars

    # -- quotes ------------------------------------------------------------

    def refresh_quotes(self, quotes, asset_class: AssetClass, now: dt.datetime) -> None:
        """Push a fresh mark for every instrument of `asset_class` in ONE batch.

        Symbols are filtered against the loaded markets first: ccxt raises
        BadSymbol on an unknown pair and one delisted name would drop the whole
        batch, unpricing every other instrument along with it.

        A symbol whose ticker is missing, unpriced, or older than
        `quote_max_age` has its quote REMOVED rather than left at the previous
        value. A stale mark is the failure this whole module exists to prevent,
        and the paper adapter already rejects an order with no quote — absence
        is a defined, loud state, where a leftover price is not.

        That applies to a TRANSPORT failure too, and it is the case that used
        to slip through: the marks are retired BEFORE the blackout is raised.
        Raising with the previous prices still sitting in the book is loud and
        wrong at the same time — nothing trades on that tick, but the book is
        still marked off frozen numbers by everything that reads the quote
        source afterwards.
        """
        self._require_bound()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("OkxFeed.refresh_quotes needs a timezone-aware `now`")
        instruments = self._class_instruments(asset_class)

        # Filter against the LOADED MARKETS before the batch call, not after.
        # ccxt raises BadSymbol on an unknown pair and the exception takes the
        # whole request with it, so one name delisted since bind() would unprice
        # every other instrument in the class. A name that fails this filter is
        # reported as unpriced, individually.
        markets = self._markets or {}
        wanted: dict[str, Instrument] = {}
        failures: dict[str, str] = {}
        for instrument in instruments:
            market = self.market_symbol(instrument.symbol)
            if self._usable(market, markets):
                wanted[market] = instrument
            else:
                failures[instrument.symbol] = (
                    f"{market} is no longer an active OKX spot market")
                self._drop_quote(quotes, instrument)

        if not wanted:
            with self._lock:
                self._unpriced = dict(failures)
            raise FeedUnavailableError(
                f"no active OKX spot market for any {asset_class.value} instrument: "
                + "; ".join(f"{k}: {v}" for k, v in sorted(failures.items()))
            )

        try:
            tickers = self._ex.fetch_tickers(sorted(wanted))
        except Exception as exc:
            # Every mark this call was supposed to re-establish goes FIRST.
            # Raising straight out would be loud and wrong at the same time:
            # the tick that failed does not trade, but the quote source is
            # what `adapter.equity()` marks the book against at the top of the
            # NEXT tick, before anything refetches, and the risk circuit
            # advances on whatever number that produces. A frozen mark there
            # under-reports the drawdown for exactly as long as the venue is
            # unreachable.
            self._blackout(quotes, wanted, f"OKX fetch_tickers failed: {exc!r}")
            raise FeedUnavailableError(
                f"OKX fetch_tickers({sorted(wanted)}) failed: {exc!r}"
            ) from exc
        if not isinstance(tickers, Mapping):
            self._blackout(
                quotes, wanted,
                f"OKX fetch_tickers returned {type(tickers).__name__}")
            raise FeedUnavailableError(
                f"OKX fetch_tickers returned {type(tickers).__name__}, not a mapping"
            )

        priced = 0
        for market, instrument in wanted.items():
            price, why = self._mark_from_ticker(tickers.get(market), now)
            if price is None:
                failures[instrument.symbol] = why
                self._drop_quote(quotes, instrument)
                continue
            quotes.set(instrument.symbol, price)
            priced += 1

        with self._lock:
            self._unpriced = dict(failures)

        if failures:
            LOG.error("OkxFeed could not price %s: %s",
                      ", ".join(sorted(failures)),
                      "; ".join(f"{k}: {v}" for k, v in sorted(failures.items())))
        if priced == 0:
            # Every requested symbol failed. That is not a per-symbol condition,
            # it is a blackout, and returning quietly would let the tick proceed
            # against a quote source we just emptied.
            raise FeedUnavailableError(
                f"OKX returned no usable ticker for any of {sorted(wanted)}: "
                + "; ".join(f"{k}: {v}" for k, v in sorted(failures.items()))
            )

    def _mark_from_ticker(self, ticker: Any, now: dt.datetime) -> tuple[Optional[Decimal], str]:
        """(price, why-not). `last`, else the bid/ask mid — and only if the
        ticker carries a timestamp we can age. A ticker with no timestamp
        cannot be shown to be current, and "probably fine" is exactly the
        assumption that lets a frozen feed through."""
        if ticker is None:
            return None, "no ticker in the batch response"
        if not isinstance(ticker, Mapping):
            return None, f"ticker is {type(ticker).__name__}, not a mapping"

        stamped = _ms_to_dt(ticker.get("timestamp"))
        if stamped is None:
            return None, "ticker carries no timestamp; freshness cannot be established"
        age = now - stamped
        if age > self._quote_max_age:
            return None, (f"ticker is {age.total_seconds():.1f}s old "
                          f"(tolerance {self._quote_max_age.total_seconds():.0f}s)")
        if -age > self._quote_max_age:
            return None, (f"ticker is stamped {(-age).total_seconds():.1f}s in the future; "
                          "clock or venue disagreement")

        for key in ("last", "close"):
            price = _dec_opt(ticker.get(key))
            if price is not None and price > 0:
                return price, ""
        bid, ask = _dec_opt(ticker.get("bid")), _dec_opt(ticker.get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            if ask < bid:
                return None, f"crossed book: bid {bid} > ask {ask}"
            return (bid + ask) / 2, ""
        return None, "ticker carries no positive last, close or bid/ask"

    def _blackout(self, quotes, wanted: Mapping[str, Instrument],
                  why: str) -> None:
        """Retire every mark a failed batch was supposed to re-establish.

        Called on the paths that raise before any per-symbol answer exists. A
        mark is a claim about right now; a transport failure is the moment we
        stopped being able to make it, and leaving it in place turns "we lost
        the venue" into "the price has not moved".
        """
        failures = dict(self._unpriced)
        for instrument in wanted.values():
            failures[instrument.symbol] = why
            self._drop_quote(quotes, instrument)
        with self._lock:
            self._unpriced = failures
        LOG.error("OkxFeed blackout, marks removed for %s: %s",
                  ", ".join(sorted(i.symbol for i in wanted.values())), why)

    @staticmethod
    def _drop_quote(quotes, instrument: Instrument) -> None:
        """Remove a quote we can no longer vouch for.

        Not a swallowed failure: the end state is 'this symbol has no price',
        which the paper adapter turns into a REJECTED order and a live adapter
        into a refusal to size. Leaving the old number there is the only
        outcome that would be silent.
        """
        remove = getattr(quotes, "remove", None)
        if remove is None:
            raise FeedUnavailableError(
                f"cannot clear the stale quote for {instrument.symbol}: this quote "
                f"source ({type(quotes).__name__}) has no remove()"
            )
        if quotes.quote(instrument) is not None:
            remove(instrument.symbol)

    # -- extra -------------------------------------------------------------

    def extra_provider(self, asset_class: AssetClass, now: dt.datetime) -> dict:
        """`ctx.extra` for one class: the bar book, a vol read, and what failed.

        Per-symbol staleness omits that symbol from the bar book rather than
        blinding the whole class. That is the more accurate signal and it is
        still loud: `StrategyBase.bars_for(must_have=True)` raises
        MissingDataError when a position is open in a symbol with no bars, and
        an absent symbol produces no new entries. The omission is also
        published under `feed_health` so it travels with the context instead of
        vanishing.

        A transport failure is different and propagates: we know nothing, so
        nothing may be decided. And if EVERY symbol is stale, that is a
        blackout, not a per-symbol condition — it raises rather than handing
        back an empty book that reads as a quiet market.
        """
        self._require_bound()
        instruments = self._class_instruments(asset_class)

        book: dict[str, list[Bar]] = {}
        stale: dict[str, str] = {}
        for instrument in instruments:
            try:
                book[instrument.symbol] = self.bars(instrument.symbol, now)
            except FeedStaleError as exc:
                # Recorded and logged, never discarded. FeedUnavailableError is
                # deliberately NOT caught here: an unreachable venue is not a
                # per-symbol condition.
                stale[instrument.symbol] = str(exc)
                LOG.error("OkxFeed: %s has no fresh bars: %s", instrument.symbol, exc)

        with self._lock:
            self._stale_bars = dict(stale)
            unpriced = dict(self._unpriced)

        if not book:
            raise FeedStaleError(
                f"no fresh bars for any {asset_class.value} instrument "
                f"({', '.join(sorted(stale))}); refusing to publish an empty bar "
                "book that reads as a quiet market"
            )

        vol: dict[str, dict] = {}
        for sym, sym_bars in book.items():
            read = self._vol_read(sym, sym_bars, now)
            # A symbol with too little history is ABSENT from the map. Present
            # with a zero would read as an extremely calm market.
            if read is not None:
                vol[sym] = read

        return {
            BARS_KEY: book,
            VOL_REGIME_KEY: vol,
            FEED_HEALTH_KEY: {
                "source": "okx",
                "timeframe": self._timeframe,
                "as_of": now,
                "served": sorted(book),
                "stale_bars": stale,
                "unpriced": unpriced,
            },
        }

    # -- vol ---------------------------------------------------------------

    def _vol_read(self, symbol: str, bars: Sequence[Bar], now: dt.datetime) -> Optional[dict]:
        """Realised volatility of closed bars. A measurement, not a signal.

        Returns None — and the symbol is then ABSENT from the map — when there
        is not enough history to measure. A zero vol would read as a real,
        extremely calm market and size positions accordingly.
        """
        window = self._vol_window
        if len(bars) < window + 1:
            return None
        recent = self._realised_vol(bars[-(window + 1):])
        if recent is None:
            return None

        baseline_bars = bars[-(self._vol_baseline_window + 1):]
        baseline = (self._realised_vol(baseline_bars)
                    if len(baseline_bars) >= window + 1 else None)

        # 24/7 market: every calendar interval is a trading interval.
        per_year = Decimal(365 * 24 * 3600) / Decimal(int(self._interval.total_seconds()))
        annualised = recent * per_year.sqrt()

        ratio: Optional[Decimal] = None
        state = "unknown"
        if baseline is not None and baseline > 0:
            ratio = recent / baseline
            if ratio < self._vol_quiet:
                state = "quiet"
            elif ratio > self._vol_elevated:
                state = "elevated"
            else:
                state = "normal"

        return {
            "as_of": bars[-1].ts,
            "source": "okx",
            "window": window,
            "baseline_window": min(self._vol_baseline_window, len(bars) - 1),
            "realised_vol": recent,          # per bar, fraction
            "annualised_vol": annualised,    # fraction
            "baseline_vol": baseline,
            "ratio": ratio,
            # `state` is "unknown" until there is a baseline to compare against.
            # It is never defaulted to "normal": "I have not measured this" and
            # "I measured this and it is ordinary" are different claims.
            "state": state,
            "quiet_below": self._vol_quiet,
            "elevated_above": self._vol_elevated,
        }

    @staticmethod
    def _realised_vol(bars: Sequence[Bar]) -> Optional[Decimal]:
        """Root-mean-square of simple close-to-close returns.

        Zero-mean by construction, which is the standard realised-volatility
        estimator over a short window and avoids pretending a 20-bar drift is a
        reliable mean. All-Decimal: no float ever touches a price here.
        """
        if len(bars) < 2:
            return None
        total = Decimal(0)
        n = 0
        for i in range(1, len(bars)):
            prev = bars[i - 1].close
            if prev <= 0:                     # Bar already rejects this; belt and braces
                return None
            r = (bars[i].close - prev) / prev
            total += r * r
            n += 1
        if n == 0:
            return None
        return (total / Decimal(n)).sqrt()

    # -- introspection -----------------------------------------------------

    @property
    def stale_bars(self) -> dict[str, str]:
        """Symbols the last `extra_provider()` could not serve, and why."""
        with self._lock:
            return dict(self._stale_bars)

    @property
    def unpriced(self) -> dict[str, str]:
        """Symbols the last `refresh_quotes()` could not price, and why."""
        with self._lock:
            return dict(self._unpriced)

    def invalidate(self, symbol: Optional[str] = None) -> None:
        """Drop cached bars. Used by an operator or a test; never used to paper
        over a failed fetch."""
        with self._lock:
            if symbol is None:
                self._bars_cache.clear()
                return
            self._bars_cache.pop((self.market_symbol(symbol), self._timeframe), None)
