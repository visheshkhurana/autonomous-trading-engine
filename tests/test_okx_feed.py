"""
Tests for the OKX market data feed.

Two of these matter more than the rest, and both are written so that they FAIL
if the safety rule is removed rather than merely exercising the happy path:

  * `test_forming_bar_is_never_published` and its siblings — delete
    `drop_forming_bar()` from `_fetch_bars` and they fail. The offline fixture
    reproduces exactly what live OKX does: the trailing candle's close is the
    ticker's last price.
  * `test_*stale*` — delete `assert_fresh()` from `bars()` and they fail,
    including the one that goes through the CACHE, which is the path where a
    dead feed would otherwise keep serving a live-looking price.

Everything here runs against a FAKE exchange except the handful marked
`@pytest.mark.network`, which read public OKX endpoints and can be deselected
with `-m "not network"`. Nothing in this file authenticates to anything.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

import pytest

from adapters.paper import InMemoryQuoteSource
from core.types import AssetClass, Instrument, Mode
from feeds.base import FeedStaleError, FeedUnavailableError
from feeds.okx_feed import (
    BARS_KEY,
    FEED_HEALTH_KEY,
    VOL_REGIME_KEY,
    MarketsUnavailable,
    OkxFeed,
    OkxFeedError,
    SymbolNotListed,
)

UTC = dt.timezone.utc
M15 = dt.timedelta(minutes=15)

BTC = Instrument("BTC/USDT", AssetClass.CRYPTO, "BTC/USDT",
                 tick_size=Decimal("0.1"), qty_precision=6,
                 min_notional=Decimal("5"))
ETH = Instrument("ETH/USDT", AssetClass.CRYPTO, "ETH/USDT",
                 tick_size=Decimal("0.01"), qty_precision=5,
                 min_notional=Decimal("5"))
RELIANCE = Instrument("RELIANCE", AssetClass.EQUITY, "PAPER-RELIANCE")


# ---------------------------------------------------------------------------
# fake exchange
# ---------------------------------------------------------------------------

def spot_market(symbol: str, active: bool = True) -> dict:
    return {"symbol": symbol, "spot": True, "active": active,
            "precision": {"amount": 8, "price": 1}, "limits": {}}


DEFAULT_MARKETS = {
    "BTC/USDT": spot_market("BTC/USDT"),
    "ETH/USDT": spot_market("ETH/USDT"),
    # A swap with a spot-looking base. Must never be used to price spot.
    "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "spot": False, "swap": True,
                      "active": True},
}


def candles(anchor: dt.datetime, n: int, *, start: str = "100",
            step: str = "1", interval: dt.timedelta = M15) -> list[list]:
    """`n` rows [ts_ms, o, h, l, c, v], oldest first, ENDING at `anchor`.

    Ends at `anchor` so a caller can make the last row the forming one simply by
    anchoring it to the current, not-yet-closed period.
    """
    rows: list[list] = []
    base = Decimal(start)
    for i in range(n):
        ts = anchor - interval * (n - 1 - i)
        close = base + Decimal(step) * i
        rows.append([
            int(ts.timestamp() * 1000),
            float(close - Decimal("0.5")),   # open
            float(close + Decimal("1")),     # high
            float(close - Decimal("1")),     # low
            float(close),                    # close
            10.0,                            # volume
        ])
    return rows


class FakeExchange:
    """The ccxt surface this feed uses, and nothing else.

    It has no apiKey, no secret and no create_order: a test cannot accidentally
    reach an account through it, and neither can the feed.
    """

    def __init__(self, markets: Optional[Mapping[str, Any]] = None,
                 ohlcv: Optional[Mapping[str, Sequence[Sequence]]] = None,
                 tickers: Optional[Mapping[str, Any]] = None) -> None:
        self._markets = dict(DEFAULT_MARKETS if markets is None else markets)
        self._ohlcv = dict(ohlcv or {})
        self._tickers = dict(tickers or {})
        self.markets = self._markets
        self.load_markets_calls = 0
        self.ohlcv_calls: list[tuple] = []
        self.tickers_calls: list[list[str]] = []
        self.ticker_calls: list[str] = []          # must stay empty: batch only
        self.load_markets_error: Optional[Exception] = None
        self.ohlcv_error: Optional[Exception] = None
        self.tickers_error: Optional[Exception] = None

    # ccxt surface -------------------------------------------------------
    def load_markets(self, reload: bool = False) -> dict:
        self.load_markets_calls += 1
        if self.load_markets_error is not None:
            raise self.load_markets_error
        return self._markets

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", since=None,
                    limit=None) -> list:
        self.ohlcv_calls.append((symbol, timeframe, since, limit))
        if self.ohlcv_error is not None:
            raise self.ohlcv_error
        if symbol not in self._markets:
            raise ValueError(f"okx does not have market symbol {symbol}")  # ccxt BadSymbol
        return [list(r) for r in self._ohlcv.get(symbol, [])]

    def fetch_tickers(self, symbols=None) -> dict:
        self.tickers_calls.append(list(symbols or []))
        if self.tickers_error is not None:
            raise self.tickers_error
        for s in symbols or []:
            if s not in self._markets:
                raise ValueError(f"okx does not have market symbol {s}")
        return {s: self._tickers[s] for s in (symbols or self._tickers)
                if s in self._tickers}

    def fetch_ticker(self, symbol: str) -> dict:
        self.ticker_calls.append(symbol)
        return self._tickers[symbol]

    # helpers ------------------------------------------------------------
    def set_ohlcv(self, symbol: str, rows: Sequence[Sequence]) -> None:
        self._ohlcv[symbol] = [list(r) for r in rows]

    def set_ticker(self, symbol: str, *, last, at: dt.datetime,
                   bid=None, ask=None) -> None:
        self._tickers[symbol] = {
            "symbol": symbol, "timestamp": int(at.timestamp() * 1000),
            "last": last, "close": last, "bid": bid, "ask": ask,
        }


def build(now: dt.datetime, *, instruments=(BTC, ETH), history: int = 60,
          n: int = 60, forming: bool = True, **kw) -> tuple[OkxFeed, FakeExchange]:
    """A bound feed whose fake venue is healthy as of `now`.

    `forming=True` anchors the last candle to the CURRENT, still-open period —
    which is what OKX actually returns.
    """
    anchor = _floor(now, M15) if forming else _floor(now, M15) - M15
    ex = FakeExchange()
    for inst in instruments:
        if inst.asset_class is AssetClass.CRYPTO:
            ex.set_ohlcv(inst.broker_id, candles(anchor, n))
            ex.set_ticker(inst.broker_id, last=100.0, at=now)
    feed = OkxFeed(list(instruments), exchange=ex, history=history, **kw)
    feed.bind({AssetClass.CRYPTO: Mode.LIVE})
    return feed, ex


def _floor(ts: dt.datetime, step: dt.timedelta) -> dt.datetime:
    secs = int(step.total_seconds())
    epoch = int(ts.timestamp())
    return dt.datetime.fromtimestamp(epoch - (epoch % secs), UTC)


NOW = dt.datetime(2026, 8, 27, 19, 40, 41, tzinfo=UTC)


# ===========================================================================
# RULE 1 — the forming bar is never published
# ===========================================================================

def test_forming_bar_is_never_published():
    """The trailing candle's period has not closed, so it must not be a Bar.

    Delete `drop_forming_bar()` from `_fetch_bars` and this fails: the newest
    bar becomes 19:30, whose period ends at 19:45, after `now`.
    """
    feed, ex = build(NOW)                       # NOW = 19:40:41, so 19:30 is open
    bars = feed.bars("BTC/USDT", NOW)

    assert bars, "a healthy feed must return bars"
    newest = bars[-1]
    assert newest.ts == dt.datetime(2026, 8, 27, 19, 15, tzinfo=UTC)
    assert newest.ts + M15 <= NOW, "published bar's period must have closed"
    # The row the venue sent last is simply absent.
    assert all(b.ts != dt.datetime(2026, 8, 27, 19, 30, tzinfo=UTC) for b in bars)
    assert len(bars) == len(ex._ohlcv["BTC/USDT"]) - 1


def test_forming_bar_close_equals_ticker_last_and_is_still_dropped():
    """The live-verified shape, reproduced offline.

    On 2026-08-27T19:40Z live OKX returned a trailing 15m candle closing at
    79985.8 while `fetch_ticker` reported last=79985.8 in the same second — the
    forming candle's "close" IS the current price. That number must never reach
    a strategy as a closed bar.
    """
    anchor = _floor(NOW, M15)                   # 19:30, still forming
    ex = FakeExchange()
    rows = candles(anchor, 40)
    rows[-1][4] = 79985.8                       # forming close
    ex.set_ohlcv("BTC/USDT", rows)
    ex.set_ticker("BTC/USDT", last=79985.8, at=NOW)   # identical, as live
    feed = OkxFeed([BTC], exchange=ex, history=40).bind({AssetClass.CRYPTO: Mode.LIVE})

    bars = feed.bars("BTC/USDT", NOW)
    assert Decimal("79985.8") not in [b.close for b in bars]

    # And the mark the quote source gets IS that live price — the two are
    # different things and the feed keeps them apart.
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC).price == Decimal("79985.8")


def test_a_closed_trailing_bar_is_kept():
    """The drop is conditional, not a blanket 'lose the last row'."""
    just_after = _floor(NOW, M15) + M15 + dt.timedelta(seconds=1)   # 19:45:01
    feed, ex = build(just_after)
    bars = feed.bars("BTC/USDT", just_after)
    # anchor is 19:45 (open); newest closed is 19:30, whose period ended 19:45.
    assert bars[-1].ts == dt.datetime(2026, 8, 27, 19, 30, tzinfo=UTC)
    assert bars[-1].ts + M15 <= just_after


def test_bars_are_tz_aware_decimals():
    feed, _ = build(NOW)
    for b in feed.bars("ETH/USDT", NOW):
        assert b.ts.tzinfo is not None and b.ts.utcoffset() is not None
        for field in ("open", "high", "low", "close", "volume"):
            assert isinstance(getattr(b, field), Decimal)


# ===========================================================================
# RULE 2 — a stale feed fails, it does not serve old prices
# ===========================================================================

def test_stale_bars_raise_rather_than_being_served():
    """Delete `assert_fresh()` from `bars()` and this fails."""
    feed, ex = build(NOW)
    old = NOW - dt.timedelta(hours=6)
    ex.set_ohlcv("BTC/USDT", candles(_floor(old, M15), 40))
    feed.invalidate()

    with pytest.raises(FeedStaleError) as err:
        feed.bars("BTC/USDT", NOW)
    assert "stale" in str(err.value).lower()


def test_freshness_is_checked_on_the_cache_hit_path_too():
    """A cache is exactly the mechanism by which a dead feed keeps serving a
    live-looking price, so the freshness question is asked of what `bars()` is
    about to RETURN, not of what it happened to fetch.

    Delete `assert_fresh()` from `bars()` and this fails on the second call —
    the one that never touches the venue.
    """
    import feeds.okx_feed as mod

    feed, ex = build(NOW)
    checked: list[tuple] = []
    real = mod.assert_fresh

    def spy(bars, symbol, interval, now, max_age_intervals=3):
        checked.append((symbol, now, len(bars)))
        return real(bars, symbol, interval, now, max_age_intervals)

    mod.assert_fresh = spy
    try:
        feed.bars("BTC/USDT", NOW)
        fetches = len(ex.ohlcv_calls)
        feed.bars("BTC/USDT", NOW + dt.timedelta(seconds=30))
    finally:
        mod.assert_fresh = real

    assert len(ex.ohlcv_calls) == fetches, "second call must be a cache hit"
    assert len(checked) == 2, "the cache hit was returned unchecked"


def test_a_tolerance_that_would_refuse_every_healthy_read_is_rejected():
    """Dropping the forming bar already leaves the newest CLOSED bar up to two
    intervals behind `now`, so a 1-interval tolerance would fail on live data
    that is perfectly fine — a feed that always raises teaches an operator to
    ignore it."""
    with pytest.raises(ValueError) as err:
        OkxFeed([BTC], exchange=FakeExchange(), max_age_intervals=1)
    assert "2" in str(err.value)


def test_the_cache_is_refetched_before_it_can_go_stale():
    """The structural reason a cache hit can never be stale: the refetch
    threshold (2 intervals) sits inside the staleness tolerance (3). If someone
    widens the cache window past the tolerance, this fails."""
    feed, ex = build(NOW)
    feed.bars("BTC/USDT", NOW)
    fetches = len(ex.ohlcv_calls)

    # One second before the staleness tolerance would bite, a refetch has
    # already happened rather than the cache being served.
    edge = NOW + 3 * M15 - dt.timedelta(seconds=1)
    ex.set_ohlcv("BTC/USDT", candles(_floor(edge, M15), 60))
    feed.bars("BTC/USDT", edge)
    assert len(ex.ohlcv_calls) == fetches + 1


def test_a_frozen_venue_raises_instead_of_replaying_the_last_good_bars():
    """The venue answers, but with the same candles it gave an hour ago."""
    feed, ex = build(NOW)
    good = feed.bars("BTC/USDT", NOW)
    assert good

    much_later = NOW + dt.timedelta(hours=2)
    with pytest.raises(FeedStaleError):
        feed.bars("BTC/USDT", much_later)           # refetch, same frozen rows


def test_no_closed_bars_is_stale_not_empty():
    """A venue that returns only a forming candle has given us nothing."""
    anchor = _floor(NOW, M15)
    ex = FakeExchange()
    ex.set_ohlcv("BTC/USDT", candles(anchor, 1))
    feed = OkxFeed([BTC], exchange=ex, history=10).bind({AssetClass.CRYPTO: Mode.LIVE})
    with pytest.raises(FeedStaleError):
        feed.bars("BTC/USDT", NOW)


def test_empty_candle_response_raises_and_returns_nothing():
    feed, ex = build(NOW)
    ex.set_ohlcv("BTC/USDT", [])
    feed.invalidate()
    with pytest.raises(FeedStaleError):
        feed.bars("BTC/USDT", NOW)


# ===========================================================================
# transport failure — raise, never [], never the cache
# ===========================================================================

def test_network_failure_raises_feed_unavailable():
    feed, ex = build(NOW)
    ex.ohlcv_error = ConnectionError("okx GET /candles: connection reset")
    feed.invalidate()
    with pytest.raises(FeedUnavailableError) as err:
        feed.bars("BTC/USDT", NOW)
    assert "fetch_ohlcv" in str(err.value)


def test_network_failure_does_not_fall_back_to_the_cache():
    """The cache exists to save calls, never to cover a failure."""
    feed, ex = build(NOW)
    warm = feed.bars("BTC/USDT", NOW)
    assert warm

    ex.ohlcv_error = TimeoutError("okx timed out")
    later = NOW + dt.timedelta(hours=1)             # past the refetch threshold
    with pytest.raises(FeedUnavailableError):
        feed.bars("BTC/USDT", later)


def test_non_list_candle_response_raises():
    feed, ex = build(NOW)
    ex._ohlcv["BTC/USDT"] = []
    feed.invalidate()
    ex.fetch_ohlcv = lambda *a, **k: None           # type: ignore[assignment]
    with pytest.raises(FeedUnavailableError):
        feed.bars("BTC/USDT", NOW)


def test_out_of_order_candles_raise():
    """Duplicated or reversed stamps mean two pages were merged or a gap was
    replayed; every indicator downstream would compute a confident number on
    nonsense."""
    feed, ex = build(NOW)
    rows = candles(_floor(NOW, M15), 20)
    rows[5][0] = rows[4][0]                         # duplicate timestamp
    ex.set_ohlcv("BTC/USDT", rows)
    feed.invalidate()
    with pytest.raises(FeedUnavailableError) as err:
        feed.bars("BTC/USDT", NOW)
    assert "increasing" in str(err.value)


# ===========================================================================
# caching
# ===========================================================================

def test_cache_avoids_refetching_inside_the_same_bar():
    feed, ex = build(NOW)
    feed.bars("BTC/USDT", NOW)
    n = len(ex.ohlcv_calls)
    for offset in (1, 30, 60, 120):
        feed.bars("BTC/USDT", NOW + dt.timedelta(seconds=offset))
    assert len(ex.ohlcv_calls) == n, "must not hammer the venue every tick"


def test_cache_refetches_once_a_new_bar_could_have_closed():
    feed, ex = build(NOW)
    feed.bars("BTC/USDT", NOW)
    n = len(ex.ohlcv_calls)

    later = NOW + 2 * M15
    ex.set_ohlcv("BTC/USDT", candles(_floor(later, M15), 60))
    feed.bars("BTC/USDT", later)
    assert len(ex.ohlcv_calls) == n + 1


def test_cache_is_per_symbol():
    feed, ex = build(NOW)
    feed.bars("BTC/USDT", NOW)
    feed.bars("ETH/USDT", NOW)
    assert {c[0] for c in ex.ohlcv_calls} == {"BTC/USDT", "ETH/USDT"}


def test_clock_going_backwards_refetches_rather_than_trusting_the_cache():
    feed, ex = build(NOW)
    feed.bars("BTC/USDT", NOW)
    n = len(ex.ohlcv_calls)
    earlier = NOW - dt.timedelta(hours=3)
    ex.set_ohlcv("BTC/USDT", candles(_floor(earlier, M15), 60))
    feed.bars("BTC/USDT", earlier)
    assert len(ex.ohlcv_calls) == n + 1


# ===========================================================================
# bind
# ===========================================================================

def test_bind_accepts_a_live_crypto_class():
    """Crypto is 24/7, so a live class is legitimate for this feed — that is the
    whole reason it exists."""
    ex = FakeExchange()
    ex.set_ohlcv("BTC/USDT", candles(_floor(NOW, M15), 40))
    feed = OkxFeed([BTC], exchange=ex)
    assert feed.bind({AssetClass.CRYPTO: Mode.LIVE}) is feed
    assert feed.bars("BTC/USDT", NOW)


def test_bind_refuses_when_markets_cannot_be_loaded():
    ex = FakeExchange()
    ex.load_markets_error = ConnectionError("okx unreachable")
    with pytest.raises(MarketsUnavailable):
        OkxFeed([BTC], exchange=ex).bind({AssetClass.CRYPTO: Mode.PAPER})


def test_bind_refuses_an_empty_market_table():
    """Zero markets is a failed read, not a venue that lists nothing. Accepting
    {} would stand every symbol down for a reason that is not true."""
    ex = FakeExchange(markets={})
    ex.markets = {}
    with pytest.raises(MarketsUnavailable):
        OkxFeed([BTC], exchange=ex).bind({AssetClass.CRYPTO: Mode.PAPER})


def test_bind_refuses_an_instrument_the_venue_does_not_list():
    ghost = Instrument("GHOST/USDT", AssetClass.CRYPTO, "GHOST/USDT")
    ex = FakeExchange()
    with pytest.raises(SymbolNotListed) as err:
        OkxFeed([BTC, ghost], exchange=ex).bind({AssetClass.CRYPTO: Mode.LIVE})
    assert "GHOST/USDT" in str(err.value)


def test_bind_refuses_a_delisted_instrument():
    markets = dict(DEFAULT_MARKETS)
    markets["ETH/USDT"] = spot_market("ETH/USDT", active=False)
    ex = FakeExchange(markets=markets)
    with pytest.raises(SymbolNotListed):
        OkxFeed([BTC, ETH], exchange=ex).bind({AssetClass.CRYPTO: Mode.LIVE})


def test_bind_will_not_price_spot_off_a_swap():
    swap_only = Instrument("BTCPERP", AssetClass.CRYPTO, "BTC/USDT:USDT")
    ex = FakeExchange()
    with pytest.raises(SymbolNotListed):
        OkxFeed([swap_only], exchange=ex).bind({AssetClass.CRYPTO: Mode.LIVE})


def test_feed_refuses_to_serve_before_bind():
    ex = FakeExchange()
    ex.set_ohlcv("BTC/USDT", candles(_floor(NOW, M15), 40))
    feed = OkxFeed([BTC], exchange=ex)
    with pytest.raises(OkxFeedError):
        feed.bars("BTC/USDT", NOW)
    with pytest.raises(OkxFeedError):
        feed.extra_provider(AssetClass.CRYPTO, NOW)
    with pytest.raises(OkxFeedError):
        feed.refresh_quotes(InMemoryQuoteSource(), AssetClass.CRYPTO, NOW)


def test_constructor_rejects_an_interval_okx_does_not_publish():
    with pytest.raises(ValueError) as err:
        OkxFeed([BTC], exchange=FakeExchange(), bar_interval=dt.timedelta(minutes=7))
    assert "7" in str(err.value) or "no" in str(err.value).lower()


def test_constructor_rejects_a_non_crypto_instrument():
    with pytest.raises(ValueError):
        OkxFeed([RELIANCE], exchange=FakeExchange())


def test_constructor_rejects_an_empty_universe():
    with pytest.raises(ValueError):
        OkxFeed([], exchange=FakeExchange())


def test_default_exchange_carries_no_credentials():
    """The default client is PUBLIC. It cannot sign a request, so it cannot
    reach an account even if something later asked it to."""
    ccxt = pytest.importorskip("ccxt")
    ex = OkxFeed._default_exchange()
    assert isinstance(ex, ccxt.okx)
    assert not getattr(ex, "apiKey", None)
    assert not getattr(ex, "secret", None)
    assert not getattr(ex, "password", None)
    assert ex.enableRateLimit is True


def test_unknown_symbol_is_refused_not_guessed():
    feed, _ = build(NOW)
    with pytest.raises(OkxFeedError):
        feed.bars("DOGE/USDT", NOW)


# ===========================================================================
# quotes
# ===========================================================================

def test_refresh_quotes_uses_one_batch_call():
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert len(ex.tickers_calls) == 1
    assert ex.tickers_calls[0] == ["BTC/USDT", "ETH/USDT"]
    assert ex.ticker_calls == [], "no per-symbol fetch_ticker fallback"
    assert quotes.quote(BTC).price == Decimal("100")
    assert quotes.quote(ETH).price == Decimal("100")


def test_one_delisted_name_does_not_drop_the_whole_batch():
    """ccxt raises BadSymbol on an unknown pair and takes the request with it.
    Filtering against the loaded markets first is what keeps one dead name from
    unpricing every other instrument."""
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(ETH) is not None

    ex._markets["ETH/USDT"] = spot_market("ETH/USDT", active=False)
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)

    assert ex.tickers_calls[-1] == ["BTC/USDT"], "the dead name never reaches the venue"
    assert quotes.quote(BTC) is not None, "the live name is still priced"
    assert quotes.quote(ETH) is None, "the dead name is unpriced, not stale"
    assert "ETH/USDT" in feed.unpriced


def test_a_stale_ticker_removes_the_quote_rather_than_leaving_the_old_one():
    """The failure this module exists to prevent. Remove the age check and the
    previous price survives, and the book trades on a frozen quote."""
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC).price == Decimal("100")

    later = NOW + dt.timedelta(minutes=30)
    ex.set_ticker("BTC/USDT", last=100.0, at=NOW)         # frozen
    ex.set_ticker("ETH/USDT", last=222.0, at=later)        # moving
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, later)

    assert quotes.quote(BTC) is None
    assert quotes.quote(ETH).price == Decimal("222")
    assert "BTC/USDT" in feed.unpriced


def test_a_ticker_with_no_timestamp_is_unusable():
    """Freshness cannot be established, so the price cannot be vouched for."""
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    ex._tickers["BTC/USDT"] = {"symbol": "BTC/USDT", "last": 100.0, "timestamp": None}
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None
    assert "timestamp" in feed.unpriced["BTC/USDT"]
    assert quotes.quote(ETH) is not None


def test_a_ticker_missing_from_the_batch_is_unpriced():
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    del ex._tickers["BTC/USDT"]
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None
    assert quotes.quote(ETH) is not None


def test_bid_ask_mid_is_used_when_last_is_absent():
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    ex._tickers["BTC/USDT"] = {
        "symbol": "BTC/USDT", "timestamp": int(NOW.timestamp() * 1000),
        "last": None, "close": None, "bid": 99.0, "ask": 101.0,
    }
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC).price == Decimal("100")


def test_a_crossed_book_is_refused():
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    ex._tickers["BTC/USDT"] = {
        "symbol": "BTC/USDT", "timestamp": int(NOW.timestamp() * 1000),
        "last": None, "close": None, "bid": 101.0, "ask": 99.0,
    }
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None
    assert "crossed" in feed.unpriced["BTC/USDT"]


def test_a_zero_price_is_not_a_price():
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    ex._tickers["BTC/USDT"] = {
        "symbol": "BTC/USDT", "timestamp": int(NOW.timestamp() * 1000),
        "last": 0.0, "close": 0.0, "bid": 0.0, "ask": 0.0,
    }
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None


def test_a_total_ticker_blackout_raises():
    """Every symbol failing is not a per-symbol condition."""
    feed, ex = build(NOW)
    quotes = InMemoryQuoteSource()
    ex._tickers.clear()
    with pytest.raises(FeedUnavailableError):
        feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None and quotes.quote(ETH) is None


def test_ticker_transport_failure_raises():
    feed, ex = build(NOW)
    ex.tickers_error = ConnectionError("okx reset")
    with pytest.raises(FeedUnavailableError):
        feed.refresh_quotes(InMemoryQuoteSource(), AssetClass.CRYPTO, NOW)


def test_refresh_quotes_refuses_a_class_this_feed_does_not_hold():
    """An empty no-op here would read downstream as a quiet market."""
    feed, _ = build(NOW)
    with pytest.raises(FeedUnavailableError):
        feed.refresh_quotes(InMemoryQuoteSource(), AssetClass.EQUITY, NOW)


# ===========================================================================
# extra_provider
# ===========================================================================

def test_extra_provider_publishes_the_bar_book():
    feed, _ = build(NOW)
    extra = feed.extra_provider(AssetClass.CRYPTO, NOW)
    assert set(extra[BARS_KEY]) == {"BTC/USDT", "ETH/USDT"}
    assert extra[FEED_HEALTH_KEY]["source"] == "okx"
    assert extra[FEED_HEALTH_KEY]["stale_bars"] == {}


def test_extra_provider_omits_a_stale_symbol_and_reports_it():
    """One frozen pair must not blind the class, and must not vanish either:
    the omission is published, and a strategy holding that symbol gets a
    MissingDataError out of `bars_for(must_have=True)`."""
    feed, ex = build(NOW)
    ex.set_ohlcv("BTC/USDT", candles(_floor(NOW - dt.timedelta(hours=8), M15), 40))
    feed.invalidate()

    extra = feed.extra_provider(AssetClass.CRYPTO, NOW)
    assert set(extra[BARS_KEY]) == {"ETH/USDT"}
    assert "BTC/USDT" in extra[FEED_HEALTH_KEY]["stale_bars"]
    assert "BTC/USDT" in feed.stale_bars


def test_extra_provider_raises_when_every_symbol_is_stale():
    """An empty bar book reads as a quiet market. It is a blackout."""
    feed, ex = build(NOW)
    old = _floor(NOW - dt.timedelta(hours=8), M15)
    for sym in ("BTC/USDT", "ETH/USDT"):
        ex.set_ohlcv(sym, candles(old, 40))
    feed.invalidate()
    with pytest.raises(FeedStaleError):
        feed.extra_provider(AssetClass.CRYPTO, NOW)


def test_extra_provider_propagates_a_transport_failure():
    """Unreachable is not a per-symbol condition: nothing may be decided."""
    feed, ex = build(NOW)
    ex.ohlcv_error = ConnectionError("okx unreachable")
    feed.invalidate()
    with pytest.raises(FeedUnavailableError):
        feed.extra_provider(AssetClass.CRYPTO, NOW)


def test_extra_provider_refuses_a_class_this_feed_does_not_hold():
    feed, _ = build(NOW)
    with pytest.raises(FeedUnavailableError):
        feed.extra_provider(AssetClass.EQUITY, NOW)


def test_extra_provider_never_returns_an_empty_bar_book():
    feed, ex = build(NOW)
    ex.set_ohlcv("BTC/USDT", [])
    ex.set_ohlcv("ETH/USDT", [])
    feed.invalidate()
    with pytest.raises(FeedStaleError):
        feed.extra_provider(AssetClass.CRYPTO, NOW)


# ===========================================================================
# vol regime — a measurement, not a signal
# ===========================================================================

def test_vol_regime_is_absent_without_enough_history():
    """Absent, never zero. A zero vol reads as an extremely calm market and
    sizes positions accordingly."""
    feed, _ = build(NOW, n=12, history=12, vol_window=20)
    extra = feed.extra_provider(AssetClass.CRYPTO, NOW)
    assert extra[VOL_REGIME_KEY] == {}
    assert extra[BARS_KEY], "bars are still served; only the vol read is withheld"


def test_vol_regime_matches_a_hand_computed_rms():
    """A flat +1 per bar series on a rising base: the realised vol is the RMS of
    the simple returns, computed here independently of the feed."""
    feed, _ = build(NOW, instruments=(BTC,), n=40, history=40,
                    vol_window=5, vol_baseline_window=10)
    bars = feed.bars("BTC/USDT", NOW)
    read = feed.extra_provider(AssetClass.CRYPTO, NOW)[VOL_REGIME_KEY]["BTC/USDT"]

    window = bars[-6:]
    total = Decimal(0)
    for i in range(1, len(window)):
        r = (window[i].close - window[i - 1].close) / window[i - 1].close
        total += r * r
    expected = (total / Decimal(5)).sqrt()
    assert read["realised_vol"] == expected
    assert read["as_of"] == bars[-1].ts
    assert read["window"] == 5


def test_vol_regime_state_is_unknown_without_a_baseline():
    """'I have not measured this' and 'I measured it and it is ordinary' are
    different claims, so the state is never defaulted to normal."""
    feed, _ = build(NOW, instruments=(BTC,), n=8, history=8,
                    vol_window=5, vol_baseline_window=5)
    read = feed.extra_provider(AssetClass.CRYPTO, NOW)[VOL_REGIME_KEY]["BTC/USDT"]
    assert read["baseline_vol"] is not None or read["state"] == "unknown"


def test_vol_regime_flags_an_elevated_window():
    anchor = _floor(NOW, M15)
    rows = candles(anchor, 60)                      # calm: +1 per bar off 100
    for row in rows[-6:]:                           # last five closed bars jump
        row[1] = row[1] * 1.20
        row[2] = row[2] * 1.35
        row[3] = row[3] * 1.05
        row[4] = row[4] * 1.30
    ex = FakeExchange()
    ex.set_ohlcv("BTC/USDT", rows)
    ex.set_ticker("BTC/USDT", last=100.0, at=NOW)
    feed = OkxFeed([BTC], exchange=ex, history=60, vol_window=5,
                   vol_baseline_window=50).bind({AssetClass.CRYPTO: Mode.LIVE})
    read = feed.extra_provider(AssetClass.CRYPTO, NOW)[VOL_REGIME_KEY]["BTC/USDT"]
    assert read["state"] == "elevated"
    assert read["ratio"] > read["elevated_above"]


def test_vol_regime_numbers_are_decimal():
    feed, _ = build(NOW, vol_window=5, vol_baseline_window=20, n=40, history=40)
    for read in feed.extra_provider(AssetClass.CRYPTO, NOW)[VOL_REGIME_KEY].values():
        for field in ("realised_vol", "annualised_vol", "baseline_vol", "ratio"):
            assert read[field] is None or isinstance(read[field], Decimal)


# ===========================================================================
# protocol conformance
# ===========================================================================

def test_okx_feed_satisfies_the_market_feed_protocol():
    from feeds.base import MarketFeed
    feed, _ = build(NOW)
    assert isinstance(feed, MarketFeed)


def test_signature_matches_the_synthetic_feed_the_runner_already_uses():
    import inspect
    import runner

    for name in ("bind", "bars", "refresh_quotes", "extra_provider"):
        ours = list(inspect.signature(getattr(OkxFeed, name)).parameters)
        theirs = list(inspect.signature(getattr(runner.SyntheticBarFeed, name)).parameters)
        assert ours == theirs, f"{name}: {ours} != {theirs}"


# ===========================================================================
# LIVE — public OKX endpoints. Deselect with -m "not network".
# ===========================================================================

@pytest.mark.network
def test_live_okx_serves_fresh_closed_bars():
    pytest.importorskip("ccxt")
    now = dt.datetime.now(UTC)
    feed = OkxFeed([BTC, ETH], history=120).bind({AssetClass.CRYPTO: Mode.LIVE})

    for symbol in ("BTC/USDT", "ETH/USDT"):
        bars = feed.bars(symbol, now)
        assert len(bars) >= 100
        assert bars[-1].ts + M15 <= now, "a forming bar reached a strategy"
        assert bars[-1].close > 0
        for i in range(1, len(bars)):
            assert bars[i].ts - bars[i - 1].ts == M15


@pytest.mark.network
def test_live_forming_candle_close_is_the_ticker_last_and_is_dropped():
    """The observation this whole module is built around, checked against the
    real venue: the trailing candle is not a bar, it is the current price."""
    ccxt = pytest.importorskip("ccxt")
    ex = ccxt.okx({"enableRateLimit": True})
    now = dt.datetime.now(UTC)
    raw = ex.fetch_ohlcv("BTC/USDT", "15m", None, 5)
    last_ts = dt.datetime.fromtimestamp(raw[-1][0] / 1000, UTC)

    feed = OkxFeed([BTC], exchange=ex, history=120).bind({AssetClass.CRYPTO: Mode.LIVE})
    bars = feed.bars("BTC/USDT", now)

    if last_ts + M15 > now:                     # the row is still forming
        assert bars[-1].ts < last_ts
        assert all(b.ts != last_ts for b in bars)


@pytest.mark.network
def test_live_okx_prices_both_pairs_in_one_batch():
    pytest.importorskip("ccxt")
    now = dt.datetime.now(UTC)
    feed = OkxFeed([BTC, ETH], history=60).bind({AssetClass.CRYPTO: Mode.LIVE})
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, now)
    assert feed.unpriced == {}
    assert quotes.quote(BTC).price > 0
    assert quotes.quote(ETH).price > 0


@pytest.mark.network
def test_live_okx_refuses_a_stale_read():
    """Same live data, asked for at a `now` an hour ahead: the answer is a
    refusal, not the last known price."""
    pytest.importorskip("ccxt")
    now = dt.datetime.now(UTC)
    feed = OkxFeed([BTC], history=60).bind({AssetClass.CRYPTO: Mode.LIVE})
    assert feed.bars("BTC/USDT", now)
    with pytest.raises(FeedStaleError):
        feed.bars("BTC/USDT", now + dt.timedelta(hours=1))


@pytest.mark.network
def test_live_okx_publishes_a_vol_read():
    pytest.importorskip("ccxt")
    now = dt.datetime.now(UTC)
    feed = OkxFeed([BTC, ETH], history=200).bind({AssetClass.CRYPTO: Mode.LIVE})
    extra = feed.extra_provider(AssetClass.CRYPTO, now)
    for symbol in ("BTC/USDT", "ETH/USDT"):
        read = extra[VOL_REGIME_KEY][symbol]
        assert read["realised_vol"] > 0
        assert read["state"] in {"quiet", "normal", "elevated"}
