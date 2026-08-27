"""
Adversarial tests for the feed layer.

Every test in here was written from the attacker's side first: construct the
call sequence that puts a LIVE book on bad data, prove it works against the
implementation as written, then fix the implementation until the test is the
thing standing in the way.

The five properties under attack:

  1. a bar whose period has not closed must never reach a strategy;
  2. a stale price must FAIL, never be served;
  3. a live-armed class must never be priced by SyntheticBarFeed;
  4. a failed read must raise — never [], never a last known value;
  5. no quote may outlive the moment we could last vouch for it.

Nothing here authenticates, and no test places, modifies or cancels an order.
Both feeds are driven through injected fakes.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

import runner
from adapters.paper import InMemoryQuoteSource
from core.profiles import IST
from core.types import AssetClass, Instrument, Mode
from feeds.base import FeedStaleError
from feeds.dhan_feed import DhanFeed, FeedUnavailableError
from feeds.okx_feed import OkxFeed
from strategies.base import BARS_KEY, Bar

UTC = dt.timezone.utc
INTERVAL = dt.timedelta(minutes=15)

# A Wednesday. 12:00 IST is mid-session — the window in which every equity
# freshness guard is supposed to be armed.
OPEN_NOW = dt.datetime(2025, 6, 11, 12, 0, tzinfo=IST)
CRYPTO_NOW = dt.datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def eq(symbol: str = "RELIANCE", security_id: str = "2885",
       expiry: dt.date | None = None) -> Instrument:
    return Instrument(symbol, AssetClass.EQUITY, security_id,
                      lot_size=1, tick_size=Decimal("0.05"), expiry=expiry)


BTC = Instrument("BTC/USDT", AssetClass.CRYPTO, "BTC/USDT",
                 lot_size=1, tick_size=Decimal("0.1"), qty_precision=6)


class FakeDhan:
    """Only the three read methods DhanFeed uses. No token, no network."""

    def __init__(self, candles=None, quotes=None, ltps=None) -> None:
        self.candles = candles
        self.quotes = {} if quotes is None else quotes
        self.ltps = {} if ltps is None else ltps
        self.ohlc_calls: list[dict] = []

    def get_ohlc(self, symbol, interval, from_date, to_date, exchange="NSE"):
        self.ohlc_calls.append({"symbol": symbol, "interval": interval})
        if isinstance(self.candles, dict):
            return self.candles.get(symbol)
        return self.candles

    def get_fno_quotes(self, security_ids, exchange_segment="NSE_FNO"):
        wanted = set(security_ids)
        return {k: v for k, v in self.quotes.items() if k in wanted}

    def get_fno_ltps(self, security_ids, exchange_segment="NSE_FNO"):
        wanted = set(security_ids)
        return {k: v for k, v in self.ltps.items() if k in wanted}


class FakeOkx:
    """The ccxt surface OkxFeed touches. Public endpoints only."""

    def __init__(self, ohlcv=None, tickers=None, markets=None) -> None:
        self._ohlcv = [] if ohlcv is None else ohlcv
        self._tickers = {} if tickers is None else tickers
        self._markets = markets or {"BTC/USDT": {"spot": True, "active": True}}
        self.ohlcv_calls = 0

    def load_markets(self):
        return self._markets

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.ohlcv_calls += 1
        return self._ohlcv

    def fetch_tickers(self, symbols):
        return {s: self._tickers[s] for s in symbols if s in self._tickers}


def dhan_rows(last_open: dt.datetime, count: int, *, price: float = 100.0,
              interval: dt.timedelta = INTERVAL) -> list[dict]:
    """`count` candles, oldest first, the LAST one opening at `last_open`."""
    return [
        {"date": int((last_open - interval * i).timestamp()),
         "open": price, "high": price + 1.0, "low": price - 1.0,
         "close": price, "volume": 1000}
        for i in range(count - 1, -1, -1)
    ]


def okx_rows(last_open: dt.datetime, count: int, *, price: float = 100.0,
             interval: dt.timedelta = INTERVAL) -> list[list]:
    return [
        [int((last_open - interval * i).timestamp() * 1000),
         price, price + 1.0, price - 1.0, price, 5.0]
        for i in range(count - 1, -1, -1)
    ]


def dhan_feed(client, instruments=None, **kwargs) -> DhanFeed:
    return DhanFeed(instruments or [eq()], client, **kwargs).bind(
        {AssetClass.EQUITY: Mode.PAPER})


def okx_feed(exchange, instruments=None, **kwargs) -> OkxFeed:
    return OkxFeed(instruments or [BTC], exchange=exchange, **kwargs).bind(
        {AssetClass.CRYPTO: Mode.LIVE})


def price_of(quotes: InMemoryQuoteSource, instrument: Instrument):
    quote = quotes.quote(instrument)
    return None if quote is None else quote.price


# ===========================================================================
# 1 + 2. A bar whose period has not closed, and the freshness blind spot
#        that lets it through.
#
# `drop_forming_bar()` pops AT MOST ONE trailing row, and `assert_fresh()`
# compares `now - bars[-1].ts` against a positive limit — so a response whose
# trailing rows run AHEAD of `now` publishes bars that have not closed and
# sails through the staleness check on a NEGATIVE age. A venue clock that
# runs fast, a replayed `now`, or a page of candles stamped in another
# timezone all produce exactly this shape.
# ===========================================================================

def test_dhan_never_publishes_a_bar_whose_period_has_not_closed():
    """ATTACK: the venue opens its newest candle one second before our clock
    reaches the grid boundary. drop_forming_bar() removes that ONE row — and
    the row behind it has not closed either, by one second. It is handed to
    the strategy as a settled bar, and `assert_fresh()` cannot object because
    its age is positive.

    Delete the unclosed-bar trim in DhanFeed.bars() and this fails.
    """
    client = FakeDhan(candles=dhan_rows(OPEN_NOW + dt.timedelta(seconds=1), 30))
    bars = dhan_feed(client).bars("RELIANCE", OPEN_NOW)
    assert bars, "trimming must not silently empty the series"
    assert bars[-1].ts + INTERVAL <= OPEN_NOW, (
        f"published a bar opening {bars[-1].ts} that does not close until "
        f"{bars[-1].ts + INTERVAL}, after now={OPEN_NOW}")


def test_okx_never_publishes_a_bar_whose_period_has_not_closed():
    """Same attack against OKX. ccxt returns the forming candle last, and one
    second of clock disagreement makes the candle behind it forming too."""
    exchange = FakeOkx(ohlcv=okx_rows(CRYPTO_NOW + dt.timedelta(seconds=1), 30))
    bars = okx_feed(exchange).bars("BTC/USDT", CRYPTO_NOW)
    assert bars
    assert bars[-1].ts + INTERVAL <= CRYPTO_NOW, (
        f"published a bar opening {bars[-1].ts} that has not closed at "
        f"now={CRYPTO_NOW}")


def test_a_feed_stamped_entirely_in_the_future_raises_rather_than_passing_fresh():
    """ATTACK: every candle is stamped after `now`, so `assert_fresh()` sees a
    NEGATIVE age and cannot fire. A feed we cannot age must fail, not pass.

    This is the freshness blind spot: `age > limit` is False for every
    negative age, so the louder the timestamp error the safer the feed looks.
    """
    exchange = FakeOkx(ohlcv=okx_rows(CRYPTO_NOW + dt.timedelta(hours=6), 4))
    with pytest.raises((FeedStaleError, FeedUnavailableError)):
        okx_feed(exchange).bars("BTC/USDT", CRYPTO_NOW)

    client = FakeDhan(candles=dhan_rows(OPEN_NOW + dt.timedelta(hours=6), 4))
    with pytest.raises((FeedStaleError, FeedUnavailableError)):
        dhan_feed(client).bars("RELIANCE", OPEN_NOW)


def test_a_venue_running_a_full_bar_ahead_raises_rather_than_reading_as_fresh():
    """The reason trimming alone is not enough, and the nastiest shape here.

    A uniformly shifted feed — every timestamp 45m ahead — is a feed that
    FROZE 45m ago wearing current stamps. Trim the future rows and what is
    left reads as perfectly fresh: `assert_fresh()` is handed timestamps that
    are simply lies, and the tidier the trim the more convincing the lie. So
    a disagreement wider than one bar interval of skew must raise instead.

    Delete the overshoot bound in _drop_unclosed() and this fails: the feed
    quietly serves 45-minute-old prices stamped as current.
    """
    exchange = FakeOkx(ohlcv=okx_rows(CRYPTO_NOW + dt.timedelta(minutes=45), 30))
    with pytest.raises(FeedUnavailableError, match="disagree"):
        okx_feed(exchange).bars("BTC/USDT", CRYPTO_NOW)

    client = FakeDhan(candles=dhan_rows(OPEN_NOW + dt.timedelta(minutes=45), 30))
    with pytest.raises(FeedUnavailableError, match="disagree"):
        dhan_feed(client).bars("RELIANCE", OPEN_NOW)


def test_a_seconds_wide_clock_skew_does_not_take_the_feed_down():
    """The counterweight to the two tests above.

    A candle opens on the venue's grid boundary; if our clock is three seconds
    behind, the trailing TWO rows are 'unclosed' by our reckoning. Trimming
    them is correct. RAISING on them would refuse every read from a healthy
    venue over a three-second skew, and a feed that always fails is worth no
    more than one that never works.
    """
    skewed = CRYPTO_NOW + dt.timedelta(seconds=3)
    exchange = FakeOkx(ohlcv=okx_rows(skewed, 30))
    bars = okx_feed(exchange).bars("BTC/USDT", CRYPTO_NOW)
    assert bars[-1].ts + INTERVAL <= CRYPTO_NOW
    # One extra bar of lag, not an outage.
    assert CRYPTO_NOW - bars[-1].ts < INTERVAL * 3


def test_okx_cache_cannot_outlive_its_own_freshness_window():
    """NOT a fix — a regression guard on a property that already holds.

    A cache is exactly the mechanism by which a dead feed keeps serving a
    live-looking price. Freeze the venue and walk the clock forward: the feed
    must refetch and then RAISE, never replay the cached bars past tolerance.
    """
    exchange = FakeOkx(ohlcv=okx_rows(CRYPTO_NOW - INTERVAL, 30))
    feed = okx_feed(exchange)
    assert feed.bars("BTC/USDT", CRYPTO_NOW)
    with pytest.raises(FeedStaleError):
        feed.bars("BTC/USDT", CRYPTO_NOW + dt.timedelta(minutes=45))
    assert exchange.ohlcv_calls > 1, "a stale cache must be refetched, not replayed"


# ===========================================================================
# 5 + 6. A quote that outlives the moment we could vouch for it.
#
# The Dhan quote path has NO freshness check by design (an LTP carries no
# timestamp) and the bar path is the stated backstop. But a quote that simply
# stops coming back was OMITTED FROM THE UPDATE rather than removed from the
# book, so the previous tick's price stood — indefinitely, during market
# hours, with the engine reading it as a live mark.
# ===========================================================================

def test_dhan_removes_a_mark_it_can_no_longer_vouch_for():
    """ATTACK: two names priced on tick 1; on tick 2 INFY drops out of the
    venue's response entirely. refresh_quotes() returns normally (RELIANCE
    came back), the tick proceeds, and INFY is still carrying tick 1's price.

    Delete the drop in DhanFeed.refresh_quotes() and this fails.
    """
    ins = [eq("RELIANCE", "2885"), eq("INFY", "1594")]
    client = FakeDhan(quotes={"2885": {"ltp": 1400.0}, "1594": {"ltp": 1600.0}})
    feed = dhan_feed(client, ins)
    quotes = InMemoryQuoteSource()

    feed.refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert price_of(quotes, ins[1]) == Decimal("1600.0")

    client.quotes = {"2885": {"ltp": 1425.0}}          # INFY goes silent
    feed.refresh_quotes(quotes, AssetClass.EQUITY,
                        OPEN_NOW + dt.timedelta(minutes=5))

    assert price_of(quotes, ins[0]) == Decimal("1425.0")
    assert price_of(quotes, ins[1]) is None, (
        "INFY is still priced at the previous tick's mark; a book that cannot "
        "be priced must be UNPRICED, not priced off a memory")


def test_dhan_option_leg_mark_does_not_freeze_during_the_session():
    """The market-hours exemption's blast radius.

    `assert_fresh()` runs only while the session is open — but it runs on the
    BAR path, and extra_provider() deliberately fetches no bars for derivative
    legs. An option leg therefore has no freshness check anywhere, so a frozen
    LTP is the only number pricing a defined-risk structure, mid-session.

    Removing the drop is enough to make this fail.
    """
    leg = eq("NIFTY25JUN24000CE", "43211", expiry=dt.date(2025, 6, 26))
    cash = eq("RELIANCE", "2885")
    client = FakeDhan(quotes={"2885": {"ltp": 1400.0}, "43211": {"ltp": 120.0}})
    feed = dhan_feed(client, [cash, leg])
    quotes = InMemoryQuoteSource()

    feed.refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert price_of(quotes, leg) == Decimal("120.0")

    client.quotes = {"2885": {"ltp": 1400.0}}          # the leg stops quoting
    feed.refresh_quotes(quotes, AssetClass.EQUITY,
                        OPEN_NOW + dt.timedelta(minutes=5))
    assert price_of(quotes, leg) is None, (
        "an option leg is still marked at a frozen LTP during market hours, "
        "and nothing on the bar path will ever check it")
    assert client.ohlc_calls == [], "no bar was fetched to paper over the gap"


def test_dhan_zero_ltp_clears_the_previous_mark():
    """A zero is not a price, and it must not leave the old one standing
    either — the two outcomes are indistinguishable downstream otherwise."""
    ins = [eq("RELIANCE", "2885"), eq("INFY", "1594")]
    client = FakeDhan(quotes={"2885": {"ltp": 1400.0}, "1594": {"ltp": 1600.0}})
    feed = dhan_feed(client, ins)
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)

    client.quotes = {"2885": {"ltp": 1400.0}, "1594": {"ltp": 0}}
    feed.refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert price_of(quotes, ins[1]) is None


def test_dhan_blackout_empties_the_book_before_it_raises():
    """A total quote blackout raises — and must not leave the last tick's
    prices sitting in the book for whatever reads it next."""
    ins = [eq("RELIANCE", "2885"), eq("INFY", "1594")]
    client = FakeDhan(quotes={"2885": {"ltp": 1400.0}, "1594": {"ltp": 1600.0}})
    feed = dhan_feed(client, ins)
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert len(quotes) == 2

    client.quotes, client.ltps = {}, {}
    with pytest.raises(FeedUnavailableError):
        feed.refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert len(quotes) == 0, "a blackout left the previous marks standing"


def test_okx_partial_quote_failure_removes_the_mark():
    """NOT a fix — the behaviour DhanFeed was measured against. Kept so a
    refactor cannot quietly regress OKX to Dhan's old shape."""
    eth = Instrument("ETH/USDT", AssetClass.CRYPTO, "ETH/USDT",
                     lot_size=1, tick_size=Decimal("0.01"), qty_precision=5)
    stamp = int(CRYPTO_NOW.timestamp() * 1000)
    exchange = FakeOkx(
        tickers={"BTC/USDT": {"timestamp": stamp, "last": "70000"},
                 "ETH/USDT": {"timestamp": stamp, "last": "2500"}},
        markets={"BTC/USDT": {"spot": True, "active": True},
                 "ETH/USDT": {"spot": True, "active": True}})
    feed = okx_feed(exchange, [BTC, eth])
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, CRYPTO_NOW)
    assert len(quotes) == 2

    # ETH's ticker freezes an hour behind the tolerance.
    exchange._tickers["ETH/USDT"] = {
        "timestamp": int((CRYPTO_NOW - dt.timedelta(hours=1)).timestamp() * 1000),
        "last": "2500"}
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, CRYPTO_NOW)
    assert price_of(quotes, eth) is None


# ===========================================================================
# 3. A live-armed class on generated prices.
#
# select_feeds() + CompositeFeed.bind() were swept exhaustively and hold (see
# the report). What did NOT hold: two of SyntheticBarFeed's three MarketFeed
# methods served without bind() ever having run, so the live refusal was only
# ever enforced by the caller's discipline in going through CompositeFeed.
# ===========================================================================

def test_synthetic_feed_will_not_serve_bars_before_the_live_check():
    """ATTACK: skip bind() and ask for bars anyway. bind() is the ONLY place
    the live refusal lives, so a caller that never binds is a caller that
    never asked."""
    feed = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO))
    with pytest.raises(RuntimeError, match="before bind"):
        feed.bars("BTC/USDT", CRYPTO_NOW)


def test_synthetic_feed_will_not_price_a_book_before_the_live_check():
    """The same hole on the quote path, which is the one that actually reaches
    an adapter: generated prices land in the quote source a live adapter would
    size against."""
    feed = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO))
    quotes = InMemoryQuoteSource()
    with pytest.raises(RuntimeError, match="before bind"):
        feed.refresh_quotes(quotes, AssetClass.CRYPTO, CRYPTO_NOW)
    assert len(quotes) == 0


def test_no_configuration_puts_a_live_class_on_generated_prices():
    """NOT a fix — the sweep, kept as a regression guard.

    Every (TRADER_FEED x asset classes x arming) combination select_feeds()
    accepts is bound and inspected: no LIVE class may be routed to
    SyntheticBarFeed.
    """
    exchange = FakeOkx()
    combos = 0
    for choice in runner.FeedChoice:
        for classes in ((AssetClass.CRYPTO,), (AssetClass.EQUITY,),
                        (AssetClass.CRYPTO, AssetClass.EQUITY)):
            for live in (frozenset(), frozenset({AssetClass.CRYPTO}),
                         frozenset({AssetClass.EQUITY}), frozenset(classes)):
                config = runner.RunnerConfig(
                    requested_mode=Mode.LIVE, asset_classes=classes,
                    feed=choice,
                    dhan_security_ids={"RELIANCE": "2885", "INFY": "1594"})
                resolved = {a: (Mode.LIVE if a in live else Mode.PAPER)
                            for a in classes}
                instruments = {a: runner.default_instruments(a) for a in classes}
                try:
                    routes, _ = runner.select_feeds(
                        config, resolved, instruments,
                        dhan_client=FakeDhan(), okx_exchange=exchange)
                    runner.CompositeFeed(routes, instruments).bind(resolved)
                except Exception:
                    continue          # refused to boot: the correct outcome
                combos += 1
                for asset_class, feed in routes.items():
                    assert not (isinstance(feed, runner.SyntheticBarFeed)
                                and resolved[asset_class] is Mode.LIVE), (
                        f"{asset_class.value} is LIVE on generated prices with "
                        f"TRADER_FEED={choice.value}")
    assert combos > 0, "the sweep refused everything; it is proving nothing"


# ===========================================================================
# 4. A failed read that reads downstream as "no signal".
# ===========================================================================

def test_composite_refuses_an_extra_with_no_bar_book():
    """ATTACK: route a class to a feed that cannot serve it. DhanFeed answers
    `extra_provider(CRYPTO)` with `{}` — not None, so the composite's only
    check passes it through, and a strategy reads a context with no bar book
    as a quiet market rather than as the wiring failure it is.
    """
    class BlankFeed:
        def bind(self, resolved_modes):
            return self

        def bars(self, symbol, now):
            raise AssertionError("not reached")

        def refresh_quotes(self, quotes, asset_class, now):
            return None

        def extra_provider(self, asset_class, now):
            return {}

    composite = runner.CompositeFeed(
        {AssetClass.CRYPTO: BlankFeed()},
        {AssetClass.CRYPTO: runner.default_instruments(AssetClass.CRYPTO)},
    ).bind({AssetClass.CRYPTO: Mode.PAPER})

    with pytest.raises(Exception) as excinfo:
        composite.extra_provider(AssetClass.CRYPTO, CRYPTO_NOW)
    assert BARS_KEY in str(excinfo.value)


def test_composite_still_passes_a_real_bar_book_through():
    """The counterweight: the check above must not reject a healthy feed."""
    exchange = FakeOkx(
        ohlcv=okx_rows(CRYPTO_NOW - INTERVAL, 60),
        tickers={"BTC/USDT": {"timestamp": int(CRYPTO_NOW.timestamp() * 1000),
                              "last": "70000"}})
    composite = runner.CompositeFeed(
        {AssetClass.CRYPTO: OkxFeed([BTC], exchange=exchange)},
        {AssetClass.CRYPTO: (BTC,)},
    ).bind({AssetClass.CRYPTO: Mode.LIVE})
    extra = composite.extra_provider(AssetClass.CRYPTO, CRYPTO_NOW)
    assert isinstance(extra[BARS_KEY]["BTC/USDT"][0], Bar)


# ===========================================================================
# 7. Decimal precision.
#
# NOT a fix — no float was found anywhere on either price path. Pinned so a
# refactor to float() fails here instead of in a live book's fill price.
# ===========================================================================

def test_no_price_loses_precision_on_the_way_to_a_strategy():
    exact = "1400.123456789012345678"

    client = FakeDhan(candles=dhan_rows(OPEN_NOW - INTERVAL, 30),
                      quotes={"2885": {"ltp": exact}})
    quotes = InMemoryQuoteSource()
    dhan_feed(client).refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert price_of(quotes, eq()) == Decimal(exact)

    rows = dhan_rows(OPEN_NOW - INTERVAL, 30)
    for row in rows:
        row["open"] = row["high"] = row["low"] = row["close"] = exact
    bars = dhan_feed(FakeDhan(candles=rows)).bars("RELIANCE", OPEN_NOW)
    assert bars[-1].close == Decimal(exact)

    okx_raw = okx_rows(CRYPTO_NOW - INTERVAL, 30)
    for row in okx_raw:
        row[1] = row[2] = row[3] = row[4] = exact
    okx_bars = okx_feed(FakeOkx(ohlcv=okx_raw)).bars("BTC/USDT", CRYPTO_NOW)
    assert okx_bars[-1].close == Decimal(exact)
