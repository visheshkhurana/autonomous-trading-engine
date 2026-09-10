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
        self._markets = markets or {
            "BTC/USDT": {"spot": True, "active": True},
            "ETH/USDT": {"spot": True, "active": True},
            "SOL/USDT": {"spot": True, "active": True},
        }
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
                        dhan_client=FakeDhan(), okx_exchange=exchange,
                        kraken_exchange=exchange)
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


# ===========================================================================
# 3, second pass. The THIRD route onto SyntheticBarFeed.
#
# The direct route (`bars()` before `bind()`) and the composite route (a live
# class in the mode map) both refuse. What neither of them proves is that a
# bind which SUCCEEDED vouched for the class actually being served.
#
# `bind()` checked the modes it was handed and then set one boolean. It kept
# no record of WHICH classes those were, so `_require_bound()` only ever
# established that some bind had happened somewhere. Three consequences, all
# reachable:
#
#   * `bind({})` vouches for nothing and unlocks everything;
#   * a feed vouched for CRYPTO/paper prices EQUITY, and in the shipped wiring
#     the synthetic instance is SHARED and holds every class's instruments
#     (`select_feeds`'s `synthetic_universe`), so the equity names really are
#     in there waiting to be priced;
#   * a REBIND that is refused leaves the earlier vouch standing — the
#     operator arms the class, the re-bind raises exactly as designed, and the
#     feed carries on serving generated prices to a now-live book.
# ===========================================================================

def test_synthetic_bind_for_one_class_does_not_license_pricing_another():
    """ATTACK: bind for the paper class, then ask it to price the live one.

    This is the shared-instance shape `select_feeds` actually builds: crypto
    falls back to synthetic, equity is armed and routed to Dhan, and the one
    SyntheticBarFeed holds both universes.
    """
    universe = (runner.default_instruments(AssetClass.CRYPTO)
                + runner.default_instruments(AssetClass.EQUITY))
    feed = runner.SyntheticBarFeed(universe)
    feed.bind({AssetClass.CRYPTO: Mode.PAPER})

    quotes = InMemoryQuoteSource()
    with pytest.raises(RuntimeError, match="equity"):
        feed.refresh_quotes(quotes, AssetClass.EQUITY, CRYPTO_NOW)
    assert len(quotes) == 0, "generated equity prices reached the quote source"

    with pytest.raises(RuntimeError, match="equity"):
        feed.extra_provider(AssetClass.EQUITY, CRYPTO_NOW)
    with pytest.raises(RuntimeError, match="equity"):
        feed.bars("RELIANCE", CRYPTO_NOW)


def test_synthetic_bind_with_an_empty_mode_map_vouches_for_nothing():
    """ATTACK: `bind({})`. No mode in the map is LIVE, so the live refusal has
    nothing to refuse, and the feed comes back bound and willing to price
    anything at all."""
    feed = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO))
    with pytest.raises(RuntimeError, match="no modes"):
        feed.bind({})
    quotes = InMemoryQuoteSource()
    with pytest.raises(RuntimeError):
        feed.refresh_quotes(quotes, AssetClass.CRYPTO, CRYPTO_NOW)
    assert len(quotes) == 0


def test_a_refused_rebind_does_not_leave_the_earlier_vouch_standing():
    """ATTACK: bind paper, arm the class, re-bind. The re-bind raises — and
    the composite, already bound, keeps serving the synthetic feed to a class
    the engine now resolves LIVE.

    A bind() that fails must leave the feed UNUSABLE. Anything else means the
    refusal was cosmetic.
    """
    instruments = runner.default_instruments(AssetClass.CRYPTO)
    synthetic = runner.SyntheticBarFeed(instruments)
    composite = runner.CompositeFeed(
        {AssetClass.CRYPTO: synthetic}, {AssetClass.CRYPTO: instruments},
    ).bind({AssetClass.CRYPTO: Mode.PAPER})

    with pytest.raises(RuntimeError, match="live"):
        composite.bind({AssetClass.CRYPTO: Mode.LIVE})

    quotes = InMemoryQuoteSource()
    with pytest.raises(RuntimeError, match="bind"):
        composite.refresh_quotes(quotes, AssetClass.CRYPTO, CRYPTO_NOW)
    assert len(quotes) == 0, "a live class was priced off generated data"
    with pytest.raises(RuntimeError, match="bind"):
        composite.bars("BTC/USDT", CRYPTO_NOW)


def test_synthetic_feed_will_not_invent_a_series_for_a_symbol_it_never_had():
    """`_series_shape()` hashes whatever string it is given, so every symbol
    in the world has a price. Both real feeds refuse an unknown symbol by
    name; this one answered a typo with a confident series."""
    feed = runner.SyntheticBarFeed(
        runner.default_instruments(AssetClass.CRYPTO)
    ).bind({AssetClass.CRYPTO: Mode.PAPER})
    with pytest.raises(RuntimeError, match="not in this feed"):
        feed.bars("RELAINCE", CRYPTO_NOW)


def test_synthetic_mark_is_not_a_way_round_the_live_check():
    """`mark()` hands out a price and had no bind guard — the one public
    price-producing method the live refusal did not cover."""
    feed = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO))
    with pytest.raises(RuntimeError, match="before bind"):
        feed.mark("BTC/USDT", CRYPTO_NOW)


# ===========================================================================
# 2 + 4. A mark that outlives the failure which retired it.
#
# DhanFeed drops every mark it cannot re-establish BEFORE it raises the
# blackout ("a failure that left the last tick's prices sitting in the book
# would be loud and wrong at the same time"). OkxFeed does not: a transport
# failure on fetch_tickers raises straight out and every previous mark stays
# in the quote source.
#
# The tick that failed does not trade — but `adapter.equity()` marks the book
# off that quote source at the TOP of the next tick, before anything refetches,
# and the risk circuit advances on the number it gets. A frozen mark there
# under-reports the drawdown for as long as the venue is unreachable, which is
# precisely when it matters.
# ===========================================================================

def test_okx_does_not_leave_a_mark_standing_through_the_blackout():
    """ATTACK: price the book, then take the venue away. The blackout must
    raise AND leave nothing behind that reads as a current mark."""
    class Flaky(FakeOkx):
        down = False

        def fetch_tickers(self, symbols):
            if self.down:
                raise ConnectionError("okx unreachable")
            return super().fetch_tickers(symbols)

    exchange = Flaky(tickers={"BTC/USDT": {
        "timestamp": int(CRYPTO_NOW.timestamp() * 1000), "last": "70000"}})
    feed = okx_feed(exchange)
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, CRYPTO_NOW)
    assert price_of(quotes, BTC) == Decimal("70000")

    exchange.down = True
    with pytest.raises(FeedUnavailableError):
        feed.refresh_quotes(quotes, AssetClass.CRYPTO,
                            CRYPTO_NOW + dt.timedelta(hours=6))
    assert price_of(quotes, BTC) is None, (
        "the mark the feed can no longer vouch for outlived the failure that "
        "retired it")


def test_dhan_will_not_publish_an_empty_bar_book_for_a_live_class():
    """ATTACK: an all-derivative universe — the defined-risk option book this
    system is built to trade.

    `extra_provider()` skips derivative legs (correctly: they have no equity
    candle history), so with nothing else in the registry it published
    `{bars: {}}`. That is not None and it does carry BARS_KEY, so
    CompositeFeed's guard passes it through, and every bar-driven strategy
    reads it as a market with nothing to say while `refresh_quotes` keeps the
    legs priced and the class keeps trading.
    """
    leg = eq("NIFTY25JUN24000CE", "43211", expiry=dt.date(2025, 6, 26))
    client = FakeDhan(quotes={"43211": {"ltp": "101.50"}})
    feed = DhanFeed([leg], client).bind({AssetClass.EQUITY: Mode.LIVE})

    with pytest.raises(FeedUnavailableError) as excinfo:
        feed.extra_provider(AssetClass.EQUITY, OPEN_NOW)
    assert "bar" in str(excinfo.value).lower()
    assert client.ohlc_calls == [], "an option leg must never be asked for candles"


def test_dhan_still_publishes_a_bar_book_when_it_has_one():
    """The counterweight: the guard above must not fail a mixed universe."""
    ins = [eq(), eq("NIFTY25JUN24000CE", "43211", expiry=dt.date(2025, 6, 26))]
    client = FakeDhan(candles=dhan_rows(OPEN_NOW - INTERVAL, 40))
    extra = DhanFeed(ins, client).bind(
        {AssetClass.EQUITY: Mode.PAPER}).extra_provider(
            AssetClass.EQUITY, OPEN_NOW)
    assert set(extra[BARS_KEY]) == {"RELIANCE"}


# ===========================================================================
# 2, the engine seam. The marks a strategy is handed are the PREVIOUS tick's.
#
# `build_stack` wires the refresh into `extra_provider` and says so in a
# comment: "Marks and bars come from ONE source and are refreshed together. A
# quote from a different clock than the bars is how a strategy's stop is
# evaluated against a price the entry never saw."
#
# The engine's tick does not honour that. `_trade()` snapshots the quotes off
# the adapter FIRST and calls `extra_provider` — the thing that refreshes them
# — second, so every ctx carries this tick's bars beside last tick's marks.
# `IndexOptionTrendStrategy` and `EquityMeanReversionStrategy` both price
# their marketable LIMITs off `ctx.quote()`, and the gate's notional caps and
# exposure sum read the same stale numbers.
#
# The sharp edge is not the lag itself but what it does to a retirement: on
# the tick where the feed decides it can no longer vouch for a mark and
# REMOVES it, the strategy is still handed the removed price, because the
# snapshot was taken before the removal.
# ===========================================================================

def _one_class_engine(feed, quotes, strategy, clock):
    """The smallest engine that exercises the real build_stack seam."""
    import engine as engine_mod
    from adapters.paper import PaperAdapter
    from core.gate import GateConfig, OrderGate
    from core.risk import RiskCircuit
    from core.store import SQLitePositionStore

    class _NoGateState:
        def seen_recently(self, *a, **k): return None
        def record_submission(self, *a, **k): return None
        def prune(self, *a, **k): return None

    store = SQLitePositionStore(":memory:")
    circuit = RiskCircuit(store, Mode.PAPER, clock=clock)
    adapter = PaperAdapter(AssetClass.CRYPTO, quotes, Decimal("100000"),
                           now=clock, sleep=lambda s: None)

    def extra_provider(asset_class, now):
        feed.refresh_quotes(quotes, asset_class, now)
        return feed.extra_provider(asset_class, now)

    wiring = engine_mod.ClassWiring(
        asset_class=AssetClass.CRYPTO, paper_adapter=adapter, risk=circuit,
        strategies=(strategy,), instruments=(BTC,), tick_interval_s=0.01,
        extra_provider=extra_provider)
    return engine_mod.Engine(
        wirings=[wiring], store=store,
        gate=OrderGate(position_store=store, gate_state=_NoGateState(),
                       risk_circuit=circuit,
                       config=GateConfig(requested_mode=Mode.PAPER), env={}),
        config=engine_mod.EngineConfig(requested_mode=Mode.PAPER),
        clock=clock, env={})


def test_the_marks_a_strategy_sees_come_from_this_tick_not_the_last_one():
    """ATTACK: move the price between ticks and read what the strategy got.

    Tick 1 must not hand a strategy an empty quote map while the venue is
    quoting, and tick 2 must not hand it tick 1's price.
    """
    import asyncio

    class Clock:
        def __init__(self, at): self.at = at
        def __call__(self): return self.at

    class MovingOkx(FakeOkx):
        def __init__(self):
            super().__init__(ohlcv=okx_rows(CRYPTO_NOW - INTERVAL, 60))
            self.at = CRYPTO_NOW
            self.last = "70000"

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            self.ohlcv_calls += 1
            return okx_rows(self.at - INTERVAL, 60)

        def fetch_tickers(self, symbols):
            return {s: {"timestamp": int(self.at.timestamp() * 1000),
                        "last": self.last} for s in symbols}

    class Spy:
        id, asset_class = "spy", AssetClass.CRYPTO

        def __init__(self): self.seen = []

        def propose(self, ctx):
            self.seen.append(ctx.quote("BTC/USDT"))
            return []

    exchange = MovingOkx()
    feed = okx_feed(exchange)
    quotes = InMemoryQuoteSource()
    clock = Clock(CRYPTO_NOW)
    spy = Spy()
    eng = _one_class_engine(feed, quotes, spy, clock)

    asyncio.run(eng.tick_once(AssetClass.CRYPTO))
    exchange.last = "80000"
    clock.at = exchange.at = CRYPTO_NOW + dt.timedelta(seconds=5)
    asyncio.run(eng.tick_once(AssetClass.CRYPTO))

    assert spy.seen == [Decimal("70000"), Decimal("80000")], (
        "the strategy was handed the previous tick's marks alongside this "
        f"tick's bars: {spy.seen}")


def test_a_mark_the_feed_retired_this_tick_does_not_reach_the_strategy():
    """The consequence that actually costs money.

    Tick 1: the venue is quoting and the strategy must see that mark — not the
    empty map a pre-refresh snapshot hands it. Tick 2: the ticker arrives with
    no timestamp, the feed refuses it, removes the mark and raises, and no
    strategy may be evaluated at all. What must never appear in `seen` is a
    price on the tick the feed retired it.
    """
    import asyncio

    class Clock:
        def __init__(self, at): self.at = at
        def __call__(self): return self.at

    class GoesQuiet(FakeOkx):
        def __init__(self):
            super().__init__(ohlcv=okx_rows(CRYPTO_NOW - INTERVAL, 60))
            self.at = CRYPTO_NOW
            self.quiet = False

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            self.ohlcv_calls += 1
            return okx_rows(self.at - INTERVAL, 60)

        def fetch_tickers(self, symbols):
            if self.quiet:
                # A ticker with no timestamp: present, unageable, refused.
                return {s: {"last": "70000"} for s in symbols}
            return {s: {"timestamp": int(self.at.timestamp() * 1000),
                        "last": "70000"} for s in symbols}

    class Spy:
        id, asset_class = "spy", AssetClass.CRYPTO

        def __init__(self): self.seen = []

        def propose(self, ctx):
            self.seen.append(ctx.quote("BTC/USDT"))
            return []

    exchange = GoesQuiet()
    feed = okx_feed(exchange)
    quotes = InMemoryQuoteSource()
    clock = Clock(CRYPTO_NOW)
    spy = Spy()
    eng = _one_class_engine(feed, quotes, spy, clock)

    asyncio.run(eng.tick_once(AssetClass.CRYPTO))
    exchange.quiet = True
    clock.at = exchange.at = CRYPTO_NOW + dt.timedelta(seconds=5)
    # The feed empties the book and says so; the tick stops there.
    with pytest.raises(FeedUnavailableError):
        asyncio.run(eng.tick_once(AssetClass.CRYPTO))

    assert price_of(quotes, BTC) is None
    assert spy.seen == [Decimal("70000")], (
        "tick 1 must hand the strategy the mark the venue was actually "
        f"quoting, and tick 2 must not evaluate it at all: {spy.seen}")


# ===========================================================================
# 3, third pass. `Mode` is a str Enum, and every live refusal in the feed
# layer is written `is Mode.LIVE`.
#
# `Mode("live") is Mode.LIVE` is True, so a map built through the enum is
# safe. A map built from a config file, a JSON payload or an API round trip
# carries the bare string `"live"` — which is EQUAL to Mode.LIVE everywhere it
# is printed, logged or compared with `==`, and IS NOT it. Every identity
# check in the feed layer then quietly answers "not live", and the refusal
# that the whole synthetic feed is built around stops firing while every log
# line still reads `crypto->live`.
# ===========================================================================

def test_a_mode_that_is_only_equal_to_live_does_not_get_past_the_refusal():
    """ATTACK: hand bind() the string "live" instead of the enum member."""
    instruments = runner.default_instruments(AssetClass.CRYPTO)
    quotes = InMemoryQuoteSource()

    feed = runner.SyntheticBarFeed(instruments)
    with pytest.raises(RuntimeError, match="Mode"):
        feed.bind({AssetClass.CRYPTO: "live"})
    with pytest.raises(RuntimeError):
        feed.refresh_quotes(quotes, AssetClass.CRYPTO, CRYPTO_NOW)
    assert len(quotes) == 0

    composite = runner.CompositeFeed(
        {AssetClass.CRYPTO: runner.SyntheticBarFeed(instruments)},
        {AssetClass.CRYPTO: instruments})
    with pytest.raises((RuntimeError, ValueError), match="Mode"):
        composite.bind({AssetClass.CRYPTO: "live"})

    with pytest.raises((RuntimeError, ValueError), match="Mode"):
        DhanFeed([eq()], FakeDhan()).bind({AssetClass.EQUITY: "live"})


def test_the_shared_synthetic_select_feeds_builds_refuses_the_armed_class():
    """The same hole, reached the way the shipped wiring actually reaches it.

    `TRADER_FEED=okx` with both classes configured and crypto ARMED: OKX takes
    crypto, equity has no OKX market so it falls back to synthetic — and
    `select_feeds` builds that synthetic over `synthetic_universe`, which is
    every class's instruments, crypto included. The composite binds it with
    `{equity: paper}` alone, correctly, and it comes back holding BTC/USDT and
    a boolean saying it had been checked.
    """
    exchange = FakeOkx(
        ohlcv=okx_rows(CRYPTO_NOW - INTERVAL, 60),
        tickers={"BTC/USDT": {"timestamp": int(CRYPTO_NOW.timestamp() * 1000),
                              "last": "70000"},
                 "ETH/USDT": {"timestamp": int(CRYPTO_NOW.timestamp() * 1000),
                              "last": "2500"},
                 "SOL/USDT": {"timestamp": int(CRYPTO_NOW.timestamp() * 1000),
                              "last": "140"}},
        markets={"BTC/USDT": {"spot": True, "active": True},
                 "ETH/USDT": {"spot": True, "active": True},
                 "SOL/USDT": {"spot": True, "active": True}})
    classes = (AssetClass.CRYPTO, AssetClass.EQUITY)
    config = runner.RunnerConfig(asset_classes=classes, feed=runner.FeedChoice.OKX)
    resolved = {AssetClass.CRYPTO: Mode.LIVE, AssetClass.EQUITY: Mode.PAPER}
    instruments = {a: runner.default_instruments(a) for a in classes}

    routes, _ = runner.select_feeds(config, resolved, instruments,
                                    okx_exchange=exchange)
    runner.CompositeFeed(routes, instruments).bind(resolved)

    synthetic = routes[AssetClass.EQUITY]
    assert isinstance(synthetic, runner.SyntheticBarFeed)
    assert any(i.asset_class is AssetClass.CRYPTO
               for i in synthetic._instruments), (
        "this test is not exercising the shared-universe shape any more")

    quotes = InMemoryQuoteSource()
    with pytest.raises(RuntimeError, match="crypto"):
        synthetic.refresh_quotes(quotes, AssetClass.CRYPTO, CRYPTO_NOW)
    assert len(quotes) == 0, "the ARMED class was priced off generated data"
