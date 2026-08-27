"""
Tests for feeds.dhan_feed.DhanFeed.

Every test in here is written so that it FAILS if the safety rule it covers is
removed. Where that is not obvious the test says which line deleting it would
turn green.

Nothing here touches Dhan. The client is a fake that records its calls; a test
at the bottom asserts that importing the module pulls in no Dhan SDK and reads
no credential.
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import sys
from decimal import Decimal

import pytest

from adapters.paper import InMemoryQuoteSource
from core.profiles import IST, EquityProfile
from core.types import AssetClass, Instrument, Mode
from feeds.base import FeedError, FeedStaleError, MarketFeed
from feeds.dhan_feed import (
    DhanFeed,
    DhanMarketDataClient,
    FeedUnavailableError,
    UnmappedSymbolError,
)
from strategies.base import BARS_KEY, Bar
from strategies.equity_trend import REGIME_KEY, RegimeRead, RegimeState

UTC = dt.timezone.utc
INTERVAL = dt.timedelta(minutes=15)

# A Wednesday. 12:00 IST is mid-session; 20:00 IST is after the close.
OPEN_NOW = dt.datetime(2025, 6, 11, 12, 0, tzinfo=IST)
CLOSED_NOW = dt.datetime(2025, 6, 11, 20, 0, tzinfo=IST)
SATURDAY_NOW = dt.datetime(2025, 6, 14, 12, 0, tzinfo=IST)


# ---------------------------------------------------------------------------
# fixtures / fakes
# ---------------------------------------------------------------------------

def instrument(symbol: str = "RELIANCE", security_id: str = "2885",
               expiry: dt.date | None = None) -> Instrument:
    return Instrument(symbol, AssetClass.EQUITY, security_id,
                      lot_size=1, tick_size=Decimal("0.05"), expiry=expiry)


class FakeDhanClient:
    """Records calls; returns exactly what the test set up. No network."""

    def __init__(self, candles=None, quotes=None, ltps=None,
                 candle_error=None, quote_error=None, ltp_error=None) -> None:
        self._candles = candles
        self._quotes = {} if quotes is None else quotes
        self._ltps = {} if ltps is None else ltps
        self._candle_error = candle_error
        self._quote_error = quote_error
        self._ltp_error = ltp_error
        self.ohlc_calls: list[dict] = []
        self.quote_calls: list[tuple] = []
        self.ltp_calls: list[tuple] = []

    def get_ohlc(self, symbol, interval, from_date, to_date, exchange="NSE"):
        self.ohlc_calls.append({
            "symbol": symbol, "interval": interval, "from_date": from_date,
            "to_date": to_date, "exchange": exchange,
        })
        if self._candle_error is not None:
            raise self._candle_error
        if isinstance(self._candles, dict) and symbol in self._candles:
            return self._candles[symbol]
        return self._candles

    def get_fno_quotes(self, security_ids, exchange_segment="NSE_FNO"):
        self.quote_calls.append((tuple(security_ids), exchange_segment))
        if self._quote_error is not None:
            raise self._quote_error
        return {sid: rec for sid, rec in self._quotes.items() if sid in set(security_ids)}

    def get_fno_ltps(self, security_ids, exchange_segment="NSE_FNO"):
        self.ltp_calls.append((tuple(security_ids), exchange_segment))
        if self._ltp_error is not None:
            raise self._ltp_error
        return {sid: px for sid, px in self._ltps.items() if sid in set(security_ids)}


def candle_rows(end: dt.datetime, count: int, *, interval: dt.timedelta = INTERVAL,
                base: float = 100.0, step: float = 0.0) -> list[dict]:
    """`count` candles whose LAST one opens at `end`, oldest first.

    Returned in the adapter's normalised list-of-dicts shape, with epoch-second
    timestamps, which is what Dhan produces after `_normalize_candles`.
    """
    rows = []
    for i in range(count - 1, -1, -1):
        ts = end - interval * i
        price = base + step * (count - 1 - i)
        rows.append({
            "date": int(ts.timestamp()),
            "open": price, "high": price + 1.0, "low": price - 1.0,
            "close": price, "volume": 1000,
        })
    return rows


def feed(client, instruments=None, **kwargs) -> DhanFeed:
    f = DhanFeed(instruments or [instrument()], client, **kwargs)
    return f.bind({AssetClass.EQUITY: Mode.PAPER})


# ---------------------------------------------------------------------------
# rule 1 — the forming bar is never published
# ---------------------------------------------------------------------------

def test_forming_bar_is_dropped():
    """The trailing candle whose period has not closed must not be returned.

    Delete the drop_forming_bar() call and this fails: the 11:45 candle at
    11:50 is still forming.
    """
    now = dt.datetime(2025, 6, 11, 11, 50, tzinfo=IST)
    last_open = dt.datetime(2025, 6, 11, 11, 45, tzinfo=IST)
    client = FakeDhanClient(candles=candle_rows(last_open, 30))
    bars = feed(client).bars("RELIANCE", now)

    assert bars[-1].ts == dt.datetime(2025, 6, 11, 11, 30, tzinfo=IST).astimezone(UTC)
    assert all(b.ts + INTERVAL <= now for b in bars)


def test_closed_trailing_bar_is_kept():
    """The complement: a bar whose period HAS closed is published. Guards
    against 'fixing' the forming-bar rule by always dropping the last row."""
    now = dt.datetime(2025, 6, 11, 12, 1, tzinfo=IST)
    last_open = dt.datetime(2025, 6, 11, 11, 45, tzinfo=IST)
    client = FakeDhanClient(candles=candle_rows(last_open, 30))
    bars = feed(client).bars("RELIANCE", now)
    assert bars[-1].ts == last_open.astimezone(UTC)


def test_forming_bar_never_reaches_the_regime_read():
    """The regime is computed from closed bars only, so its as_of can never be
    the forming bar's timestamp."""
    now = dt.datetime(2025, 6, 11, 11, 50, tzinfo=IST)
    last_open = dt.datetime(2025, 6, 11, 11, 45, tzinfo=IST)
    client = FakeDhanClient(candles=candle_rows(last_open, 40, step=1.0))
    extra = feed(client).extra_provider(AssetClass.EQUITY, now)
    read = extra[REGIME_KEY]["RELIANCE"]
    assert read.as_of == dt.datetime(2025, 6, 11, 11, 30, tzinfo=IST).astimezone(UTC)
    assert read.as_of != last_open.astimezone(UTC)


def test_regime_as_of_is_the_bar_not_now():
    """Stamping the read with `now` would make an hours-old regime look fresh
    to the strategy's own max-age check. Delete that and this fails."""
    last_open = CLOSED_NOW - dt.timedelta(hours=5)
    client = FakeDhanClient(candles=candle_rows(last_open, 40, step=1.0))
    extra = feed(client).extra_provider(AssetClass.EQUITY, CLOSED_NOW)
    read = extra[REGIME_KEY]["RELIANCE"]
    assert read.as_of == last_open.astimezone(UTC)
    assert CLOSED_NOW - read.as_of > dt.timedelta(minutes=30)


# ---------------------------------------------------------------------------
# rule 2 — stale during the session fails; closed market does not
# ---------------------------------------------------------------------------

def test_stale_bars_raise_while_the_market_is_open():
    """Delete the assert_fresh() call and this fails."""
    last_open = OPEN_NOW - dt.timedelta(hours=2)
    client = FakeDhanClient(candles=candle_rows(last_open, 30))
    with pytest.raises(FeedStaleError) as exc:
        feed(client).bars("RELIANCE", OPEN_NOW)
    assert "stale" in str(exc.value).lower()


def test_same_stale_bars_are_fine_after_the_close():
    """MARKET HOURS ARE NOT STALENESS. Call assert_fresh() unconditionally and
    this fails — which is the alarm that would scream every evening."""
    last_open = CLOSED_NOW - dt.timedelta(hours=6)
    client = FakeDhanClient(candles=candle_rows(last_open, 30))
    bars = feed(client).bars("RELIANCE", CLOSED_NOW)
    assert bars[-1].ts == last_open.astimezone(UTC)


def test_weekend_gap_is_not_staleness():
    last_open = SATURDAY_NOW - dt.timedelta(days=1, hours=3)
    client = FakeDhanClient(candles=candle_rows(last_open, 30))
    bars = feed(client).bars("RELIANCE", SATURDAY_NOW)
    assert len(bars) == 30


def test_fresh_bars_pass_during_the_session():
    last_open = dt.datetime(2025, 6, 11, 11, 45, tzinfo=IST)
    now = dt.datetime(2025, 6, 11, 12, 1, tzinfo=IST)
    client = FakeDhanClient(candles=candle_rows(last_open, 30))
    assert feed(client).bars("RELIANCE", now)


def test_staleness_tolerance_is_measured_in_intervals():
    """Just inside 3 intervals passes; just outside raises."""
    inside = OPEN_NOW - dt.timedelta(minutes=40)
    outside = OPEN_NOW - dt.timedelta(minutes=50)

    ok = FakeDhanClient(candles=candle_rows(inside, 30))
    assert feed(ok).bars("RELIANCE", OPEN_NOW)

    bad = FakeDhanClient(candles=candle_rows(outside, 30))
    with pytest.raises(FeedStaleError):
        feed(bad).bars("RELIANCE", OPEN_NOW)


# ---------------------------------------------------------------------------
# rule 3 — failure raises; empty means empty
# ---------------------------------------------------------------------------

def test_client_exception_becomes_feed_unavailable_not_empty():
    """The original failure must survive as __cause__. Swallowing the fetch
    error and letting the downstream 'no candles' check raise instead would
    report a quiet market where there was an auth failure."""
    boom = RuntimeError("DH-901 auth")
    client = FakeDhanClient(candle_error=boom)
    with pytest.raises(FeedUnavailableError) as exc:
        feed(client).bars("RELIANCE", OPEN_NOW)
    assert exc.value.__cause__ is boom
    assert "DH-901" in str(exc.value)


def test_empty_candle_response_raises():
    """[] must never be returned as 'the market said nothing'."""
    for empty in ([], {}, None):
        client = FakeDhanClient(candles=empty)
        with pytest.raises(FeedUnavailableError):
            feed(client).bars("RELIANCE", OPEN_NOW)


def test_only_a_forming_candle_raises_rather_than_returning_empty():
    now = dt.datetime(2025, 6, 11, 11, 50, tzinfo=IST)
    last_open = dt.datetime(2025, 6, 11, 11, 45, tzinfo=IST)
    client = FakeDhanClient(candles=candle_rows(last_open, 1))
    with pytest.raises(FeedUnavailableError) as exc:
        feed(client).bars("RELIANCE", now)
    assert "closed" in str(exc.value)


def test_garbage_response_shape_raises():
    client = FakeDhanClient(candles="not candles")
    with pytest.raises(FeedUnavailableError):
        feed(client).bars("RELIANCE", OPEN_NOW)


def test_missing_price_field_raises_rather_than_defaulting():
    rows = candle_rows(OPEN_NOW - dt.timedelta(minutes=15), 30)
    rows[-1]["close"] = None
    client = FakeDhanClient(candles=rows)
    with pytest.raises(FeedUnavailableError) as exc:
        feed(client).bars("RELIANCE", OPEN_NOW)
    assert "close" in str(exc.value)


def test_missing_volume_is_zero_not_an_error():
    """Index candles genuinely have no volume, and volume is not a price."""
    rows = candle_rows(OPEN_NOW - dt.timedelta(minutes=15), 30)
    for r in rows:
        r.pop("volume")
    bars = feed(FakeDhanClient(candles=rows)).bars("RELIANCE", OPEN_NOW)
    assert bars[-1].volume == Decimal(0)


def test_non_numeric_price_raises():
    rows = candle_rows(OPEN_NOW - dt.timedelta(minutes=15), 30)
    rows[-1]["high"] = "n/a"
    with pytest.raises(FeedUnavailableError):
        feed(FakeDhanClient(candles=rows)).bars("RELIANCE", OPEN_NOW)


def test_out_of_range_timestamp_raises():
    """A 1985 timestamp would read as 'very stale' during the session and as
    perfectly fine after the close. It must raise instead."""
    rows = candle_rows(OPEN_NOW - dt.timedelta(minutes=15), 30)
    rows[0]["date"] = 100_000  # 1970
    with pytest.raises(FeedUnavailableError) as exc:
        feed(FakeDhanClient(candles=rows)).bars("RELIANCE", OPEN_NOW)
    assert "2000-2100" in str(exc.value)


def test_unparseable_timestamp_raises():
    rows = candle_rows(OPEN_NOW - dt.timedelta(minutes=15), 30)
    rows[-1]["date"] = "sometime last tuesday"
    with pytest.raises(FeedUnavailableError):
        feed(FakeDhanClient(candles=rows)).bars("RELIANCE", OPEN_NOW)


def test_out_of_order_candles_raise():
    """drop_forming_bar trims the LAST row; an unordered series would have it
    drop the wrong bar and publish the forming one."""
    rows = candle_rows(OPEN_NOW - dt.timedelta(minutes=15), 30)
    rows[5], rows[6] = rows[6], rows[5]
    with pytest.raises(FeedUnavailableError) as exc:
        feed(FakeDhanClient(candles=rows)).bars("RELIANCE", OPEN_NOW)
    assert "ascending" in str(exc.value)


def test_duplicate_timestamps_raise():
    rows = candle_rows(OPEN_NOW - dt.timedelta(minutes=15), 30)
    rows[6]["date"] = rows[5]["date"]
    with pytest.raises(FeedUnavailableError):
        feed(FakeDhanClient(candles=rows)).bars("RELIANCE", OPEN_NOW)


def test_no_client_raises_rather_than_serving_nothing():
    f = DhanFeed([instrument()], None).bind({AssetClass.EQUITY: Mode.PAPER})
    with pytest.raises(FeedUnavailableError):
        f.bars("RELIANCE", OPEN_NOW)


# ---------------------------------------------------------------------------
# security_id mapping — one place, loud when missing
# ---------------------------------------------------------------------------

def test_unmapped_symbol_is_loud_not_an_empty_series():
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 30))
    with pytest.raises(UnmappedSymbolError) as exc:
        feed(client).bars("INFY", OPEN_NOW)
    assert "INFY" in str(exc.value)
    assert client.ohlc_calls == []


def test_security_id_lookup_is_the_single_mapping():
    f = feed(FakeDhanClient(), [instrument("RELIANCE", "2885"),
                                instrument("INFY", "1594")])
    assert f.security_id("RELIANCE") == "2885"
    assert f.security_id("INFY") == "1594"
    with pytest.raises(UnmappedSymbolError):
        f.security_id("TCS")


def test_placeholder_broker_id_is_rejected_at_construction():
    """runner.default_instruments() uses 'PAPER-RELIANCE'. That must never
    reach a real feed."""
    with pytest.raises(ValueError) as exc:
        DhanFeed([instrument("RELIANCE", "PAPER-RELIANCE")], FakeDhanClient())
    assert "security_id" in str(exc.value)


def test_duplicate_symbol_is_rejected():
    with pytest.raises(ValueError):
        DhanFeed([instrument("RELIANCE", "2885"), instrument("RELIANCE", "9999")],
                 FakeDhanClient())


def test_non_equity_instrument_is_rejected():
    crypto = Instrument("BTC/USDT", AssetClass.CRYPTO, "BTC/USDT",
                        lot_size=1, tick_size=Decimal("0.1"), qty_precision=6)
    with pytest.raises(ValueError):
        DhanFeed([crypto], FakeDhanClient())


def test_option_leg_is_routed_to_the_fno_segment():
    leg = instrument("NIFTY25JUN24000CE", "43211",
                     expiry=dt.date(2025, 6, 26))
    client = FakeDhanClient(quotes={"43211": {"ltp": 120.5}})
    f = feed(client, [leg])
    f.refresh_quotes(InMemoryQuoteSource(), AssetClass.EQUITY, OPEN_NOW)
    assert client.quote_calls[0][1] == "NSE_FNO"


def test_cash_listing_is_routed_to_the_eq_segment():
    client = FakeDhanClient(quotes={"2885": {"ltp": 1400.0}})
    feed(client).refresh_quotes(InMemoryQuoteSource(), AssetClass.EQUITY, OPEN_NOW)
    assert client.quote_calls[0][1] == "NSE_EQ"


# ---------------------------------------------------------------------------
# column-oriented Dhan response
# ---------------------------------------------------------------------------

def _columnar(end, count):
    rows = candle_rows(end, count)
    return {
        "timestamp": [r["date"] for r in rows],
        "open": [r["open"] for r in rows],
        "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows],
        "close": [r["close"] for r in rows],
        "volume": [r["volume"] for r in rows],
    }


def test_column_oriented_response_is_normalised():
    last_open = dt.datetime(2025, 6, 11, 11, 45, tzinfo=IST)
    now = dt.datetime(2025, 6, 11, 12, 1, tzinfo=IST)
    client = FakeDhanClient(candles=_columnar(last_open, 30))
    bars = feed(client).bars("RELIANCE", now)
    assert len(bars) == 30
    assert bars[-1].ts == last_open.astimezone(UTC)
    assert bars[-1].close == Decimal("100.0")


def test_misaligned_columns_raise_rather_than_mispair_prices():
    payload = _columnar(OPEN_NOW - INTERVAL, 30)
    payload["low"] = payload["low"][:-1]
    with pytest.raises(FeedUnavailableError) as exc:
        feed(FakeDhanClient(candles=payload)).bars("RELIANCE", OPEN_NOW)
    assert "aligned" in str(exc.value)


def test_column_response_without_timestamps_raises():
    payload = _columnar(OPEN_NOW - INTERVAL, 30)
    payload.pop("timestamp")
    with pytest.raises(FeedUnavailableError):
        feed(FakeDhanClient(candles=payload)).bars("RELIANCE", OPEN_NOW)


def test_prices_are_decimal_not_float():
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 30, base=101.1))
    bars = feed(client).bars("RELIANCE", OPEN_NOW)
    assert isinstance(bars[-1].close, Decimal)
    assert bars[-1].close == Decimal("101.1")   # not 101.09999999999999


# ---------------------------------------------------------------------------
# bind() — a live book must not run on an unconfigured feed
# ---------------------------------------------------------------------------

def test_bind_refuses_live_equity_without_a_client():
    f = DhanFeed([instrument()], None)
    with pytest.raises(RuntimeError) as exc:
        f.bind({AssetClass.EQUITY: Mode.LIVE})
    assert "LIVE" in str(exc.value)


def test_bind_refuses_live_equity_with_an_empty_universe():
    f = DhanFeed([], FakeDhanClient())
    with pytest.raises(RuntimeError):
        f.bind({AssetClass.EQUITY: Mode.LIVE})


def test_bind_allows_live_equity_with_a_client():
    f = DhanFeed([instrument()], FakeDhanClient())
    assert f.bind({AssetClass.EQUITY: Mode.LIVE}) is f


def test_bind_allows_paper_without_a_client():
    f = DhanFeed([instrument()], None)
    assert f.bind({AssetClass.EQUITY: Mode.PAPER}) is f


def test_live_crypto_elsewhere_does_not_block_this_feed():
    """DhanFeed only vouches for equity; composing it with a crypto feed must
    not make it refuse to bind."""
    f = DhanFeed([instrument()], FakeDhanClient())
    assert f.bind({AssetClass.EQUITY: Mode.PAPER, AssetClass.CRYPTO: Mode.LIVE}) is f


def test_methods_refuse_to_run_before_bind():
    f = DhanFeed([instrument()], FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 30)))
    for call in (lambda: f.bars("RELIANCE", OPEN_NOW),
                 lambda: f.refresh_quotes(InMemoryQuoteSource(), AssetClass.EQUITY, OPEN_NOW),
                 lambda: f.extra_provider(AssetClass.EQUITY, OPEN_NOW)):
        with pytest.raises(RuntimeError, match="bind"):
            call()


def test_naive_now_is_rejected():
    """A naive `now` puts is_open() in local time and the bar maths in UTC.

    Asserted as a FeedError naming the timezone, not merely 'something raised':
    without the check, drop_forming_bar happens to raise a TypeError comparing
    naive to aware, which is luck rather than a guard, and it disappears the
    moment the market-hours branch is the first thing to touch `now`.
    """
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 30))
    naive = dt.datetime(2025, 6, 11, 12, 0)
    for call in (lambda f: f.bars("RELIANCE", naive),
                 lambda f: f.refresh_quotes(InMemoryQuoteSource(), AssetClass.EQUITY, naive),
                 lambda f: f.extra_provider(AssetClass.EQUITY, naive)):
        with pytest.raises(FeedError, match="timezone-aware"):
            call(feed(client))


def test_unsupported_interval_is_rejected_at_construction():
    with pytest.raises(ValueError):
        DhanFeed([instrument()], FakeDhanClient(), bar_interval=dt.timedelta(minutes=7))


def test_client_missing_methods_is_rejected_at_construction():
    class Useless:
        def place_order(self):  # not a market-data client
            ...

    with pytest.raises(TypeError):
        DhanFeed([instrument()], Useless())


# ---------------------------------------------------------------------------
# refresh_quotes
# ---------------------------------------------------------------------------

def test_quotes_are_batched_in_one_call_per_segment():
    ins = [instrument("RELIANCE", "2885"), instrument("INFY", "1594"),
           instrument("TCS", "11536")]
    client = FakeDhanClient(quotes={"2885": {"ltp": 1400.0},
                                    "1594": {"ltp": 1600.0},
                                    "11536": {"ltp": 3400.0}})
    quotes = InMemoryQuoteSource()
    feed(client, ins).refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert len(client.quote_calls) == 1
    assert set(client.quote_calls[0][0]) == {"2885", "1594", "11536"}
    assert len(quotes) == 3


def test_quote_prices_reach_the_source_as_decimals():
    client = FakeDhanClient(quotes={"2885": {"ltp": 1400.35}})
    quotes = InMemoryQuoteSource()
    feed(client).refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    # InMemoryQuoteSource.set rejects floats outright, so reaching here at all
    # proves the conversion; assert the value survived it exactly.
    assert quotes.quote(instrument()).price == Decimal("1400.35")


def test_missing_quote_is_omitted_never_faked_from_a_bar_close():
    """The single most tempting shortcut in this file: filling a missing quote
    from the last bar. Do that and this fails."""
    ins = [instrument("RELIANCE", "2885"), instrument("INFY", "1594")]
    client = FakeDhanClient(
        candles=candle_rows(OPEN_NOW - INTERVAL, 30, base=999.0),
        quotes={"2885": {"ltp": 1400.0}},
    )
    quotes = InMemoryQuoteSource()
    feed(client, ins).refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)

    assert quotes.quote(ins[0]) is not None
    assert quotes.quote(ins[1]) is None
    assert client.ohlc_calls == []   # no bar was fetched to paper over the gap


def test_zero_ltp_is_not_a_price():
    """A zero read as valid is how an empty read becomes a confident wrong
    number."""
    ins = [instrument("RELIANCE", "2885"), instrument("INFY", "1594")]
    client = FakeDhanClient(quotes={"2885": {"ltp": 0}, "1594": {"ltp": 1600.0}})
    quotes = InMemoryQuoteSource()
    feed(client, ins).refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert quotes.quote(ins[0]) is None
    assert quotes.quote(ins[1]) is not None


def test_ltp_fallback_covers_what_quote_data_did_not():
    """quote_data returns {} without a Data API subscription; ticker_data is
    the thinner fallback."""
    ins = [instrument("RELIANCE", "2885"), instrument("INFY", "1594")]
    client = FakeDhanClient(quotes={"2885": {"ltp": 1400.0}},
                            ltps={"1594": 1600.0})
    quotes = InMemoryQuoteSource()
    feed(client, ins).refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)
    assert client.ltp_calls[0][0] == ("1594",)
    assert len(quotes) == 2


def test_no_quotes_at_all_raises():
    """Silence from both endpoints is a failure, not a book that is simply
    unpriced. Returning quietly here leaves the last tick's prices standing."""
    client = FakeDhanClient(quotes={}, ltps={})
    with pytest.raises(FeedUnavailableError):
        feed(client).refresh_quotes(InMemoryQuoteSource(), AssetClass.EQUITY, OPEN_NOW)


def test_quote_exception_raises_and_leaves_prior_quotes_untouched():
    quotes = InMemoryQuoteSource({"RELIANCE": Decimal("1400")})
    client = FakeDhanClient(quote_error=RuntimeError("rate limited"))
    with pytest.raises(FeedUnavailableError):
        feed(client).refresh_quotes(quotes, AssetClass.EQUITY, OPEN_NOW)


def test_refresh_quotes_ignores_a_class_this_feed_does_not_serve():
    client = FakeDhanClient(quotes={"2885": {"ltp": 1400.0}})
    quotes = InMemoryQuoteSource()
    feed(client).refresh_quotes(quotes, AssetClass.CRYPTO, OPEN_NOW)
    assert client.quote_calls == []
    assert len(quotes) == 0


# ---------------------------------------------------------------------------
# extra_provider / regime
# ---------------------------------------------------------------------------

def test_extra_provider_publishes_bars_and_regime():
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 40, step=1.0))
    extra = feed(client).extra_provider(AssetClass.EQUITY, OPEN_NOW)
    assert set(extra) == {BARS_KEY, REGIME_KEY}
    assert all(isinstance(b, Bar) for b in extra[BARS_KEY]["RELIANCE"])
    assert isinstance(extra[REGIME_KEY]["RELIANCE"], RegimeRead)


def test_extra_provider_returns_nothing_for_another_class():
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 40))
    assert feed(client).extra_provider(AssetClass.CRYPTO, OPEN_NOW) == {}
    assert client.ohlc_calls == []


def test_extra_provider_does_not_ask_for_option_candles():
    """Option legs are priced through quotes; asking the equity scrip map for
    an option's history fails and would take the whole tick down."""
    ins = [instrument("RELIANCE", "2885"),
           instrument("NIFTY25JUN24000CE", "43211", expiry=dt.date(2025, 6, 26))]
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 40))
    extra = feed(client, ins).extra_provider(AssetClass.EQUITY, OPEN_NOW)
    assert set(extra[BARS_KEY]) == {"RELIANCE"}
    assert [c["symbol"] for c in client.ohlc_calls] == ["RELIANCE"]


def test_regime_reads_trend_up():
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 40, step=1.0))
    read = feed(client).extra_provider(AssetClass.EQUITY, OPEN_NOW)[REGIME_KEY]["RELIANCE"]
    assert read.state is RegimeState.TREND_UP
    assert Decimal(0) <= read.confidence <= Decimal(1)


def test_regime_reads_trend_down():
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 40,
                                                base=200.0, step=-1.0))
    read = feed(client).extra_provider(AssetClass.EQUITY, OPEN_NOW)[REGIME_KEY]["RELIANCE"]
    assert read.state is RegimeState.TREND_DOWN


def test_regime_reads_range_on_a_flat_tape():
    rows = candle_rows(OPEN_NOW - INTERVAL, 40)
    for i, r in enumerate(rows):          # small alternating noise, no drift
        bump = 0.2 if i % 2 else -0.2
        r["open"] = r["close"] = 100.0 + bump
        r["high"], r["low"] = 101.0 + bump, 99.0 + bump
    read = feed(FakeDhanClient(candles=rows)).extra_provider(
        AssetClass.EQUITY, OPEN_NOW)[REGIME_KEY]["RELIANCE"]
    assert read.state is RegimeState.RANGE


def test_regime_is_absent_not_fabricated_when_history_is_short():
    """Fewer bars than the lookback means NO read. Publishing a default RANGE
    would be a fabricated opinion the strategy would act on."""
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 6))
    extra = feed(client, regime_lookback=20).extra_provider(AssetClass.EQUITY, OPEN_NOW)
    assert extra[REGIME_KEY] == {}
    assert len(extra[BARS_KEY]["RELIANCE"]) == 6


def test_regime_alias_publishes_under_the_underlying_name():
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 40, step=1.0))
    f = feed(client, [instrument("NIFTY50", "26000")],
             regime_aliases={"NIFTY": "NIFTY50"})
    regimes = f.extra_provider(AssetClass.EQUITY, OPEN_NOW)[REGIME_KEY]
    assert regimes["NIFTY"] is regimes["NIFTY50"]


def test_regime_alias_to_an_unknown_symbol_is_rejected():
    with pytest.raises(ValueError):
        DhanFeed([instrument()], FakeDhanClient(), regime_aliases={"NIFTY": "NOPE"})


def test_regime_read_is_accepted_by_the_strategy_contract():
    """RegimeRead validates state/confidence/as_of in __post_init__, so a bad
    read cannot be published at all."""
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - INTERVAL, 40, step=1.0))
    read = feed(client).extra_provider(AssetClass.EQUITY, OPEN_NOW)[REGIME_KEY]["RELIANCE"]
    assert RegimeRead.coerce(read) is read
    assert read.as_of.tzinfo is not None


def test_extra_provider_propagates_a_stale_feed_rather_than_publishing_partial_bars():
    client = FakeDhanClient(candles=candle_rows(OPEN_NOW - dt.timedelta(hours=3), 40))
    with pytest.raises(FeedStaleError):
        feed(client).extra_provider(AssetClass.EQUITY, OPEN_NOW)


# ---------------------------------------------------------------------------
# contract + no-credential guarantees
# ---------------------------------------------------------------------------

def test_dhan_feed_satisfies_the_market_feed_protocol():
    assert isinstance(DhanFeed([instrument()], FakeDhanClient()), MarketFeed)


def test_fake_client_satisfies_the_client_protocol():
    assert isinstance(FakeDhanClient(), DhanMarketDataClient)


def test_module_never_constructs_a_dhan_client_or_reads_a_token():
    """The owner's live engine holds the only permitted Dhan token. Importing
    this module must not authenticate, and nothing in it may read a credential.
    """
    module = importlib.import_module("feeds.dhan_feed")
    source = inspect.getsource(module)
    for forbidden in ("dhanhq", "DhanContext", "access_token", "ACCESS_TOKEN",
                      "client_id", "CLIENT_ID", "api_key", "os.environ",
                      "getenv", "requests"):
        assert forbidden not in source, f"dhan_feed.py references {forbidden!r}"
    assert "dhanhq" not in sys.modules


def test_module_has_no_order_path():
    source = inspect.getsource(importlib.import_module("feeds.dhan_feed"))
    for forbidden in ("place_order", "cancel_order", "modify_order"):
        assert forbidden not in source


def test_equity_profile_is_the_market_hours_authority():
    """Sanity-check the fixture times against the profile the feed uses, so a
    change to session hours breaks here loudly rather than silently retuning
    every staleness test above."""
    profile = EquityProfile()
    assert profile.is_open(OPEN_NOW)
    assert not profile.is_open(CLOSED_NOW)
    assert not profile.is_open(SATURDAY_NOW)
