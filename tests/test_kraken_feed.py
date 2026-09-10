"""Kraken public feed. Injected fake only — no auth, no account."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from adapters.paper import InMemoryQuoteSource
from core.types import AssetClass, Instrument, Mode
from feeds.base import FeedStaleError
from feeds.kraken_feed import KrakenFeed, SymbolNotListed
from tests.test_okx_feed import (
    BTC, ETH, FakeExchange, M15, UTC, candles, spot_market,
)

NOW = dt.datetime(2026, 8, 28, 12, 15, tzinfo=UTC)


def test_kraken_feed_refuses_equity_instruments() -> None:
    equity = Instrument("RELIANCE", AssetClass.EQUITY, "PAPER-RELIANCE")
    with pytest.raises(ValueError, match="crypto"):
        KrakenFeed([equity], exchange=FakeExchange())


def test_bind_resolves_spot_and_skips_swaps() -> None:
    ex = FakeExchange()
    feed = KrakenFeed([BTC, ETH], exchange=ex, bar_interval=M15)
    feed.bind({AssetClass.CRYPTO: Mode.PAPER})
    assert feed.market_symbol("BTC/USDT") == "BTC/USDT"
    assert feed.market_symbol("ETH/USDT") == "ETH/USDT"


def test_unlisted_symbol_fails_at_bind() -> None:
    luna = Instrument("LUNA/USDT", AssetClass.CRYPTO, "LUNA/USDT")
    feed = KrakenFeed([luna], exchange=FakeExchange())
    with pytest.raises(SymbolNotListed):
        feed.bind({AssetClass.CRYPTO: Mode.PAPER})


def test_forming_bar_is_never_published() -> None:
    # Last row is the currently-forming 12:00 candle; now is 12:15 so it
    # has not closed. The feed must drop it.
    forming_anchor = dt.datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
    now = dt.datetime(2026, 8, 28, 12, 7, tzinfo=UTC)
    rows = candles(forming_anchor, 10)
    ex = FakeExchange(ohlcv={"BTC/USDT": rows})
    feed = KrakenFeed([BTC], exchange=ex, history=10)
    feed.bind({AssetClass.CRYPTO: Mode.PAPER})
    bars = feed.bars("BTC/USDT", now)
    assert bars[-1].ts < now
    assert bars[-1].ts == forming_anchor - M15


def test_default_exchange_is_unsigned() -> None:
    """Construction with no exchange injects a public client. Skip if no ccxt."""
    pytest.importorskip("ccxt")
    feed = KrakenFeed([BTC])
    ex = feed._ex
    assert not getattr(ex, "apiKey", None)
    assert not getattr(ex, "secret", None)


# ---------------------------------------------------------------------------
# refresh_quotes / _mark_from_ticker. Nothing above exercised this path, which
# is how a venue that could not price a single symbol shipped with a green
# suite. `set_ticker` always stamps; real Kraken never does.
# ---------------------------------------------------------------------------

def _unstamped(ex: FakeExchange, symbol: str, *, last, bid=None, ask=None) -> None:
    """The shape ccxt actually returns for a Kraken ticker: timestamp=None."""
    ex._tickers[symbol] = {"symbol": symbol, "timestamp": None, "datetime": None,
                           "last": last, "close": last, "bid": bid, "ask": ask}


def _bound(ex: FakeExchange) -> KrakenFeed:
    feed = KrakenFeed([BTC, ETH], exchange=ex, bar_interval=M15)
    feed.bind({AssetClass.CRYPTO: Mode.PAPER})
    return feed


def test_unstamped_kraken_ticker_is_priced_from_receipt() -> None:
    """THE bug: the real Kraken shape must produce a mark, not a refusal."""
    ex = FakeExchange()
    _unstamped(ex, "BTC/USDT", last=78638.0)
    _unstamped(ex, "ETH/USDT", last=2489.53)
    quotes = InMemoryQuoteSource()
    _bound(ex).refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is not None
    assert quotes.quote(ETH) is not None


def test_stamped_but_stale_ticker_is_still_refused() -> None:
    """The venue-clock path must not have been loosened alongside the fix.
    A stamped ticker 10 minutes old fails a 2-minute tolerance."""
    ex = FakeExchange()
    ex.set_ticker("BTC/USDT", last=78638.0, at=NOW - dt.timedelta(minutes=10))
    ex.set_ticker("ETH/USDT", last=2489.53, at=NOW)
    quotes = InMemoryQuoteSource()
    _bound(ex).refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None
    assert quotes.quote(ETH) is not None


def test_unstamped_ticker_with_no_price_is_still_refused() -> None:
    """Receipt-aging establishes freshness, not a price. No price, no mark."""
    ex = FakeExchange()
    _unstamped(ex, "BTC/USDT", last=None)
    _unstamped(ex, "ETH/USDT", last=2489.53)
    quotes = InMemoryQuoteSource()
    _bound(ex).refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None
    assert quotes.quote(ETH) is not None


def test_receipt_aged_mark_records_when_it_was_received() -> None:
    """"Received this tick" is a claim. Record it, so a later reader can age
    the mark instead of trusting it. The paper adapter ignores as_of today
    (it gates on `live`), so this pins only that the claim is written."""
    ex = FakeExchange()
    _unstamped(ex, "BTC/USDT", last=78638.0)
    _unstamped(ex, "ETH/USDT", last=2489.53)
    quotes = InMemoryQuoteSource()
    _bound(ex).refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC).as_of == NOW
