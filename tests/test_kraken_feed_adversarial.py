"""Adversarial review of 81b5ecb: Kraken marks are aged from receipt.

Before that commit `_mark_from_ticker` returned at the timestamp check for
every Kraken ticker (ccxt reports timestamp=None), so nothing below that line
ever ran against Kraken data. Now it does, and two hazards that the OKX copy
has always carried become reachable on Kraken:

1. A single unreadable field (`last`, `close`, `bid`, `ask`, or `timestamp`)
   raises out of the per-symbol loop in `refresh_quotes`. The loop is
   abandoned with every LATER symbol still carrying last tick's mark, and
   `_drop_quote` is skipped for the offending symbol too. That is the frozen
   price this module says it exists to prevent — a transport failure blacks
   out; a parse failure leaves the old numbers in place.

2. `Decimal("Infinity")` passes `price > 0` and is written to the quote
   source as a mark.

The tests marked HOLE fail on 81b5ecb. The tests marked PIN pass and guard
the stamped branch, which the commit must not have loosened.

Injected fake only — no network, no auth.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from adapters.paper import InMemoryQuoteSource
from core.types import AssetClass, Mode
from feeds.kraken_feed import KrakenFeed
from tests.test_okx_feed import BTC, ETH, FakeExchange, M15, UTC

NOW = dt.datetime(2026, 8, 28, 12, 15, tzinfo=UTC)
TICK = dt.timedelta(seconds=5)

TICK1_BTC = Decimal("78638.0")
TICK1_ETH = Decimal("2489.53")
TICK2_ETH = Decimal("2600.0")


def _unstamped(ex: FakeExchange, symbol: str, **fields) -> None:
    """The shape ccxt 4.5.78 returns for a Kraken ticker: timestamp=None."""
    ticker = {"symbol": symbol, "timestamp": None, "datetime": None,
              "last": None, "close": None, "bid": None, "ask": None}
    ticker.update(fields)
    ex._tickers[symbol] = ticker


def _bound(ex: FakeExchange) -> KrakenFeed:
    feed = KrakenFeed([BTC, ETH], exchange=ex, bar_interval=M15)
    feed.bind({AssetClass.CRYPTO: Mode.PAPER})
    return feed


def _after_healthy_tick() -> tuple[KrakenFeed, FakeExchange, InMemoryQuoteSource]:
    """Tick 1: both symbols priced. The next refresh is where the attack lands."""
    ex = FakeExchange()
    _unstamped(ex, "BTC/USDT", last=float(TICK1_BTC))
    _unstamped(ex, "ETH/USDT", last=float(TICK1_ETH))
    feed = _bound(ex)
    quotes = InMemoryQuoteSource()
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC).price == TICK1_BTC
    assert quotes.quote(ETH).price == TICK1_ETH
    return feed, ex, quotes


# ---------------------------------------------------------------------------
# HOLE 1: one unreadable field strands every later symbol on last tick's mark
# ---------------------------------------------------------------------------

POISON = [
    # (label, BTC ticker fields on tick 2)
    pytest.param("last-str", {"last": "abc"}, id="last='abc' -> ValueError"),
    pytest.param("last-bool", {"last": True}, id="last=True -> TypeError"),
    pytest.param("last-list", {"last": [1]}, id="last=[1] -> TypeError"),
    pytest.param("last-nan", {"last": float("nan")},
                 id="last=nan -> decimal.InvalidOperation (not TypeError/ValueError)"),
    pytest.param("bid-str", {"last": None, "close": None, "bid": "x", "ask": 1.0},
                 id="bid='x' -> ValueError"),
    # These three were reachable before 81b5ecb too (the stamp is parsed
    # first). Listed because the same loop abandonment applies.
    pytest.param("ts-bool", {"last": float(TICK1_BTC), "timestamp": False},
                 id="timestamp=False -> TypeError [pre-existing]"),
    pytest.param("ts-nan", {"last": float(TICK1_BTC), "timestamp": float("nan")},
                 id="timestamp=nan -> ValueError [pre-existing]"),
    pytest.param("ts-huge", {"last": float(TICK1_BTC), "timestamp": 1e300},
                 id="timestamp=1e300 -> OverflowError [pre-existing]"),
]


@pytest.mark.parametrize("label, btc_fields", POISON)
def test_one_unreadable_field_unprices_one_symbol_not_the_batch(
        label: str, btc_fields: dict) -> None:
    """HOLE. BTC's ticker carries one garbage field on tick 2; ETH's is
    healthy and has MOVED. The correct end state is BTC unpriced (with the
    reason recorded) and ETH marked at the tick-2 price.

    On 81b5ecb the parse error escapes `refresh_quotes`. BTC's tick-1 mark
    is never dropped and ETH's tick-1 mark is never replaced: the quote
    source now carries two prices the venue no longer shows, and the next
    tick's `adapter.equity()` marks the book against them.
    """
    feed, ex, quotes = _after_healthy_tick()
    _unstamped(ex, "BTC/USDT", **btc_fields)
    _unstamped(ex, "ETH/USDT", last=float(TICK2_ETH))

    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW + TICK)

    assert quotes.quote(BTC) is None, (
        f"[{label}] BTC's tick-1 mark {quotes.quote(BTC)} survived a ticker "
        "that could not be read; an unreadable price is no price")
    assert quotes.quote(ETH) is not None and quotes.quote(ETH).price == TICK2_ETH, (
        f"[{label}] ETH is still marked at tick 1 ({quotes.quote(ETH)}); the "
        "batch was abandoned before ETH was re-marked")
    assert "BTC/USDT" in feed.unpriced and "ETH/USDT" not in feed.unpriced


def test_unreadable_field_on_every_symbol_is_a_blackout_not_a_crash() -> None:
    """HOLE. If no symbol can be read the answer is the same as for a failed
    fetch: marks removed, FeedUnavailableError raised. Not a bare TypeError
    with last tick's marks left in the book."""
    from feeds.base import FeedUnavailableError

    feed, ex, quotes = _after_healthy_tick()
    _unstamped(ex, "BTC/USDT", last=True)
    _unstamped(ex, "ETH/USDT", last="n/a")

    with pytest.raises(FeedUnavailableError):
        feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW + TICK)
    assert quotes.quote(BTC) is None and quotes.quote(ETH) is None


# ---------------------------------------------------------------------------
# HOLE 2: a non-finite price is accepted as a mark
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["last", "close"])
def test_infinite_price_is_refused(field: str) -> None:
    """HOLE. `Decimal("Infinity") > 0` is True, so an infinite `last` is
    written to the quote source. Nothing downstream should ever see it: the
    engine's `_quotes` filter checks `> 0` but not `is_finite()`, so the
    strategy context would carry it. Refuse it at the feed."""
    feed, ex, quotes = _after_healthy_tick()
    _unstamped(ex, "BTC/USDT", **{field: float("inf")})
    _unstamped(ex, "ETH/USDT", last=float(TICK2_ETH))

    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW + TICK)

    assert quotes.quote(BTC) is None, f"infinite {field} became mark {quotes.quote(BTC)}"
    assert quotes.quote(ETH).price == TICK2_ETH


def test_infinite_bid_ask_mid_is_refused() -> None:
    """HOLE (same shape, mid path)."""
    feed, ex, quotes = _after_healthy_tick()
    _unstamped(ex, "BTC/USDT", last=None, close=None, bid=float("inf"), ask=float("inf"))
    _unstamped(ex, "ETH/USDT", last=float(TICK2_ETH))

    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW + TICK)

    assert quotes.quote(BTC) is None


# ---------------------------------------------------------------------------
# PIN: the stamped branch was not loosened
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zero", [0, "0", 0.0],
                         ids=["int-0", "str-0", "float-0"])
def test_epoch_zero_timestamp_is_refused_not_read_as_absent(zero) -> None:
    """PIN. `timestamp: 0` is a stamp (1970), not an absent stamp. It must
    take the aged branch and be refused, not the receipt branch."""
    ex = FakeExchange()
    _unstamped(ex, "BTC/USDT", last=float(TICK1_BTC), timestamp=zero)
    _unstamped(ex, "ETH/USDT", last=float(TICK1_ETH))
    quotes = InMemoryQuoteSource()
    _bound(ex).refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None
    assert quotes.quote(ETH).price == TICK1_ETH


def test_future_stamp_is_still_refused() -> None:
    """PIN. The committed tests cover a stale stamp; this covers the other
    side of the window."""
    ex = FakeExchange()
    ex.set_ticker("BTC/USDT", last=float(TICK1_BTC), at=NOW + dt.timedelta(minutes=10))
    ex.set_ticker("ETH/USDT", last=float(TICK1_ETH), at=NOW)
    quotes = InMemoryQuoteSource()
    _bound(ex).refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC) is None
    assert quotes.quote(ETH).price == TICK1_ETH


def test_stamp_inside_tolerance_is_accepted_on_both_sides() -> None:
    """PIN. Within +/- quote_max_age a stamped ticker prices."""
    ex = FakeExchange()
    ex.set_ticker("BTC/USDT", last=float(TICK1_BTC), at=NOW - dt.timedelta(seconds=90))
    ex.set_ticker("ETH/USDT", last=float(TICK1_ETH), at=NOW + dt.timedelta(seconds=90))
    quotes = InMemoryQuoteSource()
    _bound(ex).refresh_quotes(quotes, AssetClass.CRYPTO, NOW)
    assert quotes.quote(BTC).price == TICK1_BTC
    assert quotes.quote(ETH).price == TICK1_ETH


def test_failed_fetch_after_a_priced_tick_still_blacks_out() -> None:
    """PIN. The justification for receipt-aging rests on this: a failed
    fetch retires the marks before `_mark_from_ticker` could run."""
    from feeds.base import FeedUnavailableError

    feed, ex, quotes = _after_healthy_tick()
    ex.tickers_error = ConnectionError("venue down")
    with pytest.raises(FeedUnavailableError):
        feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW + TICK)
    assert quotes.quote(BTC) is None and quotes.quote(ETH) is None


def test_symbol_missing_from_the_batch_after_a_priced_tick_is_dropped() -> None:
    """PIN. A symbol the venue stopped returning loses its previous mark."""
    feed, ex, quotes = _after_healthy_tick()
    del ex._tickers["BTC/USDT"]
    _unstamped(ex, "ETH/USDT", last=float(TICK2_ETH))
    feed.refresh_quotes(quotes, AssetClass.CRYPTO, NOW + TICK)
    assert quotes.quote(BTC) is None
    assert quotes.quote(ETH).price == TICK2_ETH
