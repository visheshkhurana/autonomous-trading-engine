"""
The Dhan bar cache's freshness vouch.

`_cache_vouches` gates every cache hit. Its primary check is
`vouched != last_closed`, which catches the ordinary case: the clock moved into
a new bar bucket, so refetch. The freshness re-check behind it looks redundant
and is not — it is the only thing standing between a live book and a cache
entry whose bars were ALREADY stale when they were stored, which happens when
the venue itself lags: `vouched` is stamped from `now`, while `bars[-1].ts`
comes from Dhan.

Removing the re-check left the whole suite green on both an independent
implementation and this one, so it is pinned directly here.
"""

from __future__ import annotations

import datetime as dt

from core.profiles import IST
from core.types import AssetClass, Instrument, Mode
from feeds.dhan_feed import DhanFeed, _last_closed_start
from strategies.base import Bar
from decimal import Decimal

NOW = dt.datetime(2026, 8, 27, 11, 0, tzinfo=IST)   # inside the NSE session


class _Client:
    def get_ohlc(self, symbol, interval, from_date, to_date, exchange="NSE"):
        return []

    def get_fno_quotes(self, security_ids, exchange_segment="NSE_FNO"):
        return {str(s): {"ltp": 100.5} for s in security_ids}

    def get_fno_ltps(self, security_ids, exchange_segment="NSE_FNO"):
        return {str(s): 100.5 for s in security_ids}


def _feed() -> DhanFeed:
    feed = DhanFeed(
        client=_Client(),
        instruments=[Instrument("CASH0", AssetClass.EQUITY, "1000", lot_size=1)],
    )
    return feed.bind({AssetClass.EQUITY: Mode.PAPER})


def _bar(ts: dt.datetime) -> Bar:
    return Bar(ts=ts, open=Decimal("100"), high=Decimal("101"),
               low=Decimal("99"), close=Decimal("100.5"), volume=Decimal("1000"))


def test_a_cache_entry_stamped_now_but_holding_stale_bars_is_refused() -> None:
    """The venue lagged when the entry was stored.

    `vouched` equals the current last-closed boundary, so the primary check
    passes and the entry looks current. Only the freshness re-check notices the
    bars themselves are hours old. Delete that re-check and this test fails.
    """
    feed = _feed()
    last_closed = _last_closed_start(NOW, feed._interval)

    # Stored under the CURRENT boundary, but the bars are four hours old.
    stale = (( _bar(NOW - dt.timedelta(hours=4)), ), last_closed)

    assert feed._cache_vouches(stale, last_closed, NOW, "CASH0") is False


def test_a_cache_entry_with_current_bars_is_still_served() -> None:
    """Counterweight: the guard must not refuse a legitimately fresh entry, or
    the cache degrades into no cache at all and the rate limit returns."""
    feed = _feed()
    last_closed = _last_closed_start(NOW, feed._interval)
    fresh = ((_bar(last_closed),), last_closed)

    assert feed._cache_vouches(fresh, last_closed, NOW, "CASH0") is True


def test_outside_the_session_old_bars_are_the_correct_answer() -> None:
    """Market hours are not staleness. At 20:00 IST the newest closed bar is
    legitimately hours old; refusing it would make the feed scream every
    evening, and an alarm that fires when nothing is wrong gets ignored."""
    feed = _feed()
    after_hours = dt.datetime(2026, 8, 27, 20, 0, tzinfo=IST)
    last_closed = _last_closed_start(after_hours, feed._interval)
    old = ((_bar(after_hours - dt.timedelta(hours=5)),), last_closed)

    assert feed._cache_vouches(old, last_closed, after_hours, "CASH0") is True
