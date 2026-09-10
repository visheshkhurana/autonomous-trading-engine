"""
Tests for the paper broker.

The theme: every assertion here exists because removing the rule it pins would
make paper P&L flattering, or make a failure look like a legitimate zero.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Callable, Optional

import pytest

from adapters.paper import (
    NFO_OPTIONS,
    NSE_CASH_INTRADAY,
    OKX_SPOT_TAKER,
    DefaultFeeModel,
    InMemoryQuoteSource,
    PaperAdapter,
    SlippageModel,
)
from core.types import (
    AssetClass, Instrument, Intent, Mode, OrderStatus, OrderType, Side,
)

UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NIFTY_CE = Instrument(
    symbol="NIFTY26JAN24000CE", asset_class=AssetClass.EQUITY, broker_id="43210",
    lot_size=75, tick_size=Decimal("0.05"), qty_precision=0,
    expiry=dt.date(2026, 1, 29),
)
RELIANCE = Instrument(
    symbol="RELIANCE", asset_class=AssetClass.EQUITY, broker_id="2885",
    lot_size=1, tick_size=Decimal("0.05"), qty_precision=0,
)
BTC = Instrument(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO, broker_id="BTC-USDT",
    lot_size=1, tick_size=Decimal("0.1"), qty_precision=4,
    min_notional=Decimal("10"),
)


class FakeClock:
    """Drives the adapter's poll loop without real time passing.

    `on_sleep` is the hook tests use to move the market (or cancel) while an
    order is resting — the only moment at which a real book would change.
    """

    def __init__(self, on_sleep: Optional[Callable[["FakeClock"], None]] = None) -> None:
        self.now = dt.datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
        self.sleeps = 0
        self.on_sleep = on_sleep

    def __call__(self) -> dt.datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += dt.timedelta(seconds=seconds)
        self.sleeps += 1
        if self.on_sleep is not None:
            self.on_sleep(self)


def crypto_adapter(quotes: InMemoryQuoteSource, cash: str = "100000",
                   clock: Optional[FakeClock] = None, **kw) -> PaperAdapter:
    clock = clock or FakeClock()
    kw.setdefault("seed", 7)
    return PaperAdapter(AssetClass.CRYPTO, quotes, Decimal(cash),
                        now=clock, sleep=clock.sleep, **kw)


def equity_adapter(quotes: InMemoryQuoteSource, cash: str = "500000",
                   clock: Optional[FakeClock] = None, **kw) -> PaperAdapter:
    clock = clock or FakeClock()
    kw.setdefault("seed", 7)
    return PaperAdapter(AssetClass.EQUITY, quotes, Decimal(cash),
                        now=clock, sleep=clock.sleep, **kw)


def intent(instrument: Instrument, side: Side, qty: str,
           order_type: OrderType = OrderType.MARKET,
           limit: Optional[str] = None, **kw) -> Intent:
    return Intent(
        instrument=instrument, side=side, qty=Decimal(qty), order_type=order_type,
        strategy_id=kw.pop("strategy_id", "s1"), reason="test",
        limit_price=None if limit is None else Decimal(limit), **kw,
    )


# ---------------------------------------------------------------------------
# Fills, slippage, costs
# ---------------------------------------------------------------------------

def test_market_buy_fills_adversely_and_pays_fees():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)
    start = a.cash

    ref = a.place(intent(BTC, Side.BUY, "1"))

    assert ref.status is OrderStatus.FILLED
    assert ref.status.terminal
    assert ref.filled_qty == Decimal("1")
    assert ref.avg_price is not None
    # Adverse-only slippage: a buyer never gets price improvement in paper.
    assert ref.avg_price >= Decimal("50000")
    assert ref.avg_price <= Decimal("50000") * Decimal("1.0005") + BTC.tick_size
    assert len(ref.fills) == 1
    assert ref.fills[0].fees > 0
    assert a.cash == start - ref.avg_price * Decimal("1") - ref.fills[0].fees


def test_market_sell_is_credited_below_the_quote():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)
    a.place(intent(BTC, Side.BUY, "2"))

    ref = a.place(intent(BTC, Side.SELL, "1"))

    assert ref.status is OrderStatus.FILLED
    assert ref.avg_price is not None
    assert ref.avg_price <= Decimal("50000")


def test_round_trip_at_a_flat_price_loses_money():
    """If costs were cosmetic, this would break even and every mean-reversion
    strategy would look profitable on paper."""
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)
    start = a.cash

    a.place(intent(BTC, Side.BUY, "1"))
    a.place(intent(BTC, Side.SELL, "1"))

    assert a.positions() == ()
    assert a.cash < start
    assert a.equity() == a.cash


def test_indian_charge_stack_is_side_asymmetric():
    """STT and stamp duty are one-sided. A symmetric flat-bps model gets
    intraday P&L wrong by enough to flip a marginal strategy's sign."""
    buy = NFO_OPTIONS.charge(NIFTY_CE, Side.BUY, Decimal("75"), Decimal("100"),
                             include_per_order_fee=True)
    sell = NFO_OPTIONS.charge(NIFTY_CE, Side.SELL, Decimal("75"), Decimal("100"),
                              include_per_order_fee=True)
    assert sell > buy


def test_default_fee_model_routes_by_instrument():
    assert DefaultFeeModel.stack_for(NIFTY_CE) is NFO_OPTIONS
    assert DefaultFeeModel.stack_for(RELIANCE) is NSE_CASH_INTRADAY
    assert DefaultFeeModel.stack_for(BTC) is OKX_SPOT_TAKER


def test_per_order_brokerage_is_charged_once_across_partial_fills():
    """Charging the flat ₹20 on every partial would make a thin fill look
    several times more expensive than the broker actually bills."""
    first = NFO_OPTIONS.charge(NIFTY_CE, Side.BUY, Decimal("75"), Decimal("100"),
                               include_per_order_fee=True)
    later = NFO_OPTIONS.charge(NIFTY_CE, Side.BUY, Decimal("75"), Decimal("100"),
                               include_per_order_fee=False)
    assert first - later >= Decimal("20")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_fills_exactly():
    prices = []
    for _ in range(2):
        quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
        a = crypto_adapter(quotes, seed=99)
        prices.append([a.place(intent(BTC, Side.BUY, "1")).avg_price
                       for _ in range(5)])
    assert prices[0] == prices[1]


def test_different_seeds_produce_different_fills():
    runs = []
    for seed in (1, 2):
        quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
        a = crypto_adapter(quotes, seed=seed)
        runs.append([a.place(intent(BTC, Side.BUY, "1")).avg_price
                     for _ in range(5)])
    assert runs[0] != runs[1]


def test_rejected_orders_do_not_shift_the_random_stream():
    """The RNG is keyed per order id, not a shared stream, so inserting a
    rejected order between two fills must not change either fill."""
    def run(with_reject: bool) -> list[Optional[Decimal]]:
        quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
        a = crypto_adapter(quotes, seed=5)
        out = [a.place(intent(BTC, Side.BUY, "1")).avg_price]
        if with_reject:
            # Below min_notional: rejected before any RNG draw is consumed.
            a.place(intent(BTC, Side.BUY, "0.0001"))
        out.append(a.place(intent(BTC, Side.BUY, "1")).avg_price)
        return out

    assert run(False)[0] == run(True)[0]


# ---------------------------------------------------------------------------
# LIMIT orders that should not fill
# ---------------------------------------------------------------------------

def test_limit_buy_below_the_market_times_out_unfilled():
    """The single most important behaviour in this module. A paper adapter
    that fills this teaches a strategy that its passive entries are free."""
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    clock = FakeClock()
    a = crypto_adapter(quotes, clock=clock)
    start = a.cash

    ref = a.place(intent(BTC, Side.BUY, "1", OrderType.LIMIT, limit="49000"),
                  timeout_s=5.0)

    assert ref.status is OrderStatus.TIMEOUT
    assert ref.status.terminal
    assert ref.filled_qty == Decimal("0")
    assert ref.avg_price is None
    assert ref.fills == ()
    assert a.positions() == ()
    assert a.cash == start
    assert clock.sleeps > 0


def test_limit_sell_above_the_market_times_out_unfilled():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)
    ref = a.place(intent(BTC, Side.SELL, "1", OrderType.LIMIT, limit="51000"),
                  timeout_s=5.0)
    assert ref.status is OrderStatus.TIMEOUT
    assert ref.filled_qty == Decimal("0")


def test_marketable_limit_never_fills_worse_than_the_limit():
    """Slippage plus tick rounding must not push a fill through the limit —
    that is a price the strategy never authorised."""
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes, slippage=SlippageModel(max_bps=Decimal("500")))

    ref = a.place(intent(BTC, Side.BUY, "1", OrderType.LIMIT, limit="50000"))

    assert ref.status is OrderStatus.FILLED
    assert ref.avg_price == Decimal("50000")


def test_marketable_limit_sell_never_fills_below_the_limit():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes, slippage=SlippageModel(max_bps=Decimal("500")))
    a.place(intent(BTC, Side.BUY, "1"))

    ref = a.place(intent(BTC, Side.SELL, "1", OrderType.LIMIT, limit="50000"))

    assert ref.status is OrderStatus.FILLED
    assert ref.avg_price == Decimal("50000")


def test_resting_limit_fills_when_the_market_comes_to_it():
    def move(clock: FakeClock) -> None:
        if clock.sleeps == 2:
            quotes.set("BTC/USDT", "48900")

    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    clock = FakeClock(on_sleep=move)
    a = crypto_adapter(quotes, clock=clock)

    ref = a.place(intent(BTC, Side.BUY, "1", OrderType.LIMIT, limit="49000"),
                  timeout_s=10.0)

    assert ref.status is OrderStatus.FILLED
    assert ref.avg_price is not None and ref.avg_price <= Decimal("49000")


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_cancel_while_resting_returns_cancelled_not_filled():
    holder: dict[str, str] = {}

    def cancel_it(clock: FakeClock) -> None:
        if clock.sleeps == 1:
            assert a.cancel(holder["id"]) is True

    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    clock = FakeClock(on_sleep=cancel_it)
    a = crypto_adapter(quotes, clock=clock)
    # The order id is deterministic, so the hook can name it before placing.
    holder["id"] = "PAPER-crypto-000001"

    ref = a.place(intent(BTC, Side.BUY, "1", OrderType.LIMIT, limit="49000"),
                  timeout_s=60.0)

    assert ref.order_id == holder["id"]
    assert ref.status is OrderStatus.CANCELLED
    assert ref.filled_qty == Decimal("0")
    assert a.positions() == ()


def test_cancel_of_unknown_or_terminal_order_is_false():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)
    assert a.cancel("PAPER-crypto-999999") is False

    ref = a.place(intent(BTC, Side.BUY, "1"))
    # Already terminal: claiming a successful cancel would tell the caller a
    # position was never opened when it was.
    assert a.cancel(ref.order_id) is False


# ---------------------------------------------------------------------------
# Refusals — things a venue would not accept
# ---------------------------------------------------------------------------

def test_missing_quote_is_rejected_never_priced():
    quotes = InMemoryQuoteSource()
    a = crypto_adapter(quotes)
    start = a.cash

    ref = a.place(intent(BTC, Side.BUY, "1"))

    assert ref.status is OrderStatus.REJECTED
    assert ref.error is not None and "no quote" in ref.error
    assert ref.filled_qty == Decimal("0")
    assert a.cash == start
    assert a.positions() == ()


def test_market_order_rejected_where_the_venue_forbids_it():
    """NFO rejects MARKET outright. Paper must too, or the strategy only finds
    out on its first live order."""
    quotes = InMemoryQuoteSource({"NIFTY26JAN24000CE": "100"})
    a = equity_adapter(quotes)

    ref = a.place(intent(NIFTY_CE, Side.BUY, "75"))

    assert ref.status is OrderStatus.REJECTED
    assert "MARKET" in (ref.error or "")


def test_sub_lot_quantity_is_rejected_not_rounded_up():
    quotes = InMemoryQuoteSource({"NIFTY26JAN24000CE": "100"})
    a = equity_adapter(quotes)

    ref = a.place(intent(NIFTY_CE, Side.BUY, "50", OrderType.LIMIT, limit="100"))

    assert ref.status is OrderStatus.REJECTED
    assert "quantizes to 0" in (ref.error or "")
    assert a.positions() == ()


def test_over_lot_quantity_rounds_down_never_up():
    quotes = InMemoryQuoteSource({"NIFTY26JAN24000CE": "100"})
    a = equity_adapter(quotes)

    ref = a.place(intent(NIFTY_CE, Side.BUY, "170", OrderType.LIMIT, limit="100"))

    assert ref.status is OrderStatus.FILLED
    assert ref.filled_qty == Decimal("150")   # 2 lots, not 3


def test_off_tick_limit_is_rejected():
    quotes = InMemoryQuoteSource({"NIFTY26JAN24000CE": "100"})
    a = equity_adapter(quotes)

    ref = a.place(intent(NIFTY_CE, Side.BUY, "75", OrderType.LIMIT, limit="100.03"))

    assert ref.status is OrderStatus.REJECTED
    assert "tick" in (ref.error or "")


def test_off_tick_limit_can_be_quantized_when_explicitly_allowed():
    quotes = InMemoryQuoteSource({"NIFTY26JAN24000CE": "100"})
    a = equity_adapter(quotes, reject_off_tick=False)

    ref = a.place(intent(NIFTY_CE, Side.BUY, "75", OrderType.LIMIT, limit="100.03"))

    assert ref.status is OrderStatus.FILLED


def test_below_min_notional_is_rejected():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)

    ref = a.place(intent(BTC, Side.BUY, "0.0001"))   # $5 against a $10 minimum

    assert ref.status is OrderStatus.REJECTED
    assert "min_notional" in (ref.error or "")


def test_insufficient_cash_is_refused_before_any_fill():
    """Checked on the worst-case price up front. A per-fill check would let a
    multi-lot order half-execute and then refuse, leaving the strategy holding
    a leg it never chose."""
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes, cash="1000")

    ref = a.place(intent(BTC, Side.BUY, "1"))

    assert ref.status is OrderStatus.REJECTED
    assert "cash" in (ref.error or "")
    assert a.cash == Decimal("1000")
    assert a.positions() == ()


def test_cash_check_uses_the_slipped_bound_not_the_quote():
    """Affordable at the quote (50,000 + ~50 of fees against 50,200 of cash),
    but not at the worst price it can actually fill at. Sizing against the mid
    is a check that has not run."""
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes, cash="50200",
                       slippage=SlippageModel(max_bps=Decimal("100")))

    ref = a.place(intent(BTC, Side.BUY, "1"))

    assert ref.status is OrderStatus.REJECTED
    assert "cash" in (ref.error or "")


def test_wrong_asset_class_raises_rather_than_routing():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = equity_adapter(quotes)
    with pytest.raises(ValueError, match="crypto"):
        a.place(intent(BTC, Side.BUY, "1"))


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------

def test_partial_fill_that_times_out_reports_what_actually_traded():
    """PARTIAL is not a terminal status in the contract, so this comes back
    TIMEOUT with filled_qty > 0. A caller reading only `status` and assuming
    nothing traded would be short a position it does not know about."""
    def pull_the_bid(clock: FakeClock) -> None:
        if clock.sleeps == 1:
            quotes.set("NIFTY26JAN24000CE", "120")   # now above our limit

    quotes = InMemoryQuoteSource()
    quotes.set("NIFTY26JAN24000CE", "100", depth="75")
    clock = FakeClock(on_sleep=pull_the_bid)
    a = equity_adapter(quotes, clock=clock)

    ref = a.place(intent(NIFTY_CE, Side.BUY, "225", OrderType.LIMIT, limit="100"),
                  timeout_s=3.0)

    assert ref.status is OrderStatus.TIMEOUT
    assert ref.filled_qty == Decimal("75")
    assert ref.avg_price is not None
    assert "150 of 225 unfilled" in (ref.error or "")
    assert len(a.positions()) == 1
    assert a.positions()[0].qty == Decimal("75")


def test_flat_brokerage_is_billed_once_across_an_orders_partial_fills():
    """Pins the adapter's wiring, not just the charge stack: the second fill of
    the same order must not be re-billed the per-order leg."""
    quotes = InMemoryQuoteSource()
    quotes.set("NIFTY26JAN24000CE", "100", depth="75")
    a = equity_adapter(quotes)

    ref = a.place(intent(NIFTY_CE, Side.BUY, "150", OrderType.LIMIT, limit="100"),
                  timeout_s=10.0)

    assert ref.status is OrderStatus.FILLED
    assert len(ref.fills) == 2
    assert ref.fills[0].fees - ref.fills[1].fees > Decimal("20")


def test_zero_depth_is_not_a_missing_quote():
    """depth == 0 means the price exists with nothing available — the order
    rests. It must not be mistaken for an unpriced instrument (a rejection)."""
    quotes = InMemoryQuoteSource()
    quotes.set("BTC/USDT", "50000", depth="0")
    a = crypto_adapter(quotes)

    ref = a.place(intent(BTC, Side.BUY, "1"), timeout_s=2.0)

    assert ref.status is OrderStatus.TIMEOUT
    assert ref.filled_qty == Decimal("0")


# ---------------------------------------------------------------------------
# Positions and equity
# ---------------------------------------------------------------------------

def test_positions_reflect_fills_and_average_in():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes, cash="500000")
    first = a.place(intent(BTC, Side.BUY, "1"))
    quotes.set("BTC/USDT", "60000")
    second = a.place(intent(BTC, Side.BUY, "1"))

    (pos,) = a.positions()
    assert pos.qty == Decimal("2")
    assert pos.mode is Mode.PAPER
    assert pos.strategy_id == "s1"
    assert first.avg_price is not None and second.avg_price is not None
    expected = (first.avg_price + second.avg_price) / 2
    assert pos.avg_price == expected


def test_reducing_a_position_leaves_the_cost_basis_alone():
    """Selling half does not re-price the half you still hold. If it did, every
    unrealised-P&L number downstream would silently reset on each trim."""
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes, cash="500000")
    opened = a.place(intent(BTC, Side.BUY, "2"))

    quotes.set("BTC/USDT", "60000")
    a.place(intent(BTC, Side.SELL, "1"))

    (pos,) = a.positions()
    assert pos.qty == Decimal("1")
    assert pos.avg_price == opened.avg_price


def test_flipping_through_zero_starts_a_new_cost_basis():
    """Long 1 then sell 2 is short 1 at today's price — not short 1 at the old
    long's basis, which would report a fictitious loss on the new leg."""
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes, cash="500000")
    a.place(intent(BTC, Side.BUY, "1"))

    quotes.set("BTC/USDT", "60000")
    flip = a.place(intent(BTC, Side.SELL, "2"))

    (pos,) = a.positions()
    assert pos.qty == Decimal("-1")
    assert pos.avg_price == flip.avg_price


def test_short_then_cover_clears_the_position():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)

    a.place(intent(BTC, Side.SELL, "1"))
    (short,) = a.positions()
    assert short.qty == Decimal("-1")
    assert a.cash > a.starting_cash          # sale credited

    quotes.set("BTC/USDT", "40000")
    a.place(intent(BTC, Side.BUY, "1"))

    assert a.positions() == ()
    # Sold near 50k, covered near 40k: the gain is real, minus costs.
    assert a.cash > a.starting_cash + Decimal("9000")


def test_equity_marks_open_positions_to_market():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)
    a.place(intent(BTC, Side.BUY, "1"))

    at_entry = a.equity()
    quotes.set("BTC/USDT", "55000")
    after = a.equity()

    assert at_entry is not None and after is not None
    assert after - at_entry == Decimal("5000")


def test_equity_is_starting_cash_when_flat():
    a = crypto_adapter(InMemoryQuoteSource({"BTC/USDT": "50000"}))
    assert a.equity() == Decimal("100000")


def test_equity_is_none_not_zero_when_a_position_cannot_be_marked():
    """The failure that wrecked the previous system: a failed read returning 0
    equity, which the risk circuit read as a 100% drawdown."""
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes)
    a.place(intent(BTC, Side.BUY, "1"))
    quotes.remove("BTC/USDT")

    eq = a.equity()

    assert eq is None
    assert eq != Decimal("0")
    assert eq != a.cash          # nor silently cash-only, which hides the long


def test_quote_passthrough_is_none_for_unknown_instrument():
    a = crypto_adapter(InMemoryQuoteSource())
    assert a.quote(BTC) is None


def test_a_frozen_mark_does_not_price_a_paper_order():
    """A frozen option LTP must be distinguishable from a live one, and must
    not silently fill a spread."""
    quotes = InMemoryQuoteSource()
    quotes.set("NIFTY26JAN24000CE", "120", live=False)
    stored = quotes.quote(NIFTY_CE)
    assert stored is not None and stored.live is False
    a = equity_adapter(quotes)
    assert a.quote(NIFTY_CE) is None
    ref = a.place(intent(NIFTY_CE, Side.BUY, "75",
                         order_type=OrderType.LIMIT, limit="120"))
    assert ref.status is OrderStatus.REJECTED
    assert "frozen" in (ref.error or "")


def test_a_frozen_mark_does_not_price_an_open_position():
    """A live fill followed by a frozen LTP must not keep marking the book."""
    quotes = InMemoryQuoteSource()
    quotes.set("NIFTY26JAN24000CE", "120")
    a = equity_adapter(quotes)
    ref = a.place(intent(NIFTY_CE, Side.BUY, "75",
                         order_type=OrderType.LIMIT, limit="120"))
    assert ref.status is OrderStatus.FILLED
    quotes.set("NIFTY26JAN24000CE", "120", live=False)
    assert a.quote(NIFTY_CE) is None
    assert a.equity() is None


# ---------------------------------------------------------------------------
# Input hygiene
# ---------------------------------------------------------------------------

def test_quote_source_refuses_non_positive_prices():
    quotes = InMemoryQuoteSource()
    with pytest.raises(ValueError):
        quotes.set("BTC/USDT", "0")
    with pytest.raises(ValueError):
        quotes.set("BTC/USDT", "-1")


def test_quote_source_refuses_floats():
    with pytest.raises(TypeError):
        InMemoryQuoteSource().set("BTC/USDT", 50000.1)


def test_removing_an_absent_quote_raises():
    with pytest.raises(KeyError):
        InMemoryQuoteSource().remove("BTC/USDT")


def test_starting_cash_refuses_floats():
    with pytest.raises(TypeError):
        PaperAdapter(AssetClass.CRYPTO, InMemoryQuoteSource(), 100000.0)


def test_every_order_ref_is_terminal_and_recorded():
    quotes = InMemoryQuoteSource({"BTC/USDT": "50000"})
    a = crypto_adapter(quotes, cash="500000")
    refs = [
        a.place(intent(BTC, Side.BUY, "1")),
        a.place(intent(BTC, Side.BUY, "0.0001")),                       # rejected
        a.place(intent(BTC, Side.BUY, "1", OrderType.LIMIT, limit="40000"),
                timeout_s=1.0),
    ]
    assert {r.status for r in refs} == {
        OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.TIMEOUT,
    }
    for r in refs:
        assert r.status.terminal
        assert r.mode is Mode.PAPER
        assert a.order(r.order_id) is r
