"""
OkxAdapter tests.

Every safety rule in adapters/okx.py has a test here that fails if the rule is
deleted. The exchange is a fake: no network, no credentials, no venue.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Optional

import pytest

from adapters.okx import (
    CostBasisUnavailable, EquityBreakdown, MarketsUnavailable, OkxAdapter,
    OrderStateUnknown, OrderSubmitFailed, UnknownOrder,
)
from core.protocols import BrokerAdapter
from core.types import (
    AssetClass, Instrument, Intent, Mode, OrderStatus, OrderType, Side,
)

D = Decimal
UTC = dt.timezone.utc


# --------------------------------------------------------------- fake exchange

class Boom(Exception):
    """A transport failure."""


def spot(symbol: str, *, amount_step: str = "0.00000001", price_step: str = "0.01",
         min_amount: Optional[str] = None, min_cost: Optional[str] = None,
         active: bool = True, kind: str = "spot") -> dict[str, Any]:
    limits: dict[str, Any] = {}
    if min_amount is not None:
        limits["amount"] = {"min": float(min_amount)}
    if min_cost is not None:
        limits["cost"] = {"min": float(min_cost)}
    return {
        "symbol": symbol, "spot": kind == "spot", "swap": kind == "swap",
        "active": active,
        "precision": {"amount": float(amount_step), "price": float(price_step)},
        "limits": limits,
    }


class FakeExchange:
    """Mimics the slice of the ccxt surface OkxAdapter uses."""

    def __init__(self, markets=None, tickers=None, balance=None,
                 order_states=None, created=None):
        self._markets = markets if markets is not None else {}
        self.markets = self._markets
        self.tickers = tickers or {}
        self.balance = balance
        self.order_states = list(order_states or [])
        self.created = created or {"id": "OID-1"}

        self.load_markets_error: Optional[Exception] = None
        self.balance_error: Optional[Exception] = None
        self.ticker_error: Optional[Exception] = None
        self.create_error: Optional[Exception] = None
        self.fetch_order_error: Optional[Exception] = None
        self.cancel_error: Optional[Exception] = None

        self.create_calls: list[tuple] = []
        self.cancel_calls: list[tuple] = []
        self.ticker_calls: list[str] = []
        self.fetch_order_calls: list[tuple] = []

    # -- markets
    def load_markets(self, reload: bool = False):
        if self.load_markets_error:
            raise self.load_markets_error
        return self._markets

    def amount_to_precision(self, symbol, amount):
        step = D(str(self._markets[symbol]["precision"]["amount"]))
        return str((D(str(amount)) / step).to_integral_value(rounding="ROUND_DOWN") * step)

    def price_to_precision(self, symbol, price):
        step = D(str(self._markets[symbol]["precision"]["price"]))
        return str((D(str(price)) / step).to_integral_value(rounding="ROUND_DOWN") * step)

    # -- data
    def fetch_ticker(self, symbol):
        self.ticker_calls.append(symbol)
        if self.ticker_error:
            raise self.ticker_error
        return self.tickers[symbol]

    def fetch_balance(self):
        if self.balance_error:
            raise self.balance_error
        return self.balance

    # -- trading
    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.create_calls.append((symbol, type_, side, amount, price, params))
        if self.create_error:
            raise self.create_error
        return self.created

    def fetch_order(self, order_id, symbol=None):
        self.fetch_order_calls.append((order_id, symbol))
        if self.fetch_order_error:
            raise self.fetch_order_error
        if len(self.order_states) > 1:
            return self.order_states.pop(0)
        return self.order_states[0]

    def cancel_order(self, order_id, symbol=None):
        self.cancel_calls.append((order_id, symbol))
        if self.cancel_error:
            raise self.cancel_error
        return {"id": order_id}


class FakeClock:
    """Monotonic time that only advances when the adapter sleeps."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def btc(broker_id: str = "BTC/USDT", **kw) -> Instrument:
    return Instrument(symbol="BTC", asset_class=AssetClass.CRYPTO,
                      broker_id=broker_id, qty_precision=8, **kw)


def intent(instr: Optional[Instrument] = None, *, side=Side.BUY, qty="1",
           order_type=OrderType.MARKET, limit_price: Optional[str] = "100",
           key="k1") -> Intent:
    return Intent(
        instrument=instr or btc(), side=side, qty=D(qty), order_type=order_type,
        strategy_id="s1", reason="test", idempotency_key=key,
        limit_price=D(limit_price) if limit_price is not None else None,
    )


def adapter(exchange: FakeExchange, clock: Optional[FakeClock] = None,
            **kw) -> OkxAdapter:
    clock = clock or FakeClock()
    kw.setdefault("mode", Mode.LIVE)
    kw.setdefault("poll_interval_s", 1.0)
    return OkxAdapter(exchange, monotonic=clock.monotonic, sleep=clock.sleep,
                      now=lambda: dt.datetime(2026, 1, 1, tzinfo=UTC), **kw)


FILLED_ORDER = {
    "id": "OID-1", "symbol": "BTC/USDT", "status": "closed", "amount": 1.0,
    "filled": 1.0, "average": 100.5, "cost": 100.5, "timestamp": 1767225600000,
    "fee": {"cost": 0.1005, "currency": "USDT"},
}


# ------------------------------------------------------------------- protocol

def test_adapter_satisfies_the_broker_protocol():
    assert isinstance(adapter(FakeExchange({"BTC/USDT": spot("BTC/USDT")})), BrokerAdapter)


def test_paper_mode_refused_without_sandbox():
    # A PAPER stamp on a real venue fill is how a live book is recorded as paper.
    with pytest.raises(ValueError, match="PAPER"):
        OkxAdapter(FakeExchange(), mode=Mode.PAPER)
    OkxAdapter(FakeExchange(), mode=Mode.PAPER, sandbox=True)  # explicit demo venue


# -------------------------------------------------------------------- markets

def test_empty_market_table_raises_rather_than_delisting_everything():
    ex = FakeExchange({})
    with pytest.raises(MarketsUnavailable):
        adapter(ex).resolve_symbol(btc())


def test_market_load_failure_is_distinguishable_from_not_listed():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    ex.load_markets_error = Boom("503")
    with pytest.raises(Boom):
        adapter(ex).resolve_symbol(btc())


def test_broker_id_is_used_when_it_is_listed():
    ex = FakeExchange({"BTC/USDC": spot("BTC/USDC"), "BTC/USDT": spot("BTC/USDT")})
    assert adapter(ex).resolve_symbol(btc("BTC/USDC")) == "BTC/USDC"


def test_quote_fallback_chain_usdt_then_usd_then_usdc():
    ex = FakeExchange({"BTC/USDC": spot("BTC/USDC")})
    assert adapter(ex).resolve_symbol(btc("BTC/USDT")) == "BTC/USDC"

    ex2 = FakeExchange({"BTC/USD": spot("BTC/USD"), "BTC/USDC": spot("BTC/USDC")})
    assert adapter(ex2).resolve_symbol(btc("BTC/USDT")) == "BTC/USD"


def test_delisted_market_is_skipped_not_traded():
    ex = FakeExchange({
        "BTC/USDT": spot("BTC/USDT", active=False),
        "BTC/USD": spot("BTC/USD"),
    })
    assert adapter(ex).resolve_symbol(btc()) == "BTC/USD"


def test_a_swap_never_answers_a_spot_intent():
    ex = FakeExchange({
        "BTC/USDT:USDT": spot("BTC/USDT:USDT", kind="swap"),
        "BTC/USD": spot("BTC/USD"),
    })
    assert adapter(ex).resolve_symbol(btc("BTC/USDT:USDT")) == "BTC/USD"


def test_unlisted_name_returns_none_so_a_batch_survives_it():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    a = adapter(ex)
    assert a.resolve_symbol(btc("LUNA/USDT")) is None       # dead name
    assert a.resolve_symbol(btc()) == "BTC/USDT"            # batch continues


# ------------------------------------------------------- quantization / limits

def test_amount_is_quantized_with_the_venues_precision():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.0}},
                      order_states=[FILLED_ORDER])
    adapter(ex).place(intent(qty="0.123456789"))
    sent = ex.create_calls[0][3]
    assert sent == "0.12345678"
    assert not isinstance(sent, float)


def test_create_order_is_not_handed_a_float():
    """float() at the ccxt boundary is silent rounding. The already-quantized
    size must cross as a string (or Decimal), never as a binary float."""
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT", amount_step="0.001",
                                        price_step="0.1")},
                      order_states=[FILLED_ORDER])
    adapter(ex).place(intent(qty="1.2345", order_type=OrderType.LIMIT,
                             limit_price="100.19"))
    amount, price = ex.create_calls[0][3], ex.create_calls[0][4]
    assert amount == "1.234"
    assert price == "100.1"
    assert isinstance(amount, str) and isinstance(price, str)


def test_missing_market_precision_refuses_rather_than_float_rounding():
    markets = {"BTC/USDT": spot("BTC/USDT")}
    markets["BTC/USDT"]["precision"] = {}
    ex = FakeExchange(markets, tickers={"BTC/USDT": {"last": 100.0}})
    ref = adapter(ex).place(intent(qty="1"))
    assert ref.status is OrderStatus.REJECTED
    assert not ex.create_calls
    assert "precision" in (ref.error or "")


def test_dust_that_quantizes_to_zero_is_refused():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT", amount_step="0.001")},
                      tickers={"BTC/USDT": {"last": 100.0}})
    ref = adapter(ex).place(intent(qty="0.0004"))
    assert ref.status is OrderStatus.REJECTED
    assert not ex.create_calls


def test_min_amount_comes_from_the_market_not_a_constant():
    strict = FakeExchange({"BTC/USDT": spot("BTC/USDT", min_amount="1")},
                          tickers={"BTC/USDT": {"last": 100.0}})
    assert adapter(strict).place(intent(qty="0.5")).status is OrderStatus.REJECTED
    assert not strict.create_calls

    loose = FakeExchange({"BTC/USDT": spot("BTC/USDT", min_amount="0.0001")},
                         tickers={"BTC/USDT": {"last": 100.0}},
                         order_states=[FILLED_ORDER])
    assert adapter(loose).place(intent(qty="0.5")).status is OrderStatus.FILLED


def test_min_cost_comes_from_the_market_not_a_constant():
    # Same order, same code path; only the venue's published minimum differs.
    strict = FakeExchange({"BTC/USDT": spot("BTC/USDT", min_cost="10")},
                          tickers={"BTC/USDT": {"last": 100.0}})
    ref = adapter(strict).place(intent(qty="0.05"))          # 5 USDT notional
    assert ref.status is OrderStatus.REJECTED and "below the minimum" in ref.error
    assert not strict.create_calls

    loose = FakeExchange({"BTC/USDT": spot("BTC/USDT", min_cost="1")},
                         tickers={"BTC/USDT": {"last": 100.0}},
                         order_states=[FILLED_ORDER])
    assert adapter(loose).place(intent(qty="0.05")).status is OrderStatus.FILLED


def test_instrument_min_notional_tightens_but_never_loosens_the_venue_minimum():
    markets = {"BTC/USDT": spot("BTC/USDT", min_cost="1")}
    tightened = btc(min_notional=D("50"))
    ex = FakeExchange(markets, tickers={"BTC/USDT": {"last": 100.0}})
    assert adapter(ex).place(intent(tightened, qty="0.05")).status is OrderStatus.REJECTED

    # The caller asking for 0 must not disable the venue's own floor.
    ex2 = FakeExchange({"BTC/USDT": spot("BTC/USDT", min_cost="10")},
                       tickers={"BTC/USDT": {"last": 100.0}})
    assert adapter(ex2).place(intent(btc(min_notional=D("0")), qty="0.05")).status \
        is OrderStatus.REJECTED


# ------------------------------------------------------------------ drift guard

def test_market_order_aborts_when_price_ran_away_from_the_reference():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 105.0}})
    ref = adapter(ex).place(intent(limit_price="100"))       # +5% vs 0.5% cap
    assert ref.status is OrderStatus.REJECTED
    assert "drift" in ref.error
    assert not ex.create_calls, "a drifted market order must never be sent"


def test_drift_guard_is_symmetric():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 95.0}})
    assert adapter(ex).place(intent(limit_price="100")).status is OrderStatus.REJECTED


def test_market_order_inside_the_drift_limit_is_sent():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.4}},
                      order_states=[FILLED_ORDER])
    ref = adapter(ex).place(intent(limit_price="100"))
    assert ref.status is OrderStatus.FILLED and ex.create_calls


def test_drift_limit_is_configurable():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 105.0}},
                      order_states=[FILLED_ORDER])
    ref = adapter(ex, max_drift_pct=D("10")).place(intent(limit_price="100"))
    assert ref.status is OrderStatus.FILLED


def test_market_order_without_a_reference_price_is_refused():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.0}})
    ref = adapter(ex).place(intent(limit_price=None))
    assert ref.status is OrderStatus.REJECTED
    assert "drift guard" in ref.error
    assert not ex.create_calls


def test_unpriceable_market_blocks_the_drift_guard_and_the_order():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    ex.ticker_error = Boom("ticker 500")
    ref = adapter(ex).place(intent(limit_price="100"))
    assert ref.status is OrderStatus.REJECTED
    assert not ex.create_calls, "no mark means no market order"


def test_limit_orders_skip_the_drift_guard():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT, limit_price="100"))
    assert ref.status is OrderStatus.FILLED
    assert ex.ticker_calls == [], "a limit price is its own bound"
    assert ex.create_calls[0][4] == "100"


def test_unlisted_symbol_is_rejected_without_touching_the_venue():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    ref = adapter(ex).place(intent(btc("LUNA/USDT")))
    assert ref.status is OrderStatus.REJECTED and ref.order_id.startswith("unplaced:")
    assert not ex.create_calls


# ----------------------------------------------------------------- submission

def test_market_buy_states_the_target_currency_explicitly():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.0}},
                      order_states=[FILLED_ORDER])
    adapter(ex).place(intent(side=Side.BUY, limit_price="100"))
    # Without tgtCcy=base_ccy OKX reads `amount` as quote units on a spot buy.
    assert ex.create_calls[0][5]["tgtCcy"] == "base_ccy"


def test_client_order_id_is_sanitised_for_okx():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    adapter(ex).place(intent(order_type=OrderType.LIMIT, key="s1:btc-2026/01/01#7"))
    assert ex.create_calls[0][5]["clOrdId"] == "s1btc202601017"


def test_submission_failure_raises_and_is_never_reported_as_rejected():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    ex.create_error = Boom("timeout after send")
    # REJECTED would read as "no order exists" and invite a doubling retry.
    with pytest.raises(OrderSubmitFailed):
        adapter(ex).place(intent(order_type=OrderType.LIMIT))


def test_missing_order_id_raises_rather_than_returning_an_untrackable_ref():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, created={"info": {}})
    with pytest.raises(OrderSubmitFailed):
        adapter(ex).place(intent(order_type=OrderType.LIMIT))


def test_equity_intent_from_another_asset_class_is_refused():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    equity_instrument = Instrument(symbol="RELIANCE", asset_class=AssetClass.EQUITY,
                                   broker_id="2885")
    with pytest.raises(ValueError):
        adapter(ex).place(intent(equity_instrument, order_type=OrderType.LIMIT))


# -------------------------------------------------------------- terminal state

def test_place_blocks_until_the_order_is_terminal():
    working = {"id": "OID-1", "symbol": "BTC/USDT", "status": "open",
               "amount": 1.0, "filled": 0.0}
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      order_states=[working, working, FILLED_ORDER])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT), timeout_s=30)
    assert ref.status is OrderStatus.FILLED
    assert len(ex.fetch_order_calls) == 3
    assert ref.filled_qty == D("1.0") and ref.avg_price == D("100.5")
    assert ref.fills[0].fees == D("0.1005")
    assert ref.fills[0].at == dt.datetime(2026, 1, 1, tzinfo=UTC)
    assert ref.fills[0].mode is Mode.LIVE


def test_partial_fill_that_the_venue_closed_keeps_its_real_filled_quantity():
    part = {"id": "OID-1", "symbol": "BTC/USDT", "status": "closed", "amount": 1.0,
            "filled": 0.4, "average": 100.0, "cost": 40.0}
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[part])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT))
    assert ref.status is OrderStatus.CANCELLED       # terminal, not "partial"
    assert ref.filled_qty == D("0.4") and ref.avg_price == D("100.0")


def test_timeout_cancels_and_reports_what_actually_filled():
    working = {"id": "OID-1", "symbol": "BTC/USDT", "status": "open",
               "amount": 1.0, "filled": 0.4, "average": 100.0}
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[working])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT), timeout_s=3)
    assert ref.status is OrderStatus.TIMEOUT
    assert ref.filled_qty == D("0.4")
    assert ex.cancel_calls == [("OID-1", "BTC/USDT")]
    assert "still open" in ref.error


def test_unreadable_order_state_raises_with_the_id_never_a_clean_ref():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    ex.fetch_order_error = Boom("gateway")
    with pytest.raises(OrderStateUnknown) as caught:
        adapter(ex).place(intent(order_type=OrderType.LIMIT), timeout_s=3)
    assert caught.value.order_id == "OID-1"


def test_a_transient_poll_error_is_retried_not_treated_as_a_verdict():
    class Flaky(FakeExchange):
        calls = 0

        def fetch_order(self, order_id, symbol=None):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise Boom("blip")
            return FILLED_ORDER

    ex = Flaky({"BTC/USDT": spot("BTC/USDT")})
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT), timeout_s=10)
    assert ref.status is OrderStatus.FILLED


def test_rejected_by_the_venue_is_reported_as_rejected():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[
        {"id": "OID-1", "symbol": "BTC/USDT", "status": "rejected",
         "amount": 1.0, "filled": 0.0},
    ])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT))
    assert ref.status is OrderStatus.REJECTED and ref.filled_qty == 0
    assert ref.fills == () and ref.avg_price is None


def test_a_fill_without_a_price_reports_unknown_not_zero():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[
        {"id": "OID-1", "symbol": "BTC/USDT", "status": "closed",
         "amount": 1.0, "filled": 1.0, "price": 999.0},   # price is NOT a fill
    ])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT))
    assert ref.filled_qty == D("1.0")
    assert ref.avg_price is None, "a reference price must never become a fill price"
    assert ref.fills == () and "no average price" in ref.error


def test_average_is_derived_from_cost_when_the_venue_omits_it():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[
        {"id": "OID-1", "symbol": "BTC/USDT", "status": "closed",
         "amount": 2.0, "filled": 2.0, "cost": 201.0},
    ])
    ref = adapter(ex).place(intent(qty="2", order_type=OrderType.LIMIT))
    assert ref.avg_price == D("100.5")


def test_per_trade_fills_are_parsed():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[
        {"id": "OID-1", "symbol": "BTC/USDT", "status": "closed", "amount": 1.0,
         "filled": 1.0, "average": 100.5, "trades": [
             {"amount": 0.6, "price": 100.0, "timestamp": 1767225600000,
              "fee": {"cost": 0.06, "currency": "USDT"}},
             {"amount": 0.4, "price": 101.25, "timestamp": 1767225601000,
              "fee": {"cost": 0.0405, "currency": "USDT"}},
         ]},
    ])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT))
    assert [f.qty for f in ref.fills] == [D("0.6"), D("0.4")]
    assert ref.fills[1].price == D("101.25")


def test_a_fee_charged_in_the_bought_coin_is_converted_not_added_to_dollars():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[
        {"id": "OID-1", "symbol": "BTC/USDT", "status": "closed", "amount": 1.0,
         "filled": 1.0, "average": 100.0,
         "fee": {"cost": -0.001, "currency": "BTC"}},     # OKX reports a deduction
    ])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT))
    assert ref.fills[0].fees == D("0.1")                  # 0.001 BTC * 100 USDT


def test_a_fee_in_an_unconvertible_currency_is_flagged_not_silently_dropped():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[
        {"id": "OID-1", "symbol": "BTC/USDT", "status": "closed", "amount": 1.0,
         "filled": 1.0, "average": 100.0,
         "fee": {"cost": 0.4, "currency": "OKB"}},
    ])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT))
    assert ref.status is OrderStatus.FILLED               # the fill is still real
    assert "OKB" in ref.error and ref.fills[0].fees == D("0")


# ---------------------------------------------------------------- cancellation

def _placed(ex) -> OkxAdapter:
    a = adapter(ex)
    a.place(intent(order_type=OrderType.LIMIT))
    return a


def test_cancel_of_an_unknown_order_raises_rather_than_returning_false():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    with pytest.raises(UnknownOrder):
        adapter(ex).cancel("OID-999")


def test_cancel_is_confirmed_against_the_venue_not_the_call_result():
    still_open = {"id": "OID-1", "symbol": "BTC/USDT", "status": "open",
                  "amount": 1.0, "filled": 0.0}
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      order_states=[{"id": "OID-1", "symbol": "BTC/USDT",
                                     "status": "closed", "amount": 1.0,
                                     "filled": 1.0, "average": 100.0}])
    a = _placed(ex)
    ex.order_states = [still_open]
    assert a.cancel("OID-1") is False, "cancel_order returning ok is not proof"


def test_cancel_true_only_when_the_venue_shows_it_cancelled():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    a = _placed(ex)
    ex.order_states = [{"id": "OID-1", "symbol": "BTC/USDT", "status": "canceled",
                        "amount": 1.0, "filled": 0.0}]
    assert a.cancel("OID-1") is True


def test_an_order_that_filled_before_the_cancel_is_not_reported_as_cancelled():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    a = _placed(ex)
    assert a.cancel("OID-1") is False
    assert a.cancel_detail("OID-1") is OrderStatus.FILLED


def test_unconfirmable_cancel_raises():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    a = _placed(ex)
    ex.fetch_order_error = Boom("gateway")
    with pytest.raises(OrderStateUnknown):
        a.cancel("OID-1")


# --------------------------------------------------------------------- quotes

def test_quote_returns_none_for_an_unlisted_name():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    assert adapter(ex).quote(btc("LUNA/USDT")) is None


def test_quote_falls_back_to_the_mid_when_last_is_missing():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"bid": 99.0, "ask": 101.0}})
    assert adapter(ex).quote(btc()) == D("100")


def test_quote_propagates_a_transport_failure_rather_than_faking_calm():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    ex.ticker_error = Boom("500")
    with pytest.raises(Boom):
        adapter(ex).quote(btc())


# --------------------------------------------------------------------- equity

MARKETS = {
    "BTC/USDT": spot("BTC/USDT"),
    "ETH/USDT": spot("ETH/USDT"),
}
TICKERS = {"BTC/USDT": {"last": 60000.0}, "ETH/USDT": {"last": 3000.0}}


def test_equity_sums_crypto_and_stables():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "USDC": 500.0, "BTC": 0.5}})
    assert adapter(ex).equity() == D("31500.0")


def test_a_pegged_fiat_balance_with_no_crypto_pair_is_priced_not_nulled():
    # The regression: AED nulled total equity and blocked every buy.
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "AED": 3672.5}})
    breakdown = adapter(ex).equity_breakdown()
    assert breakdown.total == D("2000")
    assert [c.source for c in breakdown.components if c.currency == "AED"] == ["fiat-peg"]


def test_a_floating_fiat_is_reported_unpriceable_rather_than_guessed():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "EUR": 900.0}})
    breakdown = adapter(ex).equity_breakdown()
    assert breakdown.total is None and "EUR" in breakdown.reason


def test_equity_is_none_when_anything_is_unpriceable_never_a_short_total():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "WEIRD": 42.0}})
    a = adapter(ex)
    assert a.equity() is None                       # not 1000
    assert "WEIRD" in a.equity_breakdown().reason


def test_a_short_total_cannot_even_be_constructed():
    with pytest.raises(ValueError):
        EquityBreakdown("USDT", D("1000"), unpriceable=(("WEIRD", D("42")),))


def test_an_unknown_equity_must_carry_its_reason():
    with pytest.raises(ValueError):
        EquityBreakdown("USDT", None)


def test_equity_read_failure_is_none_not_zero():
    ex = FakeExchange(MARKETS, tickers=TICKERS, balance={"total": {"USDT": 1.0}})
    ex.balance_error = Boom("401")
    a = adapter(ex)
    assert a.equity() is None                       # zero would read as -100%
    assert "fetch_balance failed" in a.equity_breakdown().reason


def test_an_empty_balance_map_is_treated_as_a_failed_read():
    ex = FakeExchange(MARKETS, tickers=TICKERS, balance={"total": {}})
    assert adapter(ex).equity() is None


def test_a_ticker_failure_while_pricing_equity_nulls_the_total_not_the_balance():
    ex = FakeExchange(MARKETS, balance={"total": {"USDT": 1000.0, "BTC": 0.5}})
    ex.ticker_error = Boom("500")
    a = adapter(ex)
    assert a.equity() is None
    assert "BTC" in a.equity_breakdown().reason


def test_zero_balances_are_ignored():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "DOGE": 0.0}})
    assert adapter(ex).equity() == D("1000.0")


def test_explicitly_ignored_currencies_are_recorded_not_silently_dropped():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "SCAMAIRDROP": 1e9}})
    breakdown = adapter(ex, ignore_currencies=["SCAMAIRDROP"]).equity_breakdown()
    assert breakdown.total == D("1000.0")
    assert breakdown.excluded == (("SCAMAIRDROP", D("1000000000.0")),)


def test_an_inverse_market_prices_the_balance():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT"), "USDT/TRY": spot("USDT/TRY")},
                      tickers={"USDT/TRY": {"last": 40.0}},
                      balance={"total": {"USDT": 100.0, "TRY": 400.0}})
    assert adapter(ex).equity() == D("110")


# ------------------------------------------------------------------ positions

def test_positions_refuse_to_invent_a_cost_basis():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "BTC": 0.5}})
    with pytest.raises(CostBasisUnavailable):
        adapter(ex).positions()          # an empty list would read as "flat"


def test_positions_use_the_supplied_cost_basis_and_exclude_cash():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "AED": 10.0, "BTC": 0.5}})
    a = adapter(ex, cost_basis=lambda c: D("58000") if c == "BTC" else None)
    positions = a.positions()
    assert [p.instrument.symbol for p in positions] == ["BTC"]
    assert positions[0].qty == D("0.5") and positions[0].avg_price == D("58000")
    assert positions[0].mode is Mode.LIVE


def test_a_holding_with_no_cost_basis_raises_rather_than_disappearing():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"BTC": 0.5, "ETH": 2.0}})
    a = adapter(ex, cost_basis=lambda c: D("58000") if c == "BTC" else None)
    with pytest.raises(CostBasisUnavailable, match="ETH"):
        a.positions()


def test_holdings_report_raw_venue_truth():
    ex = FakeExchange(MARKETS, tickers=TICKERS,
                      balance={"total": {"USDT": 1000.0, "BTC": 0.5, "DOGE": 0.0}})
    assert adapter(ex).holdings() == {"USDT": D("1000.0"), "BTC": D("0.5")}


# ------------------------------------------------------------------ wiring

def test_importing_the_adapter_needs_neither_ccxt_nor_credentials():
    import ast
    import adapters.okx as module
    source = open(module.__file__).read()
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert "ccxt" not in imported, "the exchange is injected, never imported here"
    assert "environ" not in source and "getenv" not in source


def test_a_clean_fill_carries_no_error_note():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT))
    assert ref.error is None


def test_markets_already_cached_on_the_exchange_are_used():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    ex.load_markets = lambda *a, **k: {}      # ccxt returns {} once already loaded
    assert adapter(ex).resolve_symbol(btc()) == "BTC/USDT"


def test_the_reference_price_requirement_can_be_relaxed_deliberately():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.0}},
                      order_states=[FILLED_ORDER])
    a = adapter(ex, require_reference_price=False)
    assert a.place(intent(limit_price=None)).status is OrderStatus.FILLED


def test_drift_check_refuses_a_nonsense_reference():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.0}})
    with pytest.raises(ValueError):
        adapter(ex).drift_check("BTC/USDT", D("0"))


def test_a_zero_mark_never_passes_the_drift_guard():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 0.0}})
    ref = adapter(ex).place(intent(limit_price="100"))
    assert ref.status is OrderStatus.REJECTED and not ex.create_calls


def test_no_order_minimum_is_hardcoded_in_the_adapter():
    # A market publishing no limits must not acquire one from this file.
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.0}},
                      order_states=[FILLED_ORDER])
    assert adapter(ex).place(intent(qty="0.00001")).status is OrderStatus.FILLED


def test_timeout_reports_a_venue_cancel_that_actually_landed():
    working = {"id": "OID-1", "symbol": "BTC/USDT", "status": "open",
               "amount": 1.0, "filled": 0.0}
    cancelled = {"id": "OID-1", "symbol": "BTC/USDT", "status": "canceled",
                 "amount": 1.0, "filled": 0.0}
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      order_states=[working, working, working, cancelled])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT), timeout_s=2)
    assert ref.status is OrderStatus.CANCELLED
    assert ex.cancel_calls == [("OID-1", "BTC/USDT")]


def test_a_failed_poll_window_still_yields_the_venues_own_final_state():
    """Polling failed throughout, but the confirmation read worked: the ref must
    describe what OKX actually holds, not what the failed polls implied."""
    class LateReader(FakeExchange):
        polls = 0

        def fetch_order(self, order_id, symbol=None):
            LateReader.polls += 1
            if LateReader.polls <= 3:
                raise Boom("gateway")
            return {"id": "OID-1", "symbol": "BTC/USDT", "status": "closed",
                    "amount": 1.0, "filled": 0.4, "average": 100.0}

    ex = LateReader({"BTC/USDT": spot("BTC/USDT")})
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT), timeout_s=2)
    assert ref.status is OrderStatus.CANCELLED and ref.filled_qty == D("0.4")


def test_a_poll_window_lost_entirely_to_an_outage_still_raises():
    """The confirmation read must fail too before the state counts as unknown."""
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    ex.fetch_order_error = Boom("gateway")
    with pytest.raises(OrderStateUnknown):
        adapter(ex).place(intent(order_type=OrderType.LIMIT), timeout_s=2)


def test_a_holding_listed_only_against_usd_still_resolves():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT"), "SOL/USD": spot("SOL/USD")},
                      tickers={"BTC/USDT": {"last": 60000.0},
                               "SOL/USD": {"last": 150.0}},
                      balance={"total": {"SOL": 4.0}})
    a = adapter(ex, cost_basis=lambda c: D("120"))
    assert a.positions()[0].instrument.broker_id == "SOL/USD"


def test_a_coin_listed_only_against_usd_is_priced_not_treated_as_unpriceable():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT"), "SOL/USD": spot("SOL/USD")},
                      tickers={"SOL/USD": {"last": 150.0}},
                      balance={"total": {"USDT": 100.0, "SOL": 4.0}})
    breakdown = adapter(ex).equity_breakdown()
    assert breakdown.total == D("700")
    assert [c.source for c in breakdown.components if c.currency == "SOL"] \
        == ["market:SOL/USD"]


def test_a_bridge_with_a_broken_ticker_still_nulls_the_total():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT"), "SOL/USD": spot("SOL/USD")},
                      tickers={"SOL/USD": {}},
                      balance={"total": {"USDT": 100.0, "SOL": 4.0}})
    assert adapter(ex).equity() is None
