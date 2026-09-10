"""
KrakenAdapter contract tests.

The exchange is a fake: no network, no credentials, no venue. These fail if
the safety rules in adapters/kraken.py are deleted. They do not touch OkxAdapter.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Optional

import pytest

from adapters.kraken import (
    FundsMovementForbidden, KrakenAdapter, MarketsUnavailable,
    OrderSubmitFailed, UnknownOrder, clip_spot_sell_qty, is_funds_movement,
)
from core.protocols import BrokerAdapter
from core.types import (
    AssetClass, Instrument, Intent, Mode, OrderStatus, OrderType, Side,
)
from tests.test_okx_adapter import (
    FILLED_ORDER, FakeClock, FakeExchange, btc, intent, spot,
)

D = Decimal
UTC = dt.timezone.utc


def adapter(exchange: FakeExchange, clock: Optional[FakeClock] = None,
            **kw) -> KrakenAdapter:
    clock = clock or FakeClock()
    kw.setdefault("mode", Mode.LIVE)
    kw.setdefault("poll_interval_s", 1.0)
    return KrakenAdapter(
        exchange, monotonic=clock.monotonic, sleep=clock.sleep,
        now=lambda: dt.datetime(2026, 1, 1, tzinfo=UTC), **kw,
    )


def test_adapter_satisfies_the_broker_protocol():
    assert isinstance(
        adapter(FakeExchange({"BTC/USDT": spot("BTC/USDT")})), BrokerAdapter)


def test_paper_mode_refused_without_sandbox():
    with pytest.raises(ValueError, match="PAPER"):
        KrakenAdapter(FakeExchange(), mode=Mode.PAPER)
    KrakenAdapter(FakeExchange(), mode=Mode.PAPER, sandbox=True)


def test_a_swap_never_answers_a_spot_intent():
    ex = FakeExchange({
        "BTC/USDT:USDT": spot("BTC/USDT:USDT", kind="swap"),
        "BTC/USD": spot("BTC/USD"),
    })
    assert adapter(ex).resolve_symbol(btc("BTC/USDT:USDT")) == "BTC/USD"


def test_empty_market_table_raises_rather_than_delisting_everything():
    with pytest.raises(MarketsUnavailable):
        adapter(FakeExchange({})).resolve_symbol(btc())


def test_market_order_without_a_reference_price_is_refused():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.0}})
    ref = adapter(ex).place(intent(limit_price=None))
    assert ref.status is OrderStatus.REJECTED
    assert "drift guard" in (ref.error or "")
    assert not ex.create_calls


def test_place_returns_a_terminal_ref_not_fire_and_forget():
    working = {"id": "OID-1", "symbol": "BTC/USDT", "status": "open",
               "amount": 1.0, "filled": 0.0}
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      order_states=[working, working, FILLED_ORDER])
    ref = adapter(ex).place(intent(order_type=OrderType.LIMIT), timeout_s=30)
    assert ref.status is OrderStatus.FILLED
    assert ref.status.terminal
    assert len(ex.fetch_order_calls) == 3


def test_market_buy_does_not_send_okx_tgtccy():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")},
                      tickers={"BTC/USDT": {"last": 100.0}},
                      order_states=[FILLED_ORDER])
    adapter(ex).place(intent(side=Side.BUY, limit_price="100"))
    params = ex.create_calls[0][5]
    assert "tgtCcy" not in params
    assert "userref" in params


def test_submission_failure_raises_and_is_never_reported_as_rejected():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    ex.create_error = RuntimeError("timeout after send")
    with pytest.raises(OrderSubmitFailed):
        adapter(ex).place(intent(order_type=OrderType.LIMIT))


def test_equity_intent_from_another_asset_class_is_refused():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")})
    equity = Instrument(symbol="RELIANCE", asset_class=AssetClass.EQUITY,
                        broker_id="2885")
    with pytest.raises(ValueError):
        adapter(ex).place(intent(equity, order_type=OrderType.LIMIT))


def test_cancel_of_an_unknown_order_raises_rather_than_returning_false():
    ex = FakeExchange({"BTC/USDT": spot("BTC/USDT")}, order_states=[FILLED_ORDER])
    with pytest.raises(UnknownOrder):
        adapter(ex).cancel("OID-999")


def test_withdraw_is_classified_as_funds_movement():
    assert is_funds_movement("withdraw")
    assert is_funds_movement("private_post_withdraw")
    assert is_funds_movement("transferFunds")
    assert is_funds_movement("earn_allocate")
    assert not is_funds_movement("create_order")
    assert not is_funds_movement("fetch_balance")


def test_funds_move_guard_blocks_withdraw_transfer_and_earn():
    class FundMover:
        def __init__(self) -> None:
            self.moved: list[tuple] = []

        def load_markets(self, reload: bool = False):
            return {"BTC/USDT": spot("BTC/USDT")}

        def create_order(self, *args, **kwargs):
            return {"id": "1"}

        def withdraw(self, *args, **kwargs):
            self.moved.append(("withdraw", args, kwargs))
            return {"id": "W1"}

        def transfer(self, *args, **kwargs):
            self.moved.append(("transfer", args, kwargs))
            return {"id": "T1"}

        def earn(self, *args, **kwargs):
            self.moved.append(("earn", args, kwargs))
            return {"id": "E1"}

    inner = FundMover()
    a = adapter(inner)
    with pytest.raises(FundsMovementForbidden, match="withdraw"):
        a._ex.withdraw("BTC", 1, "addr")
    with pytest.raises(FundsMovementForbidden, match="transfer"):
        a._ex.transfer("USDT", 10, "spot", "funding")
    with pytest.raises(FundsMovementForbidden, match="earn"):
        a._ex.earn("ETH")
    assert hasattr(a._ex, "withdraw") is False
    assert hasattr(a._ex, "create_order") is True
    assert inner.moved == []


def test_clip_spot_sell_qty_is_min_of_book_and_free():
    book, free = D("0.02077"), D("0.02064538")
    assert clip_spot_sell_qty(book, free) == free
    assert clip_spot_sell_qty(D("0"), free) == D("0")
