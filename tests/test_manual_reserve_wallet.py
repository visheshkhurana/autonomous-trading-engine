"""Reserve wallet: a balance that is only the manual reserve must not be sold."""

from decimal import Decimal as D

from adapters.kraken import KrakenAdapter
from core.types import Intent, Mode, OrderStatus, OrderType, Side
from tests.test_okx_adapter import FakeClock, FakeExchange, spot
from tests.test_safety_guards import BTC, WHEN


def _adapter(free):
    ex = FakeExchange(
        {"BTC/USDT": spot("BTC/USDT", amount_step="0.00000001", price_step="0.1",
                          min_amount="0.00000001")},
        tickers={"BTC/USDT": {"last": 100000}},
        balance={"free": {"BTC": format(free, "f"), "USDT": "10000"},
                 "total": {"BTC": format(free, "f"), "USDT": "10000"}},
    )
    clock = FakeClock()
    adapter = KrakenAdapter(
        ex, mode=Mode.LIVE, poll_interval_s=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep,
        now=lambda: WHEN, key_freeze=lambda: False,
        manual_reserved="BTC:0.0880257350",
    )
    return adapter, ex


def test_sell_with_only_the_reserve_in_the_wallet_sends_nothing():
    adapter, ex = _adapter(D("0.0880257350"))
    ref = adapter.place(Intent(
        instrument=BTC, side=Side.SELL, qty=D("0.001"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="x",
        limit_price=D("100000"), idempotency_key="k1",
    ))
    assert ref.status is not OrderStatus.FILLED
    assert ex.create_calls == []


def test_stop_with_only_the_reserve_in_the_wallet_is_refused():
    adapter, ex = _adapter(D("0.0880257350"))
    result = adapter.place_native_stop(BTC, D("0.001"), D("90000"))
    assert result.ok is False and ex.create_calls == []


def test_malformed_reserve_refuses_the_sell():
    adapter, ex = _adapter(D("0.0890257350"))
    adapter._manual_reserved_raw = "BTC:abc"
    ref = adapter.place(Intent(
        instrument=BTC, side=Side.SELL, qty=D("0.001"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="x",
        limit_price=D("100000"), idempotency_key="k2",
    ))
    assert ref.status is not OrderStatus.FILLED and ex.create_calls == []
