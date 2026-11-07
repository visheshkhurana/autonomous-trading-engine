"""
51008-class incident, reproduced on KrakenAdapter.

OKX took a BUY fee in ETH so venue free was smaller than the engine book.
Sending book qty is insufficient funds and flatten sticks. Kraken can take
fees in the base the same way. These tests are the same contract: a spot
SELL is min(book, venue free) and unplaceable dust is written off.

No network, no credentials, no CRYPTO_LIVE / EQUITY_LIVE / TRADER_MODE /
KRAKEN_LIVE. Buys are not shrunk. Caps default to 100 USDT / clip, 500 / day.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

import pytest

import runner
from adapters.kraken import KrakenAdapter, OrderSubmitFailed, clip_spot_sell_qty
from core.types import AssetClass, Intent, Mode, OrderStatus, OrderType, Side
from tests.test_okx_adapter import FakeClock, FakeExchange, spot
from tests.test_okx_fee_in_base_sell import BOOK, FREE, MARK, NOW, ORDER_ID

ETH = runner.default_instruments(AssetClass.CRYPTO)[1]


class InsufficientFunds(Exception):
    """Venue-shaped insufficient funds. The adapter wraps this if it still fires."""


def _eth_market() -> dict[str, Any]:
    return spot(
        "ETH/USDT",
        amount_step="0.00000001",
        price_step="0.01",
        min_amount="0.00001",
        min_cost="5",
    )


class IncidentExchange(FakeExchange):
    def __init__(self, *, free_eth: Decimal = FREE, mark: Decimal = MARK):
        super().__init__(
            {"ETH/USDT": _eth_market()},
            tickers={"ETH/USDT": {"last": format(mark, "f")}},
            balance={
                "total": {"USDT": "10000", "ETH": format(free_eth, "f")},
                "free": {"USDT": "10000", "ETH": format(free_eth, "f")},
            },
        )
        self.free_eth = free_eth
        self.mark = mark

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        amt = Decimal(str(amount))
        self.create_calls.append((symbol, type_, side, amount, price, params))
        if side == "sell" and amt > self.free_eth:
            raise InsufficientFunds("EOrder:Insufficient funds")
        if self.create_error:
            raise self.create_error
        filled = format(amt, "f")
        mark = format(self.mark, "f")
        self.order_states = [{
            "id": ORDER_ID, "symbol": symbol, "status": "closed",
            "amount": filled, "filled": filled,
            "average": mark, "cost": format(amt * self.mark, "f"),
            "timestamp": 1756832407000,
            "fee": {"cost": "0", "currency": "USDT"},
        }]
        return {"id": ORDER_ID}


def _adapter(exchange: FakeExchange, **kw) -> KrakenAdapter:
    clock = FakeClock()
    kw.setdefault("mode", Mode.PAPER)
    kw.setdefault("sandbox", True)
    kw.setdefault("poll_interval_s", 1.0)
    return KrakenAdapter(
        exchange, monotonic=clock.monotonic, sleep=clock.sleep,
        now=lambda: NOW, **kw,
    )


def _sell_intent(qty: Decimal = BOOK, *, key: str = "exit-eth") -> Intent:
    return Intent(
        instrument=ETH, side=Side.SELL, qty=qty,
        order_type=OrderType.MARKET, strategy_id="crypto_mandate",
        reason="incident exit", idempotency_key=key,
        limit_price=MARK,
    )


def _buy_intent(qty: Decimal = BOOK, *, key: str = "entry-eth") -> Intent:
    return Intent(
        instrument=ETH, side=Side.BUY, qty=qty,
        order_type=OrderType.MARKET, strategy_id="crypto_mandate",
        reason="incident entry", idempotency_key=key,
        limit_price=MARK,
    )


def _sent_amount(ex: FakeExchange) -> Decimal:
    assert ex.create_calls, "expected a venue create_order"
    return Decimal(str(ex.create_calls[0][3]))


def test_unclipped_book_qty_is_insufficient_funds_at_the_venue() -> None:
    ex = IncidentExchange()
    with pytest.raises(InsufficientFunds):
        ex.create_order("ETH/USDT", "market", "sell", format(BOOK, "f"), None, {})


def test_clip_spot_sell_qty_is_min_of_book_and_free() -> None:
    assert clip_spot_sell_qty(BOOK, FREE) == FREE
    assert clip_spot_sell_qty(FREE, FREE) == FREE
    assert clip_spot_sell_qty(BOOK, Decimal("0")) == Decimal("0")


def test_spot_sell_is_clipped_to_free_base_and_does_not_insufficient_funds() -> None:
    ex = IncidentExchange()
    ref = _adapter(ex).place(_sell_intent())
    assert ref.status is OrderStatus.FILLED
    sent = _sent_amount(ex)
    assert sent == FREE
    assert sent < BOOK
    assert ref.filled_qty == FREE
    assert "clipped" in (ref.error or "")


def test_sell_is_not_inflated_to_a_larger_free_btc_balance() -> None:
    """A manual 0.088 BTC in the same wallet must not become the sell size.

    Flatten and strategy exits pass the engine ledger qty as the intent.
    Venue free is only a ceiling: free = engine qty + 0.088 still sends
    the engine qty, not the wallet.
    """
    engine_qty = Decimal("0.001")
    manual = Decimal("0.088")
    free = manual + engine_qty
    btc = runner.default_instruments(AssetClass.CRYPTO)[0]
    mark = Decimal("100000")
    ex = IncidentExchange(free_eth=free, mark=mark)
    ex._markets = {"BTC/USDT": spot(
        "BTC/USDT", amount_step="0.00000001", price_step="0.1",
        min_amount="0.00001", min_cost="5",
    )}
    ex.markets = ex._markets
    ex.tickers = {"BTC/USDT": {"last": format(mark, "f")}}
    ex.balance = {
        "total": {"USDT": "10000", "BTC": format(free, "f")},
        "free": {"USDT": "10000", "BTC": format(free, "f")},
    }
    intent = Intent(
        instrument=btc, side=Side.SELL, qty=engine_qty,
        order_type=OrderType.MARKET, strategy_id="crypto_mandate",
        reason="engine exit", idempotency_key="exit-btc-manual",
        limit_price=mark,
    )
    ref = _adapter(ex).place(intent)
    assert ref.status is OrderStatus.FILLED
    sent = _sent_amount(ex)
    assert sent == engine_qty
    assert sent < manual
    assert sent < free
    assert "clipped" not in (ref.error or "")


def test_spot_buy_is_not_shrunk_to_free_base() -> None:
    ex = IncidentExchange(free_eth=Decimal("0"))
    ref = _adapter(ex).place(_buy_intent())
    assert ref.status is OrderStatus.FILLED
    assert _sent_amount(ex) == BOOK
    assert "tgtCcy" not in (ex.create_calls[0][5] or {})


def test_unreadable_free_refuses_a_sell_rather_than_sending_book_qty() -> None:
    ex = IncidentExchange()
    ex.balance = {"total": {"ETH": format(FREE, "f")}}
    ref = _adapter(ex).place(_sell_intent())
    assert ref.status is OrderStatus.REJECTED
    assert not ex.create_calls
    assert "unreadable" in (ref.error or "")


def test_dust_leftover_after_a_clipped_sell_is_written_off_on_the_ref() -> None:
    leftover = BOOK - FREE
    assert leftover * MARK < ETH.min_notional
    ex = IncidentExchange()
    ref = _adapter(ex).place(_sell_intent())
    assert ref.filled_qty == FREE
    dust = [f for f in ref.fills if f.qty == leftover]
    assert len(dust) == 1
    assert dust[0].side is Side.SELL
    assert dust[0].fees == Decimal("0")
    assert "dust leftover" in (ref.error or "")


def test_a_sellable_remainder_is_not_written_off() -> None:
    book = Decimal("1.0")
    free = Decimal("0.5")
    ex = IncidentExchange(free_eth=free)
    ref = _adapter(ex).place(_sell_intent(book, key="locked-half"))
    assert _sent_amount(ex) == free
    assert ref.filled_qty == free
    assert sum((f.qty for f in ref.fills), Decimal("0")) == free
    assert "dust leftover" not in (ref.error or "")


def test_place_still_raises_order_submit_failed_when_the_venue_rejects() -> None:
    ex = IncidentExchange()
    ex.create_error = InsufficientFunds("after clip")
    ex.free_eth = BOOK
    ex.balance["free"]["ETH"] = format(BOOK, "f")
    with pytest.raises(OrderSubmitFailed):
        _adapter(ex).place(_sell_intent(key="still-fails"))


def test_caps_default_to_one_hundred_five_hundred_and_one_open() -> None:
    assert runner.CRYPTO_CLIP_USDT == Decimal("100")
    assert runner.CRYPTO_DAILY_USDT == Decimal("500")
    assert runner.CRYPTO_CLASS_LIMITS.max_open_positions == 1
    assert runner.KRAKEN_SLEEVE_A_MAX_OPEN == 1
