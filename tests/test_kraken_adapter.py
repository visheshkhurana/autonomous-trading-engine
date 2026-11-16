"""
KrakenAdapter contract tests.

The exchange is a fake: no network, no credentials, no venue. These fail if
the safety rules in adapters/kraken.py are deleted. They do not touch OkxAdapter.
"""

from __future__ import annotations

import datetime as dt
import json
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


def _ledger_row(refid, kind, amount, timestamp, *, direction=None, asset="USDT",
                info_amount=None, info_type=None):
    return {
        "id": refid,
        "timestamp": timestamp,
        "datetime": dt.datetime.fromtimestamp(timestamp / 1000, tz=UTC).isoformat(),
        "type": kind,
        "direction": direction or ("out" if kind == "withdrawal" else "in"),
        "currency": asset,
        "amount": abs(float(amount)),
        "status": "ok",
        "info": {
            "refid": refid,
            "type": info_type or kind,
            "asset": asset,
            "amount": info_amount if info_amount is not None else str(amount),
        },
    }


class _LedgerExchange(FakeExchange):
    def __init__(self, history, later):
        super().__init__()
        self.history = list(history)
        self.later = list(later)
        self.calls = []

    def fetch_ledger(self, code=None, since=None, limit=None, params=None):
        self.calls.append({"since": since, "limit": limit, "params": params or {}})
        if since is None:
            return list(self.history)
        return [row for row in self.history + self.later
                if int(row["timestamp"]) >= int(since)]


def test_parse_spot_cash_flow_keeps_spot_stables_and_drops_derivatives():
    from adapters.kraken import parse_spot_cash_flow
    stamp = 1_763_140_000_000
    withdrawal = parse_spot_cash_flow(_ledger_row(
        "FTGEVAr-syDy6EaOEIqh66t8bwUHlx", "withdrawal", "-10", stamp,
        info_amount="-10.0000",
    ))
    assert withdrawal is not None
    assert withdrawal.amount == D("-10.0000")
    assert withdrawal.asset == "USDT"
    deposit = parse_spot_cash_flow(_ledger_row(
        "dep-1", "deposit", "1000", stamp + 1, info_amount="1000.00",
    ))
    assert deposit is not None and deposit.amount == D("1000.00")
    transfer = parse_spot_cash_flow(_ledger_row(
        "xfer-1", "transfer", "5", stamp + 2, direction="out", info_amount="5",
    ))
    assert transfer is not None and transfer.amount == D("-5")
    deriv = parse_spot_cash_flow(_ledger_row(
        "DERIV-1", "derivativeswithdrawal", "-50", stamp,
        info_type="derivativeswithdrawal", info_amount="-50",
    ))
    assert deriv is not None and deriv.amount == D("-50")
    assert parse_spot_cash_flow(_ledger_row(
        "trade-1", "trade", "-3", stamp, info_amount="-3",
    )) is None
    assert parse_spot_cash_flow(_ledger_row(
        "btc-in", "deposit", "0.01", stamp, asset="BTC", info_amount="0.01",
    )) is None
    from adapters.kraken import classify_ledger_row
    assert classify_ledger_row(_ledger_row(
        "d-mid", "transfer", "-1", stamp,
        info_type="spot-to-derivatives", info_amount="-1",
    ))[0] == "flow"


def test_futures_usdt_moves_shift_the_mark_for_both_label_shapes():
    """A futures-wallet or derivatives USDT move is cash, not an ignored row."""
    from core.risk import RiskCircuit
    from tests.test_risk import NOW as RISK_NOW
    from tests.test_risk import FakeStore, state_at

    stamp = int(dt.datetime(2026, 8, 27, 6, 0, tzinfo=UTC).timestamp() * 1000)
    shapes = [
        _ledger_row(
            "fut-wallet", "futures", "25", stamp,
            direction="in", asset="USDT", info_amount="25",
        ) | {"account": "futures", "info": {
            "refid": "fut-wallet", "type": "futures", "wallet": "futures",
            "asset": "USDT", "amount": "25",
        }},
        _ledger_row(
            "fut-account", "futures", "25", stamp,
            direction="in", asset="USDT", info_amount="25",
        ) | {"account": "futures"},
        _ledger_row(
            "der-sub", "transfer", "25", stamp,
            direction="in", asset="USDT", info_amount="25",
        ),
    ]
    # The helper puts type on info. Force the derivatives subtype on the last row.
    shapes[2]["info"]["subtype"] = "derivatives"
    for row in shapes:
        refid = row["id"]

        class _Pages(FakeExchange):
            def fetch_ledger(self, code=None, since=None, limit=None, params=None):
                if since is None:
                    return [_ledger_row("trade-1", "trade", "-1", 1_000_000, info_amount="-1")]
                return [row]

        store = FakeStore(state_at("10000", asset_class=AssetClass.CRYPTO))
        wired = RiskCircuit(
            store, Mode.LIVE, clock=lambda: RISK_NOW, cash_flows=adapter(_Pages()),
        )
        wired.advance(AssetClass.CRYPTO, D("10000"))
        decision = wired.advance(AssetClass.CRYPTO, D("10025"))
        written = store.writes[-1]
        assert written.high_water == D("10025"), refid
        assert written.day_open_equity == D("10025"), refid
        audit = json.loads(written.hwm_rebase_json)
        assert any(
            item.get("reason") == "external_cash_flow" and item.get("refid") == refid
            for item in audit
        ), refid
        assert decision.entries_allowed is True


def _advance_over(rows):
    """Baseline, then one page of ``rows``, against a 10000 crypto book."""
    from core.risk import RiskCircuit
    from tests.test_risk import NOW as RISK_NOW
    from tests.test_risk import FakeStore, state_at

    class _Pages(FakeExchange):
        def fetch_ledger(self, code=None, since=None, limit=None, params=None):
            if since is None:
                return [_ledger_row("trade-1", "trade", "-1", 1_000_000, info_amount="-1")]
            return list(rows)

    store = FakeStore(state_at("10000", asset_class=AssetClass.CRYPTO))
    wired = RiskCircuit(
        store, Mode.LIVE, clock=lambda: RISK_NOW, cash_flows=adapter(_Pages()),
    )
    wired.advance(AssetClass.CRYPTO, D("10000"))
    return store, wired


def test_hourly_funding_rows_leave_the_mark_unchanged():
    stamp = int(dt.datetime(2026, 8, 27, 6, 0, tzinfo=UTC).timestamp() * 1000)
    hour = 3_600_000
    rows = [
        _ledger_row(
            "fund-1", "derivativesfundingratechange", "1.25", stamp,
            direction="in", asset="USDT", info_amount="1.25",
            info_type="derivativesfundingratechange",
        ),
        _ledger_row(
            "fund-2", "derivativesfundingratechange", "-0.40", stamp + hour,
            direction="out", asset="USDC", info_amount="-0.40",
            info_type="derivativesfundingratechange",
        ),
    ]
    store, wired = _advance_over(rows)
    decision = wired.advance(AssetClass.CRYPTO, D("10000"))
    written = store.writes[-1]
    assert written.high_water == D("10000")
    assert written.day_open_equity == D("10000")
    assert written.unpriced_cash_flow != "1"
    assert written.flow_watermark_ms == str(stamp + hour)
    assert decision.entries_allowed is True


def test_futures_trades_do_not_latch():
    stamp = int(dt.datetime(2026, 8, 27, 6, 0, tzinfo=UTC).timestamp() * 1000)
    zero = _ledger_row(
        "ft-zero", "derivativesfuturestrade", "0", stamp,
        direction="out", asset="USDT", info_amount="0",
        info_type="derivativesfuturestrade",
    )
    coin = _ledger_row(
        "ft-btc", "derivativesfuturestrade", "0.01", stamp + 1,
        direction="out", asset="BTC", info_amount="-0.01",
        info_type="derivativesfuturestrade",
    )
    store, wired = _advance_over([zero, coin])
    decision = wired.advance(AssetClass.CRYPTO, D("10000"))
    written = store.writes[-1]
    assert written.high_water == D("10000")
    assert written.day_open_equity == D("10000")
    assert written.unpriced_cash_flow != "1"
    assert written.ledger_blind != "1"
    assert decision.entries_allowed is True
    assert "unpriced_cash_flow" not in decision.reason


def test_a_spot_to_futures_transfer_is_booked_as_a_flow():
    stamp = int(dt.datetime(2026, 8, 27, 6, 0, tzinfo=UTC).timestamp() * 1000)
    row = _ledger_row(
        "s2f", "spot-to-futures", "-100", stamp,
        direction="out", asset="USDT", info_amount="-100",
        info_type="spot-to-futures",
    )
    store, wired = _advance_over([row])
    decision = wired.advance(AssetClass.CRYPTO, D("9900"))
    written = store.writes[-1]
    assert written.high_water == D("9900")
    assert written.day_open_equity == D("9900")
    audit = json.loads(written.hwm_rebase_json)
    assert any(
        item.get("reason") == "external_cash_flow"
        and item.get("refid") == "s2f"
        and item.get("amount") == "-100"
        for item in audit
    )
    assert decision.entries_allowed is True


def test_a_btc_deposit_holds_the_watermark_and_blocks_entries():
    """A non-stable deposit must not be skipped and then booked as a new high."""
    from core.risk import RiskCircuit
    from tests.test_risk import NOW as RISK_NOW
    from tests.test_risk import FakeStore, state_at

    trade = _ledger_row("trade-1", "trade", "-3", 1_000_000, info_amount="-3")
    btc = _ledger_row(
        "btc-in", "deposit", "0.01", 2_000_000, asset="BTC", info_amount="0.01",
    )

    class _Pages(FakeExchange):
        def fetch_ledger(self, code=None, since=None, limit=None, params=None):
            if since is None:
                return [trade]
            return [btc]

    book = adapter(_Pages())
    store = FakeStore(state_at("10000", asset_class=AssetClass.CRYPTO))
    wired = RiskCircuit(
        store, Mode.LIVE, clock=lambda: RISK_NOW, cash_flows=book,
    )
    first = wired.advance(AssetClass.CRYPTO, D("10000"))
    assert first.entries_allowed is True
    watermark = store.writes[-1].flow_watermark_ms
    assert watermark == "1000001"
    blocked = wired.advance(AssetClass.CRYPTO, D("15000"))
    assert blocked.entries_allowed is False
    assert "unpriced_cash_flow" in blocked.reason
    assert "ledger_blind" in blocked.reason
    assert store.writes[-1].high_water == D("10000")
    assert store.writes[-1].flow_watermark_ms == watermark
    assert store.writes[-1].ledger_blind == "1"


def test_fetch_spot_cash_flows_checkpoints_history_then_returns_new_flows():
    old = 1_700_000_000_000
    newest = old + 86_400_000
    later = newest + 60_000
    history = [
        _ledger_row("FTGEVAr-syDy6EaOEIqh66t8bwUHlx", "withdrawal", "-10", old,
                    info_amount="-10"),
        _ledger_row("DERIV-1", "derivativeswithdrawal", "-50", old + 1,
                    info_type="derivativeswithdrawal", info_amount="-50"),
        _ledger_row("trade-1", "trade", "-3", newest, info_amount="-3"),
    ]
    arrived = [_ledger_row("dep-new", "deposit", "25", later, info_amount="25")]
    exchange = _LedgerExchange(history, arrived)
    book = adapter(exchange)
    flows, watermark = book.fetch_spot_cash_flows(since_ms=None)
    assert flows == []
    assert watermark == newest + 1
    newer, again = book.fetch_spot_cash_flows(since_ms=watermark)
    assert [item.refid for item in newer] == ["dep-new"]
    assert newer[0].amount == D("25")
    assert again == later
    assert all(item.refid != "DERIV-1" for item in newer)
