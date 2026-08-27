"""
Tests for adapters/dhan.py.

The whole point of this file is the failure paths. Every safety rule in the
adapter has a test here that goes red if the rule is deleted:

  - acceptance is never a fill                    (test_timeout_*)
  - a poll that errors is never a fill            (test_poll_errors_never_become_a_fill)
  - equity is None on failure, never 0            (test_equity_*)
  - a position we cannot read raises, never 0     (test_positions_*)
  - security ids are resolved, never guessed      (test_*security_id*)
  - MARKET is never sent                          (test_market_intent_becomes_marketable_limit)
  - protective BUY leg goes on first              (test_structure_places_buy_before_sell)
  - a failed leg unwinds shorts before longs      (test_unwind_closes_short_before_long)
  - a failed BUY leg never lets a SELL through    (test_failed_buy_leg_never_sends_the_short)

No live client is constructed anywhere in this file.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Optional

import pytest

from adapters.dhan import (
    BROKER_POSITION_STRATEGY_ID,
    DhanAdapter,
    DhanApiError,
    DhanSecurityMap,
    InvalidOrderSize,
    OrderSubmitFailed,
    QuoteUnavailable,
    SecurityIdUnknown,
    StructureResult,
    marketable_limit,
    parse_scrip_master,
)
from core.protocols import BrokerAdapter
from core.types import (
    AssetClass,
    Instrument,
    Intent,
    Mode,
    OrderStatus,
    OrderType,
    Side,
)

# ── fixtures / fakes ────────────────────────────────────────────────────────

NIFTY_CE = Instrument(
    symbol="NIFTY-Aug2026-25000-CE",
    asset_class=AssetClass.EQUITY,
    broker_id="43001",
    lot_size=75,
    tick_size=Decimal("0.05"),
    expiry=dt.date(2026, 8, 27),
)
NIFTY_PE = Instrument(
    symbol="NIFTY-Aug2026-25000-PE",
    asset_class=AssetClass.EQUITY,
    broker_id="43002",
    lot_size=75,
    tick_size=Decimal("0.05"),
    expiry=dt.date(2026, 8, 27),
)
NIFTY_CE_WING = Instrument(
    symbol="NIFTY-Aug2026-25400-CE",
    asset_class=AssetClass.EQUITY,
    broker_id="43003",
    lot_size=75,
    tick_size=Decimal("0.05"),
    expiry=dt.date(2026, 8, 27),
)
RELIANCE = Instrument(
    symbol="RELIANCE",
    asset_class=AssetClass.EQUITY,
    broker_id="2885",
    lot_size=1,
    tick_size=Decimal("0.05"),
)


def intent(instrument: Instrument = NIFTY_CE, side: Side = Side.BUY,
           qty: str = "75", order_type: OrderType = OrderType.MARKET,
           limit_price: Optional[str] = None,
           structure_id: Optional[str] = None,
           key: str = "k1") -> Intent:
    return Intent(
        instrument=instrument,
        side=side,
        qty=Decimal(qty),
        order_type=order_type,
        strategy_id="fno_daily",
        reason="test",
        limit_price=Decimal(limit_price) if limit_price is not None else None,
        structure_id=structure_id,
        idempotency_key=key,
    )


class FakeDhanClient:
    """Stands in for the dhanhq SDK. Scripted, and records every call so the
    tests can assert on ORDERING, which is where the multi-leg safety lives."""

    def __init__(self, *, ltp: dict[str, float] | None = None,
                 funds: Any = None, positions: Any = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._ltp = ltp or {}
        self._funds = funds
        self._positions = positions
        self._next_order = 0
        # security_id -> list of status records, consumed one per poll (the
        # last one repeats forever)
        self.status_script: dict[str, list[dict[str, Any]]] = {}
        self.order_to_sid: dict[str, str] = {}
        self.submit_error: dict[str, Exception] = {}       # sid -> raise on submit
        self.submit_failure: dict[str, dict[str, Any]] = {}  # sid -> failure payload
        self.submit_no_id: set[str] = set()
        self.cancel_error: Optional[Exception] = None
        self.cancelled: list[str] = []

    # -- scripting helpers -------------------------------------------------

    def script(self, security_id: str, statuses: list[dict[str, Any]]) -> None:
        self.status_script[security_id] = list(statuses)

    def filled(self, security_id: str, qty: int, price: float) -> None:
        self.script(security_id, [{"orderStatus": "TRADED",
                                   "filledQty": qty,
                                   "averageTradedPrice": price}])

    def resting(self, security_id: str, filled_qty: int = 0) -> None:
        rec: dict[str, Any] = {"orderStatus": "PENDING"}
        if filled_qty:
            rec = {"orderStatus": "PART_TRADED", "filledQty": filled_qty,
                   "averageTradedPrice": 100.0}
        self.script(security_id, [rec])

    # -- SDK surface -------------------------------------------------------

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("place_order", kwargs))
        sid = kwargs["security_id"]
        if sid in self.submit_error:
            raise self.submit_error[sid]
        if sid in self.submit_failure:
            return self.submit_failure[sid]
        if sid in self.submit_no_id:
            return {"status": "success", "data": {}}
        self._next_order += 1
        oid = f"O{self._next_order}"
        self.order_to_sid[oid] = sid
        return {"status": "success", "data": {"orderId": oid}}

    def get_order_by_id(self, order_id: str) -> dict[str, Any]:
        self.calls.append(("get_order_by_id", order_id))
        sid = self.order_to_sid.get(order_id, "")
        script = self.status_script.get(sid)
        if not script:
            return {"status": "success", "data": {"orderStatus": "PENDING"}}
        rec = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(rec, Exception):
            raise rec
        return {"status": "success", "data": rec}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self.calls.append(("cancel_order", order_id))
        if self.cancel_error is not None:
            raise self.cancel_error
        self.cancelled.append(order_id)
        return {"status": "success", "data": {"orderId": order_id}}

    def ticker_data(self, payload: dict[str, list[int]]) -> dict[str, Any]:
        self.calls.append(("ticker_data", payload))
        segment, ids = next(iter(payload.items()))
        seg_map: dict[str, Any] = {}
        for i in ids:
            px = self._ltp.get(str(i))
            if px is not None:
                seg_map[str(i)] = {"last_price": px}
        return {"status": "success", "data": {"data": {segment: seg_map}}}

    def get_fund_limits(self) -> Any:
        self.calls.append(("get_fund_limits", None))
        if isinstance(self._funds, Exception):
            raise self._funds
        return self._funds

    def get_positions(self) -> Any:
        self.calls.append(("get_positions", None))
        if isinstance(self._positions, Exception):
            raise self._positions
        return self._positions

    # -- assertions --------------------------------------------------------

    def placed_sids(self) -> list[str]:
        return [k["security_id"] for name, k in self.calls if name == "place_order"]

    def placed(self) -> list[dict[str, Any]]:
        return [k for name, k in self.calls if name == "place_order"]


class FakeClock:
    """Monotonic time the tests drive by hand, so nothing sleeps for real."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.001)


def adapter(client: FakeDhanClient, **kwargs: Any) -> DhanAdapter:
    clock = FakeClock()
    kwargs.setdefault("clock", clock)
    kwargs.setdefault("sleep", clock.sleep)
    kwargs.setdefault("poll_interval_s", 1.0)
    a = DhanAdapter(client, **kwargs)
    a._test_clock = clock  # type: ignore[attr-defined]
    return a


# ── construction / mode ─────────────────────────────────────────────────────

def test_satisfies_the_broker_adapter_protocol() -> None:
    a = adapter(FakeDhanClient())
    assert isinstance(a, BrokerAdapter)
    assert a.mode is Mode.LIVE
    assert a.asset_class is AssetClass.EQUITY


def test_paper_mode_is_refused() -> None:
    """SAFETY: a live venue adapter must never carry a PAPER label. Delete the
    guard and paper-labelled fills come back from a real exchange."""
    with pytest.raises(ValueError, match="paper"):
        DhanAdapter(FakeDhanClient(), mode=Mode.PAPER)


def test_client_is_required() -> None:
    with pytest.raises(ValueError):
        DhanAdapter(None)


def test_bad_product_type_is_refused() -> None:
    with pytest.raises(ValueError):
        DhanAdapter(FakeDhanClient(), product="NRML")


# ── symbol -> security_id ───────────────────────────────────────────────────

SCRIP_CSV = (
    "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,"
    "SEM_TRADING_SYMBOL,SEM_LOT_UNITS,SEM_EXPIRY_DATE,SEM_TICK_SIZE,SEM_SERIES\n"
    "NSE,E,2885,EQUITY,RELIANCE,1,,0.05,EQ\n"
    "NSE,D,43001,OPTIDX,NIFTY-Aug2026-25000-CE,75,2026-08-27,0.05,\n"
    "NSE,E,,EQUITY,BROKENROW,1,,0.05,EQ\n"
    "BSE,E,500325,EQUITY,RELIANCE,1,,0.05,A\n"
)


def test_scrip_master_parses_and_skips_unusable_rows() -> None:
    recs = parse_scrip_master(SCRIP_CSV)
    symbols = {r.symbol for r in recs}
    assert "BROKENROW" not in symbols, "a row with no security_id is not tradable"
    assert len(recs) == 3


def test_security_map_resolves_symbol_to_security_id() -> None:
    smap = DhanSecurityMap.from_csv(SCRIP_CSV)
    assert smap.resolve("RELIANCE") == "2885"
    assert smap.resolve("RELIANCE", "BSE") == "500325"
    assert smap.series("RELIANCE", "BSE") == "A"
    assert smap.record("NIFTY-Aug2026-25000-CE").segment == "NSE_FNO"


def test_security_map_builds_the_instrument_from_one_scrip_row() -> None:
    smap = DhanSecurityMap.from_csv(SCRIP_CSV)
    inst = smap.instrument("NIFTY-Aug2026-25000-CE")
    assert inst.broker_id == "43001"
    assert inst.lot_size == 75
    assert inst.expiry == dt.date(2026, 8, 27)


def test_unknown_symbol_raises_instead_of_guessing() -> None:
    """SAFETY: a guessed security_id routes a real order at the wrong contract."""
    smap = DhanSecurityMap.from_csv(SCRIP_CSV)
    with pytest.raises(SecurityIdUnknown):
        smap.resolve("NOSUCHSCRIP")
    assert smap.try_resolve("NOSUCHSCRIP") is None


def test_place_resolves_symbol_when_broker_id_is_empty() -> None:
    unmapped = Instrument(symbol="RELIANCE", asset_class=AssetClass.EQUITY,
                          broker_id="", lot_size=1)
    client = FakeDhanClient(ltp={"2885": 1400.0})
    client.filled("2885", 1, 1400.10)
    a = adapter(client, security_map=DhanSecurityMap.from_csv(SCRIP_CSV))
    ref = a.place(intent(unmapped, Side.BUY, "1"))
    assert ref.status is OrderStatus.FILLED
    assert client.placed()[0]["security_id"] == "2885"
    assert client.placed()[0]["exchange_segment"] == "NSE_EQ"


def test_unresolvable_instrument_never_reaches_the_broker() -> None:
    """SAFETY: nothing is sent when the instrument cannot be identified."""
    unmapped = Instrument(symbol="GHOST", asset_class=AssetClass.EQUITY, broker_id="")
    client = FakeDhanClient()
    a = adapter(client, security_map=DhanSecurityMap.from_csv(SCRIP_CSV))
    with pytest.raises(SecurityIdUnknown):
        a.place(intent(unmapped, Side.BUY, "1"))
    assert client.placed() == []


def test_non_numeric_security_id_is_rejected() -> None:
    bad = Instrument(symbol="WEIRD", asset_class=AssetClass.EQUITY,
                     broker_id="NSE:WEIRD")
    client = FakeDhanClient()
    with pytest.raises(SecurityIdUnknown):
        adapter(client).place(intent(bad, Side.BUY, "1"))
    assert client.placed() == []


def test_segment_falls_back_to_expiry_when_no_map_is_present() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0, "2885": 1400.0})
    a = adapter(client)
    assert a.exchange_segment(NIFTY_CE) == "NSE_FNO"
    assert a.exchange_segment(RELIANCE) == "NSE_EQ"


# ── pricing ─────────────────────────────────────────────────────────────────

def test_marketable_limit_crosses_in_both_directions() -> None:
    tick = Decimal("0.05")
    assert marketable_limit(Decimal("120.00"), Side.BUY, tick, 2) == Decimal("120.10")
    assert marketable_limit(Decimal("120.00"), Side.SELL, tick, 2) == Decimal("119.90")


def test_marketable_limit_rounding_never_pulls_the_order_back_inside() -> None:
    """SAFETY: quantizing a BUY down (or a SELL up) parks the order inside the
    spread, where it rests instead of filling — and a resting protective leg is
    exactly what leaves a short naked."""
    tick = Decimal("0.05")
    ltp = Decimal("120.03")
    buy = marketable_limit(ltp, Side.BUY, tick, 1)
    sell = marketable_limit(ltp, Side.SELL, tick, 1)
    assert buy >= ltp + tick
    assert sell <= ltp - tick
    assert buy % tick == 0 and sell % tick == 0


def test_marketable_limit_never_goes_to_zero_or_negative() -> None:
    tick = Decimal("0.05")
    assert marketable_limit(Decimal("0.05"), Side.SELL, tick, 20) == tick


def test_marketable_limit_refuses_a_non_positive_ltp() -> None:
    with pytest.raises(QuoteUnavailable):
        marketable_limit(Decimal("0"), Side.BUY, Decimal("0.05"), 2)


def test_market_intent_becomes_marketable_limit() -> None:
    """SAFETY: NFO rejects MARKET outright. If this regresses, every F&O entry
    is refused by the venue."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.filled("43001", 75, 120.10)
    a = adapter(client, limit_ticks=2)
    ref = a.place(intent(NIFTY_CE, Side.BUY, "75", OrderType.MARKET))
    sent = client.placed()[0]
    assert sent["order_type"] == "LIMIT"
    assert sent["price"] == pytest.approx(120.10)
    assert ref.status is OrderStatus.FILLED


def test_explicit_limit_is_not_made_more_aggressive() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.filled("43001", 75, 118.00)
    a = adapter(client)
    a.place(intent(NIFTY_CE, Side.BUY, "75", OrderType.LIMIT, limit_price="118.03"))
    assert client.placed()[0]["price"] == pytest.approx(118.00)  # rounded DOWN


def test_no_quote_means_no_order() -> None:
    """SAFETY: an unpriceable order is not sent at a guessed price."""
    client = FakeDhanClient(ltp={})           # feed carries nothing
    a = adapter(client)
    with pytest.raises(QuoteUnavailable):
        a.place(intent(NIFTY_CE, Side.BUY, "75", OrderType.MARKET))
    assert client.placed() == []


def test_sub_lot_quantity_is_refused_and_nothing_is_sent() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    a = adapter(client)
    with pytest.raises(InvalidOrderSize):
        a.place(intent(NIFTY_CE, Side.BUY, "40"))   # lot_size is 75
    assert client.placed() == []


def test_quantity_is_rounded_down_to_whole_lots() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.filled("43001", 150, 120.10)
    a = adapter(client)
    a.place(intent(NIFTY_CE, Side.BUY, "190"))      # 2 lots, not 3
    assert client.placed()[0]["quantity"] == 150


# ── fill confirmation ───────────────────────────────────────────────────────

def test_filled_order_reports_the_fill() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.filled("43001", 75, 120.05)
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"))
    assert ref.status is OrderStatus.FILLED
    assert ref.filled_qty == Decimal("75")
    assert ref.avg_price == Decimal("120.05")
    assert len(ref.fills) == 1
    assert ref.fills[0].mode is Mode.LIVE


def test_rejected_order_reports_rejected_with_no_fill() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.script("43001", [{"orderStatus": "REJECTED"}])
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"))
    assert ref.status is OrderStatus.REJECTED
    assert ref.filled_qty == Decimal("0")
    assert ref.fills == ()


def test_acceptance_is_not_a_fill() -> None:
    """SAFETY: the core rule. A resting order reported as filled is how a short
    goes on without its hedge. Delete the polling loop and this goes red."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.resting("43001")
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"), timeout_s=3.0)
    assert ref.status is OrderStatus.TIMEOUT
    assert ref.complete is False
    assert ref.filled_qty == Decimal("0")


def test_timeout_cancels_the_resting_order() -> None:
    """SAFETY: a timed-out order left resting can fill later, unwatched."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.resting("43001")
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"), timeout_s=3.0)
    assert client.cancelled == [ref.order_id]


def test_order_that_fills_on_a_later_poll_is_confirmed() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.script("43001", [
        {"orderStatus": "PENDING"},
        {"orderStatus": "PENDING"},
        {"orderStatus": "TRADED", "filledQty": 75, "averageTradedPrice": 120.10},
    ])
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"), timeout_s=30.0)
    assert ref.status is OrderStatus.FILLED
    assert client.cancelled == []


def test_partial_fill_at_timeout_is_reported_not_hidden() -> None:
    """SAFETY: a half-filled leg is real exposure. Reporting filled_qty=0 there
    means the residue never gets unwound."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.resting("43001", filled_qty=75)      # PART_TRADED, 1 of 2 lots
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "150"), timeout_s=3.0)
    assert ref.status is OrderStatus.TIMEOUT
    assert ref.filled_qty == Decimal("75")


def test_fill_landing_between_the_last_poll_and_the_cancel_is_caught() -> None:
    """SAFETY: the re-read after cancel. Without it a real position is reported
    as a no-fill and nobody tracks it."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.script("43001", [
        {"orderStatus": "PENDING"},                 # t=0
        {"orderStatus": "PENDING"},                 # t=1
        {"orderStatus": "PENDING"},                 # t=2 -> budget spent, TIMEOUT
        # the cancel goes out, and the re-read finds it traded after all
        {"orderStatus": "TRADED", "filledQty": 75, "averageTradedPrice": 120.10},
    ])
    a = adapter(client, poll_interval_s=1.0)
    ref = a.place(intent(NIFTY_CE, Side.BUY, "75"), timeout_s=2.0)
    assert client.cancelled == [ref.order_id]
    assert ref.status is OrderStatus.FILLED
    assert ref.filled_qty == Decimal("75")


def test_filled_status_for_less_than_the_order_is_not_a_fill() -> None:
    """SAFETY: the status and the quantity disagree, and the quantity wins.
    Believing 'TRADED' over a short filledQty books a half-filled leg as done
    and leaves its residue unwound forever."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.script("43001", [
        {"orderStatus": "TRADED", "filledQty": 75, "averageTradedPrice": 120.10},
    ])
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "150"), timeout_s=2.0)
    assert ref.status is not OrderStatus.FILLED
    assert ref.complete is False
    assert ref.filled_qty == Decimal("75")
    assert client.cancelled == [ref.order_id]


def test_poll_errors_never_become_a_fill() -> None:
    """SAFETY: "I could not ask" is not "it filled". This is the exact class of
    bug that wrecked the previous system."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.script("43001", [RuntimeError("status endpoint down")])
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"), timeout_s=3.0)
    assert ref.status is OrderStatus.TIMEOUT
    assert ref.filled_qty == Decimal("0")
    assert ref.error and "down" in ref.error


def test_unrecognised_status_can_only_time_out_never_fill() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.script("43001", [{"orderStatus": "SOMETHING_NEW"}])
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"), timeout_s=2.0)
    assert ref.status is OrderStatus.TIMEOUT


def test_fill_without_a_price_emits_no_fabricated_fill() -> None:
    """SAFETY: a Fill at an invented price poisons every P&L number downstream."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.script("43001", [{"orderStatus": "TRADED", "filledQty": 75}])
    ref = adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"))
    assert ref.status is OrderStatus.FILLED
    assert ref.filled_qty == Decimal("75")
    assert ref.avg_price is None
    assert ref.fills == ()
    assert ref.error and "average traded price" in ref.error


def test_fee_estimator_is_used_when_injected() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.filled("43001", 75, 120.00)
    a = adapter(client, fee_estimator=lambda i, s, q, p: Decimal("23.50"))
    ref = a.place(intent(NIFTY_CE, Side.BUY, "75"))
    assert ref.fills[0].fees == Decimal("23.50")


# ── submit failures ─────────────────────────────────────────────────────────

def test_explicit_broker_rejection_is_unambiguous() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.submit_failure["43001"] = {"status": "failure", "remarks": "RMS blocked"}
    with pytest.raises(OrderSubmitFailed) as exc:
        adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"))
    assert exc.value.ambiguous is False
    assert "RMS blocked" in str(exc.value)


def test_in_flight_submit_failure_is_flagged_ambiguous() -> None:
    """SAFETY: a submit that dies mid-flight may still be live. Claiming it is
    not is how a duplicate or an orphan gets created."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.submit_error["43001"] = ConnectionError("socket reset")
    with pytest.raises(OrderSubmitFailed) as exc:
        adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"))
    assert exc.value.ambiguous is True


def test_accepted_submit_with_no_order_id_is_ambiguous() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.submit_no_id.add("43001")
    with pytest.raises(OrderSubmitFailed) as exc:
        adapter(client).place(intent(NIFTY_CE, Side.BUY, "75"))
    assert exc.value.ambiguous is True


# ── cancel ──────────────────────────────────────────────────────────────────

def test_cancel_returns_false_when_the_broker_refuses() -> None:
    client = FakeDhanClient()
    client.cancel_error = RuntimeError("already traded")
    assert adapter(client).cancel("O1") is False


def test_cancel_returns_true_on_confirmation() -> None:
    assert adapter(FakeDhanClient()).cancel("O1") is True


# ── equity ──────────────────────────────────────────────────────────────────

def test_equity_sums_available_and_utilised() -> None:
    client = FakeDhanClient(funds={"status": "success", "data": {
        "availabelBalance": 412000.5, "utilizedAmount": 88000.0}})
    assert adapter(client).equity() == Decimal("500000.5")


def test_equity_is_none_when_the_read_raises() -> None:
    """SAFETY: 0 here reads as a 100% drawdown and latches the circuit into
    flatten. None makes it fail closed instead."""
    client = FakeDhanClient(funds=ConnectionError("dhan down"))
    assert adapter(client).equity() is None


def test_equity_is_none_when_the_broker_returns_a_failure_payload() -> None:
    client = FakeDhanClient(funds={"status": "failure", "remarks": "DH-901"})
    assert adapter(client).equity() is None


def test_equity_is_none_when_a_component_is_missing() -> None:
    """SAFETY: available cash alone understates equity, which reads as a
    drawdown that did not happen."""
    client = FakeDhanClient(funds={"status": "success",
                                   "data": {"availabelBalance": 412000.0}})
    assert adapter(client).equity() is None


def test_equity_is_none_when_a_component_is_unparseable() -> None:
    client = FakeDhanClient(funds={"status": "success", "data": {
        "availabelBalance": "n/a", "utilizedAmount": 0}})
    assert adapter(client).equity() is None


def test_a_genuinely_empty_account_reports_zero_not_none() -> None:
    """A real zero balance is a fact; it must stay distinguishable from a
    failed read, which is None."""
    client = FakeDhanClient(funds={"status": "success", "data": {
        "availabelBalance": 0, "utilizedAmount": 0}})
    assert adapter(client).equity() == Decimal("0")


# ── positions ───────────────────────────────────────────────────────────────

def _pos_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "success", "data": rows}


def test_positions_maps_long_and_short() -> None:
    client = FakeDhanClient(positions=_pos_payload([
        {"securityId": "43001", "tradingSymbol": "NIFTY-CE", "positionType": "LONG",
         "netQty": 75, "buyAvg": 120.5},
        {"securityId": "43002", "tradingSymbol": "NIFTY-PE", "positionType": "SHORT",
         "netQty": 150, "sellAvg": 98.25},
    ]))
    positions = adapter(client).positions()
    assert [p.qty for p in positions] == [Decimal("75"), Decimal("-150")]
    assert positions[1].avg_price == Decimal("98.25")
    assert positions[0].mode is Mode.LIVE
    assert positions[0].strategy_id == BROKER_POSITION_STRATEGY_ID


def test_positions_skips_closed_and_zero_rows() -> None:
    client = FakeDhanClient(positions=_pos_payload([
        {"securityId": "43001", "tradingSymbol": "X", "positionType": "CLOSED",
         "netQty": 0, "buyAvg": 1},
        {"securityId": "43002", "tradingSymbol": "Y", "netQty": 0, "buyAvg": 1},
    ]))
    assert adapter(client).positions() == ()


def test_empty_book_is_an_empty_tuple() -> None:
    client = FakeDhanClient(positions=_pos_payload([]))
    assert adapter(client).positions() == ()


def test_positions_read_failure_raises_instead_of_reporting_flat() -> None:
    """SAFETY: a failed read reported as [] means ten live positions become
    invisible. That is the bug that wrecked the previous system."""
    client = FakeDhanClient(positions=ConnectionError("dhan down"))
    with pytest.raises(ConnectionError):
        adapter(client).positions()

    client = FakeDhanClient(positions={"status": "failure", "remarks": "DH-901"})
    with pytest.raises(DhanApiError):
        adapter(client).positions()


def test_position_without_a_readable_average_price_raises() -> None:
    """SAFETY: avg_price 0 reads as infinite profit."""
    client = FakeDhanClient(positions=_pos_payload([
        {"securityId": "43001", "tradingSymbol": "X", "positionType": "LONG",
         "netQty": 75},
    ]))
    with pytest.raises(DhanApiError, match="average price"):
        adapter(client).positions()


def test_position_without_a_security_id_raises() -> None:
    client = FakeDhanClient(positions=_pos_payload([
        {"tradingSymbol": "X", "positionType": "LONG", "netQty": 75, "buyAvg": 10},
    ]))
    with pytest.raises(DhanApiError, match="securityId"):
        adapter(client).positions()


def test_position_without_a_readable_quantity_raises() -> None:
    client = FakeDhanClient(positions=_pos_payload([
        {"securityId": "43001", "tradingSymbol": "X", "positionType": "LONG",
         "netQty": "-", "buyAvg": 10},
    ]))
    with pytest.raises(DhanApiError, match="netQty"):
        adapter(client).positions()


# ── quotes ──────────────────────────────────────────────────────────────────

def test_quote_returns_the_last_price() -> None:
    client = FakeDhanClient(ltp={"43001": 120.25})
    assert adapter(client).quote(NIFTY_CE) == Decimal("120.25")


def test_quote_is_none_when_the_feed_carries_no_price() -> None:
    assert adapter(FakeDhanClient(ltp={})).quote(NIFTY_CE) is None


def test_quote_raises_when_the_feed_itself_is_broken() -> None:
    """SAFETY: a broken feed must not read as "no trades" — the two lead to
    different decisions."""
    class Broken(FakeDhanClient):
        def ticker_data(self, payload):  # type: ignore[override]
            raise ConnectionError("md down")

    with pytest.raises(ConnectionError):
        adapter(Broken()).quote(NIFTY_CE)


# ── multi-leg structures ────────────────────────────────────────────────────

def _spread(structure_id: str = "s1") -> list[Intent]:
    """A short call spread: sell the 25000 CE, buy the 25400 CE as protection.
    Passed SELL-first on purpose, so the ordering assertions mean something."""
    return [
        intent(NIFTY_CE, Side.SELL, "75", structure_id=structure_id, key="short"),
        intent(NIFTY_CE_WING, Side.BUY, "75", structure_id=structure_id, key="wing"),
    ]


def test_structure_places_buy_before_sell() -> None:
    """SAFETY: the protective leg goes on FIRST. Reverse this and a failure
    part-way through leaves a naked short."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.filled("43001", 75, 119.90)
    client.filled("43003", 75, 40.10)
    result = adapter(client).place_structure(_spread())
    assert result.ok is True
    assert client.placed_sids() == ["43003", "43001"]   # wing (BUY) first


def test_structure_confirms_the_buy_before_sending_the_sell() -> None:
    """SAFETY: not just placed first — CONFIRMED first. The BUY's status poll
    must land before the SELL is submitted."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.filled("43001", 75, 119.90)
    client.filled("43003", 75, 40.10)
    adapter(client).place_structure(_spread())

    names = [name for name, _ in client.calls]
    first_place = names.index("place_order")
    first_status = names.index("get_order_by_id")
    second_place = names.index("place_order", first_place + 1)
    assert first_status < second_place


def test_failed_buy_leg_never_sends_the_short() -> None:
    """SAFETY: the most expensive failure mode in the system. If the protective
    BUY does not fill, no SELL may be sent at all."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.resting("43003")            # the wing never fills
    client.filled("43001", 75, 119.90)
    result = adapter(client).place_structure(_spread(), timeout_s=2.0)

    assert result.ok is False
    assert "43001" not in client.placed_sids(), "a naked short was sent"
    assert result.failed_intent is not None
    assert result.failed_intent.instrument is NIFTY_CE_WING


def test_failed_buy_leg_with_a_partial_fill_is_unwound() -> None:
    """SAFETY: the partial residue on the failed leg is exposure too."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.resting("43003", filled_qty=75)     # 1 of 2 lots traded, then stuck
    a = adapter(client)
    legs = [
        intent(NIFTY_CE, Side.SELL, "150", structure_id="s1", key="short"),
        intent(NIFTY_CE_WING, Side.BUY, "150", structure_id="s1", key="wing"),
    ]
    # the unwind SELL of the 75 that did fill
    client.status_script["43003"] = [
        {"orderStatus": "PART_TRADED", "filledQty": 75, "averageTradedPrice": 40.0},
        {"orderStatus": "PART_TRADED", "filledQty": 75, "averageTradedPrice": 40.0},
        {"orderStatus": "PART_TRADED", "filledQty": 75, "averageTradedPrice": 40.0},
        {"orderStatus": "TRADED", "filledQty": 75, "averageTradedPrice": 39.5},
    ]
    result = a.place_structure(legs, timeout_s=2.0)

    assert result.ok is False
    assert "43001" not in client.placed_sids()
    unwind_orders = [p for p in client.placed()
                     if p["security_id"] == "43003" and p["transaction_type"] == "SELL"]
    assert len(unwind_orders) == 1
    assert unwind_orders[0]["quantity"] == 75


def test_failed_sell_leg_unwinds_the_protective_long() -> None:
    """The task's requirement: a later leg failing unwinds what was already
    placed, rather than leaving half a structure on the book."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.filled("43003", 75, 40.10)          # wing goes on
    client.script("43001", [{"orderStatus": "REJECTED"}])   # short refused
    result = adapter(client).place_structure(_spread(), timeout_s=2.0)

    assert result.ok is False
    assert result.manual_action_required is False
    sends = [(p["security_id"], p["transaction_type"]) for p in client.placed()]
    assert sends == [("43003", "BUY"), ("43001", "SELL"), ("43003", "SELL")]
    assert len(result.unwound) == 1
    assert result.unwound[0].status is OrderStatus.FILLED


def test_unwind_closes_short_before_long() -> None:
    """SAFETY: buying the short back BEFORE selling the long that covers it.
    Unwind in the other order and you re-create the naked leg you are removing."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43002": 98.0, "43003": 40.0})
    client.filled("43003", 75, 40.10)          # protective long fills
    client.filled("43001", 75, 119.90)         # first short fills
    client.resting("43002")                    # second short never fills -> abort

    legs = [
        intent(NIFTY_CE_WING, Side.BUY, "75", structure_id="s1", key="wing"),
        intent(NIFTY_CE, Side.SELL, "75", structure_id="s1", key="call"),
        intent(NIFTY_PE, Side.SELL, "75", structure_id="s1", key="put"),
    ]
    result = adapter(client).place_structure(legs, timeout_s=2.0)

    assert result.ok is False
    sends = [(p["security_id"], p["transaction_type"]) for p in client.placed()]
    assert sends[:3] == [("43003", "BUY"), ("43001", "SELL"), ("43002", "SELL")]
    unwinds = sends[3:]
    # 43001 is the live short: it must be bought back BEFORE the long that
    # covers it is sold.
    assert unwinds[0] == ("43001", "BUY")
    assert ("43003", "SELL") in unwinds
    assert unwinds.index(("43001", "BUY")) < unwinds.index(("43003", "SELL"))


def test_unwind_failure_demands_manual_action() -> None:
    """SAFETY: a failed unwind is an operator page, never a swallowed log line."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.filled("43003", 75, 40.10)
    client.script("43001", [{"orderStatus": "REJECTED"}])
    a = adapter(client)

    calls = {"n": 0}
    original = client.place_order

    def flaky(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 3:          # the unwind order
            raise ConnectionError("gone")
        return original(**kwargs)

    client.place_order = flaky  # type: ignore[method-assign]
    result = a.place_structure(_spread(), timeout_s=2.0)

    assert result.ok is False
    assert result.manual_action_required is True
    assert "UNWIND INCOMPLETE" in (result.error or "")


def test_ambiguous_submit_stops_the_sequence_and_pages() -> None:
    """SAFETY: with no order id there is nothing to cancel, so the adapter must
    stop and say so rather than firing a blind offsetting order."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.submit_error["43003"] = ConnectionError("socket reset")
    result = adapter(client).place_structure(_spread(), timeout_s=2.0)

    assert result.ok is False
    assert result.manual_action_required is True
    assert "43001" not in client.placed_sids(), "a naked short was sent"


def test_unwind_crosses_harder_than_the_entry() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.filled("43003", 75, 40.10)
    client.script("43001", [{"orderStatus": "REJECTED"}])
    a = adapter(client, limit_ticks=2, unwind_ticks=10)
    a.place_structure(_spread(), timeout_s=2.0)

    entry = client.placed()[0]["price"]      # BUY wing at ltp 40.00 + 2 ticks
    unwind = client.placed()[2]["price"]     # SELL wing at ltp 40.00 - 10 ticks
    assert entry == pytest.approx(40.10)
    assert unwind == pytest.approx(39.50)


def test_structure_requires_one_shared_structure_id() -> None:
    """SAFETY: unwind semantics apply to the whole batch. Batching unrelated
    orders through here would flatten positions that were never part of it."""
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    a = adapter(client)
    with pytest.raises(ValueError, match="structure_id"):
        a.place_structure([intent(NIFTY_CE, Side.SELL, "75", structure_id="s1"),
                           intent(NIFTY_CE_WING, Side.BUY, "75", structure_id="s2")])
    with pytest.raises(ValueError, match="structure_id"):
        a.place_structure([intent(NIFTY_CE, Side.SELL, "75")])
    assert client.placed() == []


def test_empty_structure_is_refused() -> None:
    with pytest.raises(ValueError):
        adapter(FakeDhanClient()).place_structure([])


def test_successful_structure_returns_a_ref_per_leg() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0, "43003": 40.0})
    client.filled("43001", 75, 119.90)
    client.filled("43003", 75, 40.10)
    result: StructureResult = adapter(client).place_structure(_spread())
    assert result.ok is True
    assert len(result.refs) == 2
    assert all(r.status is OrderStatus.FILLED for r in result.refs)
    assert result.unwound == ()
    assert result.manual_action_required is False


def test_idempotency_key_rides_along_as_the_dhan_tag() -> None:
    """Dhan's correlation field is `tag` and caps at 25 characters — a longer
    one is rejected outright, turning a good order into a failed submit."""
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.filled("43001", 75, 120.10)
    long_key = "fno_daily:2026-08-27:iron_fly:leg-short-call:retry-3"
    adapter(client).place(intent(NIFTY_CE, Side.BUY, "75", key=long_key))
    assert client.placed()[0]["tag"] == long_key[:25]
    assert len(client.placed()[0]["tag"]) == 25


def test_no_tag_is_sent_when_there_is_no_idempotency_key() -> None:
    client = FakeDhanClient(ltp={"43001": 120.0})
    client.filled("43001", 75, 120.10)
    adapter(client).place(intent(NIFTY_CE, Side.BUY, "75", key=""))
    assert "tag" not in client.placed()[0]
