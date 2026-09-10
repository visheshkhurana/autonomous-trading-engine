"""
Tests for OrderGate.

Every safety rule in gate.py has a test here that fails if the rule is deleted.
The fakes deliberately have two behaviours: a working mode and a FAILING mode,
because the interesting question for this module is never "does it approve a
good order" but "what does it do when it cannot establish something".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence

import pytest

from core.gate import (
    ClassLimits, GateConfig, GateStateUnavailable, OrderGate,
)
from core.profiles import IST
from core.protocols import OrderGateProto
from core.types import (
    AssetClass, Fill, Instrument, Intent, MarketContext, Mode, OrderRef,
    OrderType, Position, Side,
)

# Thursday 2026-08-27, 10:00 IST — NSE is open.
OPEN_NOW = dt.datetime(2026, 8, 27, 10, 0, tzinfo=IST)
# Saturday.
WEEKEND_NOW = dt.datetime(2026, 8, 29, 10, 0, tzinfo=IST)
AFTER_HOURS = dt.datetime(2026, 8, 27, 20, 0, tzinfo=IST)

NIFTY = Instrument(
    symbol="NIFTY26SEP24000CE", asset_class=AssetClass.EQUITY,
    broker_id="43210", lot_size=25, tick_size=Decimal("0.05"),
    qty_precision=0, min_notional=Decimal("0"),
)
RELIANCE = Instrument(
    symbol="RELIANCE", asset_class=AssetClass.EQUITY, broker_id="2885",
    lot_size=1, tick_size=Decimal("0.05"),
)
TCS = Instrument(
    symbol="TCS", asset_class=AssetClass.EQUITY, broker_id="11536",
    lot_size=1, tick_size=Decimal("0.05"),
)
BTC = Instrument(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO, broker_id="BTC/USDT",
    lot_size=1, tick_size=Decimal("0.1"), qty_precision=6,
    min_notional=Decimal("10"),
)


# ------------------------------------------------------------------ fakes


class FakePositionStore:
    """PositionStoreProto. `fail` makes positions() raise, which is what a real
    store must do rather than returning [] — [] means 'flat'."""

    def __init__(self, positions: Sequence[Position] = (), fail: bool = False) -> None:
        self._positions = tuple(positions)
        self.fail = fail

    def apply(self, fill: Fill, strategy_id: str,
              structure_id: Optional[str] = None) -> None:
        raise AssertionError("the gate must never write positions")

    def positions(self, mode: Mode,
                  asset_class: Optional[AssetClass] = None) -> Sequence[Position]:
        if self.fail:
            raise RuntimeError("position store unreachable")
        return [
            p for p in self._positions
            if p.mode is mode
            and (asset_class is None or p.instrument.asset_class is asset_class)
        ]

    def realized_pnl(self, mode: Mode, asset_class: AssetClass,
                     since_iso: str) -> Decimal:
        raise AssertionError("not used by the gate")

    def record_order(self, ref: OrderRef) -> None:
        raise AssertionError("the gate must never record orders")


class FakeGateState:
    def __init__(self, *, seen: Sequence[str] = (), count: int = 0,
                 fail_on: Sequence[str] = (), count_value: object = None,
                 spent_notional: Decimal = Decimal("0")) -> None:
        self._seen = set(seen)
        self._count = count
        self._fail_on = set(fail_on)
        self._count_value = count_value
        self._spent_notional = spent_notional
        self.submissions: list[tuple[str, AssetClass, str]] = []

    def seen_recently(self, idempotency_key: str, window_s: float,
                      now: dt.datetime) -> bool:
        if "seen_recently" in self._fail_on:
            raise GateStateUnavailable("redis down")
        if "seen_recently_type" in self._fail_on:
            return "no"          # a store that answers with garbage
        return idempotency_key in self._seen

    def order_count(self, asset_class: AssetClass, day_key: str) -> int:
        if "order_count" in self._fail_on:
            raise GateStateUnavailable("redis down")
        if self._count_value is not None:
            return self._count_value          # type: ignore[return-value]
        return self._count

    def daily_notional(self, asset_class: AssetClass, day_key: str) -> Decimal:
        if "daily_notional" in self._fail_on:
            raise GateStateUnavailable("redis down")
        return self._spent_notional

    def record_submission(self, idempotency_key: str, asset_class: AssetClass,
                          day_key: str, now: dt.datetime,
                          notional: Decimal = Decimal("0")) -> None:
        if "record_submission" in self._fail_on:
            raise GateStateUnavailable("redis down")
        self._seen.add(idempotency_key)
        self._count += 1
        self._spent_notional += notional
        self.submissions.append((idempotency_key, asset_class, day_key))


@dataclass(frozen=True)
class FakeDecision:
    state: str
    reason: str = ""


@dataclass(frozen=True)
class ThrottleDecision:
    """Shaped like core.risk.CircuitDecision: an explicit entries_allowed and a
    size_multiplier the gate is expected to apply."""
    action: str
    entries_allowed: bool
    size_multiplier: object
    reason: str = ""


class FakeCircuit:
    def __init__(self, state: object = "normal", fail: bool = False) -> None:
        self.state = state
        self.fail = fail
        self.reads: list[tuple[AssetClass, Optional[Decimal]]] = []

    def read(self, asset_class: AssetClass, equity: Optional[Decimal]) -> object:
        if self.fail:
            raise RuntimeError("risk state unreadable")
        self.reads.append((asset_class, equity))
        if isinstance(self.state, str):
            return FakeDecision(self.state)
        return self.state

    def advance(self, asset_class: AssetClass, equity: Decimal) -> object:
        raise AssertionError("the gate must never advance the circuit")


# --------------------------------------------------------------- builders


def build_gate(*, positions: Sequence[Position] = (), store_fail: bool = False,
               state: Optional[FakeGateState] = None,
               circuit: Optional[FakeCircuit] = None,
               config: Optional[GateConfig] = None,
               env: Optional[dict[str, str]] = None) -> OrderGate:
    return OrderGate(
        position_store=FakePositionStore(positions, fail=store_fail),
        gate_state=state if state is not None else FakeGateState(),
        risk_circuit=circuit if circuit is not None else FakeCircuit(),
        config=config if config is not None else GateConfig(),
        env={} if env is None else env,
    )


def equity_intent(*, qty: str = "25", side: Side = Side.BUY,
                  price: str = "100", key: str = "k1",
                  instrument: Instrument = NIFTY,
                  order_type: OrderType = OrderType.LIMIT) -> Intent:
    return Intent(
        instrument=instrument, side=side, qty=Decimal(qty),
        order_type=order_type, strategy_id="s1", reason="unit test",
        limit_price=Decimal(price) if order_type is OrderType.LIMIT else None,
        idempotency_key=key,
    )


def ctx(equity: Optional[str] = "1000000", now: dt.datetime = OPEN_NOW,
        quotes: Optional[dict[str, Decimal]] = None) -> MarketContext:
    return MarketContext(
        now=now,
        quotes=quotes or {},
        equity=Decimal(equity) if equity is not None else None,
    )


# ------------------------------------------------------------------ tests


def test_gate_satisfies_the_protocol() -> None:
    assert isinstance(build_gate(), OrderGateProto)


def test_clean_order_is_approved_and_records_exactly_one_submission() -> None:
    state = FakeGateState()
    gate = build_gate(state=state)
    approval = gate.check(equity_intent(), ctx())
    assert approval.approved, approval.reason
    assert approval.reason == "approved"
    assert state.submissions == [("k1", AssetClass.EQUITY, "2026-08-27")]
    # Every check ran and is visible to the UI.
    names = [c.name for c in approval.checks]
    assert names[0] == "sanity" and names[-1] == "mode_coercion"
    assert len(names) == 11


# --- 1. sanity -----------------------------------------------------------


def test_limit_order_without_a_price_is_refused_even_if_it_was_constructed() -> None:
    intent = equity_intent()
    # Intent.__post_init__ forbids this; a serialisation round-trip does not.
    object.__setattr__(intent, "limit_price", None)
    approval = build_gate().check(intent, ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "sanity"
    assert "limit_price" in approval.rejections[0].detail


def test_zero_qty_is_refused_even_if_it_was_constructed() -> None:
    intent = equity_intent()
    object.__setattr__(intent, "qty", Decimal("0"))
    approval = build_gate().check(intent, ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "sanity"


def test_instrument_without_broker_id_cannot_be_routed() -> None:
    broken = Instrument(symbol="X", asset_class=AssetClass.EQUITY, broker_id="  ")
    approval = build_gate().check(equity_intent(instrument=broken), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "sanity"
    assert "broker_id" in approval.rejections[0].detail


def test_market_order_is_refused_for_equity_because_the_venue_rejects_it() -> None:
    intent = equity_intent(order_type=OrderType.MARKET)
    approval = build_gate().check(intent, ctx(quotes={NIFTY.symbol: Decimal("100")}))
    assert not approval.approved
    assert approval.rejections[0].name == "sanity"
    assert "MARKET" in approval.rejections[0].detail


def test_naive_now_is_refused_because_it_silently_shifts_the_session() -> None:
    naive = MarketContext(now=dt.datetime(2026, 8, 27, 10, 0),
                          equity=Decimal("1000000"))
    approval = build_gate().check(equity_intent(), naive)
    assert not approval.approved
    assert approval.rejections[0].name == "sanity"
    assert "timezone-aware" in approval.rejections[0].detail


# --- 2. market hours -----------------------------------------------------


def test_weekend_equity_order_is_refused() -> None:
    approval = build_gate().check(equity_intent(), ctx(now=WEEKEND_NOW))
    assert not approval.approved
    assert approval.rejections[0].name == "market_open"


def test_after_hours_equity_order_is_refused() -> None:
    approval = build_gate().check(equity_intent(), ctx(now=AFTER_HOURS))
    assert not approval.approved
    assert approval.rejections[0].name == "market_open"


def test_crypto_trades_when_the_equity_market_is_shut() -> None:
    intent = Intent(
        instrument=BTC, side=Side.BUY, qty=Decimal("0.01"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="c1",
    )
    approval = build_gate().check(
        intent,
        ctx(equity="100000", now=WEEKEND_NOW,
            quotes={BTC.symbol: Decimal("50000")}),
    )
    assert approval.approved, approval.reason


# --- 3. quantization -----------------------------------------------------


def test_qty_that_snaps_to_zero_is_refused() -> None:
    approval = build_gate().check(equity_intent(qty="10"), ctx())  # lot_size 25
    assert not approval.approved
    assert approval.rejections[0].name == "quantization"


def test_approval_carries_the_snapped_qty_not_the_requested_one() -> None:
    approval = build_gate().check(equity_intent(qty="40"), ctx())
    assert approval.approved, approval.reason
    # 40 -> 25 (one lot). Handing back 40 would make the check decorative.
    assert approval.intent.qty == Decimal("25")


def test_absolute_order_notional_is_tighter_than_the_equity_fraction() -> None:
    """1 BTC at 78k would pass a 10% of 2M cap. The 50 USDT clip must bind."""
    config = GateConfig(limits={
        AssetClass.CRYPTO: ClassLimits(
            max_order_notional=Decimal("50"),
            max_daily_notional=Decimal("250"),
            max_open_positions=1,
        ),
    })
    intent = Intent(
        instrument=BTC, side=Side.BUY, qty=Decimal("1"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="c1",
    )
    approval = build_gate(config=config).check(
        intent, ctx(equity="2000000", quotes={BTC.symbol: Decimal("78000")}))
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("0.000641")
    assert approval.intent.qty * Decimal("78000") <= Decimal("50")


def test_a_reducing_crypto_exit_does_not_consume_daily_notional() -> None:
    config = GateConfig(limits={
        AssetClass.CRYPTO: ClassLimits(
            max_order_notional=Decimal("50"),
            max_daily_notional=Decimal("250"),
            max_open_positions=1,
        ),
    })
    state = FakeGateState(spent_notional=Decimal("250"))
    book = [Position(instrument=BTC, qty=Decimal("0.000641"),
                     avg_price=Decimal("78000"), mode=Mode.PAPER, strategy_id="s1")]
    intent = Intent(
        instrument=BTC, side=Side.SELL, qty=Decimal("0.000641"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="exit1",
    )
    approval = build_gate(positions=book, config=config, state=state).check(
        intent, ctx(equity="2000000", quotes={BTC.symbol: Decimal("78000")}))
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("0.000641")
    assert state._spent_notional == Decimal("250")


def test_absolute_daily_notional_refuses_when_the_day_is_spent() -> None:
    config = GateConfig(limits={
        AssetClass.CRYPTO: ClassLimits(
            max_order_notional=Decimal("50"),
            max_daily_notional=Decimal("250"),
            max_open_positions=1,
        ),
    })
    state = FakeGateState(spent_notional=Decimal("250"))
    intent = Intent(
        instrument=BTC, side=Side.BUY, qty=Decimal("0.000641"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="c1",
    )
    approval = build_gate(config=config, state=state).check(
        intent, ctx(equity="2000000", quotes={BTC.symbol: Decimal("78000")}))
    assert not approval.approved
    assert approval.rejections[0].name == "daily_notional_cap"


def test_crypto_qty_is_snapped_to_precision() -> None:
    intent = Intent(
        instrument=BTC, side=Side.BUY, qty=Decimal("0.0123456789"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="c1",
    )
    approval = build_gate().check(
        intent, ctx(equity="100000", quotes={BTC.symbol: Decimal("50000")}))
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("0.012345")


# --- 4. min notional -----------------------------------------------------


def test_order_below_instrument_min_notional_is_refused() -> None:
    intent = Intent(
        instrument=BTC, side=Side.BUY, qty=Decimal("0.0001"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="c1",
    )
    approval = build_gate().check(
        intent, ctx(equity="100000", quotes={BTC.symbol: Decimal("50000")}))
    assert not approval.approved
    assert approval.rejections[0].name == "min_notional"


def test_market_order_without_a_quote_is_refused_rather_than_sized_on_a_guess() -> None:
    intent = Intent(
        instrument=BTC, side=Side.BUY, qty=Decimal("1"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="c1",
    )
    approval = build_gate().check(intent, ctx(equity="100000", quotes={}))
    assert not approval.approved
    assert approval.rejections[0].name == "min_notional"
    assert "price" in approval.rejections[0].detail


def test_configured_min_notional_floor_is_enforced() -> None:
    config = GateConfig(default_limits=ClassLimits(min_notional_floor=Decimal("5000")))
    approval = build_gate(config=config).check(
        equity_intent(qty="25", price="100"), ctx())   # notional 2500
    assert not approval.approved
    assert approval.rejections[0].name == "min_notional"


# --- 5. dedup ------------------------------------------------------------


def test_repeat_idempotency_key_inside_the_window_is_refused() -> None:
    gate = build_gate(state=FakeGateState(seen=["k1"]))
    approval = gate.check(equity_intent(key="k1"), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "dedup"


def test_intent_without_an_idempotency_key_is_refused() -> None:
    approval = build_gate().check(equity_intent(key=""), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "dedup"


def test_second_identical_intent_is_refused_by_the_first_ones_record() -> None:
    state = FakeGateState()
    gate = build_gate(state=state)
    assert gate.check(equity_intent(key="same"), ctx()).approved
    second = gate.check(equity_intent(key="same"), ctx())
    assert not second.approved
    assert second.rejections[0].name == "dedup"


# --- the counter is spent only on approval -------------------------------


def test_rejected_orders_do_not_burn_the_daily_quota() -> None:
    """The defect that motivated this gate: both old gates incremented the
    daily counter before the checks, so rejected orders locked out good ones."""
    state = FakeGateState()
    gate = build_gate(state=state)
    for _ in range(50):
        rejected = gate.check(equity_intent(qty="10", key="dup-reject"), ctx())
        assert not rejected.approved
    assert state.submissions == []
    assert state.order_count(AssetClass.EQUITY, "2026-08-27") == 0
    # And the quota is still fully available afterwards.
    assert gate.check(equity_intent(key="good"), ctx()).approved


def test_a_rejection_does_not_claim_the_idempotency_key() -> None:
    state = FakeGateState()
    gate = build_gate(state=state, circuit=FakeCircuit("flatten"))
    assert not gate.check(equity_intent(key="k1"), ctx()).approved
    # Circuit clears; the same decision must still be placeable.
    gate2 = build_gate(state=state, circuit=FakeCircuit("normal"))
    assert gate2.check(equity_intent(key="k1"), ctx()).approved


def test_a_failure_to_record_refuses_rather_than_placing_an_unaccounted_order() -> None:
    state = FakeGateState(fail_on=["record_submission"])
    approval = build_gate(state=state).check(equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "dedup"


# --- 6/7. equity-denominated caps ---------------------------------------


def test_order_above_the_notional_fraction_of_live_equity_is_sized_down() -> None:
    # 10% of 1,000,000 = 100,000. RELIANCE lot=1, 25 * 5000 = 125,000 -> 20.
    approval = build_gate().check(
        equity_intent(qty="25", price="5000", instrument=RELIANCE), ctx())
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("20")
    cap = [c for c in approval.checks if c.name == "notional_cap"][0]
    assert cap.passed
    assert "20" in cap.detail


def test_a_scale_that_snaps_to_zero_lots_is_still_refused() -> None:
    """NIFTY lot=25. 100_000 cap / 5000 = 20, which is not a whole lot."""
    approval = build_gate().check(equity_intent(qty="25", price="5000"), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "notional_cap"
    assert "nothing to place" in approval.rejections[0].detail


def test_sizing_does_not_run_when_equity_is_unknown() -> None:
    """Fail closed: a missing equity cannot be a licence to size against a guess."""
    approval = build_gate().check(
        equity_intent(qty="25", price="5000", instrument=RELIANCE),
        ctx(equity=None))
    assert not approval.approved
    assert approval.rejections[0].name == "notional_cap"
    assert "equity unavailable" in approval.rejections[0].detail


def test_reducing_orders_are_not_sized_down_to_the_cap() -> None:
    """Shrinking an exit leaves a tail on."""
    book = [Position(instrument=RELIANCE, qty=Decimal("25"),
                     avg_price=Decimal("5000"), mode=Mode.PAPER, strategy_id="s1")]
    approval = build_gate(positions=book).check(
        equity_intent(qty="25", price="5000", instrument=RELIANCE, side=Side.SELL),
        ctx())
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("25")


def test_sizing_does_not_change_paper_vs_live() -> None:
    """Paper vs live is an adapter choice. Sizing an order must not arm it."""
    approval = build_gate().check(
        equity_intent(qty="25", price="5000", instrument=RELIANCE), ctx())
    assert approval.approved
    assert approval.mode is Mode.PAPER
    live = build_gate(env={"EQUITY_LIVE": "true"},
                      config=GateConfig(requested_mode=Mode.LIVE)).check(
        equity_intent(qty="25", price="5000", instrument=RELIANCE, key="k2"),
        ctx())
    assert live.approved
    assert live.mode is Mode.LIVE
    assert live.intent.qty == Decimal("20")


def test_the_notional_cap_shrinks_with_equity() -> None:
    """A cap that is a fraction of live equity, not a constant: the same order
    that passes on a big account is sized down (or refused, if it cannot
    print a lot) on a halved one."""
    intent = equity_intent(qty="25", price="300", instrument=RELIANCE)  # 7,500
    assert build_gate().check(intent, ctx(equity="100000")).approved
    approval = build_gate().check(intent, ctx(equity="50000"))  # cap 5,000
    assert approval.approved, approval.reason
    # 5000/300 = 16.666 -> 16 lots of 1.
    assert approval.intent.qty == Decimal("16")


def test_unavailable_equity_refuses_instead_of_assuming_headroom() -> None:
    approval = build_gate().check(equity_intent(), ctx(equity=None))
    assert not approval.approved
    assert approval.rejections[0].name == "notional_cap"
    assert "equity unavailable" in approval.rejections[0].detail


def test_zero_equity_refuses_instead_of_reading_as_a_valid_account() -> None:
    approval = build_gate().check(equity_intent(), ctx(equity="0"))
    assert not approval.approved
    assert approval.rejections[0].name == "notional_cap"


def test_float_equity_refuses_because_the_caps_are_decimal_arithmetic() -> None:
    bad = MarketContext(now=OPEN_NOW, equity=1_000_000.0)  # type: ignore[arg-type]
    approval = build_gate().check(equity_intent(), bad)
    assert not approval.approved
    assert approval.rejections[0].name == "notional_cap"


def test_the_daily_counter_rolls_on_the_asset_classes_own_day_boundary() -> None:
    """02:00 IST is still the previous UTC day. Recording crypto against the IST
    date would reset the crypto cap in the middle of its trading day."""
    state = FakeGateState()
    intent = Intent(
        instrument=BTC, side=Side.BUY, qty=Decimal("0.01"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="c1",
    )
    now = dt.datetime(2026, 8, 28, 2, 0, tzinfo=IST)   # 2026-08-27 20:30 UTC
    approval = build_gate(state=state).check(
        intent, ctx(equity="100000", now=now,
                    quotes={BTC.symbol: Decimal("50000")}))
    assert approval.approved, approval.reason
    assert state.submissions == [("c1", AssetClass.CRYPTO, "2026-08-27")]


def test_allocation_cap_sizes_down_to_remaining_room() -> None:
    book = [
        Position(instrument=RELIANCE, qty=Decimal("100"),
                 avg_price=Decimal("2500"), mode=Mode.PAPER, strategy_id="s1"),
    ]  # 250,000 exposure; cap is 30% of 1,000,000 = 300,000; room 50,000
    approval = build_gate(positions=book).check(
        equity_intent(qty="25", price="3000", instrument=RELIANCE), ctx())
    assert approval.approved, approval.reason
    # 50_000 / 3000 = 16.666 -> 16
    assert approval.intent.qty == Decimal("16")


def test_allocation_cap_sums_short_exposure_by_magnitude() -> None:
    """Netting a long against a short would let a large paired book look flat."""
    book = [
        Position(instrument=RELIANCE, qty=Decimal("100"),
                 avg_price=Decimal("2500"), mode=Mode.PAPER, strategy_id="s1"),
        Position(instrument=TCS, qty=Decimal("-100"),
                 avg_price=Decimal("2500"), mode=Mode.PAPER, strategy_id="s1"),
    ]
    approval = build_gate(positions=book).check(
        equity_intent(qty="25", price="100"), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "allocation_cap"


def test_positions_from_the_other_mode_do_not_consume_the_allocation() -> None:
    live_book = [
        Position(instrument=RELIANCE, qty=Decimal("100"),
                 avg_price=Decimal("2500"), mode=Mode.LIVE, strategy_id="s1"),
    ]
    approval = build_gate(positions=live_book).check(
        equity_intent(qty="25", price="3000"), ctx())
    assert approval.approved, approval.reason


# --- 8. max open positions ----------------------------------------------


def test_max_open_positions_blocks_a_new_instrument() -> None:
    config = GateConfig(default_limits=ClassLimits(max_open_positions=1))
    book = [Position(instrument=RELIANCE, qty=Decimal("1"),
                     avg_price=Decimal("100"), mode=Mode.PAPER, strategy_id="s1")]
    approval = build_gate(positions=book, config=config).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "max_open_positions"


def test_adding_to_an_existing_position_needs_no_new_slot() -> None:
    config = GateConfig(default_limits=ClassLimits(max_open_positions=1))
    book = [Position(instrument=NIFTY, qty=Decimal("25"),
                     avg_price=Decimal("100"), mode=Mode.PAPER, strategy_id="s1")]
    approval = build_gate(positions=book, config=config).check(
        equity_intent(), ctx())
    assert approval.approved, approval.reason


# --- 9. daily order cap --------------------------------------------------


def test_daily_order_cap_refuses_once_the_count_is_reached() -> None:
    config = GateConfig(default_limits=ClassLimits(max_orders_per_day=3))
    state = FakeGateState(count=3)
    approval = build_gate(state=state, config=config).check(equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "daily_order_cap"


def test_daily_order_cap_allows_the_last_slot() -> None:
    config = GateConfig(default_limits=ClassLimits(max_orders_per_day=3))
    state = FakeGateState(count=2)
    approval = build_gate(state=state, config=config).check(equity_intent(), ctx())
    assert approval.approved, approval.reason


def test_a_non_int_order_count_refuses_rather_than_being_coerced() -> None:
    state = FakeGateState(count_value="lots")
    approval = build_gate(state=state).check(equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "daily_order_cap"


# --- stores that fail must close the gate, never fall back to memory -----


def test_unreadable_position_store_refuses() -> None:
    approval = build_gate(store_fail=True).check(equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "allocation_cap"
    assert "unavailable" in approval.rejections[0].detail


def test_unreadable_dedup_store_refuses() -> None:
    state = FakeGateState(fail_on=["seen_recently"])
    approval = build_gate(state=state).check(equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "dedup"


def test_dedup_store_answering_with_garbage_refuses() -> None:
    state = FakeGateState(fail_on=["seen_recently_type"])
    approval = build_gate(state=state).check(equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "dedup"


def test_unreadable_order_count_refuses() -> None:
    state = FakeGateState(fail_on=["order_count"])
    approval = build_gate(state=state).check(equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "daily_order_cap"


def test_a_broken_store_never_degrades_to_an_in_memory_count() -> None:
    """Repeated calls against a broken store must never start approving."""
    state = FakeGateState(fail_on=["order_count"])
    gate = build_gate(state=state)
    for i in range(10):
        assert not gate.check(equity_intent(key=f"k{i}"), ctx()).approved


# --- 10. risk circuit ----------------------------------------------------


def test_flatten_blocks_a_new_entry() -> None:
    approval = build_gate(circuit=FakeCircuit("flatten")).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_pause_blocks_a_new_entry() -> None:
    approval = build_gate(circuit=FakeCircuit("paused")).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_flatten_still_allows_the_exit_that_carries_it_out() -> None:
    """A flatten instruction that blocks its own exits traps the book."""
    book = [Position(instrument=NIFTY, qty=Decimal("50"),
                     avg_price=Decimal("100"), mode=Mode.PAPER, strategy_id="s1")]
    approval = build_gate(positions=book, circuit=FakeCircuit("flatten")).check(
        equity_intent(qty="50", side=Side.SELL), ctx())
    assert approval.approved, approval.reason


def test_flatten_blocks_a_sell_larger_than_the_position_because_it_flips_side() -> None:
    book = [Position(instrument=NIFTY, qty=Decimal("25"),
                     avg_price=Decimal("100"), mode=Mode.PAPER, strategy_id="s1")]
    approval = build_gate(positions=book, circuit=FakeCircuit("flatten")).check(
        equity_intent(qty="100", side=Side.SELL), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_buying_to_cover_a_short_counts_as_reducing() -> None:
    book = [Position(instrument=NIFTY, qty=Decimal("-50"),
                     avg_price=Decimal("100"), mode=Mode.PAPER, strategy_id="s1")]
    approval = build_gate(positions=book, circuit=FakeCircuit("flatten")).check(
        equity_intent(qty="50", side=Side.BUY), ctx())
    assert approval.approved, approval.reason


def test_reduce_annotates_but_does_not_block() -> None:
    approval = build_gate(circuit=FakeCircuit("reduce_only")).check(
        equity_intent(), ctx())
    assert approval.approved, approval.reason
    circuit_check = next(c for c in approval.checks if c.name == "risk_circuit")
    assert "reduce_only" in circuit_check.detail


def test_unreadable_circuit_refuses() -> None:
    approval = build_gate(circuit=FakeCircuit(fail=True)).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_uninterpretable_circuit_state_refuses_instead_of_assuming_clear() -> None:
    approval = build_gate(circuit=FakeCircuit("banana")).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_a_circuit_decision_with_no_state_refuses() -> None:
    approval = build_gate(circuit=FakeCircuit(object())).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_entries_allowed_false_blocks_even_when_the_state_word_looks_fine() -> None:
    """core.risk folds 'I could not read equity' into entries_allowed. A gate
    that only pattern-matched the word 'ok' would trade on unverified numbers."""
    decision = ThrottleDecision(action="ok", entries_allowed=False,
                               size_multiplier=Decimal("0"))
    approval = build_gate(circuit=FakeCircuit(decision)).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_a_size_multiplier_is_applied_not_merely_reported() -> None:
    decision = ThrottleDecision(action="reduce", entries_allowed=True,
                               size_multiplier=Decimal("0.5"))
    approval = build_gate(circuit=FakeCircuit(decision)).check(
        equity_intent(qty="100"), ctx())
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("50")


def test_a_throttle_that_leaves_nothing_placeable_is_refused() -> None:
    decision = ThrottleDecision(action="reduce", entries_allowed=True,
                               size_multiplier=Decimal("0.1"))
    # 25 (one lot) * 0.1 = 2.5, which snaps back to zero lots.
    approval = build_gate(circuit=FakeCircuit(decision)).check(
        equity_intent(qty="25"), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_a_throttled_order_is_re_checked_against_the_min_notional() -> None:
    config = GateConfig(default_limits=ClassLimits(min_notional_floor=Decimal("4000")))
    decision = ThrottleDecision(action="reduce", entries_allowed=True,
                               size_multiplier=Decimal("0.5"))
    # 50 * 100 = 5,000 clears the floor; throttled to 25 * 100 = 2,500 does not.
    approval = build_gate(config=config, circuit=FakeCircuit(decision)).check(
        equity_intent(qty="50", price="100"), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "min_notional"


def test_a_nonsense_size_multiplier_refuses_rather_than_being_ignored() -> None:
    for bad in (Decimal("1.5"), Decimal("-1"), 0.5, "half"):
        decision = ThrottleDecision(action="ok", entries_allowed=True,
                                   size_multiplier=bad)
        approval = build_gate(circuit=FakeCircuit(decision)).check(
            equity_intent(), ctx())
        assert not approval.approved, bad
        assert approval.rejections[0].name == "risk_circuit"


def test_an_exit_is_never_throttled() -> None:
    """Halving an exit leaves a tail on in exactly the state where the circuit
    wants the book smaller."""
    book = [Position(instrument=NIFTY, qty=Decimal("100"),
                     avg_price=Decimal("100"), mode=Mode.PAPER, strategy_id="s1")]
    decision = ThrottleDecision(action="reduce", entries_allowed=True,
                               size_multiplier=Decimal("0.5"))
    approval = build_gate(positions=book, circuit=FakeCircuit(decision)).check(
        equity_intent(qty="100", side=Side.SELL), ctx())
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("100")


def test_the_circuit_is_read_with_the_live_equity_it_needs() -> None:
    circuit = FakeCircuit("normal")
    build_gate(circuit=circuit).check(equity_intent(), ctx(equity="777000"))
    assert circuit.reads == [(AssetClass.EQUITY, Decimal("777000"))]


def test_exposure_caps_do_not_trap_a_position_that_is_being_reduced() -> None:
    book = [Position(instrument=NIFTY, qty=Decimal("500"),
                     avg_price=Decimal("5000"), mode=Mode.PAPER, strategy_id="s1")]
    # Exposure 2,500,000 already breaches every cap against 1,000,000 equity.
    approval = build_gate(positions=book).check(
        equity_intent(qty="500", side=Side.SELL, price="5000"), ctx())
    assert approval.approved, approval.reason


# --- 11. mode coercion ---------------------------------------------------


def test_live_request_is_coerced_to_paper_when_the_class_is_disarmed() -> None:
    """The headline rule: a stale LIVE-flagged object cannot place a real order
    while its asset class is disarmed."""
    config = GateConfig(requested_mode=Mode.LIVE)
    approval = build_gate(config=config, env={}).check(equity_intent(), ctx())
    assert approval.approved, approval.reason
    assert approval.mode is Mode.PAPER
    coercion = next(c for c in approval.checks if c.name == "mode_coercion")
    assert "coerced to paper" in coercion.detail


def test_a_rejected_approval_never_carries_live_while_disarmed() -> None:
    config = GateConfig(requested_mode=Mode.LIVE)
    approval = build_gate(config=config, env={}).check(
        equity_intent(qty="10"), ctx())     # snaps to zero -> rejected early
    assert not approval.approved
    assert approval.mode is Mode.PAPER


def test_live_request_stays_live_when_the_class_is_armed() -> None:
    config = GateConfig(requested_mode=Mode.LIVE)
    approval = build_gate(config=config, env={"EQUITY_LIVE": "true"}).check(
        equity_intent(), ctx())
    assert approval.approved, approval.reason
    assert approval.mode is Mode.LIVE


def test_arming_crypto_does_not_arm_equity() -> None:
    config = GateConfig(requested_mode=Mode.LIVE)
    gate = build_gate(config=config, env={"CRYPTO_LIVE": "1"})
    assert gate.check(equity_intent(), ctx()).mode is Mode.PAPER
    crypto = Intent(
        instrument=BTC, side=Side.BUY, qty=Decimal("0.01"),
        order_type=OrderType.MARKET, strategy_id="s1", reason="unit test",
        idempotency_key="c1",
    )
    approval = gate.check(
        crypto, ctx(equity="100000", quotes={BTC.symbol: Decimal("50000")}))
    assert approval.approved, approval.reason
    assert approval.mode is Mode.LIVE


def test_paper_request_stays_paper_even_when_armed() -> None:
    config = GateConfig(requested_mode=Mode.PAPER)
    approval = build_gate(config=config, env={"EQUITY_LIVE": "1"}).check(
        equity_intent(), ctx())
    assert approval.mode is Mode.PAPER


def test_default_config_is_paper() -> None:
    assert GateConfig().requested_mode is Mode.PAPER


def test_arming_is_read_once_so_it_cannot_change_mid_session() -> None:
    env: dict[str, str] = {}
    config = GateConfig(requested_mode=Mode.LIVE)
    gate = build_gate(config=config, env=env)
    env["EQUITY_LIVE"] = "1"          # flipped after construction
    assert gate.check(equity_intent(), ctx()).mode is Mode.PAPER


def test_an_unparseable_arming_flag_raises_instead_of_reading_as_disarmed() -> None:
    with pytest.raises(ValueError, match="EQUITY_LIVE"):
        build_gate(env={"EQUITY_LIVE": "ture"})


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "FALSE"])
def test_falsey_arming_values_disarm(raw: str) -> None:
    config = GateConfig(requested_mode=Mode.LIVE)
    gate = build_gate(config=config, env={"EQUITY_LIVE": raw})
    assert gate.check(equity_intent(), ctx()).mode is Mode.PAPER


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_arming_values_arm(raw: str) -> None:
    config = GateConfig(requested_mode=Mode.LIVE)
    gate = build_gate(config=config, env={"EQUITY_LIVE": raw})
    assert gate.check(equity_intent(), ctx()).mode is Mode.LIVE


def test_the_gate_reads_the_book_of_the_mode_it_will_actually_execute_in() -> None:
    """Disarmed + requested live: caps must be measured against the paper book,
    not the live one."""
    live_book = [Position(instrument=RELIANCE, qty=Decimal("1000"),
                          avg_price=Decimal("2500"), mode=Mode.LIVE,
                          strategy_id="s1")]
    config = GateConfig(requested_mode=Mode.LIVE)
    approval = build_gate(positions=live_book, config=config, env={}).check(
        equity_intent(), ctx())
    assert approval.approved, approval.reason
    assert approval.mode is Mode.PAPER


# --- the gate cannot be reconfigured or bypassed -------------------------


def test_the_gate_exposes_only_check() -> None:
    public = [n for n in dir(build_gate())
              if not n.startswith("_") and callable(getattr(build_gate(), n))]
    assert public == ["check"]


def test_collaborators_cannot_be_swapped_after_construction() -> None:
    gate = build_gate()
    with pytest.raises(AttributeError):
        gate._OrderGate__config = GateConfig()          # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        gate._OrderGate__armed = {AssetClass.EQUITY: True}   # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        del gate._OrderGate__circuit                    # type: ignore[attr-defined]


def test_caps_cannot_be_widened_through_the_config_mapping() -> None:
    limits = {AssetClass.EQUITY: ClassLimits(max_orders_per_day=1)}
    config = GateConfig(limits=limits)
    limits[AssetClass.EQUITY] = ClassLimits(max_orders_per_day=999)
    assert config.limits_for(AssetClass.EQUITY).max_orders_per_day == 1
    with pytest.raises(TypeError):
        config.limits[AssetClass.EQUITY] = ClassLimits()   # type: ignore[index]


# --- config validation ---------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"max_order_notional_fraction": Decimal("1.5")},
    {"max_order_notional_fraction": Decimal("0")},
    {"max_allocation_fraction": Decimal("-0.1")},
    {"max_open_positions": 0},
    {"max_orders_per_day": 0},
    {"min_notional_floor": Decimal("-1")},
])
def test_nonsensical_limits_do_not_construct(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        ClassLimits(**kwargs)


def test_an_order_cap_above_the_allocation_cap_does_not_construct() -> None:
    with pytest.raises(ValueError):
        ClassLimits(max_order_notional_fraction=Decimal("0.5"),
                    max_allocation_fraction=Decimal("0.2"))


def test_a_zero_dedup_window_does_not_construct() -> None:
    with pytest.raises(ValueError):
        GateConfig(dedup_window_s=0)


def test_per_class_limits_override_the_default() -> None:
    config = GateConfig(
        default_limits=ClassLimits(max_orders_per_day=20),
        limits={AssetClass.EQUITY: ClassLimits(max_orders_per_day=1)},
    )
    state = FakeGateState(count=1)
    approval = build_gate(state=state, config=config).check(equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "daily_order_cap"


def test_the_gate_never_writes_to_the_position_store() -> None:
    # FakePositionStore.apply/record_order raise AssertionError if touched.
    assert build_gate().check(equity_intent(), ctx()).approved


# --- integration with the real circuit decision --------------------------
#
# The gate reads the circuit structurally so the two modules stay independent.
# These tests pin that the structure it reads is the one core.risk actually
# emits — if that contract drifts, this fails rather than the gate silently
# refusing (or worse, silently permitting) in production.


def _decision(action, *, known: bool = True, multiplier: str = "0"):
    from core.risk import CircuitDecision
    return CircuitDecision(
        action=action,
        drawdown_pct=Decimal("0.05") if known else None,
        daily_loss_pct=Decimal("0.01") if known else None,
        reason="unit test",
        equity_known=known,
        state_known=known,
        size_multiplier=Decimal(multiplier),
    )


def test_real_circuit_ok_decision_is_honoured() -> None:
    from core.risk import CircuitAction
    decision = _decision(CircuitAction.OK, multiplier="1")
    approval = build_gate(circuit=FakeCircuit(decision)).check(
        equity_intent(qty="50"), ctx())
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("50")


def test_real_circuit_reduce_decision_shrinks_the_order() -> None:
    from core.risk import CircuitAction
    decision = _decision(CircuitAction.REDUCE, multiplier="0.5")
    approval = build_gate(circuit=FakeCircuit(decision)).check(
        equity_intent(qty="100"), ctx())
    assert approval.approved, approval.reason
    assert approval.intent.qty == Decimal("50")


@pytest.mark.parametrize("name", ["PAUSE", "FLATTEN"])
def test_real_circuit_pause_and_flatten_block_new_entries(name: str) -> None:
    from core.risk import CircuitAction
    decision = _decision(getattr(CircuitAction, name))
    approval = build_gate(circuit=FakeCircuit(decision)).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_real_circuit_with_unknown_inputs_blocks_new_entries() -> None:
    """core.risk emits PAUSE with equity_known=False when it cannot see. The
    gate must block on that, not on the word alone."""
    from core.risk import CircuitAction
    decision = _decision(CircuitAction.PAUSE, known=False)
    assert decision.entries_allowed is False
    approval = build_gate(circuit=FakeCircuit(decision)).check(
        equity_intent(), ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_real_circuit_flatten_still_lets_the_exit_through() -> None:
    from core.risk import CircuitAction
    book = [Position(instrument=NIFTY, qty=Decimal("50"),
                     avg_price=Decimal("100"), mode=Mode.PAPER, strategy_id="s1")]
    decision = _decision(CircuitAction.FLATTEN)
    approval = build_gate(positions=book, circuit=FakeCircuit(decision)).check(
        equity_intent(qty="50", side=Side.SELL), ctx())
    assert approval.approved, approval.reason
