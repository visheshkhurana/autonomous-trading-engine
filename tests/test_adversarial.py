"""
Adversarial tests for the safety-critical path: core/gate.py, core/risk.py,
core/store.py.

Every test here started life as an attack. The ones that pass on first write
are regression pins for a rule that already held; the ones that failed are the
holes that were found, and each is named for the money it would have cost.

The fakes below are deliberately independent of tests/test_gate.py's fakes:
these tests exist to catch a rule being deleted, and sharing a helper with the
suite that the rule's author wrote is how two tests end up asserting the same
mistake.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Optional, Sequence

import pytest

from core.gate import ClassLimits, GateConfig, GateStateUnavailable, OrderGate
from core.profiles import IST, UTC
from core.risk import (
    CircuitAction, CircuitDecision, RiskCircuit, RiskState,
)
from core.store import SQLitePositionStore
from core.types import (
    AssetClass, Fill, Instrument, Intent, MarketContext, Mode, OrderRef,
    OrderStatus, OrderType, Position, Side,
)

D = Decimal

# Thursday 2026-08-27, 10:00 IST — NSE is open.
OPEN_NOW = dt.datetime(2026, 8, 27, 10, 0, tzinfo=IST)
UTC_NOW = dt.datetime(2026, 8, 27, 6, 30, tzinfo=UTC)   # same instant, 12:00 IST

NIFTY_LONG = Instrument(
    symbol="NIFTY26SEP24000CE", asset_class=AssetClass.EQUITY,
    broker_id="43210", lot_size=25, tick_size=D("0.05"),
)
NIFTY_SHORT = Instrument(
    symbol="NIFTY26SEP24500CE", asset_class=AssetClass.EQUITY,
    broker_id="43211", lot_size=25, tick_size=D("0.05"),
)
RELIANCE = Instrument(
    symbol="RELIANCE", asset_class=AssetClass.EQUITY, broker_id="2885",
    lot_size=1, tick_size=D("0.05"),
)
# Fourth equity leg, so an iron condor can be built out of four instruments.
NIFTY_WING = Instrument(
    symbol="NIFTY26SEP23500PE", asset_class=AssetClass.EQUITY,
    broker_id="43212", lot_size=25, tick_size=D("0.05"),
)


# ------------------------------------------------------------------ fakes


class Book:
    """PositionStoreProto. Raises on read when `fail`, because [] means flat."""

    def __init__(self, positions: Sequence[Position] = (),
                 fail: bool = False) -> None:
        self._positions = tuple(positions)
        self.fail = fail

    def apply(self, *a: Any, **k: Any) -> None:
        raise AssertionError("the gate must never write positions")

    def positions(self, mode: Mode,
                  asset_class: Optional[AssetClass] = None) -> Sequence[Position]:
        if self.fail:
            raise RuntimeError("position store unreachable")
        return [p for p in self._positions
                if p.mode is mode
                and (asset_class is None or p.instrument.asset_class is asset_class)]

    def realized_pnl(self, *a: Any, **k: Any) -> Decimal:
        raise AssertionError("the gate must never read realized pnl")

    def record_order(self, *a: Any, **k: Any) -> None:
        raise AssertionError("the gate must never record orders")


class Counters:
    """The gate's own durable state. Records every claim so a test can prove a
    rejected order cost the strategy nothing."""

    def __init__(self, *, seen: Sequence[str] = (), count: int = 0,
                 fail_on: Sequence[str] = ()) -> None:
        self.seen = set(seen)
        self.count = count
        self._fail_on = set(fail_on)
        self.submissions: list[tuple[str, AssetClass, str]] = []

    def seen_recently(self, idempotency_key: str, window_s: float,
                      now: dt.datetime) -> bool:
        if "seen_recently" in self._fail_on:
            raise GateStateUnavailable("state store down")
        return idempotency_key in self.seen

    def order_count(self, asset_class: AssetClass, day_key: str) -> int:
        if "order_count" in self._fail_on:
            raise GateStateUnavailable("state store down")
        return self.count

    def daily_notional(self, asset_class: AssetClass, day_key: str) -> Decimal:
        if "daily_notional" in self._fail_on:
            raise GateStateUnavailable("state store down")
        return Decimal("0")

    def record_submission(self, idempotency_key: str, asset_class: AssetClass,
                          day_key: str, now: dt.datetime,
                          notional: Decimal = Decimal("0")) -> None:
        if "record_submission" in self._fail_on:
            raise GateStateUnavailable("state store down")
        self.seen.add(idempotency_key)
        self.count += 1
        self.submissions.append((idempotency_key, asset_class, day_key))


@dataclass(frozen=True)
class WordDecision:
    """A circuit that only exposes a state word."""
    action: str
    reason: str = "test"


@dataclass(frozen=True)
class RichDecision:
    """A circuit that exposes both a word and an explicit entries_allowed —
    the shape core.risk uses, and the shape a third-party circuit would copy."""
    action: str
    entries_allowed: bool
    size_multiplier: object = D("1")
    reason: str = "test"


class Circuit:
    def __init__(self, decision: object = "ok", fail: bool = False) -> None:
        self.decision = decision
        self.fail = fail

    def read(self, asset_class: AssetClass,
             equity: Optional[Decimal]) -> object:
        if self.fail:
            raise RuntimeError("risk state unreadable")
        if isinstance(self.decision, str):
            return WordDecision(self.decision)
        return self.decision

    def advance(self, asset_class: AssetClass, equity: Decimal) -> object:
        raise AssertionError("the gate must never advance the circuit")


def gate(*, book: Sequence[Position] = (), book_fails: bool = False,
         counters: Optional[Counters] = None,
         circuit: object = None,
         config: Optional[GateConfig] = None,
         env: Optional[dict[str, str]] = None) -> OrderGate:
    return OrderGate(
        position_store=Book(book, fail=book_fails),
        gate_state=counters if counters is not None else Counters(),
        risk_circuit=circuit if circuit is not None else Circuit(),  # type: ignore[arg-type]
        config=config if config is not None else GateConfig(),
        env={} if env is None else env,
    )


def order(*, instrument: Instrument = NIFTY_LONG, side: Side = Side.BUY,
          qty: str = "25", price: Optional[str] = "100",
          strategy_id: str = "s1", key: str = "k1",
          structure_id: Optional[str] = None,
          order_type: OrderType = OrderType.LIMIT) -> Intent:
    return Intent(
        instrument=instrument, side=side, qty=D(qty), order_type=order_type,
        strategy_id=strategy_id, reason="adversarial test",
        limit_price=D(price) if (order_type is OrderType.LIMIT
                                 and price is not None) else None,
        structure_id=structure_id, idempotency_key=key,
    )


def market(*, equity: Optional[object] = "1000000",
           now: dt.datetime = OPEN_NOW,
           quotes: Optional[dict[str, Decimal]] = None) -> MarketContext:
    return MarketContext(
        now=now, quotes=quotes or {},
        equity=D(equity) if isinstance(equity, str) else equity,  # type: ignore[arg-type]
    )


def held(instrument: Instrument, qty: str, *, strategy_id: str = "s1",
         avg: str = "100", mode: Mode = Mode.PAPER,
         structure_id: Optional[str] = None) -> Position:
    return Position(instrument=instrument, qty=D(qty), avg_price=D(avg),
                    mode=mode, strategy_id=strategy_id,
                    structure_id=structure_id)


# ======================================================================
# 1. A real order placed while its asset class is disarmed
# ======================================================================

def test_arming_one_class_does_not_arm_the_other() -> None:
    """A crypto experiment must never arm the equity book."""
    config = GateConfig(requested_mode=Mode.LIVE)
    approval = gate(config=config, env={"CRYPTO_LIVE": "1"}).check(
        order(), market())
    assert approval.approved, approval.reason
    assert approval.mode is Mode.PAPER


def test_a_rejected_approval_still_reports_the_coerced_mode() -> None:
    """A caller that reads .mode off a rejection and retries must not be told
    LIVE while the class is disarmed."""
    config = GateConfig(requested_mode=Mode.LIVE)
    approval = gate(config=config, env={}, circuit=Circuit("flatten")).check(
        order(), market())
    assert not approval.approved
    assert approval.mode is Mode.PAPER


def test_an_unparseable_arm_flag_refuses_to_build_the_gate() -> None:
    """`EQUITY_LIVE=ture` read as False looks exactly like a correctly
    disarmed engine, and the operator never learns the flag did nothing."""
    with pytest.raises(ValueError, match="EQUITY_LIVE"):
        gate(config=GateConfig(requested_mode=Mode.LIVE),
             env={"EQUITY_LIVE": "ture"})


def test_arming_cannot_be_widened_after_construction() -> None:
    g = gate(config=GateConfig(requested_mode=Mode.LIVE), env={})
    for name in ("_OrderGate__armed", "_OrderGate__config",
                 "_OrderGate__circuit", "_OrderGate__positions"):
        with pytest.raises(AttributeError):
            setattr(g, name, {})
        with pytest.raises(AttributeError):
            delattr(g, name)
    assert g.check(order(), market()).mode is Mode.PAPER


def test_caps_cannot_be_widened_through_the_config_mapping() -> None:
    config = GateConfig(limits={AssetClass.EQUITY: ClassLimits(max_orders_per_day=1)})
    with pytest.raises(TypeError):
        config.limits[AssetClass.EQUITY] = ClassLimits(max_orders_per_day=999)  # type: ignore[index]


# ======================================================================
# 2. Laundering a new position through another strategy's book
#
# EXPLOIT: `_is_reducing` matched on symbol alone. A strategy holding nothing
# could sell against a DIFFERENT strategy's long, be classified as "reducing",
# and thereby skip the order-notional cap, the class allocation cap AND the
# circuit's entry block — repeatedly, because the first strategy's row never
# goes away.
# ======================================================================

def test_a_strategy_cannot_launder_a_new_short_through_another_strategys_long() -> None:
    book = [held(NIFTY_LONG, "500", strategy_id="momo")]
    approval = gate(book=book, circuit=Circuit("flatten")).check(
        order(side=Side.SELL, qty="500", strategy_id="rogue"), market())
    assert not approval.approved, (
        "a SELL from a strategy holding nothing is a NEW short, not a "
        "reduction; the circuit says flatten and it must be refused"
    )
    assert approval.rejections[0].name == "risk_circuit"


def test_the_laundered_short_also_skips_every_size_cap() -> None:
    """Same shape with the circuit clear: the caps must still bite."""
    book = [held(NIFTY_LONG, "500", strategy_id="momo", avg="5000")]
    # 500 * 5000 = 2,500,000 against 1,000,000 equity: past every cap.
    approval = gate(book=book).check(
        order(side=Side.SELL, qty="500", price="5000", strategy_id="rogue"),
        market())
    assert not approval.approved
    assert approval.rejections[0].name in {"notional_cap", "allocation_cap"}


def test_the_laundered_short_cannot_be_repeated_against_a_stale_long() -> None:
    """The second lap of the attack: the book now shows both sides, and the
    long row must still not authorise another short."""
    book = [held(NIFTY_LONG, "500", strategy_id="momo"),
            held(NIFTY_LONG, "-500", strategy_id="rogue")]
    approval = gate(book=book, circuit=Circuit("flatten")).check(
        order(side=Side.SELL, qty="500", strategy_id="rogue"), market())
    assert not approval.approved


def test_a_strategy_exiting_its_own_position_is_still_exempt() -> None:
    """The rule above must not trap a book behind its own safety rule."""
    book = [held(NIFTY_LONG, "500", strategy_id="momo", avg="5000")]
    approval = gate(book=book, circuit=Circuit("flatten")).check(
        order(side=Side.SELL, qty="500", price="5000", strategy_id="momo"),
        market())
    assert approval.approved, approval.reason


def test_covering_a_strategys_own_short_is_still_exempt() -> None:
    book = [held(NIFTY_LONG, "-500", strategy_id="momo", avg="5000")]
    approval = gate(book=book, circuit=Circuit("flatten")).check(
        order(side=Side.BUY, qty="500", price="5000", strategy_id="momo"),
        market())
    assert approval.approved, approval.reason


def test_an_exit_larger_than_the_position_is_not_a_reduction() -> None:
    book = [held(NIFTY_LONG, "25", strategy_id="momo")]
    approval = gate(book=book, circuit=Circuit("flatten")).check(
        order(side=Side.SELL, qty="500", strategy_id="momo"), market())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


# ======================================================================
# 3. A circuit that says "flatten" but claims entries are allowed
#
# EXPLOIT: `_circuit_allows_entries` returned the explicit `entries_allowed`
# flag before it ever looked at the state word, so a decision carrying
# action="flatten" with entries_allowed=True opened new positions. The gate
# reads the circuit structurally precisely so it keeps failing closed when the
# circuit is swapped — this was the hole in that promise.
# ======================================================================

@pytest.mark.parametrize("word", ["flatten", "halt", "stopped", "tripped",
                                  "breached", "paused"])
def test_a_blocking_state_word_beats_a_permissive_entries_allowed(word: str) -> None:
    decision = RichDecision(action=word, entries_allowed=True,
                            size_multiplier=D("1"))
    approval = gate(circuit=Circuit(decision)).check(order(), market())
    assert not approval.approved, (
        f"circuit said {word!r}; a permissive entries_allowed must not "
        "override the state word"
    )
    assert approval.rejections[0].name == "risk_circuit"


def test_entries_allowed_false_still_blocks_a_healthy_looking_word() -> None:
    decision = RichDecision(action="ok", entries_allowed=False,
                            size_multiplier=D("0"))
    approval = gate(circuit=Circuit(decision)).check(order(), market())
    assert not approval.approved
    assert approval.rejections[0].name == "risk_circuit"


def test_a_blocking_circuit_still_lets_its_own_exit_through() -> None:
    decision = RichDecision(action="flatten", entries_allowed=True,
                            size_multiplier=D("1"))
    book = [held(NIFTY_LONG, "25", strategy_id="s1")]
    approval = gate(book=book, circuit=Circuit(decision)).check(
        order(side=Side.SELL, qty="25"), market())
    assert approval.approved, approval.reason


# ======================================================================
# 4. A half-placed structure that leaves the naked half on
#
# EXPLOIT: each leg passes the gate on its own. On a debit spread the LONG leg
# is always the expensive one, so the order-notional cap rejects the protective
# leg and approves the short one — a naked short call with unbounded loss,
# every single time the spread is a little too big for the cap.
# ======================================================================

def _spread_legs(*, qty: str = "75", long_px: str = "200",
                 short_px: str = "50", sid: str = "spread-1",
                 strategy_id: str = "trend") -> tuple[Intent, Intent]:
    long_leg = order(instrument=NIFTY_LONG, side=Side.BUY, qty=qty,
                     price=long_px, strategy_id=strategy_id,
                     structure_id=sid, key=f"{sid}:long")
    short_leg = order(instrument=NIFTY_SHORT, side=Side.SELL, qty=qty,
                      price=short_px, strategy_id=strategy_id,
                      structure_id=sid, key=f"{sid}:short")
    return long_leg, short_leg


def test_the_notional_cap_sizes_the_long_leg_of_a_debit_spread() -> None:
    """The long leg of a debit spread is the expensive one. Sizing it down
    (rather than refusing) is what lets the structure trade; the short at the
    original qty must still be refused if it would outrun the sized hedge."""
    config = GateConfig(default_limits=ClassLimits(
        max_order_notional_fraction=D("0.10"), max_allocation_fraction=D("0.30")))
    long_leg, _ = _spread_legs()
    # 75 * 200 = 15,000 against a 10,000 cap (10% of 100,000).
    # lot=25 so 10000/200 = 50, which is two lots.
    approval = gate(config=config).check(long_leg, market(equity="100000"))
    assert approval.approved, approval.reason
    assert approval.intent.qty == D("50")


def test_the_short_leg_may_not_outrun_a_sized_down_hedge() -> None:
    """After the long is sized to 50, selling the original 75 would be naked."""
    book = [held(NIFTY_LONG, "50", strategy_id="trend", avg="200",
                 structure_id="spread-1")]
    _, short_leg = _spread_legs(qty="75")
    approval = gate(book=book).check(short_leg, market(equity="1000000"))
    assert not approval.approved
    assert approval.rejections[0].name == "structure_legs"


def test_the_short_leg_is_refused_when_no_other_leg_of_its_structure_is_open() -> None:
    config = GateConfig(default_limits=ClassLimits(
        max_order_notional_fraction=D("0.10"), max_allocation_fraction=D("0.30")))
    _, short_leg = _spread_legs()
    # 75 * 50 = 3,750 — comfortably under every cap on its own.
    approval = gate(config=config).check(short_leg, market(equity="100000"))
    assert not approval.approved, (
        "no hedge is in the book; placing this short alone leaves a naked "
        "call with unbounded loss"
    )
    assert approval.rejections[0].name == "structure_legs"


def test_the_short_leg_is_allowed_once_its_hedge_is_actually_in_the_book() -> None:
    book = [held(NIFTY_LONG, "75", strategy_id="trend", avg="200",
                 structure_id="spread-1")]
    _, short_leg = _spread_legs()
    approval = gate(book=book).check(short_leg, market(equity="1000000"))
    assert approval.approved, approval.reason


def test_the_short_leg_may_not_outrun_a_partially_filled_hedge() -> None:
    """The other half of the same failure: the protective leg filled 25 of 75,
    so selling the full 75 leaves 50 naked."""
    book = [held(NIFTY_LONG, "25", strategy_id="trend", avg="200",
                 structure_id="spread-1")]
    _, short_leg = _spread_legs(qty="75")
    approval = gate(book=book).check(short_leg, market(equity="1000000"))
    assert not approval.approved
    assert approval.rejections[0].name == "structure_legs"


def test_the_short_leg_may_match_a_partially_filled_hedge() -> None:
    book = [held(NIFTY_LONG, "25", strategy_id="trend", avg="200",
                 structure_id="spread-1")]
    _, short_leg = _spread_legs(qty="25")
    approval = gate(book=book).check(short_leg, market(equity="1000000"))
    assert approval.approved, approval.reason


def test_the_second_short_of_an_iron_condor_leans_on_the_second_wing() -> None:
    book = [
        held(NIFTY_LONG, "75", strategy_id="trend", avg="200",
             structure_id="condor-1"),
        held(RELIANCE, "75", strategy_id="trend", avg="200",
             structure_id="condor-1"),
        held(NIFTY_SHORT, "-75", strategy_id="trend", avg="50",
             structure_id="condor-1"),
    ]
    second_short = order(instrument=NIFTY_WING, side=Side.SELL, qty="75",
                         price="50", strategy_id="trend",
                         structure_id="condor-1", key="condor:short-2")
    approval = gate(book=book).check(second_short, market(equity="1000000"))
    assert approval.approved, approval.reason


def test_a_structure_that_is_short_by_construction_is_refused() -> None:
    """A short strangle has no protective leg to lean on. Refusing it is the
    point: this engine does not build unbounded structures by accident."""
    book = [held(NIFTY_SHORT, "-75", strategy_id="trend", avg="50",
                 structure_id="strangle-1")]
    other_side = order(instrument=NIFTY_LONG, side=Side.SELL, qty="75",
                       price="50", strategy_id="trend",
                       structure_id="strangle-1", key="strangle:2")
    approval = gate(book=book).check(other_side, market(equity="1000000"))
    assert not approval.approved
    assert approval.rejections[0].name == "structure_legs"


def test_the_long_leg_of_a_structure_is_never_blocked_by_the_rule() -> None:
    """A lone long leg is a paid debit, not an unbounded loss. It must be free
    to go on first — that ordering is the whole point."""
    long_leg, _ = _spread_legs()
    approval = gate().check(long_leg, market(equity="1000000"))
    assert approval.approved, approval.reason


def test_exiting_the_long_leg_of_a_live_structure_is_not_blocked() -> None:
    book = [held(NIFTY_LONG, "75", strategy_id="trend", avg="200",
                 structure_id="spread-1")]
    exit_long = order(instrument=NIFTY_LONG, side=Side.SELL, qty="75",
                      price="200", strategy_id="trend",
                      structure_id="spread-1", key="exit-long")
    approval = gate(book=book, circuit=Circuit("flatten")).check(
        exit_long, market(equity="1000000"))
    assert approval.approved, approval.reason


def test_an_outright_short_is_untouched_by_the_structure_rule() -> None:
    approval = gate().check(
        order(instrument=RELIANCE, side=Side.SELL, qty="10", price="2500"),
        market(equity="1000000"))
    assert approval.approved, approval.reason


# ======================================================================
# 5. Unknown inputs treated as benign
# ======================================================================

def test_unknown_equity_refuses_rather_than_sizing_against_a_default() -> None:
    approval = gate().check(order(), market(equity=None))
    assert not approval.approved
    assert "unavailable" in approval.reason


def test_zero_equity_refuses_rather_than_reading_as_unlimited_room() -> None:
    approval = gate().check(order(), market(equity="0"))
    assert not approval.approved


def test_an_unreadable_position_store_refuses_rather_than_reading_flat() -> None:
    approval = gate(book_fails=True).check(order(), market())
    assert not approval.approved
    assert approval.rejections[0].name == "allocation_cap"


def test_nan_equity_is_a_rejection_not_an_exception() -> None:
    """`Decimal('NaN') > 0` raises InvalidOperation. The gate's contract is to
    return a verdict; a caller that is not wrapped in try/except would see a
    crash where it expected a refusal."""
    approval = gate().check(order(), market(equity=D("NaN")))
    assert not approval.approved


def test_a_nan_limit_price_is_a_rejection_not_an_exception() -> None:
    # Intent.__post_init__ only asks that a LIMIT has *a* limit_price, so a NaN
    # survives construction and arrives at the gate looking like a price.
    intent = replace(order(), limit_price=D("NaN"), idempotency_key="nan")
    approval = gate().check(intent, market())
    assert not approval.approved
    assert approval.rejections[0].name == "sanity"


def test_a_nan_quote_on_a_held_position_is_a_rejection_not_an_exception() -> None:
    book = [held(RELIANCE, "10", strategy_id="other", avg="2500")]
    approval = gate(book=book).check(
        order(), market(quotes={RELIANCE.symbol: D("NaN")}))
    assert approval.approved, approval.reason
    allocation = next(c for c in approval.checks if c.name == "allocation_cap")
    # A NaN is not a mark; the row falls back to its recorded cost basis.
    assert "exposure 25000 " in allocation.detail, allocation.detail


def test_a_position_that_cannot_be_marked_at_all_blocks_a_new_entry() -> None:
    """A position with no quote and a zero cost basis contributed ZERO to the
    class exposure — a real open position measured as no exposure, which is the
    exact shape of every bug this engine was rebuilt to remove."""
    book = [held(RELIANCE, "1000", strategy_id="other", avg="0")]
    approval = gate(book=book).check(order(), market(quotes={}))
    assert not approval.approved, (
        "class exposure cannot be measured, so it cannot be added to"
    )
    assert approval.rejections[0].name == "allocation_cap"


def test_an_unmarkable_position_does_not_trap_an_exit() -> None:
    book = [held(RELIANCE, "1000", strategy_id="other", avg="0"),
            held(NIFTY_LONG, "25", strategy_id="s1", avg="100")]
    approval = gate(book=book).check(
        order(side=Side.SELL, qty="25"), market(quotes={}))
    assert approval.approved, approval.reason


# ======================================================================
# 6. The daily counter being spent by an order that never went anywhere
# ======================================================================

REJECTION_CASES: list[tuple[str, dict[str, Any]]] = [
    ("market closed", {"ctx": {"now": dt.datetime(2026, 8, 29, 10, 0, tzinfo=IST)}}),
    ("quantized to nothing", {"intent": {"qty": "10"}}),          # lot_size 25
    ("no price", {"intent": {"order_type": OrderType.MARKET, "price": None}}),
    ("no idempotency key", {"intent": {"key": "   "}}),
    ("unknown equity", {"ctx": {"equity": None}}),
    ("notional cap", {"intent": {"qty": "25", "price": "100000"}}),
    ("circuit blocked", {"circuit": "flatten"}),
    ("uninterpretable circuit", {"circuit": "banana"}),
    ("naked structure leg", {"intent": {"side": Side.SELL, "qty": "25",
                                        "structure_id": "s-9"}}),
]


@pytest.mark.parametrize("label,case", REJECTION_CASES,
                         ids=[c[0] for c in REJECTION_CASES])
def test_a_rejected_order_never_spends_the_daily_quota(
        label: str, case: dict[str, Any]) -> None:
    counters = Counters(count=7)
    g = gate(counters=counters,
             circuit=Circuit(case.get("circuit", "ok")))
    approval = g.check(order(**case.get("intent", {})),
                       market(**case.get("ctx", {})))
    assert not approval.approved, f"{label} should have been refused"
    assert counters.submissions == [], f"{label} burned a daily order slot"
    assert counters.count == 7, f"{label} moved the counter"
    assert counters.seen == set(), f"{label} claimed the idempotency key"


def test_the_quota_is_spent_exactly_once_by_an_order_that_is_approved() -> None:
    counters = Counters()
    approval = gate(counters=counters).check(order(key="only-once"), market())
    assert approval.approved, approval.reason
    assert counters.submissions == [("only-once", AssetClass.EQUITY, "2026-08-27")]


def test_a_failed_submission_record_refuses_rather_than_placing_unaccounted() -> None:
    counters = Counters(fail_on=["record_submission"])
    approval = gate(counters=counters).check(order(), market())
    assert not approval.approved
    assert counters.submissions == []


def test_a_store_that_answers_the_dedup_read_with_garbage_refuses() -> None:
    class Liar(Counters):
        def seen_recently(self, *a: Any, **k: Any) -> bool:
            return "no"                                  # type: ignore[return-value]

    counters = Liar()
    approval = gate(counters=counters).check(order(), market())
    assert not approval.approved
    assert approval.rejections[0].name == "dedup"


# ======================================================================
# 6b. What the gate can and cannot promise about reaching an adapter
#
# The gate has no adapter reference and no way to place anything, so "an order
# reached a venue without passing check()" is not reachable from this module.
# What it CAN promise is that its own verdict is complete and self-describing,
# so the executor never has to consult the original Intent — the two properties
# below are what a caller has to honour for the chokepoint to mean anything.
# ======================================================================

def test_the_gate_touches_nothing_but_the_three_stores_it_was_given() -> None:
    """Every fake here raises on a write. An approved check must not have
    written a position, recorded an order, or advanced the circuit — the gate
    decides, it does not act."""
    approval = gate().check(order(), market())
    assert approval.approved, approval.reason


def test_an_approved_order_must_be_placed_as_the_gate_rewrote_it() -> None:
    """`Approval.intent` is snapped and throttled. A caller that places the
    intent it submitted instead makes quantization and the circuit's throttle
    decorative — so the two must be visibly different objects."""
    decision = RichDecision(action="reduce", entries_allowed=True,
                            size_multiplier=D("0.5"))
    submitted = order(qty="107")                 # lot_size 25
    approval = gate(circuit=Circuit(decision)).check(submitted, market())
    assert approval.approved, approval.reason
    assert submitted.qty == D("107")
    assert approval.intent.qty == D("50"), "107 -> 100 lots -> halved -> 50"
    assert approval.intent is not submitted


# ======================================================================
# 7. Paper rows influencing live decisions
# ======================================================================

def test_paper_positions_do_not_consume_the_live_allocation() -> None:
    config = GateConfig(requested_mode=Mode.LIVE)
    paper_book = [held(RELIANCE, "1000", avg="2500", mode=Mode.PAPER)]
    approval = gate(book=paper_book, config=config,
                    env={"EQUITY_LIVE": "1"}).check(
        order(qty="25", price="3000"), market())
    assert approval.mode is Mode.LIVE
    assert approval.approved, approval.reason


def test_live_positions_do_not_consume_the_paper_allocation() -> None:
    live_book = [held(RELIANCE, "1000", avg="2500", mode=Mode.LIVE)]
    approval = gate(book=live_book).check(
        order(qty="25", price="3000"), market())
    assert approval.mode is Mode.PAPER
    assert approval.approved, approval.reason


def test_a_live_circuit_refuses_to_judge_against_a_paper_row() -> None:
    """A row for the other book is worse than no row."""
    store = SQLitePositionStore(":memory:")
    paper = RiskState(mode=Mode.PAPER, asset_class=AssetClass.EQUITY,
                      high_water=D("100000"), day_key="2026-08-27",
                      day_open_equity=D("100000"), updated_at=UTC_NOW)
    # Write the paper row under the LIVE key, the way a mis-keyed migration or
    # a copied volume would.
    store.write_risk_state(Mode.LIVE, AssetClass.EQUITY, paper.as_row())
    circuit = RiskCircuit(store=store, mode=Mode.LIVE,
                          clock=lambda: UTC_NOW)
    decision = circuit.read(AssetClass.EQUITY, D("50000"))
    assert not decision.entries_allowed
    assert decision.state_known is False
    assert decision.drawdown_pct is None
    store.close()


def test_a_paper_fill_is_invisible_to_every_live_read() -> None:
    store = SQLitePositionStore(":memory:")
    store.apply(Fill(order_id="p1", instrument=RELIANCE, side=Side.BUY,
                     qty=D("100"), price=D("2500"), fees=D("0"),
                     at=UTC_NOW, mode=Mode.PAPER), "s1")
    assert store.positions(Mode.LIVE) == []
    assert store.positions(Mode.PAPER) != []
    assert store.position(Mode.LIVE, RELIANCE.symbol, "s1") is None
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY,
                              "2026-01-01T00:00:00+00:00") == D("0")
    assert store.get_order(Mode.LIVE, "p1") is None
    store.close()


def test_the_ledger_refuses_an_order_carrying_fills_from_the_other_mode() -> None:
    store = SQLitePositionStore(":memory:")
    intent = order(instrument=RELIANCE, qty="1", price="2500")
    ref = OrderRef(
        order_id="x1", status=OrderStatus.FILLED, intent=intent,
        mode=Mode.PAPER, filled_qty=D("1"), avg_price=D("2500"),
        fills=(Fill(order_id="x1", instrument=RELIANCE, side=Side.BUY,
                    qty=D("1"), price=D("2500"), fees=D("0"), at=UTC_NOW,
                    mode=Mode.LIVE),),
    )
    with pytest.raises(ValueError, match="fill mode"):
        store.record_order(ref)
    assert store.orders(Mode.PAPER) == []
    assert store.orders(Mode.LIVE) == []
    store.close()


# ======================================================================
# 8. The risk circuit erasing a drawdown
# ======================================================================

def _store_with(state: Optional[RiskState] = None) -> SQLitePositionStore:
    store = SQLitePositionStore(":memory:")
    if state is not None:
        store.write_risk_state(state.mode, state.asset_class, state.as_row())
    return store


def _state(high: str, *, day_key: str = "2026-08-27",
           day_open: Optional[str] = None) -> RiskState:
    return RiskState(mode=Mode.LIVE, asset_class=AssetClass.EQUITY,
                     high_water=D(high), day_key=day_key,
                     day_open_equity=D(day_open or high), updated_at=UTC_NOW)


def test_read_never_persists_a_baseline_however_high_the_equity() -> None:
    store = _store_with(_state("100000"))
    circuit = RiskCircuit(store=store, mode=Mode.LIVE, clock=lambda: UTC_NOW)
    circuit.read(AssetClass.EQUITY, D("500000"))
    row = store.read_risk_state(Mode.LIVE, AssetClass.EQUITY)
    assert row is not None and row["high_water"] == "100000"
    store.close()


def test_advance_refuses_to_write_when_the_previous_state_is_unreadable() -> None:
    """THE fail-open bug: without the previous high-water mark, writing today's
    equity as the new one erases the drawdown permanently."""
    class Broken(SQLitePositionStore):
        def read_risk_state(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("database is locked")

    store = Broken(":memory:")
    store.write_risk_state(Mode.LIVE, AssetClass.EQUITY, _state("100000").as_row())
    circuit = RiskCircuit(store=store, mode=Mode.LIVE, clock=lambda: UTC_NOW)
    decision = circuit.advance(AssetClass.EQUITY, D("50000"))
    assert not decision.entries_allowed
    assert decision.state_known is False
    # Read past the override to prove nothing was written.
    row = SQLitePositionStore.read_risk_state(store, Mode.LIVE, AssetClass.EQUITY)
    assert row is not None and row["high_water"] == "100000"
    store.close()


def test_advance_never_lowers_the_high_water_mark() -> None:
    store = _store_with(_state("100000"))
    circuit = RiskCircuit(store=store, mode=Mode.LIVE, clock=lambda: UTC_NOW)
    for equity in ("90000", "50000", "99999"):
        circuit.advance(AssetClass.EQUITY, D(equity))
        row = store.read_risk_state(Mode.LIVE, AssetClass.EQUITY)
        assert row is not None and D(row["high_water"]) == D("100000")
    store.close()


@pytest.mark.parametrize("equity", ["0", "-1", "NaN"])
def test_advance_refuses_a_baseline_it_cannot_believe(equity: str) -> None:
    store = _store_with(_state("100000"))
    circuit = RiskCircuit(store=store, mode=Mode.LIVE, clock=lambda: UTC_NOW)
    with pytest.raises(ValueError):
        circuit.advance(AssetClass.EQUITY, D(equity))
    row = store.read_risk_state(Mode.LIVE, AssetClass.EQUITY)
    assert row is not None and row["high_water"] == "100000"
    store.close()


def test_a_clock_that_steps_backwards_cannot_rebaseline_the_daily_loss() -> None:
    """EXPLOIT: the daily baseline rolled whenever the stored day_key merely
    DIFFERED from today's. An NTP step backwards (or a row written by a host in
    another timezone) therefore reset day_open to the equity at the bottom of
    the day and erased the whole day's loss."""
    # Stored day is 2026-08-28; our clock says 2026-08-27.
    store = _store_with(_state("100000", day_key="2026-08-28",
                               day_open="100000"))
    circuit = RiskCircuit(store=store, mode=Mode.LIVE, clock=lambda: UTC_NOW)
    decision = circuit.read(AssetClass.EQUITY, D("94000"))   # -6% on the day
    assert not decision.entries_allowed, (
        "the daily boundary cannot be established; entries must be refused "
        "rather than re-baselined at the bottom of the day"
    )
    assert decision.daily_loss_pct is None
    store.close()


def test_a_backwards_clock_does_not_get_a_new_baseline_written_either() -> None:
    store = _store_with(_state("100000", day_key="2026-08-28",
                               day_open="100000"))
    circuit = RiskCircuit(store=store, mode=Mode.LIVE, clock=lambda: UTC_NOW)
    decision = circuit.advance(AssetClass.EQUITY, D("94000"))
    assert not decision.entries_allowed
    row = store.read_risk_state(Mode.LIVE, AssetClass.EQUITY)
    assert row is not None
    assert row["day_key"] == "2026-08-28", "the daily baseline was rewritten"
    assert row["day_open_equity"] == "100000"
    store.close()


def test_a_genuine_day_roll_still_resets_the_daily_baseline() -> None:
    store = _store_with(_state("100000", day_key="2026-08-26"))
    circuit = RiskCircuit(store=store, mode=Mode.LIVE, clock=lambda: UTC_NOW)
    decision = circuit.advance(AssetClass.EQUITY, D("97000"))
    assert decision.daily_loss_pct == D("0")
    assert decision.drawdown_pct == D("0.03")
    row = store.read_risk_state(Mode.LIVE, AssetClass.EQUITY)
    assert row is not None and row["day_key"] == "2026-08-27"
    store.close()


def test_a_decision_cannot_advertise_a_throttle_it_does_not_apply() -> None:
    """A `reduce` carrying a multiplier of 1 is a control that reports itself
    as active and changes nothing."""
    with pytest.raises(ValueError):
        CircuitDecision(action=CircuitAction.REDUCE, drawdown_pct=D("0.06"),
                        daily_loss_pct=D("0"), reason="x",
                        equity_known=True, state_known=True,
                        size_multiplier=D("1"))


def test_an_ok_decision_cannot_carry_a_throttle() -> None:
    with pytest.raises(ValueError):
        CircuitDecision(action=CircuitAction.OK, drawdown_pct=D("0"),
                        daily_loss_pct=D("0"), reason="x",
                        equity_known=True, state_known=True,
                        size_multiplier=D("0.5"))


def test_a_blind_decision_never_reports_entries_allowed() -> None:
    for equity_known, state_known in ((False, True), (True, False), (False, False)):
        decision = CircuitDecision(
            action=CircuitAction.PAUSE, drawdown_pct=None, daily_loss_pct=None,
            reason="blind", equity_known=equity_known, state_known=state_known)
        assert not decision.entries_allowed


# ======================================================================
# 9. The ledger accepting a price that makes a position invisible
# ======================================================================

def test_the_ledger_refuses_a_zero_price_fill() -> None:
    """A position carried at avg_price 0 marks as zero exposure in every
    notional cap whenever its quote is missing."""
    store = SQLitePositionStore(":memory:")
    with pytest.raises(ValueError, match="negative price"):
        store.apply(Fill(order_id="z1", instrument=RELIANCE, side=Side.BUY,
                         qty=D("100"), price=D("0"), fees=D("0"),
                         at=UTC_NOW, mode=Mode.PAPER), "s1")
    assert store.positions(Mode.PAPER) == []
    store.close()


def test_a_failed_write_does_not_leave_the_ledger_wedged() -> None:
    """A transaction left open turns every later write into 'cannot start a
    transaction within a transaction' — reporting a lock error for what was
    really one bad row."""
    store = SQLitePositionStore(":memory:")
    with pytest.raises(ValueError):
        store.apply(Fill(order_id="bad", instrument=RELIANCE, side=Side.BUY,
                         qty=D("1"), price=D("-1"), fees=D("0"),
                         at=UTC_NOW, mode=Mode.PAPER), "s1")
    store.apply(Fill(order_id="good", instrument=RELIANCE, side=Side.BUY,
                     qty=D("1"), price=D("2500"), fees=D("0"),
                     at=UTC_NOW, mode=Mode.PAPER), "s1")
    assert len(store.positions(Mode.PAPER)) == 1
    store.close()


def test_a_gate_reading_a_real_ledger_sees_the_book_it_will_trade_into() -> None:
    """End to end on the real store: a paper fill must not appear in the live
    gate's exposure, and the live gate must not approve against it."""
    store = SQLitePositionStore(":memory:")
    store.apply(Fill(order_id="p1", instrument=RELIANCE, side=Side.BUY,
                     qty=D("100"), price=D("2500"), fees=D("0"),
                     at=UTC_NOW, mode=Mode.PAPER), "s1")
    g = OrderGate(
        position_store=store,
        gate_state=Counters(),
        risk_circuit=Circuit(),                    # type: ignore[arg-type]
        config=GateConfig(requested_mode=Mode.LIVE),
        env={"EQUITY_LIVE": "1"},
    )
    approval = g.check(order(instrument=RELIANCE, qty="10", price="2500"),
                       market(quotes={RELIANCE.symbol: D("2500")}))
    assert approval.mode is Mode.LIVE
    allocation = next(c for c in approval.checks if c.name == "allocation_cap")
    assert "exposure 0 " in allocation.detail, allocation.detail
    store.close()
