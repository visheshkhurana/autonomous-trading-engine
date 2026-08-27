"""
Tests for the risk circuit.

Each safety rule in core/risk.py has a test here that fails if the rule is
removed. The ones that matter most are grouped under "fail closed" and
"read() is pure" — those two properties are what stop the circuit from
silently disarming itself.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Mapping, Optional, Sequence

import pytest

from core.profiles import IST, UTC
from core.protocols import RiskCircuitProto
from core.risk import (
    DEFAULT_LIMITS,
    CircuitAction,
    CircuitDecision,
    RiskCircuit,
    RiskLimits,
    RiskState,
)
from core.types import AssetClass, Mode

D = Decimal


class FakeStore:
    """Stands in for the real store. Implements the risk_state slice for real
    and makes every position method explode, so any accidental coupling between
    the circuit and the position tables shows up as a failing test."""

    def __init__(self, state: Optional[RiskState] = None, *,
                 fail_read: bool = False, fail_write: bool = False,
                 read_returns: object = "unset") -> None:
        self.row: Optional[Mapping[str, str]] = None if state is None else state.as_row()
        self.key: Optional[tuple[Mode, AssetClass]] = (
            None if state is None else (state.mode, state.asset_class)
        )
        self.fail_read = fail_read
        self.fail_write = fail_write
        self.read_returns = read_returns   # override the row handed back
        self.writes: list[RiskState] = []
        self.reads = 0

    @property
    def state(self) -> Optional[RiskState]:
        return None if self.row is None else RiskState.from_row(self.row)

    # --- risk_state slice -------------------------------------------------
    def read_risk_state(self, mode: Mode,
                        asset_class: AssetClass) -> Optional[Mapping[str, str]]:
        self.reads += 1
        if self.fail_read:
            raise RuntimeError("database is locked")
        if self.read_returns != "unset":
            return self.read_returns            # type: ignore[return-value]
        if self.row is None:
            return None
        # The real table is keyed on (mode, asset_class).
        if self.key != (mode, asset_class):
            return None
        return self.row

    def write_risk_state(self, mode: Mode, asset_class: AssetClass,
                         state: Mapping[str, str]) -> None:
        if self.fail_write:
            raise RuntimeError("disk full")
        for k, v in state.items():
            # The ledger stores str->str JSON; a Decimal reaching it would be
            # round-tripped through float.
            assert isinstance(k, str) and isinstance(v, str), f"{k!r}->{v!r} not str->str"
        self.row = dict(state)
        self.key = (mode, asset_class)
        self.writes.append(RiskState.from_row(self.row))

    # --- PositionStoreProto surface: must never be touched by the circuit --
    def apply(self, *a, **k) -> None:
        raise AssertionError("risk circuit must not write positions")

    def positions(self, *a, **k) -> Sequence:
        raise AssertionError("risk circuit must not read positions")

    def realized_pnl(self, *a, **k) -> Decimal:
        raise AssertionError("risk circuit must not read realized pnl")

    def record_order(self, *a, **k) -> None:
        raise AssertionError("risk circuit must not record orders")


NOW = dt.datetime(2026, 8, 27, 6, 30, tzinfo=UTC)   # 12:00 IST, mid-session


def clock_at(when: dt.datetime):
    return lambda: when


def state_at(equity: str, *, day_key: str = "2026-08-27",
             day_open: Optional[str] = None,
             mode: Mode = Mode.LIVE,
             asset_class: AssetClass = AssetClass.EQUITY) -> RiskState:
    return RiskState(
        mode=mode,
        asset_class=asset_class,
        high_water=D(equity),
        day_key=day_key,
        day_open_equity=D(day_open if day_open is not None else equity),
        updated_at=NOW,
    )


def circuit(store: FakeStore, *, mode: Mode = Mode.LIVE,
            when: dt.datetime = NOW, limits=None) -> RiskCircuit:
    return RiskCircuit(store=store, mode=mode, limits=limits, clock=clock_at(when))


# ---------------------------------------------------------------- contract

def test_conforms_to_protocol():
    assert isinstance(circuit(FakeStore()), RiskCircuitProto)


# --------------------------------------------------------------- the ladder

@pytest.mark.parametrize("equity,expected", [
    ("100000", CircuitAction.OK),        # flat
    ("120000", CircuitAction.OK),        # new high
    ("98500",  CircuitAction.OK),        # -1.5%, below every rung
    ("98000",  CircuitAction.REDUCE),    # daily loss exactly 2%
    ("97000",  CircuitAction.PAUSE),     # daily loss exactly 3%
    ("95000",  CircuitAction.FLATTEN),   # daily loss exactly 5%
])
def test_daily_loss_ladder(equity, expected):
    store = FakeStore(state_at("100000"))
    d = circuit(store).read(AssetClass.EQUITY, D(equity))
    assert d.action is expected


@pytest.mark.parametrize("equity,expected", [
    ("96000", CircuitAction.OK),        # -4% drawdown, day baseline unaffected
    ("95000", CircuitAction.REDUCE),    # 5% drawdown
    ("90000", CircuitAction.PAUSE),     # 10%
    ("85000", CircuitAction.FLATTEN),   # 15%
])
def test_drawdown_ladder_uses_high_water_not_day_open(equity, expected):
    # High-water 100k set earlier; today opened at today's (lower) equity, so
    # the daily ladder sees nothing and only the drawdown rung can fire.
    store = FakeStore(state_at("100000", day_open=equity))
    d = circuit(store).read(AssetClass.EQUITY, D(equity))
    assert d.action is expected
    assert d.daily_loss_pct == D("0")


def test_stricter_of_the_two_ladders_wins():
    # 4% drawdown (below reduce) but 4% on the day (above pause).
    store = FakeStore(state_at("100000", day_open="100000"))
    d = circuit(store).read(AssetClass.EQUITY, D("96000"))
    assert d.action is CircuitAction.PAUSE
    assert d.drawdown_pct == D("0.04")
    assert d.daily_loss_pct == D("0.04")


def test_reported_figures_are_fractions_not_percentages():
    store = FakeStore(state_at("200000", day_open="200000"))
    d = circuit(store).read(AssetClass.EQUITY, D("190000"))
    assert d.drawdown_pct == D("0.05")
    assert d.daily_loss_pct == D("0.05")
    assert "5.00%" in d.reason      # humans get percent, callers get fractions


def test_gain_never_reports_a_negative_drawdown():
    store = FakeStore(state_at("100000"))
    d = circuit(store).read(AssetClass.EQUITY, D("150000"))
    assert d.drawdown_pct == D("0")
    assert d.daily_loss_pct == D("0")
    assert d.entries_allowed


def test_size_multiplier_matches_action():
    store = FakeStore(state_at("100000", day_open="100000"))
    assert circuit(store).read(AssetClass.EQUITY, D("100000")).size_multiplier == D("1")
    assert circuit(store).read(AssetClass.EQUITY, D("98000")).size_multiplier == D("0.5")
    assert circuit(store).read(AssetClass.EQUITY, D("97000")).size_multiplier == D("0")
    assert circuit(store).read(AssetClass.EQUITY, D("95000")).size_multiplier == D("0")


def test_limits_are_configurable_per_asset_class():
    # Crypto's wider defaults: 4% on the day is nothing for crypto, a pause for equity.
    eq = circuit(FakeStore(state_at("100000", day_open="100000")))
    cr = circuit(FakeStore(state_at("100000", day_open="100000",
                                    asset_class=AssetClass.CRYPTO)))
    assert eq.read(AssetClass.EQUITY, D("96000")).action is CircuitAction.PAUSE
    assert cr.read(AssetClass.CRYPTO, D("96000")).action is CircuitAction.REDUCE


def test_custom_limits_override_defaults():
    tight = RiskLimits(
        reduce_drawdown=D("0.01"), pause_drawdown=D("0.02"), flatten_drawdown=D("0.03"),
        reduce_daily_loss=D("0.01"), pause_daily_loss=D("0.02"), flatten_daily_loss=D("0.03"),
    )
    store = FakeStore(state_at("100000", day_open="100000"))
    c = circuit(store, limits={AssetClass.EQUITY: tight})
    assert c.read(AssetClass.EQUITY, D("97000")).action is CircuitAction.FLATTEN


def test_unconfigured_asset_class_raises_rather_than_permitting():
    store = FakeStore(state_at("100000"))
    c = circuit(store, limits={AssetClass.EQUITY: DEFAULT_LIMITS[AssetClass.EQUITY]})
    with pytest.raises(KeyError):
        c.read(AssetClass.CRYPTO, D("100000"))


# ------------------------------------------------------------- fail closed

def test_unknown_equity_never_reads_as_zero():
    store = FakeStore(state_at("100000"))
    d = circuit(store).read(AssetClass.EQUITY, None)
    assert d.equity_known is False
    assert d.state_known is True
    assert d.drawdown_pct is None          # NOT Decimal("1") — zero equity would be
    assert d.daily_loss_pct is None
    assert d.action is CircuitAction.PAUSE  # blocked, but not force-flattened
    assert d.entries_allowed is False
    assert d.size_multiplier == D("0")


def test_unknown_equity_does_not_flatten_the_book():
    # A blind circuit must not dump positions on numbers it cannot see.
    store = FakeStore(state_at("100000"))
    d = circuit(store).read(AssetClass.EQUITY, None)
    assert d.must_flatten is False


def test_nan_equity_is_treated_as_unknown_not_as_a_value():
    store = FakeStore(state_at("100000"))
    d = circuit(store).read(AssetClass.EQUITY, D("NaN"))
    assert d.equity_known is False
    assert d.entries_allowed is False


def test_non_decimal_equity_raises():
    store = FakeStore(state_at("100000"))
    with pytest.raises(TypeError):
        circuit(store).read(AssetClass.EQUITY, 100000.0)   # type: ignore[arg-type]


def test_unreadable_state_blocks_entries():
    store = FakeStore(state_at("100000"), fail_read=True)
    d = circuit(store).read(AssetClass.EQUITY, D("100000"))
    assert d.state_known is False
    assert d.equity_known is True
    assert d.drawdown_pct is None
    assert d.entries_allowed is False
    assert "database is locked" in d.reason   # the failure is reported, not hidden


@pytest.mark.parametrize("row", [
    {"high_water": "100000"},                                   # truncated row
    dict(state_at("100000").as_row(), high_water="not-a-number"),
    dict(state_at("100000").as_row(), high_water="0"),           # unusable denominator
    dict(state_at("100000").as_row(), day_key=""),
    "not-a-row",
])
def test_corrupt_stored_row_is_not_usable(row):
    # A row that cannot be interpreted is no more usable than one that could not
    # be fetched: both must block, neither may become a fresh baseline.
    store = FakeStore(read_returns=row)
    d = circuit(store).read(AssetClass.EQUITY, D("100000"))
    assert d.state_known is False
    assert d.drawdown_pct is None
    assert d.entries_allowed is False


def test_corrupt_stored_row_is_never_overwritten_by_advance():
    store = FakeStore(read_returns=dict(state_at("100000").as_row(), high_water="0"))
    d = circuit(store).advance(AssetClass.EQUITY, D("100000"))
    assert store.writes == []
    assert d.state_known is False


def test_state_belonging_to_another_book_is_refused():
    # A paper row must never become the baseline for a live decision.
    foreign = state_at("1000000", mode=Mode.PAPER)
    store = FakeStore(read_returns=foreign.as_row())
    d = circuit(store, mode=Mode.LIVE).read(AssetClass.EQUITY, D("100000"))
    assert d.state_known is False
    assert d.entries_allowed is False
    assert "mismatched" in d.reason


def test_absent_row_is_distinguishable_from_a_failed_read():
    fresh = circuit(FakeStore(None)).read(AssetClass.EQUITY, D("100000"))
    broken = circuit(FakeStore(None, fail_read=True)).read(AssetClass.EQUITY, D("100000"))
    assert fresh.state_known is True and fresh.entries_allowed is True
    assert broken.state_known is False and broken.entries_allowed is False


def test_first_run_uses_current_equity_as_baseline():
    d = circuit(FakeStore(None)).read(AssetClass.EQUITY, D("100000"))
    assert d.action is CircuitAction.OK
    assert d.drawdown_pct == D("0")


def test_first_run_with_zero_equity_refuses_to_judge():
    # No baseline and no usable denominator: say so rather than invent 0%.
    d = circuit(FakeStore(None)).read(AssetClass.EQUITY, D("0"))
    assert d.state_known is True
    assert d.drawdown_pct is None
    assert d.entries_allowed is False


def test_zero_equity_against_a_known_high_water_is_a_full_drawdown():
    store = FakeStore(state_at("100000"))
    d = circuit(store).read(AssetClass.EQUITY, D("0"))
    assert d.drawdown_pct == D("1")
    assert d.action is CircuitAction.FLATTEN


def test_decision_cannot_claim_tradable_while_blind():
    with pytest.raises(ValueError):
        CircuitDecision(action=CircuitAction.OK, drawdown_pct=None, daily_loss_pct=None,
                        reason="x", equity_known=False, state_known=True,
                        size_multiplier=D("1"))


def test_decision_cannot_report_numbers_it_did_not_measure():
    with pytest.raises(ValueError):
        CircuitDecision(action=CircuitAction.PAUSE, drawdown_pct=D("0"),
                        daily_loss_pct=D("0"), reason="x",
                        equity_known=False, state_known=True)


def test_blocking_actions_cannot_carry_size():
    with pytest.raises(ValueError):
        CircuitDecision(action=CircuitAction.PAUSE, drawdown_pct=D("0.2"),
                        daily_loss_pct=D("0"), reason="x",
                        equity_known=True, state_known=True, size_multiplier=D("0.5"))


# ------------------------------------------------------------ read is pure

def test_read_never_writes():
    for equity in ("100000", "150000", "50000", "0"):
        store = FakeStore(state_at("100000"))
        circuit(store).read(AssetClass.EQUITY, D(equity))
        assert store.writes == []


def test_read_never_writes_on_a_day_roll():
    store = FakeStore(state_at("100000", day_key="2026-08-26"))
    circuit(store).read(AssetClass.EQUITY, D("99000"))
    assert store.writes == []


def test_repeated_reads_do_not_move_the_high_water_mark():
    store = FakeStore(state_at("100000"))
    c = circuit(store)
    for _ in range(5):
        c.read(AssetClass.EQUITY, D("90000"))
    assert store.state is not None
    assert store.state.high_water == D("100000")
    # The drawdown is still visible on the next read — this is the regression
    # that a write-on-read would erase.
    assert c.read(AssetClass.EQUITY, D("90000")).drawdown_pct == D("0.10")


def test_read_does_not_touch_position_tables():
    store = FakeStore(state_at("100000"))
    circuit(store).read(AssetClass.EQUITY, D("95000"))   # FakeStore raises if touched


# --------------------------------------------------------------- advance

def test_advance_raises_the_high_water_mark_on_a_new_high():
    store = FakeStore(state_at("100000"))
    d = circuit(store).advance(AssetClass.EQUITY, D("120000"))
    assert store.writes[-1].high_water == D("120000")
    assert d.drawdown_pct == D("0")


def test_advance_never_lowers_the_high_water_mark():
    store = FakeStore(state_at("100000"))
    d = circuit(store).advance(AssetClass.EQUITY, D("80000"))
    assert store.writes[-1].high_water == D("100000")
    assert d.drawdown_pct == D("0.20")
    assert d.action is CircuitAction.FLATTEN


def test_advance_refuses_to_write_when_state_is_unreadable():
    """The fail-OPEN bug this module exists to prevent.

    If advance() wrote the current equity as the high-water mark after a failed
    read, the drawdown would be erased permanently and the circuit would report
    'ok' forever after.
    """
    store = FakeStore(state_at("100000"), fail_read=True)
    d = circuit(store).advance(AssetClass.EQUITY, D("60000"))
    assert store.writes == []
    assert d.state_known is False
    assert d.entries_allowed is False

    # And the real high-water mark survived: once the store recovers, the 40%
    # drawdown is still there.
    store.fail_read = False
    recovered = circuit(store).read(AssetClass.EQUITY, D("60000"))
    assert recovered.drawdown_pct == D("0.40")
    assert recovered.action is CircuitAction.FLATTEN


def test_advance_refuses_to_write_on_a_mismatched_row():
    store = FakeStore(read_returns=state_at("1000000", mode=Mode.PAPER).as_row())
    d = circuit(store, mode=Mode.LIVE).advance(AssetClass.EQUITY, D("100000"))
    assert store.writes == []
    assert d.state_known is False


def test_advance_rejects_unknown_equity_instead_of_persisting_zero():
    store = FakeStore(state_at("100000"))
    with pytest.raises(ValueError):
        circuit(store).advance(AssetClass.EQUITY, None)   # type: ignore[arg-type]
    assert store.writes == []


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "0", "-5"])
def test_advance_rejects_unusable_equity(bad):
    store = FakeStore(state_at("100000"))
    with pytest.raises(ValueError):
        circuit(store).advance(AssetClass.EQUITY, D(bad))
    assert store.writes == []


def test_advance_rejects_non_decimal_equity():
    store = FakeStore(state_at("100000"))
    with pytest.raises(TypeError):
        circuit(store).advance(AssetClass.EQUITY, 100000)   # type: ignore[arg-type]
    assert store.writes == []


def test_advance_surfaces_a_failed_write():
    # A write that did not land must reach the order path, not be smoothed into
    # a decision that looks persisted.
    store = FakeStore(state_at("100000"), fail_write=True)
    with pytest.raises(RuntimeError):
        circuit(store).advance(AssetClass.EQUITY, D("110000"))


def test_advance_on_a_fresh_book_seeds_the_baseline():
    store = FakeStore(None)
    d = circuit(store).advance(AssetClass.EQUITY, D("100000"))
    assert store.writes[-1].high_water == D("100000")
    assert store.writes[-1].day_open_equity == D("100000")
    assert d.action is CircuitAction.OK


def test_advance_keeps_the_day_baseline_within_the_same_day():
    store = FakeStore(state_at("100000", day_open="100000"))
    circuit(store).advance(AssetClass.EQUITY, D("99000"))
    assert store.writes[-1].day_open_equity == D("100000")


# ------------------------------------------------- per-class day boundaries

def test_equity_day_rolls_on_the_ist_session_not_utc_midnight():
    # 19:00 UTC on the 27th is 00:30 IST on the 28th: a new day for NSE.
    when = dt.datetime(2026, 8, 27, 19, 0, tzinfo=UTC)
    store = FakeStore(state_at("100000", day_key="2026-08-27", day_open="100000"))
    d = circuit(store, when=when).read(AssetClass.EQUITY, D("95000"))
    assert d.daily_loss_pct == D("0")        # baseline rolled with the IST day
    assert d.drawdown_pct == D("0.05")       # drawdown does not roll


def test_crypto_day_rolls_on_utc_and_ignores_the_ist_session():
    when = dt.datetime(2026, 8, 27, 19, 0, tzinfo=UTC)
    store = FakeStore(state_at("100000", day_key="2026-08-27", day_open="100000",
                               asset_class=AssetClass.CRYPTO))
    d = circuit(store, when=when).read(AssetClass.CRYPTO, D("95000"))
    assert d.daily_loss_pct == D("0.05")     # still the same UTC day
    assert "2026-08-27" in d.reason


def test_advance_rolls_the_daily_baseline_on_the_new_day():
    when = dt.datetime(2026, 8, 28, 6, 30, tzinfo=UTC)
    store = FakeStore(state_at("100000", day_key="2026-08-27", day_open="100000"))
    circuit(store, when=when).advance(AssetClass.EQUITY, D("94000"))
    row = store.writes[-1]
    assert row.day_key == "2026-08-28"
    assert row.day_open_equity == D("94000")   # today's baseline
    assert row.high_water == D("100000")       # yesterday's peak survives


def test_a_wiped_account_does_not_reseed_the_daily_baseline_at_zero():
    when = dt.datetime(2026, 8, 28, 6, 30, tzinfo=UTC)
    store = FakeStore(state_at("100000", day_key="2026-08-27", day_open="100000"))
    d = circuit(store, when=when).read(AssetClass.EQUITY, D("0"))
    assert d.daily_loss_pct == D("1")
    assert d.action is CircuitAction.FLATTEN


def test_naive_clock_is_rejected():
    store = FakeStore(state_at("100000"))
    c = RiskCircuit(store=store, mode=Mode.LIVE,
                    clock=lambda: dt.datetime(2026, 8, 27, 12, 0))   # no tzinfo
    with pytest.raises(ValueError):
        c.read(AssetClass.EQUITY, D("100000"))
    with pytest.raises(ValueError):
        c.advance(AssetClass.EQUITY, D("100000"))
    assert store.writes == []


def test_ist_local_clock_agrees_with_utc_clock_on_the_day_key():
    same_instant_ist = dt.datetime(2026, 8, 28, 0, 30, tzinfo=IST)
    same_instant_utc = same_instant_ist.astimezone(UTC)
    a = circuit(FakeStore(state_at("100000", day_key="2026-08-27", day_open="100000")),
                when=same_instant_ist).read(AssetClass.EQUITY, D("95000"))
    b = circuit(FakeStore(state_at("100000", day_key="2026-08-27", day_open="100000")),
                when=same_instant_utc).read(AssetClass.EQUITY, D("95000"))
    assert a.daily_loss_pct == b.daily_loss_pct


# ------------------------------------------------- configuration validation

def test_ladder_must_be_ordered():
    with pytest.raises(ValueError):
        RiskLimits(reduce_drawdown=D("0.10"), pause_drawdown=D("0.05"),
                   flatten_drawdown=D("0.15"), reduce_daily_loss=D("0.02"),
                   pause_daily_loss=D("0.03"), flatten_daily_loss=D("0.05"))


@pytest.mark.parametrize("bad", ["0", "-0.05", "1.5", "NaN"])
def test_thresholds_must_be_fractions(bad):
    with pytest.raises(ValueError):
        RiskLimits(reduce_drawdown=D(bad), pause_drawdown=D("0.10"),
                   flatten_drawdown=D("0.15"), reduce_daily_loss=D("0.02"),
                   pause_daily_loss=D("0.03"), flatten_daily_loss=D("0.05"))


def test_percentages_mistaken_for_fractions_are_rejected():
    # Someone typing 5 instead of 0.05 must not get a circuit that never fires.
    with pytest.raises(ValueError):
        RiskLimits(reduce_drawdown=D("5"), pause_drawdown=D("10"),
                   flatten_drawdown=D("15"), reduce_daily_loss=D("2"),
                   pause_daily_loss=D("3"), flatten_daily_loss=D("5"))


@pytest.mark.parametrize("bad", ["0", "-1", "NaN"])
def test_stored_state_denominators_must_be_positive(bad):
    with pytest.raises(ValueError):
        RiskState(mode=Mode.LIVE, asset_class=AssetClass.EQUITY,
                  high_water=D(bad), day_key="2026-08-27",
                  day_open_equity=D("100000"), updated_at=NOW)


def test_stored_state_requires_an_aware_timestamp():
    with pytest.raises(ValueError):
        RiskState(mode=Mode.LIVE, asset_class=AssetClass.EQUITY,
                  high_water=D("100000"), day_key="2026-08-27",
                  day_open_equity=D("100000"),
                  updated_at=dt.datetime(2026, 8, 27, 12, 0))


def test_row_round_trip_survives_a_sql_backed_store():
    original = state_at("123456.78", day_open="120000.5")
    assert RiskState.from_row(original.as_row()) == original


# --------------------------------------------------- against the real ledger

def test_round_trip_through_the_real_sqlite_store():
    """The circuit is written against the interface, but the interface has to be
    the one the ledger actually implements. This pins that."""
    from core.store import SQLitePositionStore

    with SQLitePositionStore(":memory:") as store:
        live = RiskCircuit(store=store, mode=Mode.LIVE, clock=clock_at(NOW))
        paper = RiskCircuit(store=store, mode=Mode.PAPER, clock=clock_at(NOW))

        assert live.read(AssetClass.EQUITY, D("500000")).state_known is True
        live.advance(AssetClass.EQUITY, D("500000"))

        # Peak persisted; a later dip is measured against it, not against itself.
        d = live.read(AssetClass.EQUITY, D("460000"))
        assert d.drawdown_pct == D("0.08")
        # Same session, so the 8% is also the day's loss: past the 5% daily
        # flatten rung even though the 8% drawdown alone would only pause.
        assert d.daily_loss_pct == D("0.08")
        assert d.action is CircuitAction.FLATTEN

        # advance() at the lower equity must not move the peak.
        live.advance(AssetClass.EQUITY, D("460000"))
        assert live.read(AssetClass.EQUITY, D("460000")).drawdown_pct == D("0.08")

        # Paper and live keep separate books in the same file.
        assert paper.read(AssetClass.EQUITY, D("100000")).drawdown_pct == D("0")
        paper.advance(AssetClass.EQUITY, D("100000"))
        assert live.read(AssetClass.EQUITY, D("460000")).drawdown_pct == D("0.08")

        # And so do the two asset classes.
        assert live.read(AssetClass.CRYPTO, D("2000")).drawdown_pct == D("0")


def test_action_words_stay_in_the_gate_vocabulary():
    """The gate reads decision.action structurally (it does not import this
    module) and refuses any word it does not recognise. Renaming a rung here
    would silently turn every decision into a refusal."""
    assert {a.value for a in CircuitAction} == {"ok", "reduce", "pause", "flatten"}


def test_a_blind_decision_always_carries_a_blocking_word():
    # The gate only sees the word, so the word itself has to be safe.
    for store in (FakeStore(state_at("100000"), fail_read=True),
                  FakeStore(state_at("100000"))):
        d = circuit(store).read(AssetClass.EQUITY, None)
        assert d.action.value in {"pause", "flatten"}
