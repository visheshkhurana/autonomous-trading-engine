"""The durable gate state store.

`core.gate` declares GateStateStoreProto and warns, in its own docstring, that
an in-process dict satisfies it and must never be used outside tests — N workers
would each keep their own counters, so the daily cap becomes N times what it
says. These tests pin the properties that make SQLiteGateStateStore the thing
that closes that hole:

  * a claim survives into another connection (cross-process),
  * a second claim of one key RAISES rather than succeeding,
  * every failure raises rather than returning False or 0, because False means
    "new key" and 0 means "nothing traded today" and both are legitimate.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import tempfile
from decimal import Decimal

import pytest

from core.gate import (
    GateConfig, GateStateStoreProto, GateStateUnavailable, OrderGate,
)
from core.store import SQLiteGateStateStore, SQLitePositionStore
from core.types import (
    AssetClass, Instrument, Intent, MarketContext, Mode, OrderType, Side,
)

NOW = dt.datetime(2026, 3, 2, 12, 0, tzinfo=dt.timezone.utc)
DAY = "2026-03-02"


@pytest.fixture()
def db_path(tmp_path) -> str:
    return str(tmp_path / "gate.db")


@pytest.fixture()
def state(db_path: str) -> SQLiteGateStateStore:
    store = SQLiteGateStateStore(db_path, Mode.PAPER)
    yield store
    store.close()


# --------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------

def test_it_satisfies_the_protocol_the_gate_requires(state) -> None:
    assert isinstance(state, GateStateStoreProto)


def test_a_fresh_key_is_not_seen_and_the_day_starts_at_zero(state) -> None:
    # Both of these are ANSWERS, not failures. The point of the rest of this
    # file is that a failure can never produce them.
    assert state.seen_recently("k1", 300.0, NOW) is False
    assert state.order_count(AssetClass.CRYPTO, DAY) == 0


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------

def test_a_claimed_key_reads_as_seen_inside_the_window(state) -> None:
    state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    assert state.seen_recently("k1", 300.0, NOW) is True
    assert state.seen_recently("k1", 300.0, NOW + dt.timedelta(seconds=299)) is True


def test_the_window_expires(state) -> None:
    state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    assert state.seen_recently("k1", 300.0, NOW + dt.timedelta(seconds=301)) is False


def test_a_claim_stamped_in_the_future_still_reads_as_seen(state) -> None:
    """Clock skew between two processes must fail CLOSED.

    A claim written by a host whose clock runs fast would otherwise compare as
    "not within the window" and let the same decision through a second time.
    """
    ahead = NOW + dt.timedelta(hours=1)
    state.record_submission("k1", AssetClass.CRYPTO, DAY, ahead)
    assert state.seen_recently("k1", 300.0, NOW) is True


def test_reclaiming_a_key_raises_rather_than_quietly_succeeding(state) -> None:
    """The INSERT is the real dedup: it is the only step that is atomic across
    processes. Two workers that both read "not seen" must not both claim."""
    state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    with pytest.raises(GateStateUnavailable):
        state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)


def test_reclaiming_after_the_window_expired_still_raises(state) -> None:
    """An expired dedup window does not un-approve an order that was already
    approved once. Placing it again is a second real position."""
    state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    later = NOW + dt.timedelta(hours=6)
    assert state.seen_recently("k1", 300.0, later) is False
    with pytest.raises(GateStateUnavailable):
        state.record_submission("k1", AssetClass.CRYPTO, DAY, later)


def test_a_failed_reclaim_does_not_spend_a_second_order_from_the_day(state) -> None:
    state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    with pytest.raises(GateStateUnavailable):
        state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    assert state.order_count(AssetClass.CRYPTO, DAY) == 1


# --------------------------------------------------------------------------
# the daily count
# --------------------------------------------------------------------------

def test_each_claim_spends_one_order_from_the_day(state) -> None:
    for i in range(3):
        state.record_submission(f"k{i}", AssetClass.CRYPTO, DAY, NOW)
    assert state.order_count(AssetClass.CRYPTO, DAY) == 3


def test_the_count_is_per_class_and_per_day(state) -> None:
    state.record_submission("a", AssetClass.CRYPTO, DAY, NOW)
    state.record_submission("b", AssetClass.EQUITY, DAY, NOW)
    state.record_submission("c", AssetClass.CRYPTO, "2026-03-03", NOW)
    assert state.order_count(AssetClass.CRYPTO, DAY) == 1
    assert state.order_count(AssetClass.EQUITY, DAY) == 1
    assert state.order_count(AssetClass.CRYPTO, "2026-03-03") == 1


def test_daily_notional_sums_entry_claims_and_ignores_other_days(state) -> None:
    assert state.daily_notional(AssetClass.CRYPTO, DAY) == Decimal("0")
    state.record_submission("a", AssetClass.CRYPTO, DAY, NOW, notional=Decimal("50"))
    state.record_submission("b", AssetClass.CRYPTO, DAY, NOW, notional=Decimal("50"))
    state.record_submission("c", AssetClass.EQUITY, DAY, NOW, notional=Decimal("999"))
    state.record_submission("d", AssetClass.CRYPTO, "2026-03-03", NOW,
                            notional=Decimal("50"))
    assert state.daily_notional(AssetClass.CRYPTO, DAY) == Decimal("100")
    assert state.daily_notional(AssetClass.EQUITY, DAY) == Decimal("999")
    assert state.daily_notional(AssetClass.CRYPTO, "2026-03-03") == Decimal("50")


def test_an_older_ledger_without_notional_is_migrated(db_path) -> None:
    """A reused /data/trader.db from before the clip cap must still boot."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE gate_submissions (
            idempotency_key TEXT NOT NULL,
            mode            TEXT NOT NULL,
            asset_class     TEXT NOT NULL,
            day_key         TEXT NOT NULL,
            claimed_at      TEXT NOT NULL,
            PRIMARY KEY (mode, idempotency_key)
        );
    """)
    conn.execute(
        "INSERT INTO gate_submissions VALUES (?,?,?,?,?)",
        ("old", "paper", "crypto", DAY, NOW.isoformat()),
    )
    conn.commit()
    conn.close()
    store = SQLiteGateStateStore(db_path, Mode.PAPER)
    try:
        assert store.daily_notional(AssetClass.CRYPTO, DAY) == Decimal("0")
        store.record_submission("new", AssetClass.CRYPTO, DAY, NOW,
                                notional=Decimal("50"))
        assert store.daily_notional(AssetClass.CRYPTO, DAY) == Decimal("50")
        assert store.order_count(AssetClass.CRYPTO, DAY) == 2
    finally:
        store.close()


# --------------------------------------------------------------------------
# durability — the whole reason this is not a dict
# --------------------------------------------------------------------------

def test_a_claim_is_visible_to_a_second_process_on_the_same_file(db_path) -> None:
    """The failure this replaces: N workers, N sets of counters, N times the
    daily cap, and a retry on another worker placing a second real order."""
    first = SQLiteGateStateStore(db_path, Mode.PAPER)
    first.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)

    second = SQLiteGateStateStore(db_path, Mode.PAPER)
    assert second.seen_recently("k1", 300.0, NOW) is True
    assert second.order_count(AssetClass.CRYPTO, DAY) == 1
    # And the atomic claim holds ACROSS the two connections.
    with pytest.raises(GateStateUnavailable):
        second.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    first.close()
    second.close()


def test_claims_survive_a_restart(db_path) -> None:
    store = SQLiteGateStateStore(db_path, Mode.PAPER)
    store.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    store.close()

    revived = SQLiteGateStateStore(db_path, Mode.PAPER)
    assert revived.order_count(AssetClass.CRYPTO, DAY) == 1
    assert revived.seen_recently("k1", 300.0, NOW) is True
    revived.close()


def test_paper_and_live_do_not_share_a_dedup_namespace_or_a_budget(db_path) -> None:
    """Same file, separate books. A paper session must not spend the live
    session's daily order budget, nor block its keys."""
    paper = SQLiteGateStateStore(db_path, Mode.PAPER)
    live = SQLiteGateStateStore(db_path, Mode.LIVE)
    paper.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)

    assert live.seen_recently("k1", 300.0, NOW) is False
    assert live.order_count(AssetClass.CRYPTO, DAY) == 0
    # The live side can claim the same key, because it is a different book.
    live.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    assert paper.order_count(AssetClass.CRYPTO, DAY) == 1
    assert live.order_count(AssetClass.CRYPTO, DAY) == 1
    paper.close()
    live.close()


def test_it_shares_the_ledger_file_without_disturbing_it(db_path) -> None:
    ledger = SQLitePositionStore(db_path)
    gate_state = SQLiteGateStateStore(db_path, Mode.PAPER)
    gate_state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)
    # The ledger's own reads still work on the shared file.
    assert list(ledger.positions(Mode.PAPER)) == []
    assert gate_state.order_count(AssetClass.CRYPTO, DAY) == 1
    ledger.close()
    gate_state.close()


# --------------------------------------------------------------------------
# every failure raises — none of them returns a plausible value
# --------------------------------------------------------------------------

def test_a_broken_connection_raises_instead_of_reading_as_a_new_key(state) -> None:
    state.close()
    with pytest.raises(GateStateUnavailable):
        state.seen_recently("k1", 300.0, NOW)


def test_a_broken_connection_raises_instead_of_counting_zero(state) -> None:
    state.close()
    with pytest.raises(GateStateUnavailable):
        state.order_count(AssetClass.CRYPTO, DAY)


def test_a_broken_connection_raises_instead_of_claiming_nothing(state) -> None:
    state.close()
    with pytest.raises(GateStateUnavailable):
        state.record_submission("k1", AssetClass.CRYPTO, DAY, NOW)


def test_a_dropped_table_raises_rather_than_reporting_an_empty_day(state) -> None:
    """A schema that has gone missing must not read as "nothing traded today",
    which would hand a runaway strategy the entire daily cap again."""
    state._conn.execute("DROP TABLE gate_submissions")
    with pytest.raises(GateStateUnavailable):
        state.order_count(AssetClass.CRYPTO, DAY)
    with pytest.raises(GateStateUnavailable):
        state.seen_recently("k1", 300.0, NOW)


def test_a_naive_timestamp_is_refused(state) -> None:
    """Comparing a naive stamp against tz-aware rows written by another process
    is a silent 5.5-hour error in the dedup window."""
    naive = dt.datetime(2026, 3, 2, 12, 0)
    with pytest.raises(GateStateUnavailable):
        state.seen_recently("k1", 300.0, naive)
    with pytest.raises(GateStateUnavailable):
        state.record_submission("k1", AssetClass.CRYPTO, DAY, naive)


@pytest.mark.parametrize("key", ["", "   ", None, 7])
def test_an_unusable_key_is_refused_rather_than_treated_as_new(state, key) -> None:
    with pytest.raises(GateStateUnavailable):
        state.seen_recently(key, 300.0, NOW)
    with pytest.raises(GateStateUnavailable):
        state.record_submission(key, AssetClass.CRYPTO, DAY, NOW)


@pytest.mark.parametrize("window", [0, -1, "300", None, True])
def test_a_nonsense_window_is_refused(state, window) -> None:
    with pytest.raises(GateStateUnavailable):
        state.seen_recently("k1", window, NOW)


@pytest.mark.parametrize("day", ["", "   ", None, 20260302])
def test_a_nonsense_day_key_is_refused(state, day) -> None:
    with pytest.raises(GateStateUnavailable):
        state.order_count(AssetClass.CRYPTO, day)
    with pytest.raises(GateStateUnavailable):
        state.record_submission("k1", AssetClass.CRYPTO, day, NOW)


def test_a_raw_string_asset_class_is_refused(state) -> None:
    """A string would query a class that does not exist and come back 0, which
    reads as 'nothing traded today'."""
    with pytest.raises(Exception):
        state.order_count("crypto", DAY)


def test_a_bad_mode_is_refused_at_construction(db_path) -> None:
    with pytest.raises(TypeError):
        SQLiteGateStateStore(db_path, "paper")


# --------------------------------------------------------------------------
# through the gate itself
# --------------------------------------------------------------------------

BTC = Instrument("BTC/USDT", AssetClass.CRYPTO, "BTC/USDT", 1, Decimal("0.1"),
                 qty_precision=6, min_notional=Decimal("5"))


def _intent(key: str = "k1") -> Intent:
    return Intent(instrument=BTC, side=Side.BUY, qty=Decimal("1"),
                  order_type=OrderType.MARKET, strategy_id="s1",
                  reason="test", idempotency_key=key)


class _OkCircuit:
    mode = Mode.PAPER

    def read(self, asset_class, equity):
        class D:
            state = "normal"
            size_multiplier = Decimal("1")
            reason = "ok"
        return D()

    def advance(self, asset_class, equity):  # pragma: no cover - never called
        raise AssertionError("the gate must not advance the circuit")


def _gate(db_path: str) -> tuple[OrderGate, SQLitePositionStore, SQLiteGateStateStore]:
    ledger = SQLitePositionStore(db_path)
    gate_state = SQLiteGateStateStore(db_path, Mode.PAPER)
    gate = OrderGate(position_store=ledger, gate_state=gate_state,
                     risk_circuit=_OkCircuit(), config=GateConfig(), env={})
    return gate, ledger, gate_state


def _ctx() -> MarketContext:
    return MarketContext(now=NOW, quotes={"BTC/USDT": Decimal("60000")},
                         positions=(), equity=Decimal("5000000"))


def test_the_gate_approves_against_the_real_store_and_claims_exactly_once(db_path) -> None:
    gate, ledger, gate_state = _gate(db_path)
    approval = gate.check(_intent(), _ctx())
    assert approval.approved, approval.reason
    assert gate_state.order_count(AssetClass.CRYPTO, "2026-03-02") == 1
    ledger.close()
    gate_state.close()


def test_the_gate_refuses_the_same_decision_twice(db_path) -> None:
    gate, ledger, gate_state = _gate(db_path)
    assert gate.check(_intent(), _ctx()).approved
    second = gate.check(_intent(), _ctx())
    assert not second.approved
    assert "dedup" in second.reason
    # And the refusal cost nothing from the day's budget.
    assert gate_state.order_count(AssetClass.CRYPTO, "2026-03-02") == 1
    ledger.close()
    gate_state.close()


def test_the_gate_refuses_everything_when_the_state_store_is_dead(db_path) -> None:
    """The gate turns any raise from this store into a refusal. Verified end to
    end so the fail-closed wiring cannot be broken by either side alone."""
    gate, ledger, gate_state = _gate(db_path)
    gate_state.close()
    approval = gate.check(_intent(), _ctx())
    assert not approval.approved
    assert "dedup store unavailable" in approval.reason
    ledger.close()
