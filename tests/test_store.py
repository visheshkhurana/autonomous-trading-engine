"""
Tests for core.store.

Every safety rule in the store has a test here that fails if the rule is
removed. The three that matter most: mode isolation (both directions), fill
idempotency, and loud failure instead of a plausible-looking zero/empty.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from core.store import DEFAULT_DB_PATH, SQLitePositionStore, net_fill
from core.types import (
    AssetClass, Fill, Instrument, Intent, Mode, OrderRef, OrderStatus,
    OrderType, Position, Side,
)

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 8, 27, 9, 30, tzinfo=UTC)
EPOCH = "1970-01-01T00:00:00+00:00"

RELIANCE = Instrument(
    symbol="RELIANCE", asset_class=AssetClass.EQUITY, broker_id="2885",
    lot_size=1, tick_size=Decimal("0.05"),
)
BTC = Instrument(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO, broker_id="BTC/USDT",
    lot_size=1, tick_size=Decimal("0.1"), qty_precision=6,
)


@pytest.fixture()
def store(tmp_path) -> SQLitePositionStore:
    s = SQLitePositionStore(str(tmp_path / "ledger.db"))
    yield s
    s.close()


def mk_fill(order_id: str, side: Side, qty: str, price: str, mode: Mode,
            fees: str = "0", at: dt.datetime = T0,
            instrument: Instrument = RELIANCE) -> Fill:
    return Fill(order_id=order_id, instrument=instrument, side=side,
                qty=Decimal(qty), price=Decimal(price), fees=Decimal(fees),
                at=at, mode=mode)


def mk_ref(order_id: str, mode: Mode, status: OrderStatus = OrderStatus.FILLED,
           fills: tuple[Fill, ...] = (), error: str | None = None,
           instrument: Instrument = RELIANCE) -> OrderRef:
    intent = Intent(
        instrument=instrument, side=Side.BUY, qty=Decimal("10"),
        order_type=OrderType.LIMIT, strategy_id="s1",
        reason="test entry", limit_price=Decimal("100.05"),
        idempotency_key=f"key-{order_id}",
    )
    filled = sum((f.qty for f in fills), Decimal(0))
    return OrderRef(order_id=order_id, status=status, intent=intent, mode=mode,
                    filled_qty=filled,
                    avg_price=fills[0].price if fills else None,
                    fills=fills, error=error)


# ---------------------------------------------------------------- isolation

def test_paper_fill_is_invisible_to_live_reads(store):
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.PAPER), "s1")

    assert store.positions(Mode.LIVE) == []
    assert store.position(Mode.LIVE, "RELIANCE", "s1") is None
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal(0)
    assert store.fills(Mode.LIVE) == []
    # and it IS visible in its own mode, so the assertions above mean isolation
    # rather than "nothing was written".
    assert len(store.positions(Mode.PAPER)) == 1


def test_live_fill_is_invisible_to_paper_reads(store):
    store.apply(mk_fill("o2", Side.BUY, "7", "250", Mode.LIVE, fees="3"), "s1")

    assert store.positions(Mode.PAPER) == []
    assert store.position(Mode.PAPER, "RELIANCE", "s1") is None
    assert store.realized_pnl(Mode.PAPER, AssetClass.EQUITY, EPOCH) == Decimal(0)
    assert store.fills(Mode.PAPER) == []
    assert store.positions(Mode.LIVE)[0].qty == Decimal("7")


def test_same_symbol_and_strategy_net_independently_per_mode(store):
    """The exact shape of the old bug: a paper book that looks closed must not
    close the live one."""
    store.apply(mk_fill("p1", Side.BUY, "10", "100", Mode.PAPER), "s1")
    store.apply(mk_fill("l1", Side.BUY, "10", "100", Mode.LIVE), "s1")
    # Flatten the PAPER book only.
    store.apply(mk_fill("p2", Side.SELL, "10", "120", Mode.PAPER), "s1")

    assert store.position(Mode.PAPER, "RELIANCE", "s1") is None
    live = store.position(Mode.LIVE, "RELIANCE", "s1")
    assert live is not None and live.qty == Decimal("10")
    # Paper's +200 must not show up in the live daily loss number.
    assert store.realized_pnl(Mode.PAPER, AssetClass.EQUITY, EPOCH) == Decimal("200")
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal("0")


def test_orders_are_mode_isolated(store):
    store.record_order(mk_ref("ord-1", Mode.PAPER))
    assert store.get_order(Mode.LIVE, "ord-1") is None
    assert store.orders(Mode.LIVE) == []
    assert store.get_order(Mode.PAPER, "ord-1") is not None


def test_risk_state_is_mode_isolated(store):
    store.write_risk_state(Mode.PAPER, AssetClass.EQUITY, {"peak_equity": "500000"})
    assert store.read_risk_state(Mode.LIVE, AssetClass.EQUITY) is None
    assert store.read_risk_state(Mode.PAPER, AssetClass.EQUITY) == {"peak_equity": "500000"}


def test_asset_classes_do_not_mix_in_pnl(store):
    store.apply(mk_fill("c1", Side.BUY, "1", "60000", Mode.PAPER, instrument=BTC), "c")
    store.apply(mk_fill("c2", Side.SELL, "1", "61000", Mode.PAPER, instrument=BTC), "c")
    assert store.realized_pnl(Mode.PAPER, AssetClass.CRYPTO, EPOCH) == Decimal("1000")
    assert store.realized_pnl(Mode.PAPER, AssetClass.EQUITY, EPOCH) == Decimal("0")


# -------------------------------------------------------------- idempotency

def test_applying_the_same_fill_twice_does_not_double_count(store):
    f = mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE, fees="5")
    store.apply(f, "s1")
    store.apply(f, "s1")

    pos = store.position(Mode.LIVE, "RELIANCE", "s1")
    assert pos is not None
    assert pos.qty == Decimal("10")
    assert pos.avg_price == Decimal("100")
    assert len(store.fills(Mode.LIVE)) == 1
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal("-5")


def test_replayed_closing_fill_does_not_double_realise(store):
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE), "s1")
    close = mk_fill("o2", Side.SELL, "10", "110", Mode.LIVE)
    store.apply(close, "s1")
    store.apply(close, "s1")

    assert store.position(Mode.LIVE, "RELIANCE", "s1") is None
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal("100")


def test_explicit_fill_index_dedupes_identical_partials(store):
    """Two byte-identical partials on one order are distinct only if the caller
    says so with an index; replaying either index is still a no-op."""
    f = mk_fill("o1", Side.BUY, "5", "100", Mode.LIVE)
    store.apply(f, "s1", fill_index=0)
    store.apply(f, "s1", fill_index=1)
    store.apply(f, "s1", fill_index=1)   # replay

    pos = store.position(Mode.LIVE, "RELIANCE", "s1")
    assert pos is not None and pos.qty == Decimal("10")
    assert len(store.fills(Mode.LIVE)) == 2


def test_distinct_fills_on_one_order_both_apply(store):
    store.apply(mk_fill("o1", Side.BUY, "5", "100", Mode.LIVE), "s1")
    store.apply(mk_fill("o1", Side.BUY, "5", "102", Mode.LIVE), "s1")
    pos = store.position(Mode.LIVE, "RELIANCE", "s1")
    assert pos is not None and pos.qty == Decimal("10")
    assert pos.avg_price == Decimal("101")


def test_recording_the_same_order_twice_stores_one_row(store):
    ref = mk_ref("ord-1", Mode.LIVE, OrderStatus.REJECTED, error="margin")
    store.record_order(ref)
    store.record_order(ref)
    assert len(store.orders(Mode.LIVE)) == 1


# ------------------------------------------------------------------ netting

def test_adding_to_a_long_recomputes_weighted_average(store):
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.PAPER), "s1")
    store.apply(mk_fill("o2", Side.BUY, "30", "110", Mode.PAPER), "s1")
    pos = store.position(Mode.PAPER, "RELIANCE", "s1")
    assert pos.qty == Decimal("40")
    assert pos.avg_price == Decimal("107.5")


def test_partial_close_realises_only_the_closed_portion(store):
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.PAPER), "s1")
    store.apply(mk_fill("o2", Side.SELL, "4", "110", Mode.PAPER), "s1")
    pos = store.position(Mode.PAPER, "RELIANCE", "s1")
    assert pos.qty == Decimal("6")
    assert pos.avg_price == Decimal("100")   # basis of the remainder is unchanged
    assert store.realized_pnl(Mode.PAPER, AssetClass.EQUITY, EPOCH) == Decimal("40")


def test_full_close_leaves_no_open_position(store):
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.PAPER), "s1")
    store.apply(mk_fill("o2", Side.SELL, "10", "90", Mode.PAPER), "s1")
    assert store.positions(Mode.PAPER) == []
    assert store.realized_pnl(Mode.PAPER, AssetClass.EQUITY, EPOCH) == Decimal("-100")


def test_crossing_through_zero_long_to_short(store):
    """The remainder is a NEW short at the crossing fill's price, and only the
    closed 10 realises. Carrying the old average across the flip would invent
    P&L on the short leg."""
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.PAPER, at=T0), "s1")
    store.apply(mk_fill("o2", Side.SELL, "15", "110", Mode.PAPER,
                        at=T0 + dt.timedelta(minutes=5)), "s1")

    pos = store.position(Mode.PAPER, "RELIANCE", "s1")
    assert pos.qty == Decimal("-5")
    assert pos.avg_price == Decimal("110")
    assert store.realized_pnl(Mode.PAPER, AssetClass.EQUITY, EPOCH) == Decimal("100")
    # a crossed position is a new position, so its clock restarts
    assert pos.opened_at == T0 + dt.timedelta(minutes=5)


def test_crossing_through_zero_short_to_long(store):
    store.apply(mk_fill("o1", Side.SELL, "10", "100", Mode.PAPER), "s1")
    store.apply(mk_fill("o2", Side.BUY, "25", "90", Mode.PAPER), "s1")
    pos = store.position(Mode.PAPER, "RELIANCE", "s1")
    assert pos.qty == Decimal("15")
    assert pos.avg_price == Decimal("90")
    assert store.realized_pnl(Mode.PAPER, AssetClass.EQUITY, EPOCH) == Decimal("100")


def test_short_then_cover_realises_positive_on_a_fall(store):
    store.apply(mk_fill("o1", Side.SELL, "10", "100", Mode.LIVE), "s1")
    pos = store.position(Mode.LIVE, "RELIANCE", "s1")
    assert pos.qty == Decimal("-10") and pos.avg_price == Decimal("100")
    store.apply(mk_fill("o2", Side.BUY, "4", "90", Mode.LIVE), "s1")
    pos = store.position(Mode.LIVE, "RELIANCE", "s1")
    assert pos.qty == Decimal("-6")
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal("40")


def test_fees_are_realised_on_the_fill_that_incurs_them(store):
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE, fees="20"), "s1")
    # An opening fill moves no P&L except its cost.
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal("-20")
    assert store.position(Mode.LIVE, "RELIANCE", "s1").avg_price == Decimal("100")
    store.apply(mk_fill("o2", Side.SELL, "10", "110", Mode.LIVE, fees="20"), "s1")
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal("60")


def test_positions_do_not_net_across_strategies(store):
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE), "alpha")
    store.apply(mk_fill("o2", Side.SELL, "10", "100", Mode.LIVE), "beta")
    by_strategy = {p.strategy_id: p.qty for p in store.positions(Mode.LIVE)}
    assert by_strategy == {"alpha": Decimal("10"), "beta": Decimal("-10")}


def test_crypto_fractional_quantities_survive_the_round_trip(store):
    store.apply(mk_fill("c1", Side.BUY, "0.0125", "61234.5", Mode.PAPER,
                        instrument=BTC), "c")
    pos = store.positions(Mode.PAPER, AssetClass.CRYPTO)[0]
    assert pos.qty == Decimal("0.0125")
    assert pos.avg_price == Decimal("61234.5")
    assert pos.instrument.qty_precision == 6


def test_net_fill_is_pure_and_matches_the_stored_result():
    r = net_fill(Decimal("10"), Decimal("100"), Decimal("-15"),
                 Decimal("110"), Decimal("0"))
    assert (r.qty, r.avg_price, r.realized, r.reopened) == (
        Decimal("-5"), Decimal("110"), Decimal("100"), True)


# ------------------------------------------------------------- order ledger

def test_rejection_is_persisted_in_full(store):
    ref = mk_ref("ord-9", Mode.LIVE, OrderStatus.REJECTED,
                 error="RMS: insufficient margin")
    store.record_order(ref)
    got = store.get_order(Mode.LIVE, "ord-9")
    assert got is not None
    assert got.status is OrderStatus.REJECTED
    assert got.error == "RMS: insufficient margin"
    assert got.intent.reason == "test entry"
    assert got.intent.idempotency_key == "key-ord-9"
    assert got.intent.limit_price == Decimal("100.05")
    assert got.filled_qty == Decimal("0")


def test_filled_order_round_trips_with_its_fills(store):
    f = mk_fill("ord-2", Side.BUY, "10", "100.05", Mode.LIVE, fees="1.5")
    store.record_order(mk_ref("ord-2", Mode.LIVE, OrderStatus.FILLED, fills=(f,)))
    got = store.get_order(Mode.LIVE, "ord-2")
    assert got.fills == (f,)
    assert got.filled_qty == Decimal("10")


def test_recording_an_order_never_moves_a_position(store):
    """record_order writes the ledger; apply() is the only position writer.
    If record_order also applied its fills, every filled order would count
    twice."""
    f = mk_fill("ord-3", Side.BUY, "10", "100", Mode.LIVE)
    store.record_order(mk_ref("ord-3", Mode.LIVE, OrderStatus.FILLED, fills=(f,)))
    assert store.positions(Mode.LIVE) == []
    assert store.fills(Mode.LIVE) == []


def test_a_second_different_ref_is_appended_not_overwritten(store):
    store.record_order(mk_ref("ord-4", Mode.LIVE, OrderStatus.TIMEOUT))
    store.record_order(mk_ref("ord-4", Mode.LIVE, OrderStatus.CANCELLED))
    rows = store.orders(Mode.LIVE)
    assert [r.status for r in rows] == [OrderStatus.CANCELLED, OrderStatus.TIMEOUT]
    assert store.get_order(Mode.LIVE, "ord-4").status is OrderStatus.CANCELLED


def test_order_carrying_a_fill_from_the_other_mode_is_refused(store):
    stray = mk_fill("ord-5", Side.BUY, "10", "100", Mode.PAPER)
    ref = mk_ref("ord-5", Mode.LIVE, OrderStatus.FILLED, fills=(stray,))
    with pytest.raises(ValueError, match="fill mode"):
        store.record_order(ref)
    assert store.orders(Mode.LIVE) == []


# ------------------------------------------------------- loud failure paths

def test_raw_string_mode_raises_instead_of_returning_empty(store):
    """An unknown mode string would match no rows and read as 'flat'. That is
    the class of bug this whole module exists to prevent."""
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE), "s1")
    for call in (
        lambda: store.positions("live"),
        lambda: store.position("live", "RELIANCE", "s1"),
        lambda: store.realized_pnl("live", AssetClass.EQUITY, EPOCH),
        lambda: store.fills("live"),
        lambda: store.get_order("live", "o1"),
        lambda: store.read_risk_state("live", AssetClass.EQUITY),
    ):
        with pytest.raises(TypeError, match="mode must be a Mode"):
            call()


def test_raw_string_asset_class_raises(store):
    with pytest.raises(TypeError, match="asset_class must be"):
        store.realized_pnl(Mode.LIVE, "equity", EPOCH)
    with pytest.raises(TypeError, match="asset_class must be"):
        store.positions(Mode.LIVE, "equity")


def test_naive_fill_timestamp_is_refused(store):
    naive = mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE,
                    at=dt.datetime(2026, 8, 27, 9, 30))
    with pytest.raises(ValueError, match="naive"):
        store.apply(naive, "s1")
    # and nothing was half-written
    assert store.positions(Mode.LIVE) == []
    assert store.fills(Mode.LIVE) == []


def test_naive_or_malformed_since_is_refused(store):
    with pytest.raises(ValueError, match="naive"):
        store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, "2026-08-27")
    with pytest.raises(ValueError):
        store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, "yesterday")


def test_apply_refuses_a_zero_or_negative_quantity(store):
    with pytest.raises(ValueError, match="qty must be"):
        store.apply(mk_fill("o1", Side.BUY, "0", "100", Mode.LIVE), "s1")
    with pytest.raises(ValueError, match="negative price"):
        store.apply(mk_fill("o1", Side.BUY, "1", "-100", Mode.LIVE), "s1")


def test_apply_requires_a_strategy_id(store):
    with pytest.raises(ValueError, match="strategy_id"):
        store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE), "")


def test_missing_risk_state_reads_as_none_not_zeros(store):
    """None is 'never written'. Zeros here read as 'no drawdown' and disarm the
    circuit, which is exactly how the previous system lost money."""
    assert store.read_risk_state(Mode.LIVE, AssetClass.EQUITY) is None
    store.write_risk_state(Mode.LIVE, AssetClass.EQUITY, {})
    assert store.read_risk_state(Mode.LIVE, AssetClass.EQUITY) == {}


def test_risk_state_values_must_be_strings(store):
    with pytest.raises(TypeError, match="str->str"):
        store.write_risk_state(Mode.LIVE, AssetClass.EQUITY,
                               {"peak_equity": Decimal("500000")})


def test_risk_state_overwrites_in_place(store):
    store.write_risk_state(Mode.LIVE, AssetClass.CRYPTO, {"level": "1"})
    store.write_risk_state(Mode.LIVE, AssetClass.CRYPTO, {"level": "2"})
    assert store.read_risk_state(Mode.LIVE, AssetClass.CRYPTO) == {"level": "2"}


# --------------------------------------------------------------- durability

def test_since_window_excludes_earlier_fills(store):
    t1 = T0
    t2 = T0 + dt.timedelta(days=1)
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE, at=t1), "s1")
    store.apply(mk_fill("o2", Side.SELL, "10", "110", Mode.LIVE, at=t1), "s1")
    store.apply(mk_fill("o3", Side.BUY, "10", "100", Mode.LIVE, at=t2), "s1")
    store.apply(mk_fill("o4", Side.SELL, "10", "95", Mode.LIVE, at=t2), "s1")

    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal("50")
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY,
                              t2.isoformat()) == Decimal("-50")


def test_since_window_accepts_a_non_utc_offset(store):
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE, at=T0), "s1")
    store.apply(mk_fill("o2", Side.SELL, "10", "110", Mode.LIVE, at=T0), "s1")
    # 09:30 UTC is 15:00 IST; a window starting 14:00 IST includes the fill.
    since = dt.datetime(2026, 8, 27, 14, 0, tzinfo=ist).isoformat()
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, since) == Decimal("100")
    later = dt.datetime(2026, 8, 27, 16, 0, tzinfo=ist).isoformat()
    assert store.realized_pnl(Mode.LIVE, AssetClass.EQUITY, later) == Decimal("0")


def test_state_survives_reopening_the_file(tmp_path):
    path = str(tmp_path / "ledger.db")
    s1 = SQLitePositionStore(path)
    s1.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE, fees="2"), "s1")
    s1.record_order(mk_ref("o1", Mode.LIVE))
    s1.write_risk_state(Mode.LIVE, AssetClass.EQUITY, {"peak_equity": "500000"})
    s1.close()

    s2 = SQLitePositionStore(path)
    try:
        assert s2.position(Mode.LIVE, "RELIANCE", "s1").qty == Decimal("10")
        assert s2.get_order(Mode.LIVE, "o1") is not None
        assert s2.read_risk_state(Mode.LIVE, AssetClass.EQUITY) == {"peak_equity": "500000"}
        assert s2.realized_pnl(Mode.LIVE, AssetClass.EQUITY, EPOCH) == Decimal("-2")
    finally:
        s2.close()


def test_wal_is_enabled(store):
    mode = store._read("PRAGMA journal_mode", ())[0][0]
    assert mode.lower() == "wal"


def test_default_path_is_the_documented_one():
    assert DEFAULT_DB_PATH == "/data/automoney.db"


def test_position_round_trips_as_the_core_dataclass(store):
    store.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE, at=T0), "s1",
                structure_id="spread-1")
    pos = store.positions(Mode.LIVE)[0]
    assert isinstance(pos, Position)
    assert pos.instrument == RELIANCE
    assert pos.mode is Mode.LIVE
    assert pos.structure_id == "spread-1"
    assert pos.opened_at == T0


def test_satisfies_the_position_store_protocol(store):
    from core.protocols import PositionStoreProto
    assert isinstance(store, PositionStoreProto)


def test_two_connections_on_one_file_see_each_others_writes(tmp_path):
    """The engine writes; the API reads from its own process. A write that is
    invisible to the reader shows up there as a missing position."""
    path = str(tmp_path / "ledger.db")
    writer = SQLitePositionStore(path)
    reader = SQLitePositionStore(path)
    try:
        writer.apply(mk_fill("o1", Side.BUY, "10", "100", Mode.LIVE), "s1")
        assert reader.position(Mode.LIVE, "RELIANCE", "s1").qty == Decimal("10")
        reader.apply(mk_fill("o2", Side.SELL, "4", "110", Mode.LIVE), "s1")
        assert writer.position(Mode.LIVE, "RELIANCE", "s1").qty == Decimal("6")
    finally:
        writer.close()
        reader.close()


def test_concurrent_applies_do_not_lose_a_fill(store):
    """apply() is a read-modify-write. If it were not serialised, two threads
    netting into the same position would drop one of the fills."""
    import threading

    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            store.apply(mk_fill(f"o{n}", Side.BUY, "1", "100", Mode.LIVE), "s1")
        except BaseException as exc:      # surfaced, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert store.position(Mode.LIVE, "RELIANCE", "s1").qty == Decimal("20")


def test_option_instrument_round_trips_including_expiry():
    """An F&O leg whose expiry is lost on the way into the ledger is a leg
    nobody can unwind."""
    store = SQLitePositionStore(":memory:")
    try:
        nifty = Instrument(
            symbol="NIFTY26AUG24500CE", asset_class=AssetClass.EQUITY,
            broker_id="43217", lot_size=75, tick_size=Decimal("0.05"),
            min_notional=Decimal("0"), expiry=dt.date(2026, 8, 27),
        )
        store.apply(mk_fill("f1", Side.SELL, "75", "112.35", Mode.PAPER,
                            instrument=nifty), "fno", structure_id="ic-1")
        pos = store.positions(Mode.PAPER)[0]
        assert pos.instrument == nifty
        assert pos.instrument.expiry == dt.date(2026, 8, 27)
        assert pos.qty == Decimal("-75")
        assert pos.structure_id == "ic-1"
    finally:
        store.close()
