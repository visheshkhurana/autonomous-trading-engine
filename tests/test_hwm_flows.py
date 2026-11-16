"""External cash flows move the high-water mark and are not P&L.

The 14 Nov withdrawals were booked as a drawdown because advance() only
ever raised the mark. These tests fail if a flow is counted as P&L, applied
twice, or allowed to lower the mark when the ledger cannot be read.
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from typing import Optional, Sequence

from core.hwm_flows import apply_external_flows, main
from core.risk import (
    DEFAULT_LIMITS,
    CircuitAction,
    ExternalCashFlow,
    RiskCircuit,
    RiskState,
)
from core.store import SQLitePositionStore
from core.types import AssetClass, Mode
from tests.test_risk import FakeStore, NOW, circuit, state_at

D = Decimal
UTC = dt.timezone.utc


class ScriptedLedger:
    def __init__(self, pages: Sequence[object]) -> None:
        self.pages = list(pages)
        self.since: list[Optional[int]] = []

    def fetch_spot_cash_flows(self, *, since_ms: Optional[int]):
        self.since.append(since_ms)
        item = self.pages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def flow(refid: str, amount: str, *, at: dt.datetime = NOW, asset: str = "USDT") -> ExternalCashFlow:
    return ExternalCashFlow(refid=refid, asset=asset, amount=D(amount), at=at)


def _wired(store: FakeStore, pages: Sequence[object], **kw) -> tuple[RiskCircuit, ScriptedLedger]:
    source = ScriptedLedger(pages)
    wired = circuit(store, cash_flows=source, **kw)
    return wired, source


def test_circuit_accepts_a_cash_flow_source():
    """circuit() in the risk tests grew a keyword; this pins the signature."""
    store = FakeStore(state_at("10000"))
    wired, _source = _wired(store, [([], 1)])
    wired.advance(AssetClass.EQUITY, D("10000"))
    assert store.writes[-1].high_water == D("10000")


def test_withdrawal_leaves_drawdown_unchanged():
    store = FakeStore(state_at("10000"))
    before = circuit(store).read(AssetClass.EQUITY, D("10000"))
    wired, _source = _wired(store, [([flow("w1", "-400")], 10)])
    decision = wired.advance(AssetClass.EQUITY, D("9600"))
    assert before.drawdown_pct == D("0")
    assert decision.drawdown_pct == D("0")
    assert decision.action is CircuitAction.OK
    written = store.writes[-1]
    assert written.high_water == D("9600")
    assert written.day_open_equity == D("9600")
    audit = json.loads(written.hwm_rebase_json)
    assert audit[-1]["reason"] == "external_cash_flow"
    assert audit[-1]["refid"] == "w1"
    assert audit[-1]["asset"] == "USDT"
    assert audit[-1]["amount"] == "-400"
    assert audit[-1]["old_high_water"] == "10000"
    assert audit[-1]["new_high_water"] == "9600"
    assert audit[-1]["old_day_open"] == "10000"
    assert audit[-1]["new_day_open"] == "9600"
    # A withdrawal while already in drawdown does not add the cash to the gap.
    gap_store = FakeStore(state_at("10000"))
    gap, _source = _wired(gap_store, [([flow("w2", "-1000")], 11)])
    gap_decision = gap.advance(AssetClass.EQUITY, D("8000"))
    assert gap_store.writes[-1].high_water - D("8000") == D("10000") - D("9000")
    assert gap_decision.drawdown_pct == (D("9000") - D("8000")) / D("9000")


def test_deposit_leaves_drawdown_unchanged():
    store = FakeStore(state_at("10000"))
    before = circuit(store).read(AssetClass.EQUITY, D("10000"))
    wired, _source = _wired(store, [([flow("d1", "2000")], 12)])
    decision = wired.advance(AssetClass.EQUITY, D("12000"))
    assert before.drawdown_pct == D("0")
    assert decision.drawdown_pct == D("0")
    assert store.writes[-1].high_water == D("12000")
    assert store.writes[-1].day_open_equity == D("12000")
    # Below the mark, the deposit shifts the mark instead of erasing the gap.
    gap_store = FakeStore(state_at("10000"))
    gap, _source = _wired(gap_store, [([flow("d2", "1000")], 13)])
    gap.advance(AssetClass.EQUITY, D("9000"))
    assert gap_store.writes[-1].high_water - D("9000") == D("10000") - D("8000")


def test_the_same_flow_twice_applies_once():
    duplicate = flow("same", "-400")
    store = FakeStore(state_at("10000"))
    source = ScriptedLedger([
        ([duplicate, duplicate], 20),
        ([duplicate], 21),
    ])
    first = RiskCircuit(store, Mode.LIVE, clock=lambda: NOW, cash_flows=source)
    first.advance(AssetClass.EQUITY, D("9600"))
    second = RiskCircuit(store, Mode.LIVE, clock=lambda: NOW, cash_flows=source)
    decision = second.advance(AssetClass.EQUITY, D("9600"))
    assert store.writes[-1].high_water == D("9600")
    assert decision.drawdown_pct == D("0")
    audits = [row for row in json.loads(store.writes[-1].hwm_rebase_json)
              if row.get("reason") == "external_cash_flow"]
    assert len(audits) == 1
    assert json.loads(store.writes[-1].processed_flow_refids) == ["same"]


def test_restart_mid_sequence_applies_each_flow_once():
    store = FakeStore(state_at("10000"))
    source = ScriptedLedger([
        ([flow("a", "-100")], 30),
        ([flow("a", "-100"), flow("b", "-50")], 31),
    ])
    first = RiskCircuit(store, Mode.LIVE, clock=lambda: NOW, cash_flows=source)
    first.advance(AssetClass.EQUITY, D("9900"))
    assert store.writes[-1].high_water == D("9900")
    # A new circuit is a restart: the cursor is the row, not the process.
    restarted = RiskCircuit(store, Mode.LIVE, clock=lambda: NOW, cash_flows=source)
    decision = restarted.advance(AssetClass.EQUITY, D("9850"))
    assert decision.drawdown_pct == D("0")
    assert store.writes[-1].high_water == D("9850")
    assert store.writes[-1].day_open_equity == D("9850")
    refids = [row["refid"] for row in json.loads(store.writes[-1].hwm_rebase_json)]
    assert refids == ["a", "b"]
    assert source.since == [None, 30]


def test_a_flow_before_the_stored_day_does_not_move_day_open():
    """The 16 Nov open already includes the 14 Nov withdrawals."""
    store = FakeStore(state_at(
        "14469.879350497385", day_open="9995.236160936496",
        day_key="2026-11-16", asset_class=AssetClass.CRYPTO,
    ))
    withdrawn = dt.datetime(2026, 11, 14, 18, 20, 28, tzinfo=UTC)
    clock_at = dt.datetime(2026, 11, 16, 1, 0, tzinfo=UTC)
    source = ScriptedLedger([([flow("old", "-4440", at=withdrawn)], 1)])
    wired = RiskCircuit(store, Mode.LIVE, clock=lambda: clock_at, cash_flows=source)
    equity = D("9992.67")
    decision = wired.advance(AssetClass.CRYPTO, equity)
    written = store.writes[-1]
    assert written.high_water == D("14469.879350497385") - D("4440")
    assert written.day_open_equity == D("9995.236160936496")
    assert decision.drawdown_pct == (written.high_water - equity) / written.high_water
    assert decision.daily_loss_pct == (D("9995.236160936496") - equity) / D("9995.236160936496")


def test_a_deposit_during_a_ledger_outage_is_applied_once():
    store = FakeStore(state_at("10000"))
    source = ScriptedLedger([
        ([], 1000),
        RuntimeError("ledger timeout"),
        ([flow("dep", "2000")], 2000),
    ])
    wired = RiskCircuit(store, Mode.LIVE, clock=lambda: NOW, cash_flows=source)
    wired.advance(AssetClass.EQUITY, D("10000"))
    blind = wired.advance(AssetClass.EQUITY, D("12000"))
    assert store.writes[-1].high_water == D("10000")
    assert store.writes[-1].day_open_equity == D("10000")
    assert "high-water mark not adjusted" in blind.reason
    recovered = wired.advance(AssetClass.EQUITY, D("12000"))
    assert store.writes[-1].high_water == D("12000")
    assert store.writes[-1].day_open_equity == D("12000")
    assert recovered.drawdown_pct == D("0")
    assert recovered.daily_loss_pct == D("0")


def test_a_blind_tick_does_not_open_the_new_day_on_unreconciled_equity():
    store = FakeStore(state_at("10000", day_key="2026-08-27"))
    # 19:00 UTC is 00:30 IST the next session, so the equity day has rolled.
    nxt = dt.datetime(2026, 8, 27, 19, 0, tzinfo=UTC)
    clock = {"at": NOW}

    def now() -> dt.datetime:
        return clock["at"]

    source = ScriptedLedger([
        ([], 1),
        RuntimeError("down"),
        ([flow("dep", "2000", at=nxt)], 2),
    ])
    wired = RiskCircuit(store, Mode.LIVE, clock=now, cash_flows=source)
    wired.advance(AssetClass.EQUITY, D("10000"))
    clock["at"] = nxt
    wired.advance(AssetClass.EQUITY, D("12000"))
    frozen = store.writes[-1]
    assert frozen.day_key == "2026-08-27"
    assert frozen.high_water == D("10000")
    assert frozen.day_open_equity == D("10000")
    recovered = wired.advance(AssetClass.EQUITY, D("12000"))
    assert store.writes[-1].day_key == "2026-08-28"
    assert store.writes[-1].high_water == D("12000")
    assert store.writes[-1].day_open_equity == D("12000")
    assert recovered.drawdown_pct == D("0")
    assert recovered.daily_loss_pct == D("0")


def test_a_frozen_day_keeps_an_eleven_percent_loss_as_a_flatten():
    """UTC midnight must not zero daily loss while the ledger is blind.

    Crypto flattens at 8% daily loss. 11% against the last certified open is
    a flatten. Resetting that open to the post-loss equity reports a reduce
    and leaves entries open.
    """
    store = FakeStore(state_at(
        "10000", day_key="2026-11-15", asset_class=AssetClass.CRYPTO,
    ))
    later = dt.datetime(2026, 11, 16, 0, 30, tzinfo=UTC)
    source = ScriptedLedger([RuntimeError("midnight outage")])
    wired = RiskCircuit(store, Mode.LIVE, clock=lambda: later, cash_flows=source)
    equity = D("8900")
    decision = wired.advance(AssetClass.CRYPTO, equity)
    assert store.writes[-1].day_key == "2026-11-15"
    assert store.writes[-1].day_open_equity == D("10000")
    assert decision.daily_loss_pct == D("0.11")
    assert decision.must_flatten is True
    assert decision.entries_allowed is False
    again = wired.read(AssetClass.CRYPTO, equity)
    assert again.daily_loss_pct == D("0.11")
    assert again.must_flatten is True
    assert again.entries_allowed is False


def test_a_flow_that_would_wipe_the_baseline_is_remembered():
    store = FakeStore(state_at("100"))
    wipe = flow("wipe", "-500")
    source = ScriptedLedger([([wipe], 1), ([wipe], 2)])
    wired = RiskCircuit(store, Mode.LIVE, clock=lambda: NOW, cash_flows=source)
    wired.advance(AssetClass.EQUITY, D("100"))
    assert store.writes[-1].high_water == D("100")
    assert json.loads(store.writes[-1].processed_flow_refids) == ["wipe"]
    wired.advance(AssetClass.EQUITY, D("100"))
    assert store.writes[-1].high_water == D("100")
    assert json.loads(store.writes[-1].processed_flow_refids) == ["wipe"]


def test_ledger_read_failure_leaves_the_high_water_mark_unchanged():
    store = FakeStore(state_at("10000"))
    wired, _source = _wired(store, [RuntimeError("ledger timeout")])
    decision = wired.advance(AssetClass.EQUITY, D("8000"))
    assert store.writes[-1].high_water == D("10000")
    assert store.writes[-1].day_open_equity == D("10000")
    assert decision.drawdown_pct == D("0.20")
    assert "high-water mark not adjusted" in decision.reason
    assert "ledger timeout" in decision.reason
    assert "external cash-flow ledger unreadable" in decision.reason


def test_market_loss_after_a_withdrawal_is_drawdown_from_the_adjusted_mark():
    store = FakeStore(state_at("10000"))
    source = ScriptedLedger([
        ([flow("w", "-1000")], 40),
        ([], 41),
    ])
    wired = RiskCircuit(store, Mode.LIVE, clock=lambda: NOW, cash_flows=source)
    flat = wired.advance(AssetClass.EQUITY, D("9000"))
    assert flat.drawdown_pct == D("0")
    loss = wired.advance(AssetClass.EQUITY, D("8100"))
    assert loss.drawdown_pct == D("900") / D("9000")
    assert store.writes[-1].high_water == D("9000")
    # The unadjusted mark would have called this a 19% drawdown.
    assert loss.drawdown_pct != (D("10000") - D("8100")) / D("10000")


def test_intraday_flow_shifts_day_open_so_daily_loss_is_unchanged():
    store = FakeStore(state_at("10000", day_open="10000"))
    before = circuit(store).read(AssetClass.EQUITY, D("10000"))
    wired, _source = _wired(store, [([flow("day", "-250")], 50)])
    decision = wired.advance(AssetClass.EQUITY, D("9750"))
    assert before.daily_loss_pct == D("0")
    assert decision.daily_loss_pct == D("0")
    assert store.writes[-1].day_open_equity == D("9750")
    assert store.writes[-1].day_key == "2026-08-27"


def test_thresholds_are_not_part_of_the_cash_flow_adjustment():
    assert DEFAULT_LIMITS[AssetClass.CRYPTO].flatten_drawdown == D("0.25")
    assert DEFAULT_LIMITS[AssetClass.EQUITY].flatten_drawdown == D("0.15")


def test_backfill_command_applies_the_known_withdrawals_once(tmp_path, capsys):
    path = str(tmp_path / "trader.db")
    store = SQLitePositionStore(path)
    try:
        seeded = RiskState(
            mode=Mode.LIVE,
            asset_class=AssetClass.CRYPTO,
            high_water=D("14469.879350497385"),
            day_key="2026-11-16",
            day_open_equity=D("9995.236160936496"),
            updated_at=NOW,
            equity_basis="engine_only",
        )
        store.write_risk_state(Mode.LIVE, AssetClass.CRYPTO, seeded.as_row())
    finally:
        store.close()

    code = main([
        "--db", path, "--mode", "live", "--asset-class", "crypto",
        "--apply-known-2026-11-14",
    ])
    assert code == 0
    again = main([
        "--db", path, "--mode", "live", "--asset-class", "crypto",
        "--apply-known-2026-11-14",
    ])
    assert again == 0
    captured = capsys.readouterr()
    assert "FTGEVAr-syDy6EaOEIqh66t8bwUHlx" in captured.out
    assert "FTPJF3S-WqybEyMe4YF7HfFlQZ87uD" in captured.out

    check = SQLitePositionStore(path)
    try:
        row = RiskState.from_row(check.read_risk_state(Mode.LIVE, AssetClass.CRYPTO))
    finally:
        check.close()
    assert row.high_water == D("14469.879350497385") - D("4440")
    # 16 Nov already opened after those withdrawals. The day's open stays.
    assert row.day_open_equity == D("9995.236160936496")
    audit = [item for item in json.loads(row.hwm_rebase_json)
             if item.get("reason") == "external_cash_flow"]
    assert all(item["old_day_open"] == item["new_day_open"] for item in audit)
    assert [item["refid"] for item in audit] == [
        "FTGEVAr-syDy6EaOEIqh66t8bwUHlx",
        "FTPJF3S-WqybEyMe4YF7HfFlQZ87uD",
    ]
    assert [item["amount"] for item in audit] == ["-10", "-4430"]
    # The second invocation found the refids and did not shift again.
    assert row.high_water == D("10029.879350497385")


def test_backfill_refuses_a_live_engine(tmp_path, capsys):
    """A fresh heartbeat or a held database lock means the engine is not stopped."""
    import sqlite3

    from api.server import ENGINE_SCOPE, SqliteControlStore

    path = str(tmp_path / "live.db")
    store = SQLitePositionStore(path)
    try:
        store.write_risk_state(Mode.LIVE, AssetClass.CRYPTO, state_at(
            "10000", asset_class=AssetClass.CRYPTO, mode=Mode.LIVE,
        ).as_row())
    finally:
        store.close()
    args = [
        "--db", path, "--mode", "live", "--asset-class", "crypto",
        "--apply-known-2026-11-14",
    ]
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        locked = main(args)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert locked == 1
    locked_err = capsys.readouterr().err
    assert "lock" in locked_err

    now = dt.datetime.now(UTC)
    control = SqliteControlStore(path)
    try:
        control.write(ENGINE_SCOPE, {
            "paused": "0",
            "heartbeat_at": now.isoformat(),
        })
        fresh = main(args)
        fresh_err = capsys.readouterr().err
        control.write(ENGINE_SCOPE, {
            "paused": "0",
            "heartbeat_at": (now - dt.timedelta(hours=2)).isoformat(),
        })
    finally:
        control.close()
    assert fresh == 1
    assert "heartbeat" in fresh_err
    applied = main(args)
    assert applied == 0
    check = SQLitePositionStore(path)
    try:
        row = RiskState.from_row(check.read_risk_state(Mode.LIVE, AssetClass.CRYPTO))
    finally:
        check.close()
    assert row.high_water == D("10000") - D("4440")


def test_backfill_refuses_to_invent_a_baseline(tmp_path):
    path = str(tmp_path / "empty.db")
    SQLitePositionStore(path).close()
    code = main([
        "--db", path, "--mode", "live", "--asset-class", "crypto",
        "--apply-known-2026-11-14",
    ])
    assert code == 1


def test_apply_external_flows_is_idempotent_on_the_store(tmp_path):
    path = str(tmp_path / "one.db")
    store = SQLitePositionStore(path)
    try:
        store.write_risk_state(Mode.LIVE, AssetClass.CRYPTO, state_at(
            "10000", asset_class=AssetClass.CRYPTO, mode=Mode.LIVE,
        ).as_row())
        flows = (flow("once", "-10"),)
        first = apply_external_flows(store, Mode.LIVE, AssetClass.CRYPTO, flows, now=NOW)
        second = apply_external_flows(store, Mode.LIVE, AssetClass.CRYPTO, flows, now=NOW)
        assert first["applied"] == ["once"]
        assert second["applied"] == []
        assert second["high_water"] == "9990"
    finally:
        store.close()
