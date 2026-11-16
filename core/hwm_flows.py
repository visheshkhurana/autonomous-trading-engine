"""One-shot operator backfill for external cash flows.

The engine applies spot deposits and withdrawals from the Kraken ledger
going forward. Flows that landed before that cursor existed are not
replayed, because the stored high-water mark already contains them as if
they were P&L. This command shifts the stored baselines by named flows,
once each, and appends an ``hwm_rebase_json`` row with
``reason=external_cash_flow``.

It is not imported by the engine, the runner, or boot. Do not schedule it.
Stop the engine first so a tick cannot overwrite ``risk_state`` after this
process reads it.

The high-water mark moves by the signed amount. The day's opening equity
moves only when the flow falls inside the stored day. The 14 Nov withdrawals
are before the 16 Nov open, so they move the mark and leave that open as it is.

The 14 Nov 2026 USDT withdrawals::

    python3 -m core.hwm_flows --db "$TRADER_DB" --mode live --asset-class crypto \\
        --apply-known-2026-11-14

That applies, once:

* ``FTGEVAr-syDy6EaOEIqh66t8bwUHlx``  -10 USDT  2026-11-14T18:05:50.123Z
* ``FTPJF3S-WqybEyMe4YF7HfFlQZ87uD``  -4430 USDT  2026-11-14T18:20:28.723Z

A single audited flow::

    python3 -m core.hwm_flows --db "$TRADER_DB" --mode live --asset-class crypto \\
        --refid REFID --asset USDT --amount -10 --at 2026-11-14T18:05:50.123Z

``--amount`` is signed account equity: withdrawals are negative.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence

from core.profiles import profile_for
from core.risk import (
    ExternalCashFlow, RiskState, _append_hwm_audit, _merge_refid, _parse_refids,
    fold_external_cash_flows, start_of_day_key,
)
from core.safety import ENTRY_HEARTBEAT_MAX_AGE_S
from core.store import SQLitePositionStore
from core.types import AssetClass, Mode

# Same scope the engine stamps. Duplicated so this command does not import
# the API process.
_ENGINE_SCOPE = "__engine__"

UTC = dt.timezone.utc

# The two withdrawals that produced the 14 Nov fake drawdown. Amounts are
# signed: cash left the book. Refids are the Kraken ledger ids.
KNOWN_2026_11_14_WITHDRAWALS: tuple[ExternalCashFlow, ...] = (
    ExternalCashFlow(
        refid="FTGEVAr-syDy6EaOEIqh66t8bwUHlx",
        asset="USDT",
        amount=Decimal("-10"),
        at=dt.datetime(2026, 11, 14, 18, 5, 50, 123000, tzinfo=UTC),
    ),
    ExternalCashFlow(
        refid="FTPJF3S-WqybEyMe4YF7HfFlQZ87uD",
        asset="USDT",
        amount=Decimal("-4430"),
        at=dt.datetime(2026, 11, 14, 18, 20, 28, 723000, tzinfo=UTC),
    ),
)


def apply_external_flows(
    store: SQLitePositionStore,
    mode: Mode,
    asset_class: AssetClass,
    flows: Sequence[ExternalCashFlow],
    *,
    now: Optional[dt.datetime] = None,
) -> dict[str, object]:
    """Shift the stored baselines by ``flows``. Each refid applies once.

    Does not read a venue and does not call ``advance``, so the shift is not
    then raised back to the current equity. Refuses to invent a baseline when
    the row is missing.
    """
    stamp = now or dt.datetime.now(UTC)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    row = store.read_risk_state(mode, asset_class)
    if row is None:
        raise RuntimeError(
            f"no risk_state for {mode.value}/{asset_class.value}; "
            "refusing to invent a high-water mark"
        )
    state = RiskState.from_row(row)
    before = _parse_ids(state.processed_flow_refids)
    # The stored day's open already includes cash from before that day.
    # The 14 Nov withdrawals land on a 16 Nov baseline that way: they move
    # the high-water mark and leave day_open alone.
    high_water, day_open, rebase, processed, detail = fold_external_cash_flows(
        state.high_water,
        state.day_open_equity,
        state.hwm_rebase_json,
        state.processed_flow_refids or "[]",
        flows,
        day_starts_at=start_of_day_key(profile_for(asset_class), state.day_key),
    )
    applied = [refid for refid in _parse_ids(processed) if refid not in before]
    updated = RiskState(
        mode=state.mode,
        asset_class=state.asset_class,
        high_water=high_water,
        day_key=state.day_key,
        day_open_equity=day_open,
        updated_at=stamp,
        equity_basis=state.equity_basis,
        hwm_rebase_json=rebase,
        processed_flow_refids=processed,
        flow_watermark_ms=state.flow_watermark_ms,
    )
    store.write_risk_state(mode, asset_class, updated.as_row())
    return {
        "applied": applied,
        "high_water": str(high_water),
        "day_open_equity": str(day_open),
        "detail": detail,
        "audit": json.loads(rebase) if rebase else [],
    }


def resolve_unpriced_refid(
    store: SQLitePositionStore,
    mode: Mode,
    asset_class: AssetClass,
    refid: str,
    *,
    now: Optional[dt.datetime] = None,
) -> dict[str, object]:
    """Clear one latched unpriced refid and append an audit row.

    The high-water mark is not moved here. The next engine tick may ratchet
    only after every latched refid has been resolved. Refuses an unknown refid
    so a typo cannot drop the halt.
    """
    stamp = now or dt.datetime.now(UTC)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    name = refid.strip()
    if not name:
        raise ValueError("refid is required")
    row = store.read_risk_state(mode, asset_class)
    if row is None:
        raise RuntimeError(
            f"no risk_state for {mode.value}/{asset_class.value}"
        )
    state = RiskState.from_row(row)
    latched = _parse_refids(state.unpriced_refids)
    if name not in latched:
        raise RuntimeError(f"{name} is not a latched unpriced cash flow")
    remaining = [item for item in latched if item != name]
    audit = _append_hwm_audit(state.hwm_rebase_json, {
        "at": stamp.astimezone(UTC).isoformat(),
        "refid": name,
        "reason": "unpriced_cash_flow_resolved",
        "old_high_water": str(state.high_water),
        "new_high_water": str(state.high_water),
        "old_day_open": str(state.day_open_equity),
        "new_day_open": str(state.day_open_equity),
    })
    updated = RiskState(
        mode=state.mode,
        asset_class=state.asset_class,
        high_water=state.high_water,
        day_key=state.day_key,
        day_open_equity=state.day_open_equity,
        updated_at=stamp,
        equity_basis=state.equity_basis,
        hwm_rebase_json=audit,
        processed_flow_refids=_merge_refid(state.processed_flow_refids, name),
        flow_watermark_ms=state.flow_watermark_ms,
        ledger_blind=state.ledger_blind,
        unpriced_cash_flow="1" if remaining else "0",
        unpriced_refids=json.dumps(remaining),
    )
    store.write_risk_state(mode, asset_class, updated.as_row())
    return {
        "resolved": name,
        "remaining": remaining,
        "unpriced_cash_flow": updated.unpriced_cash_flow,
        "audit": json.loads(audit) if audit else [],
    }


def _parse_ids(raw: str) -> set[str]:
    if not raw:
        return set()
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        return set()
    return {str(item) for item in parsed}


def refuse_if_engine_live(path: str, now: dt.datetime) -> Optional[str]:
    """Why this command must not write, or None when the engine is stopped.

    A fresh ``heartbeat_at`` on the engine control row means a tick is still
    landing. A database lock means another process is in a write. Either one
    refuses the backfill. A missing heartbeat, or one older than the entry
    stale threshold, is a stopped engine.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    conn = sqlite3.connect(path, timeout=0.5, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=500")
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            return f"engine database lock: {exc}"
        try:
            return _heartbeat_refusal(conn, now)
        finally:
            conn.execute("ROLLBACK")
    finally:
        conn.close()


def _heartbeat_refusal(conn: sqlite3.Connection, now: dt.datetime) -> Optional[str]:
    try:
        row = conn.execute(
            "SELECT state_json FROM engine_control WHERE scope=?",
            (_ENGINE_SCOPE,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None or not row[0]:
        return None
    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        return "engine control row is unreadable; refusing to backfill"
    if not isinstance(payload, dict):
        return "engine control row is unreadable; refusing to backfill"
    raw = payload.get("heartbeat_at")
    if not raw:
        return None
    try:
        beat = dt.datetime.fromisoformat(str(raw))
    except ValueError:
        return f"engine heartbeat {raw!r} is unreadable; refusing to backfill"
    if beat.tzinfo is None or beat.utcoffset() is None:
        return "engine heartbeat is not timezone-aware; refusing to backfill"
    age = (now - beat.astimezone(UTC)).total_seconds()
    if age <= ENTRY_HEARTBEAT_MAX_AGE_S:
        return (
            f"engine heartbeat is {age:.0f}s old; stop the engine before "
            "applying cash-flow backfill"
        )
    return None


def _parse_at(text: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError(f"{text!r} is not timezone-aware")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m core.hwm_flows",
        description=(
            "Apply audited external cash flows to the stored high-water mark. "
            "Not run by the engine. Stop the engine first."
        ),
        epilog=(
            "Known 14 Nov 2026 withdrawals: "
            "FTGEVAr-syDy6EaOEIqh66t8bwUHlx -10 USDT 2026-11-14T18:05:50.123Z; "
            "FTPJF3S-WqybEyMe4YF7HfFlQZ87uD -4430 USDT 2026-11-14T18:20:28.723Z. "
            "Pass --apply-known-2026-11-14 to apply that pair once."
        ),
    )
    parser.add_argument("--db", required=True, help="path to trader.db")
    parser.add_argument("--mode", default="live", choices=["live", "paper"])
    parser.add_argument("--asset-class", default="crypto", choices=["crypto", "equity"])
    parser.add_argument(
        "--apply-known-2026-11-14",
        action="store_true",
        help="apply the two audited 14 Nov 2026 USDT withdrawals",
    )
    parser.add_argument("--refid", help="one additional ledger refid")
    parser.add_argument("--asset", help="asset code for --refid, for example USDT")
    parser.add_argument("--amount", help="signed equity amount for --refid; withdrawals are negative")
    parser.add_argument("--at", help="timezone-aware timestamp for --refid")
    parser.add_argument(
        "--resolve-unpriced",
        help="clear one latched unpriced refid and write an audit row",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    explicit = bool(args.refid or args.asset or args.amount or args.at)
    if args.apply_known_2026_11_14 and explicit:
        parser.error(
            "--apply-known-2026-11-14 and --refid are mutually exclusive"
        )
    if args.resolve_unpriced and (args.apply_known_2026_11_14 or explicit):
        parser.error("--resolve-unpriced is mutually exclusive with applying flows")

    flows: list[ExternalCashFlow] = []
    if args.apply_known_2026_11_14:
        flows.extend(KNOWN_2026_11_14_WITHDRAWALS)
    elif explicit:
        if not all((args.refid, args.asset, args.amount, args.at)):
            parser.error("--refid, --asset, --amount and --at are required together")
        try:
            amount = Decimal(args.amount)
        except InvalidOperation:
            parser.error(f"--amount {args.amount!r} is not a decimal")
        flows.append(ExternalCashFlow(
            refid=args.refid,
            asset=args.asset.upper(),
            amount=amount,
            at=_parse_at(args.at),
        ))
    if args.resolve_unpriced:
        refusal = refuse_if_engine_live(args.db, dt.datetime.now(UTC))
        if refusal is not None:
            print(f"not applied: {refusal}", file=sys.stderr)
            return 1
        store = SQLitePositionStore(args.db)
        try:
            summary = resolve_unpriced_refid(
                store, Mode(args.mode), AssetClass(args.asset_class),
                args.resolve_unpriced,
            )
        except Exception as exc:
            print(f"not applied: {exc}", file=sys.stderr)
            return 1
        finally:
            store.close()
        print(json.dumps(summary, default=str, indent=2))
        return 0

    if not flows:
        parser.error(
            "pass --apply-known-2026-11-14, an explicit --refid flow, "
            "or --resolve-unpriced"
        )

    refusal = refuse_if_engine_live(args.db, dt.datetime.now(UTC))
    if refusal is not None:
        print(f"not applied: {refusal}", file=sys.stderr)
        return 1

    store = SQLitePositionStore(args.db)
    try:
        summary = apply_external_flows(
            store, Mode(args.mode), AssetClass(args.asset_class), flows,
        )
    except Exception as exc:
        print(f"not applied: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    print(json.dumps(summary, default=str, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
