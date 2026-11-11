"""
The ledger: SQLite-backed positions, orders, fills and risk state.

Two properties matter more than anything else in this file.

1. MODE ISOLATION. Every single read filters on `mode`. Paper rows are
   invisible to a live query and live rows are invisible to a paper query.
   In the previous system these shared a table and paper P&L force-flattened a
   live book. There is no "all modes" read here, deliberately.

2. FAILURES ARE LOUD. A malformed timestamp, an unknown mode, a corrupt row —
   all raise. Nothing here returns 0, [] or None to paper over a read it could
   not perform, because downstream those values are indistinguishable from
   "flat", "no bots" and "no drawdown", and each of those has cost real money.
   The only zeros and empties this module returns are answers it actually
   computed.

`apply()` is the single writer for positions. `record_order()` writes the
order ledger and NEVER moves a position — the broker-reported fills it carries
are stored verbatim on the order row, separately from the `fills` table, which
holds only what was actually applied to a position.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterator, Mapping, Optional, Sequence

from .gate import GateStateUnavailable
from .types import (
    AssetClass, Fill, Instrument, Intent, Mode, OrderRef, OrderStatus,
    OrderType, Position, Side,
)

DEFAULT_DB_PATH = "/data/automoney.db"
UTC = dt.timezone.utc

# Fixed-width UTC text ("2026-08-27T09:15:00.000000+00:00") so that lexical
# ordering in SQLite is chronological ordering. Any other format silently
# breaks the `since_iso` window in realized_pnl().
_TS_SPEC = "microseconds"


# --------------------------------------------------------------------------
# conversions — every one of these raises rather than guessing
# --------------------------------------------------------------------------

def _utc_text(t: dt.datetime) -> str:
    """Naive datetimes are rejected: 'is that IST or UTC?' is a 5.5 hour error
    in a daily-loss window, and it would be silent."""
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise ValueError(
            f"timestamp {t!r} is naive; the ledger requires a tz-aware instant"
        )
    return t.astimezone(UTC).isoformat(timespec=_TS_SPEC)


def _parse_since(since_iso: str) -> str:
    """Normalise a caller-supplied ISO instant into ledger text.

    Rejects naive input for the same reason as _utc_text. Note for callers:
    AssetClassProfile.day_key() returns a LOCAL calendar date (IST for equity),
    not an instant — convert it to an aware datetime before passing it here.
    """
    if not isinstance(since_iso, str):
        raise TypeError(f"since_iso must be str, got {type(since_iso).__name__}")
    parsed = dt.datetime.fromisoformat(since_iso)   # raises on malformed input
    return _utc_text(parsed)


def _mode_value(mode: Mode) -> str:
    """A raw string here would query a mode that does not exist and come back
    empty — which reads as 'flat'. Refuse instead."""
    if not isinstance(mode, Mode):
        raise TypeError(f"mode must be a Mode enum, got {type(mode).__name__}")
    return mode.value


def _asset_value(asset_class: AssetClass) -> str:
    if not isinstance(asset_class, AssetClass):
        raise TypeError(
            f"asset_class must be an AssetClass enum, got {type(asset_class).__name__}"
        )
    return asset_class.value


def _instrument_obj(inst: Instrument) -> dict[str, Any]:
    return {
        "symbol": inst.symbol,
        "asset_class": inst.asset_class.value,
        "broker_id": inst.broker_id,
        "lot_size": inst.lot_size,
        "tick_size": str(inst.tick_size),
        "qty_precision": inst.qty_precision,
        "min_notional": str(inst.min_notional),
        "expiry": inst.expiry.isoformat() if inst.expiry is not None else None,
    }


def _instrument_from_mapping(o: Mapping[str, Any]) -> Instrument:
    expiry = o["expiry"]
    return Instrument(
        symbol=str(o["symbol"]),
        asset_class=AssetClass(str(o["asset_class"])),
        broker_id=str(o["broker_id"]),
        lot_size=int(o["lot_size"]),
        tick_size=Decimal(str(o["tick_size"])),
        qty_precision=int(o["qty_precision"]),
        min_notional=Decimal(str(o["min_notional"])),
        expiry=dt.date.fromisoformat(str(expiry)) if expiry is not None else None,
    )


def _fill_obj(fill: Fill) -> dict[str, Any]:
    return {
        "order_id": fill.order_id,
        "instrument": _instrument_obj(fill.instrument),
        "side": fill.side.value,
        "qty": str(fill.qty),
        "price": str(fill.price),
        "fees": str(fill.fees),
        "at": _utc_text(fill.at),
        "mode": fill.mode.value,
    }


def _fill_from_mapping(o: Mapping[str, Any]) -> Fill:
    return Fill(
        order_id=str(o["order_id"]),
        instrument=_instrument_from_mapping(o["instrument"]),
        side=Side(str(o["side"])),
        qty=Decimal(str(o["qty"])),
        price=Decimal(str(o["price"])),
        fees=Decimal(str(o["fees"])),
        at=dt.datetime.fromisoformat(str(o["at"])),
        mode=Mode(str(o["mode"])),
    )


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------
# netting — pure, so the money maths can be tested without a database
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NetResult:
    qty: Decimal            # signed position AFTER the fill
    avg_price: Decimal      # 0 when flat
    realized: Decimal       # realised on this fill, net of its fees
    reopened: bool          # this fill opened a new position (incl. after a cross)


def net_fill(
    prev_qty: Decimal, prev_avg: Decimal,
    signed_qty: Decimal, price: Decimal, fees: Decimal,
) -> NetResult:
    """Net one fill into a position.

    Fees are charged in full against realised P&L on the fill that incurs them,
    including the fill that opens a position (which therefore realises -fees).
    Amortising them into the average price would quietly move the break-even
    the risk circuit reasons about.
    """
    if prev_qty == 0:
        return NetResult(signed_qty, price, -fees, True)

    same_side = (prev_qty > 0) == (signed_qty > 0)
    if same_side:
        new_qty = prev_qty + signed_qty
        avg = (abs(prev_qty) * prev_avg + abs(signed_qty) * price) / abs(new_qty)
        return NetResult(new_qty, avg, -fees, False)

    # Opposing side: the overlapping quantity closes and realises.
    closed = min(abs(signed_qty), abs(prev_qty))
    direction = Decimal(1) if prev_qty > 0 else Decimal(-1)
    realized = closed * (price - prev_avg) * direction - fees
    new_qty = prev_qty + signed_qty

    if new_qty == 0:
        return NetResult(Decimal(0), Decimal(0), realized, False)
    if (new_qty > 0) == (prev_qty > 0):
        # Partial reduction — the cost basis of what is left is unchanged.
        return NetResult(new_qty, prev_avg, realized, False)
    # Crossed through zero: the remainder is a NEW position on the other side,
    # priced at this fill. Carrying the old average across the flip would
    # invent P&L that never existed.
    return NetResult(new_qty, price, realized, True)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

_INSTRUMENT_COLS = """
    asset_class    TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    broker_id      TEXT NOT NULL,
    lot_size       INTEGER NOT NULL,
    tick_size      TEXT NOT NULL,
    qty_precision  INTEGER NOT NULL,
    min_notional   TEXT NOT NULL,
    expiry         TEXT
"""

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS orders (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT NOT NULL,
    mode            TEXT NOT NULL,
    status          TEXT NOT NULL,
    {_INSTRUMENT_COLS},
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    qty             TEXT NOT NULL,
    limit_price     TEXT,
    strategy_id     TEXT NOT NULL,
    reason          TEXT NOT NULL,
    structure_id    TEXT,
    idempotency_key TEXT NOT NULL,
    filled_qty      TEXT NOT NULL,
    avg_price       TEXT,
    error           TEXT,
    fills_json      TEXT NOT NULL,
    payload_hash    TEXT NOT NULL,
    recorded_at     TEXT NOT NULL,
    UNIQUE (mode, order_id, payload_hash)
);
CREATE INDEX IF NOT EXISTS idx_orders_lookup ON orders (mode, order_id, seq);

CREATE TABLE IF NOT EXISTS fills (
    mode          TEXT NOT NULL,
    order_id      TEXT NOT NULL,
    fill_index    INTEGER NOT NULL,
    dedupe_key    TEXT NOT NULL,
    {_INSTRUMENT_COLS},
    side          TEXT NOT NULL,
    qty           TEXT NOT NULL,
    price         TEXT NOT NULL,
    fees          TEXT NOT NULL,
    at_utc        TEXT NOT NULL,
    strategy_id   TEXT NOT NULL,
    structure_id  TEXT,
    realized_pnl  TEXT NOT NULL,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (mode, order_id, fill_index),
    UNIQUE (mode, order_id, dedupe_key)
);
CREATE INDEX IF NOT EXISTS idx_fills_pnl ON fills (mode, asset_class, at_utc);

CREATE TABLE IF NOT EXISTS positions (
    mode          TEXT NOT NULL,
    strategy_id   TEXT NOT NULL,
    {_INSTRUMENT_COLS},
    qty           TEXT NOT NULL,
    avg_price     TEXT NOT NULL,
    structure_id  TEXT,
    opened_at     TEXT,
    updated_at    TEXT NOT NULL,
    stop_txid     TEXT,
    stop_price    TEXT,
    stop_qty      TEXT,
    stop_status   TEXT,
    PRIMARY KEY (mode, symbol, strategy_id)
);
CREATE INDEX IF NOT EXISTS idx_positions_mode ON positions (mode, asset_class);

CREATE TABLE IF NOT EXISTS risk_state (
    mode         TEXT NOT NULL,
    asset_class  TEXT NOT NULL,
    state_json   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (mode, asset_class)
);

CREATE TABLE IF NOT EXISTS gate_submissions (
    idempotency_key TEXT NOT NULL,
    mode            TEXT NOT NULL,
    asset_class     TEXT NOT NULL,
    day_key         TEXT NOT NULL,
    claimed_at      TEXT NOT NULL,
    notional        TEXT NOT NULL DEFAULT '0',
    PRIMARY KEY (mode, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_gate_day
    ON gate_submissions (mode, asset_class, day_key);
"""


class SQLitePositionStore:
    """Implements PositionStoreProto. One file, WAL mode, one writer.

    Thread-safe: a single connection guarded by a lock, so the read-modify-write
    inside apply() cannot interleave. Cross-process writers are serialised by
    BEGIN IMMEDIATE.
    """

    def __init__(self, path: str = DEFAULT_DB_PATH,
                 now_fn: Any = None) -> None:
        self._path = path
        # Injectable only so tests can pin the ledger's receipt timestamps;
        # nothing in the money maths reads it.
        self._now = now_fn or (lambda: dt.datetime.now(UTC))
        self._lock = threading.RLock()

        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row

        journal = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if journal.lower() not in {"wal", "memory"}:
            # A silent fall back to rollback journalling (network fs, some
            # container volumes) changes the durability guarantee the ledger is
            # relying on. Say so rather than discovering it after a crash.
            raise RuntimeError(
                f"{path}: could not enable WAL, journal_mode is {journal!r}"
            )
        # The ledger is the one artifact that cannot be reconstructed from
        # anywhere else, so pay the fsync.
        self._conn.execute("PRAGMA synchronous=FULL")
        # The engine and the API are separate processes on one file. Without a
        # busy timeout the second writer gets "database is locked" and a fill
        # goes unrecorded; with it, it waits its turn.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        _ensure_position_stop_columns(self._conn)

    # -- plumbing ---------------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLitePositionStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                # Inside the guard: a COMMIT that fails (full disk, a lost
                # network volume) leaves the transaction OPEN, and then every
                # later write dies with "cannot start a transaction within a
                # transaction". The ledger would report a bogus lock error for
                # the rest of the process's life while the real failure — the
                # one the operator has to act on — scrolled past once.
                self._conn.execute("COMMIT")
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    # Nothing was left to roll back. Swallowed deliberately and
                    # narrowly: it discards no result and hides no state, it
                    # only stops a cleanup error from masking the exception
                    # that actually explains what went wrong, which is re-raised
                    # immediately below.
                    pass
                raise

    def _read(self, sql: str, params: Sequence[Any]) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)))

    # -- orders -----------------------------------------------------------

    def record_order(self, ref: OrderRef) -> None:
        """Append a terminal OrderRef — rejections and timeouts included.

        Append-only: a second, DIFFERENT ref for the same order_id is stored as
        a new row rather than overwriting the first, because a ledger that can
        be rewritten is not a ledger. Re-recording the identical ref is a no-op.
        """
        if not isinstance(ref, OrderRef):
            raise TypeError(f"record_order expects OrderRef, got {type(ref).__name__}")
        mode = _mode_value(ref.mode)
        for f in ref.fills:
            if f.mode is not ref.mode:
                # A ref carrying fills from the other mode is corrupt, and
                # writing it would put live fills under a paper order.
                raise ValueError(
                    f"order {ref.order_id}: fill mode {f.mode} != order mode {ref.mode}"
                )

        intent = ref.intent
        inst = intent.instrument
        fills_json = json.dumps([_fill_obj(f) for f in ref.fills],
                                sort_keys=True, separators=(",", ":"))
        payload = {
            "order_id": ref.order_id,
            "status": ref.status.value,
            "mode": mode,
            "instrument": _instrument_obj(inst),
            "side": intent.side.value,
            "order_type": intent.order_type.value,
            "qty": str(intent.qty),
            "limit_price": None if intent.limit_price is None else str(intent.limit_price),
            "strategy_id": intent.strategy_id,
            "reason": intent.reason,
            "structure_id": intent.structure_id,
            "idempotency_key": intent.idempotency_key,
            "filled_qty": str(ref.filled_qty),
            "avg_price": None if ref.avg_price is None else str(ref.avg_price),
            "error": ref.error,
            "fills": fills_json,
        }
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    order_id, mode, status,
                    asset_class, symbol, broker_id, lot_size, tick_size,
                    qty_precision, min_notional, expiry,
                    side, order_type, qty, limit_price,
                    strategy_id, reason, structure_id, idempotency_key,
                    filled_qty, avg_price, error, fills_json,
                    payload_hash, recorded_at
                ) VALUES (?,?,?, ?,?,?,?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?)
                ON CONFLICT (mode, order_id, payload_hash) DO NOTHING
                """,
                (
                    ref.order_id, mode, ref.status.value,
                    inst.asset_class.value, inst.symbol, inst.broker_id,
                    inst.lot_size, str(inst.tick_size), inst.qty_precision,
                    str(inst.min_notional),
                    inst.expiry.isoformat() if inst.expiry is not None else None,
                    intent.side.value, intent.order_type.value, str(intent.qty),
                    None if intent.limit_price is None else str(intent.limit_price),
                    intent.strategy_id, intent.reason, intent.structure_id,
                    intent.idempotency_key,
                    str(ref.filled_qty),
                    None if ref.avg_price is None else str(ref.avg_price),
                    ref.error, fills_json,
                    _sha(payload), _utc_text(self._now()),
                ),
            )

    def get_order(self, mode: Mode, order_id: str) -> Optional[OrderRef]:
        """Latest recorded ref for this order in THIS mode. None when the
        ledger has never seen it — which is a fact, not a failure."""
        rows = self._read(
            "SELECT * FROM orders WHERE mode=? AND order_id=? ORDER BY seq DESC LIMIT 1",
            (_mode_value(mode), order_id),
        )
        return _order_from_row(rows[0]) if rows else None

    def orders(self, mode: Mode, asset_class: Optional[AssetClass] = None,
               since_iso: Optional[str] = None,
               limit: Optional[int] = None) -> Sequence[OrderRef]:
        sql = "SELECT * FROM orders WHERE mode=?"
        params: list[Any] = [_mode_value(mode)]
        if asset_class is not None:
            sql += " AND asset_class=?"
            params.append(_asset_value(asset_class))
        if since_iso is not None:
            sql += " AND recorded_at>=?"
            params.append(_parse_since(since_iso))
        sql += " ORDER BY seq DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [_order_from_row(r) for r in self._read(sql, params)]

    # -- fills / positions -------------------------------------------------

    def apply(self, fill: Fill, strategy_id: str,
              structure_id: Optional[str] = None,
              fill_index: Optional[int] = None) -> None:
        """Net a fill into its position. The only writer of `positions`.

        Idempotency: Fill carries no index of its own, so when `fill_index` is
        omitted the fill is keyed by the hash of its content. Replaying the same
        fill is then a no-op. An adapter that can emit two byte-identical fills
        on one order (same qty, price, fees AND timestamp) MUST pass an explicit
        `fill_index`, or the second one is treated as a replay and dropped.
        Do not mix the two conventions within one order.
        """
        if not isinstance(fill, Fill):
            raise TypeError(f"apply expects Fill, got {type(fill).__name__}")
        if fill.qty <= 0:
            raise ValueError(f"fill {fill.order_id}: qty must be > 0, got {fill.qty}")
        if not fill.price.is_finite() or fill.price <= 0:
            # A zero is as unusable as a negative here. It is not a traded
            # price, and the position it creates carries avg_price 0 — which
            # every notional cap that has to fall back on the cost basis then
            # measures as no exposure at all. A real open position invisible to
            # the caps is exactly the class of bug this ledger exists to stop.
            raise ValueError(
                f"fill {fill.order_id}: zero or negative price {fill.price}"
            )
        if not strategy_id:
            raise ValueError("apply requires a strategy_id; positions are per strategy")

        mode = _mode_value(fill.mode)
        inst = fill.instrument
        at_utc = _utc_text(fill.at)
        signed = fill.qty if fill.side is Side.BUY else -fill.qty

        if fill_index is None:
            dedupe = "content:" + _sha(_fill_obj(fill))
        else:
            if fill_index < 0:
                raise ValueError(f"fill_index must be >= 0, got {fill_index}")
            dedupe = f"index:{fill_index}"

        with self._write() as conn:
            seen = conn.execute(
                "SELECT 1 FROM fills WHERE mode=? AND order_id=? AND dedupe_key=?",
                (mode, fill.order_id, dedupe),
            ).fetchone()
            if seen is not None:
                return   # already applied; positions must not move again

            if fill_index is None:
                row = conn.execute(
                    "SELECT MAX(fill_index) FROM fills WHERE mode=? AND order_id=?",
                    (mode, fill.order_id),
                ).fetchone()
                index = 0 if row[0] is None else int(row[0]) + 1
            else:
                index = fill_index

            pos = conn.execute(
                "SELECT qty, avg_price, opened_at, structure_id, "
                "stop_txid, stop_price, stop_qty, stop_status FROM positions "
                "WHERE mode=? AND symbol=? AND strategy_id=?",
                (mode, inst.symbol, strategy_id),
            ).fetchone()
            prev_qty = Decimal(pos["qty"]) if pos else Decimal(0)
            prev_avg = Decimal(pos["avg_price"]) if pos else Decimal(0)

            result = net_fill(prev_qty, prev_avg, signed, fill.price, fill.fees)

            conn.execute(
                """
                INSERT INTO fills (
                    mode, order_id, fill_index, dedupe_key,
                    asset_class, symbol, broker_id, lot_size, tick_size,
                    qty_precision, min_notional, expiry,
                    side, qty, price, fees, at_utc,
                    strategy_id, structure_id, realized_pnl, recorded_at
                ) VALUES (?,?,?,?, ?,?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?)
                """,
                (
                    mode, fill.order_id, index, dedupe,
                    inst.asset_class.value, inst.symbol, inst.broker_id,
                    inst.lot_size, str(inst.tick_size), inst.qty_precision,
                    str(inst.min_notional),
                    inst.expiry.isoformat() if inst.expiry is not None else None,
                    fill.side.value, str(fill.qty), str(fill.price),
                    str(fill.fees), at_utc,
                    strategy_id, structure_id, str(result.realized),
                    _utc_text(self._now()),
                ),
            )

            if result.qty == 0:
                # Flat. The row goes; the fills ledger keeps the history, and a
                # zero-qty row lying around is one more thing to filter out of
                # every read.
                conn.execute(
                    "DELETE FROM positions WHERE mode=? AND symbol=? AND strategy_id=?",
                    (mode, inst.symbol, strategy_id),
                )
                return

            opened_at = at_utc if (pos is None or result.reopened) else pos["opened_at"]
            struct = structure_id if (pos is None or result.reopened) else pos["structure_id"]
            # A new position does not inherit the previous one's stop. Adding
            # to a position keeps the resting stop until the engine replaces
            # it; wiping it here would forget a live txid between ticks.
            fresh = pos is None or result.reopened
            stop_txid = None if fresh else pos["stop_txid"]
            stop_price = None if fresh else pos["stop_price"]
            stop_qty = None if fresh else pos["stop_qty"]
            stop_status = None if fresh else pos["stop_status"]
            conn.execute(
                """
                INSERT INTO positions (
                    mode, strategy_id,
                    asset_class, symbol, broker_id, lot_size, tick_size,
                    qty_precision, min_notional, expiry,
                    qty, avg_price, structure_id, opened_at, updated_at,
                    stop_txid, stop_price, stop_qty, stop_status
                ) VALUES (?,?, ?,?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?)
                ON CONFLICT (mode, symbol, strategy_id) DO UPDATE SET
                    qty=excluded.qty,
                    avg_price=excluded.avg_price,
                    structure_id=excluded.structure_id,
                    opened_at=excluded.opened_at,
                    updated_at=excluded.updated_at,
                    stop_txid=excluded.stop_txid,
                    stop_price=excluded.stop_price,
                    stop_qty=excluded.stop_qty,
                    stop_status=excluded.stop_status
                """,
                (
                    mode, strategy_id,
                    inst.asset_class.value, inst.symbol, inst.broker_id,
                    inst.lot_size, str(inst.tick_size), inst.qty_precision,
                    str(inst.min_notional),
                    inst.expiry.isoformat() if inst.expiry is not None else None,
                    str(result.qty), str(result.avg_price), struct,
                    opened_at, _utc_text(self._now()),
                    stop_txid, stop_price, stop_qty, stop_status,
                ),
            )

    def positions(self, mode: Mode,
                  asset_class: Optional[AssetClass] = None) -> Sequence[Position]:
        sql = "SELECT * FROM positions WHERE mode=?"
        params: list[Any] = [_mode_value(mode)]
        if asset_class is not None:
            sql += " AND asset_class=?"
            params.append(_asset_value(asset_class))
        sql += " ORDER BY symbol, strategy_id"
        return [_position_from_row(r) for r in self._read(sql, params)]

    def position(self, mode: Mode, symbol: str,
                 strategy_id: str) -> Optional[Position]:
        """None means flat in THIS mode — an answer, not a read failure."""
        rows = self._read(
            "SELECT * FROM positions WHERE mode=? AND symbol=? AND strategy_id=?",
            (_mode_value(mode), symbol, strategy_id),
        )
        return _position_from_row(rows[0]) if rows else None

    def set_native_stop(
        self, mode: Mode, symbol: str, strategy_id: str,
        *,
        txid: Optional[str],
        stop_price: Optional[Decimal],
        stop_qty: Optional[Decimal],
        status: str,
    ) -> None:
        """Record the exchange stop that protects this ledger row.

        The qty stored here is the engine position size the stop was built
        from. It is never a wallet balance. Missing row raises: writing a
        txid onto a book we cannot see would report a stop for a position
        that is not there.
        """
        if not status.strip():
            raise ValueError("native stop status must be non-empty")
        if stop_price is not None and stop_price <= 0:
            raise ValueError(f"stop price must be > 0, got {stop_price}")
        if stop_qty is not None and stop_qty <= 0:
            raise ValueError(f"stop qty must be > 0, got {stop_qty}")
        with self._write() as conn:
            cur = conn.execute(
                """
                UPDATE positions
                   SET stop_txid=?, stop_price=?, stop_qty=?, stop_status=?,
                       updated_at=?
                 WHERE mode=? AND symbol=? AND strategy_id=?
                """,
                (
                    txid,
                    None if stop_price is None else str(stop_price),
                    None if stop_qty is None else str(stop_qty),
                    status,
                    _utc_text(self._now()),
                    _mode_value(mode), symbol, strategy_id,
                ),
            )
            if cur.rowcount != 1:
                raise LookupError(
                    f"no {mode.value} position {symbol}/{strategy_id} to attach "
                    "a native stop to"
                )

    def fills(self, mode: Mode, asset_class: Optional[AssetClass] = None,
              since_iso: Optional[str] = None) -> Sequence[Fill]:
        sql = "SELECT * FROM fills WHERE mode=?"
        params: list[Any] = [_mode_value(mode)]
        if asset_class is not None:
            sql += " AND asset_class=?"
            params.append(_asset_value(asset_class))
        if since_iso is not None:
            sql += " AND at_utc>=?"
            params.append(_parse_since(since_iso))
        sql += " ORDER BY at_utc, order_id, fill_index"
        return [_fill_from_row(r) for r in self._read(sql, params)]

    def realized_pnl(self, mode: Mode, asset_class: AssetClass,
                     since_iso: str) -> Decimal:
        """Realised P&L, net of fees, for fills at or after `since_iso`.

        Summed in Decimal rather than SQL SUM() because the amounts are stored
        as exact text; float accumulation would drift the number the daily-loss
        cap is compared against.
        """
        rows = self._read(
            "SELECT realized_pnl FROM fills "
            "WHERE mode=? AND asset_class=? AND at_utc>=?",
            (_mode_value(mode), _asset_value(asset_class), _parse_since(since_iso)),
        )
        total = Decimal(0)
        for r in rows:
            total += Decimal(r["realized_pnl"])
        return total

    # -- risk state --------------------------------------------------------

    def read_risk_state(self, mode: Mode,
                        asset_class: AssetClass) -> Optional[dict[str, str]]:
        """None means 'never written'. It does NOT mean zeros — a circuit that
        reads zeros for peak equity and drawdown concludes there is no drawdown
        and disarms itself."""
        rows = self._read(
            "SELECT state_json FROM risk_state WHERE mode=? AND asset_class=?",
            (_mode_value(mode), _asset_value(asset_class)),
        )
        if not rows:
            return None
        state = json.loads(rows[0]["state_json"])
        if not isinstance(state, dict):
            raise ValueError(
                f"risk_state for {mode}/{asset_class} is corrupt: not an object"
            )
        return {str(k): str(v) for k, v in state.items()}

    def write_risk_state(self, mode: Mode, asset_class: AssetClass,
                         state: Mapping[str, str]) -> None:
        """Values must already be strings. Decimals are stringified exactly by
        the caller so nothing round-trips through float on the way to a
        drawdown comparison."""
        for k, v in state.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError(
                    f"risk_state entries must be str->str; got {k!r}->{type(v).__name__}"
                )
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO risk_state (mode, asset_class, state_json, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT (mode, asset_class) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (_mode_value(mode), _asset_value(asset_class),
                 json.dumps(dict(state), sort_keys=True), _utc_text(self._now())),
            )


# --------------------------------------------------------------------------
# row -> dataclass
# --------------------------------------------------------------------------

class SQLiteGateStateStore:
    """Implements `core.gate.GateStateStoreProto` on the ledger's own file.

    The gate's docstring is explicit that an in-process dict satisfies the
    Protocol and must never be used outside tests: N workers would each keep
    their own counters, so the daily cap becomes N times what it says and a
    retry landing on another worker places a second real order. This is the
    implementation that makes the cap mean something — the claim and the count
    live in SQLite, so every process on the file sees the same numbers.

    `mode` is fixed at construction, exactly as `RiskCircuit` fixes it and for
    the same reason: a paper session and a live session pointed at one file
    must not share a daily order budget or a dedup namespace. Within one
    process the gate may coerce an unarmed class from live to paper, and those
    coerced orders are counted against the constructed mode. That direction is
    deliberate — it over-counts, which refuses earlier, and the alternative
    (an uncounted order) is the one that cannot be allowed.
    """

    def __init__(self, path: str = DEFAULT_DB_PATH, mode: Mode = Mode.PAPER) -> None:
        if not isinstance(mode, Mode):
            raise TypeError("mode must be a Mode")
        self._path = path
        self._mode = _mode_value(mode)
        self._lock = threading.RLock()

        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        journal = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if journal.lower() not in {"wal", "memory"}:
            raise RuntimeError(
                f"{path}: could not enable WAL, journal_mode is {journal!r}"
            )
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._ensure_gate_notional_column()

    def _ensure_gate_notional_column(self) -> None:
        """Older ledgers created gate_submissions without `notional`. A missing
        column would make daily_notional refuse every crypto entry after a
        deploy that added the absolute daily notional cap."""
        cols = {row[1] for row in self._conn.execute(
            "PRAGMA table_info(gate_submissions)")}
        if "notional" not in cols:
            self._conn.execute(
                "ALTER TABLE gate_submissions "
                "ADD COLUMN notional TEXT NOT NULL DEFAULT '0'"
            )

    @property
    def path(self) -> str:
        return self._path

    @property
    def mode(self) -> Mode:
        return Mode(self._mode)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteGateStateStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- GateStateStoreProto ----------------------------------------------

    def seen_recently(self, idempotency_key: str, window_s: float,
                      now: dt.datetime) -> bool:
        """True if this key was claimed within the window.

        Raises rather than returning False on any failure: False means "this
        key is new" and would let the gate place a duplicate of an order that
        may already be live at the venue.
        """
        key = self._key(idempotency_key)
        if not isinstance(window_s, (int, float)) or isinstance(window_s, bool):
            raise GateStateUnavailable(f"window_s must be a number, got {window_s!r}")
        if window_s <= 0:
            raise GateStateUnavailable(f"window_s must be > 0, got {window_s}")
        cutoff = _utc_text(self._instant(now) - dt.timedelta(seconds=float(window_s)))
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT claimed_at FROM gate_submissions "
                    "WHERE mode = ? AND idempotency_key = ?",
                    (self._mode, key),
                ).fetchone()
        except sqlite3.Error as exc:
            raise GateStateUnavailable(
                f"dedup read failed for {key!r}: {exc!r}") from exc
        if row is None:
            return False
        # Lexical comparison is chronological because _utc_text is fixed-width.
        # A claim stamped in the FUTURE (clock skew between processes) compares
        # greater than the cutoff and so reads as seen — the closed direction.
        return str(row["claimed_at"]) > cutoff

    def order_count(self, asset_class: AssetClass, day_key: str) -> int:
        """Approved orders claimed for this class on this trading day.

        Derived by counting the claims themselves rather than kept in a
        separate counter column: one number that can drift out of step with the
        rows it summarises is one number too many, and the cap is the thing
        that stops a runaway strategy.
        """
        if not isinstance(day_key, str) or not day_key.strip():
            raise GateStateUnavailable(f"day_key must be a non-empty str, got {day_key!r}")
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM gate_submissions "
                    "WHERE mode = ? AND asset_class = ? AND day_key = ?",
                    (self._mode, _asset_value(asset_class), day_key),
                ).fetchone()
        except sqlite3.Error as exc:
            raise GateStateUnavailable(
                f"order-count read failed for {asset_class!r}/{day_key!r}: {exc!r}"
            ) from exc
        if row is None:                       # pragma: no cover - COUNT always returns
            raise GateStateUnavailable("order-count query returned no row")
        return int(row["n"])

    def daily_notional(self, asset_class: AssetClass, day_key: str) -> Decimal:
        """Approved entry notional claimed for this class on this trading day.

        Summed from the claim rows themselves (same source as order_count). A
        separate counter that can drift from the claims is one number too many.
        """
        if not isinstance(day_key, str) or not day_key.strip():
            raise GateStateUnavailable(f"day_key must be a non-empty str, got {day_key!r}")
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT notional FROM gate_submissions "
                    "WHERE mode = ? AND asset_class = ? AND day_key = ?",
                    (self._mode, _asset_value(asset_class), day_key),
                ).fetchall()
        except sqlite3.Error as exc:
            raise GateStateUnavailable(
                f"daily-notional read failed for {asset_class!r}/{day_key!r}: "
                f"{exc!r}"
            ) from exc
        total = Decimal("0")
        for row in rows:
            raw = row["notional"]
            try:
                value = Decimal(str(raw))
            except Exception as exc:
                raise GateStateUnavailable(
                    f"daily-notional row {raw!r} is not a Decimal: {exc!r}"
                ) from exc
            if not value.is_finite() or value < 0:
                raise GateStateUnavailable(
                    f"daily-notional row {raw!r} is not a non-negative finite Decimal"
                )
            total += value
        return total

    def record_submission(self, idempotency_key: str, asset_class: AssetClass,
                          day_key: str, now: dt.datetime,
                          notional: Decimal = Decimal("0")) -> None:
        """Claim the key and, by doing so, spend one of the day's orders.

        A plain INSERT, so a key already in the table raises. That raise is the
        point: it is the ONLY step in the sequence that is atomic across
        processes, which makes it the real dedup, and `seen_recently` merely an
        early and cheaper refusal. Two workers that both read "not seen" for
        one key still cannot both claim it — the second one's INSERT loses to
        the primary key and the gate refuses instead of placing the second
        order.

        Claims are never pruned. A key whose dedup window has expired is still
        a key that was ALREADY APPROVED once, and re-approving it would place a
        second order for one decision. Strategy keys carry the deciding bar's
        timestamp and square-off keys carry a time bucket, so a legitimately
        new decision always arrives with a new key.
        """
        key = self._key(idempotency_key)
        if not isinstance(day_key, str) or not day_key.strip():
            raise GateStateUnavailable(f"day_key must be a non-empty str, got {day_key!r}")
        if not isinstance(notional, Decimal) or not notional.is_finite() or notional < 0:
            raise GateStateUnavailable(
                f"notional must be a non-negative finite Decimal, got {notional!r}")
        stamp = _utc_text(self._instant(now))
        try:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT INTO gate_submissions "
                        "(idempotency_key, mode, asset_class, day_key, claimed_at, notional) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (key, self._mode, _asset_value(asset_class), day_key, stamp,
                         str(notional)),
                    )
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
                self._conn.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            raise GateStateUnavailable(
                f"idempotency_key {key!r} is already claimed in {self._mode}; "
                "refusing to approve one decision twice"
            ) from exc
        except sqlite3.Error as exc:
            raise GateStateUnavailable(
                f"could not claim {key!r}: {exc!r}") from exc

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _key(idempotency_key: str) -> str:
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise GateStateUnavailable(
                f"idempotency_key must be a non-empty str, got {idempotency_key!r}")
        return idempotency_key.strip()

    @staticmethod
    def _instant(now: dt.datetime) -> dt.datetime:
        if not isinstance(now, dt.datetime):
            raise GateStateUnavailable(f"now must be a datetime, got {now!r}")
        if now.tzinfo is None or now.utcoffset() is None:
            # A naive timestamp would be compared against tz-aware rows written
            # by another process, and the dedup window would be wrong by the
            # local UTC offset — 5.5 hours here.
            raise GateStateUnavailable(f"now must be timezone-aware, got {now!r}")
        return now


def _instrument_from_row(row: sqlite3.Row) -> Instrument:
    return _instrument_from_mapping({
        "symbol": row["symbol"],
        "asset_class": row["asset_class"],
        "broker_id": row["broker_id"],
        "lot_size": row["lot_size"],
        "tick_size": row["tick_size"],
        "qty_precision": row["qty_precision"],
        "min_notional": row["min_notional"],
        "expiry": row["expiry"],
    })


def _row_text(row: sqlite3.Row, key: str) -> Optional[str]:
    if key not in row.keys():
        return None
    value = row[key]
    if value is None or value == "":
        return None
    return str(value)


def _ensure_position_stop_columns(conn: sqlite3.Connection) -> None:
    """Add native-stop columns to a ledger created before they existed.

    ``CREATE TABLE IF NOT EXISTS`` does not alter a table that already
    exists, and a restart onto this build must not refuse to open the book.
    """
    present = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
    for name, decl in (
        ("stop_txid", "TEXT"),
        ("stop_price", "TEXT"),
        ("stop_qty", "TEXT"),
        ("stop_status", "TEXT"),
    ):
        if name not in present:
            conn.execute(f"ALTER TABLE positions ADD COLUMN {name} {decl}")


def _position_from_row(row: sqlite3.Row) -> Position:
    opened = row["opened_at"]
    price = _row_text(row, "stop_price")
    qty = _row_text(row, "stop_qty")
    return Position(
        instrument=_instrument_from_row(row),
        qty=Decimal(row["qty"]),
        avg_price=Decimal(row["avg_price"]),
        mode=Mode(row["mode"]),
        strategy_id=row["strategy_id"],
        structure_id=row["structure_id"],
        opened_at=dt.datetime.fromisoformat(opened) if opened else None,
        stop_txid=_row_text(row, "stop_txid"),
        stop_price=Decimal(price) if price is not None else None,
        stop_qty=Decimal(qty) if qty is not None else None,
        stop_status=_row_text(row, "stop_status"),
    )


def _fill_from_row(row: sqlite3.Row) -> Fill:
    return Fill(
        order_id=row["order_id"],
        instrument=_instrument_from_row(row),
        side=Side(row["side"]),
        qty=Decimal(row["qty"]),
        price=Decimal(row["price"]),
        fees=Decimal(row["fees"]),
        at=dt.datetime.fromisoformat(row["at_utc"]),
        mode=Mode(row["mode"]),
    )


def _order_from_row(row: sqlite3.Row) -> OrderRef:
    intent = Intent(
        instrument=_instrument_from_row(row),
        side=Side(row["side"]),
        qty=Decimal(row["qty"]),
        order_type=OrderType(row["order_type"]),
        strategy_id=row["strategy_id"],
        reason=row["reason"],
        limit_price=Decimal(row["limit_price"]) if row["limit_price"] is not None else None,
        structure_id=row["structure_id"],
        idempotency_key=row["idempotency_key"],
    )
    return OrderRef(
        order_id=row["order_id"],
        status=OrderStatus(row["status"]),
        intent=intent,
        mode=Mode(row["mode"]),
        filled_qty=Decimal(row["filled_qty"]),
        avg_price=Decimal(row["avg_price"]) if row["avg_price"] is not None else None,
        fills=tuple(_fill_from_mapping(o) for o in json.loads(row["fills_json"])),
        error=row["error"],
    )


# The engine wires this by protocol, not by name; the alias keeps call sites
# readable when there is only ever one implementation in play.
PositionStore = SQLitePositionStore
