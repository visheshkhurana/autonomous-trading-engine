"""
The read-only face of the engine, plus the two operator levers that can only
ever make it safer.

Three properties shape everything in this file.

1. THIS PROCESS PLACES NO ORDERS. It holds no gate, constructs no adapter call
   that reaches a venue, and never calls `RiskCircuit.advance()` (the circuit's
   only writer, which belongs to the order path). `/api/control/flatten` does
   not square anything off — it RECORDS A REQUEST that the engine picks up and
   executes through the gate on its next tick. That is why the flatten response
   says "pending" and why `/api/state` exposes whether the engine has
   acknowledged it: an operator who sees 200 OK must not conclude the book is
   closed.

2. AUTH IS STRUCTURAL, NOT PER-ROUTE. A raw ASGI middleware sits in front of
   the router with an explicit exempt list (`/health` only). Every other path —
   including paths that do not exist, and any route a future author adds —
   requires the bearer token. There is no decorator to forget. The app REFUSES
   TO START without `ENGINE_API_TOKEN`, and refuses a CORS config containing
   "*". The previous system had three unauthenticated order endpoints and cron
   guards that failed OPEN on a missing secret; nothing here falls back to open
   access.

3. UNKNOWN IS NOT ZERO. Every number in a response is either a value the API
   actually computed or `null` with a sibling reason. `equity: null` means the
   adapter could not establish it; it never becomes 0, because 0 reads as a
   100% drawdown. An empty `positions` list from a successful ledger read is an
   answer; a ledger read that FAILED returns 503, never 200 with `[]`. An empty
   strategy list from a registry nothing ever registered with returns 503 too —
   "[] read as no bots when ten existed" is on the list of things that cost
   this system real money.

Decimals cross the wire as STRINGS. JSON numbers are IEEE doubles and a price
that round-trips through one is no longer the price that was filled.

CROSS-PROCESS CAVEAT, stated plainly: `EventBus` and `StrategyRegistry` are
in-memory and therefore only see what is published in THIS process. If the
engine runs as a separate process, `/api/events` will carry the control events
this API raises and nothing else, and `/api/strategies` will 503. The durable
surfaces — positions, orders, P&L, risk state, control state — go through
SQLite and are correct across processes. See the module report for the fix.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import itertools
import json
import os
import sqlite3
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import (
    Any, AsyncIterator, Callable, Mapping, Optional, Protocol, Sequence,
    runtime_checkable,
)

from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.profiles import AssetClassProfile, profile_for
from core.protocols import BrokerAdapter, RiskCircuitProto
from core.types import (
    Approval, AssetClass, Fill, Instrument, Mode, OrderRef, OrderStatus,
    Position,
)

UTC = dt.timezone.utc

__all__ = [
    "ApiSettings",
    "ControlState",
    "ControlStateStore",
    "ControlUnavailable",
    "EngineHandle",
    "Event",
    "EventBus",
    "GATE_REJECTION_PREFIX",
    "LedgerReader",
    "ProposalRecord",
    "SqliteControlStore",
    "StrategyRegistry",
    "StrategyStatus",
    "create_app",
    "day_start_instant",
    "gate_rejection_ref",
]

# Scope key for engine-wide (not per-asset-class) control rows.
ENGINE_SCOPE = "__engine__"

# Order ids minted for gate refusals. A gate rejection never reaches a broker,
# so it has no broker order id; the ledger still needs one to key the row.
GATE_REJECTION_PREFIX = "gate-reject:"

_MIN_TOKEN_LEN = 16
_ORDERS_DEFAULT_LIMIT = 50
_ORDERS_MAX_LIMIT = 500
_MAX_REASON_LEN = 200
# Reconnect delay the browser honours after a dropped stream.
_SSE_RETRY_MS = 3000


# ==========================================================================
# settings
# ==========================================================================

@dataclass(frozen=True)
class ApiSettings:
    """Configuration. Constructed from the environment by `from_env`, which
    raises rather than defaulting anything that guards access."""

    token: str
    cors_origins: tuple[str, ...] = ()
    # How often the SSE generator wakes to drain its queue. Small enough to feel
    # live, large enough that an idle client costs nothing.
    sse_poll_interval_s: float = 0.1
    sse_heartbeat_s: float = 15.0
    # Every stream closes itself after this long and the browser reconnects
    # with Last-Event-ID. Bounded on purpose: an SSE connection that lives
    # forever survives a deploy as a generator holding a subscription, and
    # proxies kill long-lived streams anyway — better to end them deliberately,
    # where the reconnect is a resume the backlog can fill, than to have one
    # die at a moment nobody chose.
    sse_max_stream_s: float = 900.0
    # A heartbeat older than this means the engine process is not ticking.
    engine_stale_after_s: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token.strip():
            raise ValueError("ApiSettings.token must be a non-empty string")
        if len(self.token) < _MIN_TOKEN_LEN:
            # A four-character token is not authentication, it is a speed bump
            # with a 401 attached. Refuse at startup where an operator will see
            # it, not at 3am in an access log.
            raise ValueError(
                f"ENGINE_API_TOKEN must be at least {_MIN_TOKEN_LEN} characters; "
                f"got {len(self.token)}"
            )
        for origin in self.cors_origins:
            if origin.strip() == "*":
                raise ValueError(
                    "CORS origin '*' is not allowed: this API exposes the live "
                    "book and accepts control actions. List the exact origins."
                )
            if not origin.strip():
                raise ValueError("CORS origin entries must be non-empty")
        for name in ("sse_poll_interval_s", "sse_heartbeat_s",
                     "sse_max_stream_s", "engine_stale_after_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None,
                 **overrides: Any) -> "ApiSettings":
        """Read `ENGINE_API_TOKEN` and `ENGINE_API_CORS_ORIGINS`.

        A missing token RAISES. There is deliberately no "development mode"
        branch here: the one thing a missing secret must never do is grant
        access, and a fallback that only triggers in dev is a fallback that
        eventually ships.
        """
        source = os.environ if env is None else env
        token = source.get("ENGINE_API_TOKEN")
        if token is None or not token.strip():
            raise RuntimeError(
                "ENGINE_API_TOKEN is not set. The engine API refuses to start "
                "without a bearer token; it exposes the live book and accepts "
                "control actions, and there is no unauthenticated mode."
            )
        raw_origins = source.get("ENGINE_API_CORS_ORIGINS", "")
        origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        return cls(token=token, cors_origins=origins, **overrides)


# ==========================================================================
# events
# ==========================================================================

_JSON_SCALARS = (str, int, bool, type(None))


@dataclass(frozen=True)
class Event:
    """One thing that happened, ready to serialise.

    `data` values must already be JSON scalars — Decimals stringified, datetimes
    isoformatted. Validated here so an unserialisable event fails at the
    publisher (in a test, at import-adjacent code) rather than inside the SSE
    generator, where it would silently kill a live operator's stream.
    """

    kind: str
    at: dt.datetime
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("Event.kind must be set")
        if self.at.tzinfo is None or self.at.utcoffset() is None:
            raise ValueError("Event.at must be timezone-aware")
        for key, value in self.data.items():
            if not isinstance(key, str):
                raise TypeError(f"Event.data keys must be str, got {key!r}")
            if isinstance(value, bool) or isinstance(value, _JSON_SCALARS):
                continue
            raise TypeError(
                f"Event.data[{key!r}] is {type(value).__name__}; events carry "
                "JSON scalars only. Stringify Decimals and datetimes at the "
                "publisher so the stream cannot fail mid-flight."
            )

    def payload(self) -> str:
        return json.dumps(
            {"kind": self.kind, "at": self.at.astimezone(UTC).isoformat(),
             **dict(self.data)},
            sort_keys=True, separators=(",", ":"),
        )


@dataclass(frozen=True)
class _Delivery:
    """An event with its stream sequence, or a gap marker standing in for
    events a subscriber missed."""
    seq: int
    event: Optional[Event]
    gap: int = 0


class _Subscription:
    """One SSE client's view of the bus."""

    __slots__ = ("_queue", "_lock", "_dropped", "_maxlen")

    def __init__(self, maxlen: int) -> None:
        self._queue: deque[_Delivery] = deque()
        self._lock = threading.Lock()
        self._dropped = 0
        self._maxlen = maxlen

    def _offer(self, delivery: _Delivery) -> None:
        with self._lock:
            self._queue.append(delivery)
            while len(self._queue) > self._maxlen:
                # A slow or wedged UI client must never back-pressure the
                # engine. Drop the OLDEST and count it; the count is surfaced to
                # that client as a gap marker so it knows to refetch rather than
                # believing it has seen everything.
                self._queue.popleft()
                self._dropped += 1

    def drain(self) -> list[_Delivery]:
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
            dropped, self._dropped = self._dropped, 0
        if dropped and items:
            items.insert(0, _Delivery(seq=items[0].seq, event=None, gap=dropped))
        elif dropped:
            items.append(_Delivery(seq=-1, event=None, gap=dropped))
        return items


class EventBus:
    """Fan-out for live UI updates. Publishing never blocks and never raises.

    Bounded per-subscriber queues plus a bounded shared backlog for reconnects.
    The backlog is what lets a browser that dropped its connection tell the
    difference between "nothing happened while I was away" and "I missed
    things" — without it, a reconnecting UI silently resumes with a stale view.
    """

    def __init__(self, per_client_maxlen: int = 512,
                 backlog_maxlen: int = 512) -> None:
        if per_client_maxlen < 1 or backlog_maxlen < 1:
            raise ValueError("bus queue lengths must be >= 1")
        self._lock = threading.Lock()
        self._per_client_maxlen = per_client_maxlen
        self._subs: set[_Subscription] = set()
        self._backlog: deque[_Delivery] = deque(maxlen=backlog_maxlen)
        self._seq = itertools.count(1)
        self._published = 0

    def publish(self, event: Event) -> int:
        """Deliver to every subscriber. Returns the event's sequence number.

        Deliberately infallible from the caller's point of view: this is a
        display path, and a display path must not be able to break the order
        path. `Event.__post_init__` already rejected anything unserialisable, so
        there is nothing left here that can legitimately fail.
        """
        if not isinstance(event, Event):
            raise TypeError(f"publish expects Event, got {type(event).__name__}")
        with self._lock:
            seq = next(self._seq)
            delivery = _Delivery(seq=seq, event=event)
            self._backlog.append(delivery)
            self._published += 1
            subs = list(self._subs)
        for sub in subs:
            sub._offer(delivery)
        return seq

    def subscribe(self, after_seq: Optional[int] = None,
                  per_client_maxlen: Optional[int] = None) -> _Subscription:
        """Attach a client, optionally replaying everything after `after_seq`.

        If `after_seq` predates the retained backlog the subscription opens with
        a gap marker: the client is told it missed events rather than being
        handed a silently truncated history.
        """
        sub = _Subscription(per_client_maxlen or self._per_client_maxlen)
        with self._lock:
            if after_seq is not None:
                replay = [d for d in self._backlog if d.seq > after_seq]
                oldest = self._backlog[0].seq if self._backlog else None
                if oldest is not None and after_seq + 1 < oldest:
                    sub._offer(_Delivery(seq=oldest - 1, event=None,
                                         gap=oldest - after_seq - 1))
                for d in replay:
                    sub._offer(d)
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: _Subscription) -> None:
        with self._lock:
            self._subs.discard(sub)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    @property
    def published(self) -> int:
        with self._lock:
            return self._published


# ==========================================================================
# strategy registry
# ==========================================================================

@dataclass(frozen=True)
class ProposalRecord:
    """One Intent a strategy proposed, flattened for display."""
    symbol: str
    side: str
    qty: str
    order_type: str
    reason: str
    structure_id: Optional[str] = None


@dataclass(frozen=True)
class StrategyStatus:
    strategy_id: str
    asset_class: AssetClass
    enabled: bool
    # None means "has never run in this process" — which is NOT the same as
    # "ran and did nothing". A UI showing 0 errors and no proposals for a
    # strategy that never ticked is showing a healthy-looking dead strategy.
    last_tick_at: Optional[dt.datetime] = None
    last_proposal_at: Optional[dt.datetime] = None
    last_proposals: tuple[ProposalRecord, ...] = ()
    last_reason: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[dt.datetime] = None
    disabled_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must be set")
        if not self.enabled and not (self.disabled_reason or "").strip():
            raise ValueError(
                f"{self.strategy_id}: a disabled strategy must carry a reason; "
                "an unexplained disabled bot is indistinguishable from a bug"
            )


class StrategyRegistry:
    """What the runner knows about its strategies, for display only.

    `registered` is the load-bearing field. An empty registry is reported as
    UNAVAILABLE, not as "no strategies": the failure this system was wrecked by
    was a bot list that came back `[]` while ten bots were running, one of them
    live.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, StrategyStatus] = {}

    @property
    def registered(self) -> bool:
        with self._lock:
            return bool(self._rows)

    def register(self, strategy_id: str, asset_class: AssetClass) -> None:
        with self._lock:
            if strategy_id in self._rows:
                return
            self._rows[strategy_id] = StrategyStatus(
                strategy_id=strategy_id, asset_class=asset_class, enabled=True)

    def _require(self, strategy_id: str) -> StrategyStatus:
        row = self._rows.get(strategy_id)
        if row is None:
            raise KeyError(
                f"{strategy_id} is not registered; register() it at startup so "
                "an unknown id cannot be invented by a status update"
            )
        return row

    def record_tick(self, strategy_id: str, at: dt.datetime,
                    proposals: Sequence[ProposalRecord] = (),
                    reason: Optional[str] = None) -> None:
        with self._lock:
            row = self._require(strategy_id)
            self._rows[strategy_id] = StrategyStatus(
                strategy_id=row.strategy_id,
                asset_class=row.asset_class,
                enabled=row.enabled,
                last_tick_at=at,
                last_proposal_at=at if proposals else row.last_proposal_at,
                last_proposals=tuple(proposals) if proposals else row.last_proposals,
                last_reason=reason if reason is not None else row.last_reason,
                error_count=row.error_count,
                last_error=row.last_error,
                last_error_at=row.last_error_at,
                disabled_reason=row.disabled_reason,
            )

    def record_error(self, strategy_id: str, at: dt.datetime, error: str,
                     disable: bool = False) -> None:
        with self._lock:
            row = self._require(strategy_id)
            self._rows[strategy_id] = StrategyStatus(
                strategy_id=row.strategy_id,
                asset_class=row.asset_class,
                enabled=row.enabled and not disable,
                last_tick_at=at,
                last_proposal_at=row.last_proposal_at,
                last_proposals=row.last_proposals,
                last_reason=row.last_reason,
                error_count=row.error_count + 1,
                last_error=error,
                last_error_at=at,
                disabled_reason=error if disable else row.disabled_reason,
            )

    def disable(self, strategy_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("disabling a strategy requires a reason")
        with self._lock:
            row = self._require(strategy_id)
            self._rows[strategy_id] = StrategyStatus(
                strategy_id=row.strategy_id,
                asset_class=row.asset_class,
                enabled=False,
                last_tick_at=row.last_tick_at,
                last_proposal_at=row.last_proposal_at,
                last_proposals=row.last_proposals,
                last_reason=row.last_reason,
                error_count=row.error_count,
                last_error=row.last_error,
                last_error_at=row.last_error_at,
                disabled_reason=reason,
            )

    def snapshot(self) -> tuple[StrategyStatus, ...]:
        with self._lock:
            return tuple(self._rows[k] for k in sorted(self._rows))


# ==========================================================================
# control state
# ==========================================================================

class ControlUnavailable(RuntimeError):
    """Raised by a ControlStateStore that cannot read or write.

    Never returned as a value: `paused=False` is a legitimate state and cannot
    double as an error signal. An API that reports "not paused" because it could
    not read the row is an API that tells an operator their pause took effect
    when it did not.
    """


@dataclass(frozen=True)
class ControlState:
    """Operator intent for one asset class, as persisted.

    Note what this does NOT contain: any notion of mode, arming, or limits.
    Nothing an operator can POST widens what the engine may do — `paused` and
    `flatten_requested` only ever subtract.
    """

    paused: bool = False
    paused_at: Optional[dt.datetime] = None
    paused_reason: Optional[str] = None
    # Free text supplied by the caller. There is one shared bearer token, so
    # this is a note in an audit trail, NOT an authenticated identity. Named to
    # make that impossible to misread.
    paused_actor_claimed: Optional[str] = None
    flatten_request_id: Optional[str] = None
    flatten_requested_at: Optional[dt.datetime] = None
    flatten_reason: Optional[str] = None
    flatten_actor_claimed: Optional[str] = None
    # Written by the ENGINE when it has taken the request. Until then the
    # request is a wish, and the API must say so.
    flatten_acked_id: Optional[str] = None
    flatten_acked_at: Optional[dt.datetime] = None

    @property
    def flatten_pending(self) -> bool:
        return (
            self.flatten_request_id is not None
            and self.flatten_acked_id != self.flatten_request_id
        )

    def as_row(self) -> dict[str, str]:
        row: dict[str, str] = {"paused": "1" if self.paused else "0"}
        for key, value in (
            ("paused_at", self.paused_at),
            ("flatten_requested_at", self.flatten_requested_at),
            ("flatten_acked_at", self.flatten_acked_at),
        ):
            if value is not None:
                row[key] = value.astimezone(UTC).isoformat()
        for key, value in (
            ("paused_reason", self.paused_reason),
            ("paused_actor_claimed", self.paused_actor_claimed),
            ("flatten_request_id", self.flatten_request_id),
            ("flatten_reason", self.flatten_reason),
            ("flatten_actor_claimed", self.flatten_actor_claimed),
            ("flatten_acked_id", self.flatten_acked_id),
        ):
            if value is not None:
                row[key] = value
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "ControlState":
        """Raises on a corrupt row. The caller reports that as unavailable
        rather than falling back to an all-clear default — a corrupt row read as
        `paused=False` un-pauses a paused engine."""
        raw_paused = row.get("paused", "0")
        if raw_paused not in {"0", "1"}:
            raise ValueError(f"control row has non-boolean paused={raw_paused!r}")

        def ts(key: str) -> Optional[dt.datetime]:
            raw = row.get(key)
            if raw is None:
                return None
            parsed = dt.datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                raise ValueError(f"control row {key} is not timezone-aware: {raw!r}")
            return parsed

        state = cls(
            paused=raw_paused == "1",
            paused_at=ts("paused_at"),
            paused_reason=row.get("paused_reason"),
            paused_actor_claimed=row.get("paused_actor_claimed"),
            flatten_request_id=row.get("flatten_request_id"),
            flatten_requested_at=ts("flatten_requested_at"),
            flatten_reason=row.get("flatten_reason"),
            flatten_actor_claimed=row.get("flatten_actor_claimed"),
            flatten_acked_id=row.get("flatten_acked_id"),
            flatten_acked_at=ts("flatten_acked_at"),
        )
        if state.paused and state.paused_at is None:
            raise ValueError("control row is paused with no paused_at timestamp")
        return state


@runtime_checkable
class ControlStateStore(Protocol):
    """Durable operator intent, shared between the API and the engine.

    It MUST be durable and cross-process. An in-memory implementation satisfies
    this Protocol and would be a lie: the operator pauses through the API,
    receives 200 OK, and the engine — a different process — keeps trading. Only
    a store both processes can read counts.

    `read` returns None ONLY when the scope has genuinely never been written,
    and RAISES when it cannot read. Those two must stay distinguishable.
    `mutate` must apply `fn` under a write lock so two concurrent pauses cannot
    lose one another.
    """

    def read(self, scope: str) -> Optional[Mapping[str, str]]: ...
    def write(self, scope: str, row: Mapping[str, str]) -> None: ...
    def mutate(self, scope: str,
               fn: Callable[[Optional[Mapping[str, str]]], Mapping[str, str]],
               ) -> Mapping[str, str]: ...


class SqliteControlStore:
    """SQLite-backed ControlStateStore, in its own table.

    Shares the ledger file by default so there is one thing to back up and one
    thing to mount, but keeps a separate table: `store.py` owns `positions`,
    `orders`, `fills` and `risk_state`, and nothing here touches them. The
    ledger's "single writer" rule is about positions, not about the file.

    `mutate` runs BEGIN IMMEDIATE around read-modify-write, so two API workers
    pausing at once cannot lose an update. That matters more than it looks: the
    lost update here would be a pause.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS engine_control (
        scope       TEXT PRIMARY KEY,
        state_json  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );
    """

    def __init__(self, path: str,
                 now_fn: Optional[Callable[[], dt.datetime]] = None) -> None:
        self._path = path
        self._now = now_fn or (lambda: dt.datetime.now(UTC))
        self._lock = threading.RLock()
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # The engine and the API are separate processes on one file; without
        # this the second writer gets "database is locked" and a pause is lost.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(self._SCHEMA)

    @property
    def path(self) -> str:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def read(self, scope: str) -> Optional[Mapping[str, str]]:
        with self._lock:
            rows = list(self._conn.execute(
                "SELECT state_json FROM engine_control WHERE scope=?", (scope,)))
        if not rows:
            return None
        parsed = json.loads(rows[0]["state_json"])
        if not isinstance(parsed, dict):
            raise ValueError(f"control row for {scope!r} is corrupt: not an object")
        return {str(k): str(v) for k, v in parsed.items()}

    def write(self, scope: str, row: Mapping[str, str]) -> None:
        payload = self._encode(row)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert(scope, payload)
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def mutate(self, scope: str,
               fn: Callable[[Optional[Mapping[str, str]]], Mapping[str, str]],
               ) -> Mapping[str, str]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = list(self._conn.execute(
                    "SELECT state_json FROM engine_control WHERE scope=?", (scope,)))
                current: Optional[Mapping[str, str]] = None
                if rows:
                    parsed = json.loads(rows[0]["state_json"])
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"control row for {scope!r} is corrupt: not an object")
                    current = {str(k): str(v) for k, v in parsed.items()}
                updated = fn(current)
                self._upsert(scope, self._encode(updated))
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")
        return dict(updated)

    @staticmethod
    def _encode(row: Mapping[str, str]) -> str:
        for key, value in row.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    f"control rows are str->str; got {key!r}->{type(value).__name__}")
        return json.dumps(dict(row), sort_keys=True, separators=(",", ":"))

    def _upsert(self, scope: str, payload: str) -> None:
        self._conn.execute(
            """
            INSERT INTO engine_control (scope, state_json, updated_at)
            VALUES (?,?,?)
            ON CONFLICT (scope) DO UPDATE SET
                state_json=excluded.state_json,
                updated_at=excluded.updated_at
            """,
            (scope, payload, self._now().astimezone(UTC).isoformat()),
        )


# ==========================================================================
# the handle the API reads through
# ==========================================================================

@runtime_checkable
class LedgerReader(Protocol):
    """The read half of PositionStoreProto plus the order log.

    Narrower than PositionStoreProto on purpose: this process must not be
    handed `apply()`. `SQLitePositionStore` satisfies this.
    """

    def positions(self, mode: Mode,
                  asset_class: Optional[AssetClass] = None) -> Sequence[Position]: ...
    def orders(self, mode: Mode, asset_class: Optional[AssetClass] = None,
               since_iso: Optional[str] = None,
               limit: Optional[int] = None) -> Sequence[OrderRef]: ...
    def realized_pnl(self, mode: Mode, asset_class: AssetClass,
                     since_iso: str) -> Decimal: ...


@dataclass(frozen=True)
class EngineHandle:
    """Everything the API is allowed to see. Validated at construction so a
    misconfiguration is a startup failure, not a wrong number on a dashboard."""

    mode: Mode
    asset_classes: tuple[AssetClass, ...]
    ledger: LedgerReader
    circuit: RiskCircuitProto
    control: ControlStateStore
    strategies: StrategyRegistry
    bus: EventBus
    # Missing entries are reported as unresolved by /ready rather than assumed.
    adapters: Mapping[AssetClass, BrokerAdapter] = field(default_factory=dict)
    # Whether LIVE is armed per class, as the GATE sees it. Required for every
    # configured class: displaying "not armed" for a class that is in fact armed
    # would let an operator believe they are watching paper.
    armed: Mapping[AssetClass, bool] = field(default_factory=dict)
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(UTC)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, Mode):
            raise TypeError("EngineHandle.mode must be a Mode")
        if not self.asset_classes:
            raise ValueError(
                "EngineHandle.asset_classes must not be empty; an API that "
                "reports on nothing looks identical to a healthy idle engine")
        for asset_class in self.asset_classes:
            if not isinstance(asset_class, AssetClass):
                raise TypeError(f"not an AssetClass: {asset_class!r}")
            if asset_class not in self.armed:
                raise ValueError(
                    f"armed[{asset_class.value}] is not stated. Pass the same "
                    "flags the OrderGate was constructed with; an absent flag "
                    "would render as 'paper' for a class that may be live."
                )
        circuit_mode = getattr(self.circuit, "mode", None)
        if circuit_mode is not None and circuit_mode is not self.mode:
            # Paper and live keep separate baselines. A live view reading a
            # paper circuit shows a drawdown computed from the wrong book.
            raise ValueError(
                f"circuit is bound to {circuit_mode} but the handle reports "
                f"{self.mode}; risk state would be read from the wrong book"
            )
        object.__setattr__(self, "asset_classes", tuple(self.asset_classes))
        object.__setattr__(self, "adapters", dict(self.adapters))
        object.__setattr__(self, "armed", dict(self.armed))

    def now(self) -> dt.datetime:
        moment = self.clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("EngineHandle.clock must return a tz-aware datetime")
        return moment


# ==========================================================================
# helpers shared with the runner
# ==========================================================================

def day_start_instant(profile: AssetClassProfile,
                      now: dt.datetime) -> dt.datetime:
    """The tz-aware UTC instant at which `profile.day_key(now)` began.

    Exists because `store.realized_pnl` requires a tz-aware INSTANT and raises
    on a naive one, while `day_key()` returns a calendar DATE in the class's own
    timezone. Passing the date straight through is a 5.5-hour error in the
    equity daily-loss window.

    The boundary is found by bisecting `day_key` itself rather than by looking
    up a timezone per asset class. `profiles.py` is explicit that the asset-class
    distinction lives there and nowhere else; a `{EQUITY: IST, CRYPTO: UTC}`
    table here would be a second copy of that knowledge, free to drift the day
    a profile changes.
    """
    key = profile.day_key(now)
    lo = now - dt.timedelta(days=2)
    if profile.day_key(lo) == key:
        raise ValueError(
            f"day_key {key!r} is unchanged 48h earlier; this profile's day_key "
            "is not day-granular and no daily window can be derived from it"
        )
    hi = now
    # Invariant: day_key(lo) != key and day_key(hi) == key.
    while (hi - lo) > dt.timedelta(minutes=1):
        mid = lo + (hi - lo) / 2
        if profile.day_key(mid) == key:
            hi = mid
        else:
            lo = mid
    boundary = hi.astimezone(UTC).replace(second=0, microsecond=0)
    if profile.day_key(boundary) != key:
        raise ValueError(
            f"day boundary for {key!r} is not minute-aligned; refusing to guess "
            "a P&L window that would be off by up to a minute of fills"
        )
    return boundary


def gate_rejection_ref(approval: Approval, at: dt.datetime) -> OrderRef:
    """Turn a gate refusal into a ledger row.

    A gate rejection never reaches a broker, so it has no broker order id and
    no OrderRef of its own — which is exactly why the previous system's refusals
    were invisible. `/api/orders` can only show "REJECTED with the gate's
    reason" if the RUNNER records one of these for every failed `check()`.
    Minted here rather than in the runner so the id shape and the reason text
    are identical everywhere and `origin` stays derivable.

    `at` disambiguates two refusals of the same decision: the ledger dedups on a
    hash of the row, so without it a re-refused key collapses into one entry and
    a strategy jamming against a cap looks like it tried once.
    """
    if approval.approved:
        raise ValueError("gate_rejection_ref is for refusals; this Approval passed")
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("gate_rejection_ref requires a tz-aware timestamp")
    key = approval.intent.idempotency_key or "unkeyed"
    stamp = at.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%f")
    return OrderRef(
        order_id=f"{GATE_REJECTION_PREFIX}{key}:{stamp}",
        status=OrderStatus.REJECTED,
        intent=approval.intent,
        mode=approval.mode,
        filled_qty=Decimal("0"),
        avg_price=None,
        fills=(),
        error=approval.reason,
    )


# ==========================================================================
# serialisation — Decimals become strings, never JSON numbers
# ==========================================================================

def _dec(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else str(value)


def _ts(value: Optional[dt.datetime]) -> Optional[str]:
    return None if value is None else value.astimezone(UTC).isoformat()


def _instrument_json(inst: Instrument) -> dict[str, Any]:
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


def _position_json(pos: Position) -> dict[str, Any]:
    return {
        "instrument": _instrument_json(pos.instrument),
        "symbol": pos.instrument.symbol,
        "qty": str(pos.qty),                 # signed; negative is short
        "avg_price": str(pos.avg_price),
        "mode": pos.mode.value,
        "strategy_id": pos.strategy_id,
        "structure_id": pos.structure_id,
        "opened_at": _ts(pos.opened_at),
    }


def _fill_json(fill: Fill) -> dict[str, Any]:
    return {
        "order_id": fill.order_id,
        "symbol": fill.instrument.symbol,
        "side": fill.side.value,
        "qty": str(fill.qty),
        "price": str(fill.price),
        "fees": str(fill.fees),
        "at": _ts(fill.at),
        "mode": fill.mode.value,
    }


def _order_json(ref: OrderRef) -> dict[str, Any]:
    intent = ref.intent
    return {
        "order_id": ref.order_id,
        # "gate" rows never reached a venue. Without this a UI cannot tell a
        # broker rejection from an order we refused to send, and those two have
        # completely different remedies.
        "origin": "gate" if ref.order_id.startswith(GATE_REJECTION_PREFIX) else "broker",
        "status": ref.status.value,
        "mode": ref.mode.value,
        "symbol": intent.instrument.symbol,
        "asset_class": intent.instrument.asset_class.value,
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "qty": str(intent.qty),
        "limit_price": _dec(intent.limit_price),
        "filled_qty": str(ref.filled_qty),
        "avg_price": _dec(ref.avg_price),
        "strategy_id": intent.strategy_id,
        "structure_id": intent.structure_id,
        "idempotency_key": intent.idempotency_key,
        "reason": intent.reason,
        # For a gate row this carries the failed CheckResults verbatim.
        "error": ref.error,
        "fills": [_fill_json(f) for f in ref.fills],
    }


def _decision_json(decision: Any) -> dict[str, Any]:
    """Serialise a CircuitDecision structurally, the way the gate reads it.

    Field names are translated: `drawdown_pct` on the dataclass is a FRACTION
    (0.05 means 5%), and shipping that to a UI under a "_pct" name is how a
    dashboard ends up displaying "0.05%" for a 5% drawdown.
    """
    action = getattr(decision, "action", None)
    return {
        "action": getattr(action, "value", action),
        "entries_allowed": bool(getattr(decision, "entries_allowed", False)),
        "must_flatten": bool(getattr(decision, "must_flatten", False)),
        "drawdown_fraction": _dec(getattr(decision, "drawdown_pct", None)),
        "daily_loss_fraction": _dec(getattr(decision, "daily_loss_pct", None)),
        "equity_known": bool(getattr(decision, "equity_known", False)),
        "state_known": bool(getattr(decision, "state_known", False)),
        "size_multiplier": _dec(getattr(decision, "size_multiplier", None)),
        "reason": getattr(decision, "reason", ""),
    }


def _limits_json(limits: Any) -> dict[str, Any]:
    return {
        name: _dec(getattr(limits, name))
        for name in ("reduce_drawdown", "pause_drawdown", "flatten_drawdown",
                     "reduce_daily_loss", "pause_daily_loss",
                     "flatten_daily_loss", "reduce_size_multiplier")
        if hasattr(limits, name)
    }


def _control_json(state: ControlState) -> dict[str, Any]:
    return {
        "paused": state.paused,
        "paused_at": _ts(state.paused_at),
        "paused_reason": state.paused_reason,
        "paused_actor_claimed": state.paused_actor_claimed,
        "flatten_request_id": state.flatten_request_id,
        "flatten_requested_at": _ts(state.flatten_requested_at),
        "flatten_reason": state.flatten_reason,
        "flatten_actor_claimed": state.flatten_actor_claimed,
        "flatten_acked_id": state.flatten_acked_id,
        "flatten_acked_at": _ts(state.flatten_acked_at),
        "flatten_pending": state.flatten_pending,
    }


def _strategy_json(status: StrategyStatus) -> dict[str, Any]:
    return {
        "strategy_id": status.strategy_id,
        "asset_class": status.asset_class.value,
        "enabled": status.enabled,
        "disabled_reason": status.disabled_reason,
        "last_tick_at": _ts(status.last_tick_at),
        "last_proposal_at": _ts(status.last_proposal_at),
        "last_proposals": [
            {"symbol": p.symbol, "side": p.side, "qty": p.qty,
             "order_type": p.order_type, "reason": p.reason,
             "structure_id": p.structure_id}
            for p in status.last_proposals
        ],
        "last_reason": status.last_reason,
        "error_count": status.error_count,
        "last_error": status.last_error,
        "last_error_at": _ts(status.last_error_at),
        # Explicit rather than inferred from a null timestamp: "has not run" and
        # "ran and proposed nothing" must not look the same in a UI.
        "has_ticked": status.last_tick_at is not None,
    }


# ==========================================================================
# auth
# ==========================================================================

_EXEMPT_PATHS = frozenset({"/health"})


class BearerAuthMiddleware:
    """Deny by default, for every path.

    A raw ASGI middleware rather than a dependency or a decorator: a dependency
    protects the routes someone remembered to annotate, and this protects the
    ones they did not — including routes that do not exist yet, and paths that
    404. There is no way to add an unauthenticated endpoint to this app by
    accident, which is the only property worth having here.

    BaseHTTPMiddleware is avoided deliberately: it wraps responses in an extra
    task group, which interferes with the SSE stream this app has to keep open.
    """

    def __init__(self, app: Any, token: str,
                 exempt_paths: frozenset[str] = _EXEMPT_PATHS) -> None:
        if not token or len(token) < _MIN_TOKEN_LEN:
            raise ValueError("BearerAuthMiddleware requires a real token")
        self._app = app
        self._token = token
        self._exempt = exempt_paths

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        kind = scope.get("type")
        if kind == "lifespan":
            await self._app(scope, receive, send)
            return
        if kind != "http":
            # This app serves no websockets. Refuse rather than forward an
            # unauthenticated connection to a router that may grow one.
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope.get("path") in self._exempt:
            await self._app(scope, receive, send)
            return
        if not self._authorized(scope):
            await self._unauthorized(send)
            return
        await self._app(scope, receive, send)

    def _authorized(self, scope: Any) -> bool:
        for raw_name, raw_value in scope.get("headers", ()):
            if raw_name.lower() != b"authorization":
                continue
            try:
                value = raw_value.decode("latin-1")
            except UnicodeDecodeError:
                return False
            scheme, _, presented = value.partition(" ")
            if scheme.lower() != "bearer":
                return False
            # Constant time: a token comparison that short-circuits on the first
            # differing byte is a token oracle.
            return hmac.compare_digest(presented.strip(), self._token)
        return False

    @staticmethod
    async def _unauthorized(send: Any) -> None:
        body = json.dumps({"detail": "missing or invalid bearer token"}).encode()
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": body})


# ==========================================================================
# request bodies
# ==========================================================================

class _ControlBody(BaseModel):
    """Shared shape for the control endpoints.

    `extra="forbid"` is a safety rule, not tidiness: it means a body carrying
    `{"mode": "live"}` or `{"armed": true}` is a 422 rather than a silently
    ignored field that a future author might wire up. No control endpoint may
    arm live trading or change a mode, and the schema is where that is enforced
    first.
    """

    model_config = ConfigDict(extra="forbid")

    # Mandatory, mirroring Intent.reason: an operator action nobody can explain
    # later is not operable either.
    reason: str = Field(min_length=1, max_length=_MAX_REASON_LEN)
    actor: Optional[str] = Field(default=None, max_length=_MAX_REASON_LEN)
    asset_class: Optional[AssetClass] = None

    @field_validator("reason")
    @classmethod
    def _reason_is_substantive(cls, value: str) -> str:
        # min_length alone accepts "   ", which reads as a reason in the audit
        # trail and explains nothing.
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be blank")
        return stripped


# ==========================================================================
# the app
# ==========================================================================

def create_app(handle: EngineHandle,
               settings: Optional[ApiSettings] = None,
               env: Optional[Mapping[str, str]] = None) -> FastAPI:
    """Build the API. Raises if the token is missing or CORS is wildcarded.

    `settings=None` reads the environment, which is what production does. Tests
    pass an explicit ApiSettings.
    """
    if not isinstance(handle, EngineHandle):
        raise TypeError(f"create_app expects EngineHandle, got {type(handle).__name__}")
    resolved = settings if settings is not None else ApiSettings.from_env(env)

    app = FastAPI(
        title="automoney engine API",
        docs_url=None, redoc_url=None, openapi_url=None,
    )
    app.state.handle = handle
    app.state.settings = resolved

    # Added first so it ends up INNERMOST: Starlette builds the stack with the
    # most recently added middleware outermost, and CORS must see the request
    # before auth does — a preflight OPTIONS carries no Authorization header and
    # would otherwise 401 before CORS could answer it.
    app.add_middleware(BearerAuthMiddleware, token=resolved.token)
    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_origins),
            # Bearer tokens travel in a header, so cookies are never needed and
            # allowing credentials would only widen what a compromised origin
            # could do.
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["authorization", "content-type", "last-event-id"],
        )

    _register_routes(app, handle, resolved)
    return app


def _problem(scope: str, detail: str) -> dict[str, str]:
    return {"scope": scope, "detail": detail}


def _read_control(handle: EngineHandle, scope: str) -> ControlState:
    """Current control state, or raise. Never a default-on-failure.

    A missing row is a real answer (nothing has ever been paused). A read that
    FAILED raises, and the caller turns that into a 503 — reporting "not paused"
    because the row could not be read is how a paused engine appears live.
    """
    try:
        row = handle.control.read(scope)
    except Exception as exc:
        raise ControlUnavailable(f"control state for {scope!r} unreadable: {exc!r}") from exc
    if row is None:
        return ControlState()
    try:
        return ControlState.from_row(row)
    except Exception as exc:
        raise ControlUnavailable(f"control state for {scope!r} is corrupt: {exc!r}") from exc


def _adapter_equity(handle: EngineHandle,
                    asset_class: AssetClass) -> tuple[Optional[Decimal], Optional[str]]:
    """(equity, detail). `detail` is non-None whenever equity is None.

    Never returns 0 for a failure. `BrokerAdapter.equity()` is contracted to
    return None when it cannot establish the value, and a raise is treated the
    same way — with the exception text carried through so the reason is visible
    rather than inferred.
    """
    adapter = handle.adapters.get(asset_class)
    if adapter is None:
        return None, f"no adapter resolved for {asset_class.value}"
    try:
        value = adapter.equity()
    except Exception as exc:
        return None, f"adapter.equity() raised: {exc!r}"
    if value is None:
        return None, "adapter could not establish equity"
    if not isinstance(value, Decimal):
        # A float here would silently become a JSON number and drift the
        # denominator every drawdown fraction is computed against.
        return None, (f"adapter returned {type(value).__name__} for equity; "
                      "Decimal is required")
    return value, None


def _register_routes(app: FastAPI, handle: EngineHandle,
                     settings: ApiSettings) -> None:

    # ------------------------------------------------------------- liveness

    @app.get("/health")
    def health() -> JSONResponse:
        """The one unauthenticated route. Says nothing about the book, the
        account or whether trading is armed — only that this process is up."""
        return JSONResponse({"status": "ok"})

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Ready means: the ledger answers, and every configured asset class has
        an adapter.

        Deliberately does NOT call `adapter.equity()` or `quote()`. A readiness
        probe runs on a timer; wiring it to a venue turns a Kubernetes restart
        loop into a rate-limit ban, and a broker blip into a rolling restart of
        a process holding positions.
        """
        problems: list[dict[str, str]] = []
        classes: dict[str, Any] = {}
        for asset_class in handle.asset_classes:
            adapter = handle.adapters.get(asset_class)
            if adapter is None:
                problems.append(_problem(asset_class.value, "no adapter resolved"))
                classes[asset_class.value] = {"resolved": False, "adapter": None,
                                              "adapter_mode": None,
                                              "armed_live": handle.armed[asset_class]}
                continue
            adapter_mode = getattr(adapter, "mode", None)
            entry = {
                "resolved": True,
                "adapter": type(adapter).__name__,
                "adapter_mode": getattr(adapter_mode, "value", None),
                "armed_live": handle.armed[asset_class],
            }
            if adapter_mode is Mode.LIVE and not handle.armed[asset_class]:
                # Not fatal — the gate coerces to paper — but an operator should
                # never have to deduce this from two separate screens.
                problems.append(_problem(
                    asset_class.value,
                    "a LIVE adapter is wired for a class that is not armed; the "
                    "gate will coerce its orders to paper"))
            classes[asset_class.value] = entry

        try:
            handle.ledger.positions(handle.mode)
        except Exception as exc:
            problems.append(_problem("ledger", f"positions() raised: {exc!r}"))
        try:
            _read_control(handle, ENGINE_SCOPE)
        except ControlUnavailable as exc:
            problems.append(_problem("control", str(exc)))

        payload = {
            "ready": not problems,
            "mode": handle.mode.value,
            "classes": classes,
            "problems": problems,
        }
        return JSONResponse(payload, status_code=200 if not problems else 503)

    # -------------------------------------------------------------- snapshot

    @app.get("/api/state")
    def state() -> JSONResponse:
        now = handle.now()
        problems: list[dict[str, str]] = []
        classes: dict[str, Any] = {}

        try:
            engine_row = handle.control.read(ENGINE_SCOPE)
        except Exception as exc:
            return JSONResponse(
                {"detail": f"control state unreadable: {exc!r}"}, status_code=503)
        engine_block = _engine_block(engine_row, now, settings)
        if engine_block["status"] != "alive":
            problems.append(_problem("engine", engine_block["detail"]))

        total_positions = 0
        for asset_class in handle.asset_classes:
            profile = profile_for(asset_class)
            block: dict[str, Any] = {
                "base_currency": profile.base_currency,
                "market_open": profile.is_open(now),
                "must_flatten_intraday": profile.must_flatten_intraday,
                "armed_live": handle.armed[asset_class],
            }

            try:
                positions = handle.ledger.positions(handle.mode, asset_class)
            except Exception as exc:
                # The ledger is the authority on the book. If it will not
                # answer, the whole snapshot is untrustworthy — a 200 with an
                # empty book reads as flat, and flat is a decision input.
                return JSONResponse(
                    {"detail": f"ledger read failed for {asset_class.value}: {exc!r}"},
                    status_code=503)
            block["open_positions"] = len(positions)
            total_positions += len(positions)

            equity, equity_detail = _adapter_equity(handle, asset_class)
            block["equity"] = {"value": _dec(equity),
                               "known": equity is not None,
                               "detail": equity_detail}
            if equity is None:
                problems.append(_problem(asset_class.value,
                                         equity_detail or "equity unknown"))

            try:
                since = day_start_instant(profile, now)
                realized = handle.ledger.realized_pnl(
                    handle.mode, asset_class, since.isoformat())
                block["realized_pnl_today"] = {"value": str(realized),
                                               "since": since.isoformat(),
                                               "detail": None}
            except Exception as exc:
                # Distinguishable from a genuine zero: value stays null.
                block["realized_pnl_today"] = {"value": None, "since": None,
                                               "detail": f"{exc!r}"}
                problems.append(_problem(asset_class.value,
                                         f"realized P&L unavailable: {exc!r}"))

            block["risk"] = _risk_block(handle, asset_class, equity, problems)

            try:
                block["control"] = _control_json(_read_control(handle, asset_class.value))
            except ControlUnavailable as exc:
                return JSONResponse({"detail": str(exc)}, status_code=503)

            classes[asset_class.value] = block

        return JSONResponse({
            "at": now.astimezone(UTC).isoformat(),
            "mode": handle.mode.value,
            "engine": engine_block,
            # No cross-class total. Equity is INR for the equity book and USDT
            # for crypto; adding them produces a number with no unit, and a
            # unitless "total equity" is what the risk ladder would divide by.
            "classes": classes,
            "open_positions": total_positions,
            "degraded": bool(problems),
            "problems": problems,
        })

    # ------------------------------------------------------------- positions

    @app.get("/api/positions")
    def positions(mode: Optional[Mode] = Query(default=None),
                  asset_class: Optional[AssetClass] = Query(default=None),
                  ) -> JSONResponse:
        """Open positions from the ledger, always mode-filtered.

        Carries no mark price or unrealised P&L: producing those means a quote
        per position per poll, and a dashboard refresh is not a good reason to
        hammer a venue. The runner publishes marks on the event stream instead.
        """
        target = mode or handle.mode
        try:
            rows = handle.ledger.positions(target, asset_class)
        except Exception as exc:
            return JSONResponse({"detail": f"ledger read failed: {exc!r}"},
                                status_code=503)
        return JSONResponse({
            "mode": target.value,
            "asset_class": asset_class.value if asset_class else None,
            "count": len(rows),
            "positions": [_position_json(p) for p in rows],
        })

    # ---------------------------------------------------------------- orders

    @app.get("/api/orders")
    def orders(mode: Optional[Mode] = Query(default=None),
               asset_class: Optional[AssetClass] = Query(default=None),
               limit: int = Query(default=_ORDERS_DEFAULT_LIMIT, ge=1,
                                  le=_ORDERS_MAX_LIMIT),
               since: Optional[str] = Query(default=None),
               ) -> JSONResponse:
        """Most recent orders first, rejections included.

        A gate refusal shows up here only if the runner recorded one — see
        `gate_rejection_ref`. `origin` separates "the broker refused" from "we
        refused to send", and `error` carries the gate's CheckResult text
        verbatim.
        """
        target = mode or handle.mode
        if since is not None:
            # Validated here so a malformed client parameter is a 422 rather
            # than surfacing as a 503 that reads like the ledger is broken.
            # Aware-only, matching the store: a naive instant would silently
            # shift the window by the local offset.
            try:
                parsed = dt.datetime.fromisoformat(since)
            except ValueError:
                return JSONResponse(
                    {"detail": f"since must be an ISO-8601 instant, got {since!r}"},
                    status_code=422)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                return JSONResponse(
                    {"detail": f"since must carry a timezone offset, got {since!r}"},
                    status_code=422)
        try:
            rows = handle.ledger.orders(target, asset_class, since, limit)
        except Exception as exc:
            return JSONResponse({"detail": f"ledger read failed: {exc!r}"},
                                status_code=503)
        return JSONResponse({
            "mode": target.value,
            "asset_class": asset_class.value if asset_class else None,
            "count": len(rows),
            "limit": limit,
            "orders": [_order_json(o) for o in rows],
        })

    # ------------------------------------------------------------ strategies

    @app.get("/api/strategies")
    def strategies() -> JSONResponse:
        """Per-strategy status, or 503 when nothing has registered.

        The 503 is the point. An empty list with 200 is indistinguishable from
        "the engine runs in another process and this registry has never seen
        it", and that misreading — no bots listed while ten were running — is
        the specific failure this whole codebase is a reaction to.
        """
        if not handle.strategies.registered:
            return JSONResponse({
                "detail": "no strategy has registered with this process. The "
                          "registry is in-memory; if the engine runs elsewhere "
                          "this endpoint cannot see it. An empty list here would "
                          "mean 'no strategies', which is not what is known.",
                "registered": False,
                "strategies": None,
            }, status_code=503)
        rows = handle.strategies.snapshot()
        return JSONResponse({
            "registered": True,
            "count": len(rows),
            "enabled": sum(1 for r in rows if r.enabled),
            "never_ticked": [r.strategy_id for r in rows if r.last_tick_at is None],
            "strategies": [_strategy_json(r) for r in rows],
        })

    # ------------------------------------------------------------------ risk

    @app.get("/api/risk")
    def risk() -> JSONResponse:
        """Circuit state per asset class.

        Calls `read()` only. `advance()` is the circuit's single writer and
        belongs to the order path; calling it from a polling endpoint would move
        the high-water mark on a dashboard refresh, and its read-modify-write is
        not transactional across processes.
        """
        now = handle.now()
        problems: list[dict[str, str]] = []
        classes: dict[str, Any] = {}
        for asset_class in handle.asset_classes:
            equity, equity_detail = _adapter_equity(handle, asset_class)
            if equity is None:
                problems.append(_problem(asset_class.value,
                                         equity_detail or "equity unknown"))
            block = _risk_block(handle, asset_class, equity, problems)
            block["equity"] = {"value": _dec(equity), "known": equity is not None,
                               "detail": equity_detail}
            limits_for = getattr(handle.circuit, "limits_for", None)
            if callable(limits_for):
                try:
                    block["limits"] = _limits_json(limits_for(asset_class))
                except Exception as exc:
                    block["limits"] = None
                    problems.append(_problem(asset_class.value,
                                             f"limits unreadable: {exc!r}"))
            else:
                block["limits"] = None
            classes[asset_class.value] = block
        return JSONResponse({
            "at": now.astimezone(UTC).isoformat(),
            "mode": handle.mode.value,
            "classes": classes,
            "degraded": bool(problems),
            "problems": problems,
        })

    # ---------------------------------------------------------------- events

    @app.get("/api/events")
    async def events(request: Request) -> Response:
        """Server-Sent Events for the live UI.

        Reconnects may send `Last-Event-ID`; anything still in the bus backlog
        is replayed and a `gap` event is emitted when it is not, so a UI never
        resumes believing it saw everything. Slow clients are dropped from the
        tail of their own queue, never back-pressured onto the publisher.

        The stream ends itself after `sse_max_stream_s` with a `bye` event;
        EventSource reconnects on its own and the backlog fills the seam.
        """
        after = _last_event_id(request.headers.get("last-event-id"))
        sub = handle.bus.subscribe(after_seq=after)

        async def stream() -> AsyncIterator[bytes]:
            last_beat = 0.0
            deadline = asyncio.get_running_loop().time() + settings.sse_max_stream_s
            try:
                yield f"retry: {_SSE_RETRY_MS}\n\n".encode()
                while True:
                    if await request.is_disconnected():
                        return
                    for delivery in sub.drain():
                        if delivery.event is None:
                            yield _sse_gap(delivery.gap)
                            continue
                        yield _sse_frame(delivery.seq, delivery.event)
                        last_beat = 0.0
                    if asyncio.get_running_loop().time() >= deadline:
                        # Told, not just dropped: the client reconnects with
                        # Last-Event-ID and resumes exactly where it stopped.
                        yield b"event: bye\ndata: {\"reason\":\"stream age limit\"}\n\n"
                        return
                    await asyncio.sleep(settings.sse_poll_interval_s)
                    last_beat += settings.sse_poll_interval_s
                    if last_beat >= settings.sse_heartbeat_s:
                        last_beat = 0.0
                        # A comment frame: keeps proxies from closing an idle
                        # stream, and is ignored by EventSource.
                        yield b": keep-alive\n\n"
            finally:
                handle.bus.unsubscribe(sub)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --------------------------------------------------------------- control

    @app.post("/api/control/pause")
    def control_pause(body: _ControlBody) -> JSONResponse:
        """Stop opening new positions. Closes nothing.

        Pausing and flattening are separate on purpose: the usual reason to
        reach for pause is that something looks wrong, and dumping a book on a
        suspicion is its own disaster.
        """
        return _apply_control(
            handle, settings, body, action="pause",
            apply=lambda state, now: ControlState(
                paused=True,
                paused_at=now,
                paused_reason=body.reason,
                paused_actor_claimed=body.actor,
                flatten_request_id=state.flatten_request_id,
                flatten_requested_at=state.flatten_requested_at,
                flatten_reason=state.flatten_reason,
                flatten_actor_claimed=state.flatten_actor_claimed,
                flatten_acked_id=state.flatten_acked_id,
                flatten_acked_at=state.flatten_acked_at,
            ))

    @app.post("/api/control/resume")
    def control_resume(body: _ControlBody) -> JSONResponse:
        """Clear an operator pause. The only widening action here, and bounded.

        It clears the OPERATOR pause and nothing else: the risk circuit is
        re-read by the gate on every order and is untouched by this, and a
        pending flatten blocks the resume outright (409). Resuming entries while
        a square-off has been requested and not yet carried out is the exact
        wrong order of operations.
        """
        def apply(state: ControlState, now: dt.datetime) -> ControlState:
            return ControlState(
                paused=False,
                paused_at=None,
                paused_reason=None,
                paused_actor_claimed=None,
                flatten_request_id=state.flatten_request_id,
                flatten_requested_at=state.flatten_requested_at,
                flatten_reason=state.flatten_reason,
                flatten_actor_claimed=state.flatten_actor_claimed,
                flatten_acked_id=state.flatten_acked_id,
                flatten_acked_at=state.flatten_acked_at,
            )

        def guard(state: ControlState) -> Optional[str]:
            if state.flatten_pending:
                return ("a flatten request is still unacknowledged by the engine; "
                        "resume would re-enable entries into a book that is "
                        "supposed to be closing")
            return None

        return _apply_control(handle, settings, body, action="resume",
                              apply=apply, guard=guard)

    @app.post("/api/control/flatten")
    def control_flatten(body: _ControlBody) -> JSONResponse:
        """REQUEST a square-off. Places nothing from this process.

        The API holds no gate and no order path. This records a request the
        engine picks up on its next tick and executes through the gate, and the
        response says `executed: false` until the engine writes back an
        acknowledgement. An operator reading 200 OK here must not conclude the
        book is closed — hence the explicit field rather than an implied one.

        Also pauses. Flattening while still opening positions is incoherent, and
        two separate operator actions to reach one coherent state is one action
        too many at the moment it matters.
        """
        request_id = uuid.uuid4().hex

        def apply(state: ControlState, now: dt.datetime) -> ControlState:
            return ControlState(
                paused=True,
                paused_at=state.paused_at or now,
                paused_reason=state.paused_reason or f"flatten requested: {body.reason}",
                paused_actor_claimed=state.paused_actor_claimed or body.actor,
                flatten_request_id=request_id,
                flatten_requested_at=now,
                flatten_reason=body.reason,
                flatten_actor_claimed=body.actor,
                # Deliberately dropped: a NEW request must not inherit the
                # previous request's acknowledgement, or it would report itself
                # as already carried out.
                flatten_acked_id=None,
                flatten_acked_at=None,
            )

        return _apply_control(handle, settings, body, action="flatten",
                              apply=apply, extra={"request_id": request_id,
                                                  "executed": False})


# --------------------------------------------------------------------------
# route helpers
# --------------------------------------------------------------------------

def _engine_block(row: Optional[Mapping[str, str]], now: dt.datetime,
                  settings: ApiSettings) -> dict[str, Any]:
    """Whether the engine process is ticking.

    Three states, not two. "unknown" is its own answer: a heartbeat that was
    never written does not prove the engine is dead any more than it proves it
    is alive, and a dashboard that renders an untested claim either way is worse
    than one that says it does not know.
    """
    raw = None if row is None else row.get("heartbeat_at")
    if raw is None:
        return {"heartbeat_at": None, "status": "unknown", "age_s": None,
                "detail": "no engine heartbeat has ever been recorded here"}
    try:
        beat = dt.datetime.fromisoformat(raw)
    except ValueError:
        return {"heartbeat_at": raw, "status": "unknown", "age_s": None,
                "detail": f"heartbeat timestamp is unparseable: {raw!r}"}
    if beat.tzinfo is None:
        return {"heartbeat_at": raw, "status": "unknown", "age_s": None,
                "detail": f"heartbeat timestamp is not timezone-aware: {raw!r}"}
    age = (now - beat).total_seconds()
    stale = age > settings.engine_stale_after_s
    return {
        "heartbeat_at": beat.astimezone(UTC).isoformat(),
        "status": "stale" if stale else "alive",
        "age_s": round(age, 3),
        "detail": (f"last heartbeat {age:.0f}s ago, older than the "
                   f"{settings.engine_stale_after_s:.0f}s threshold"
                   if stale else "engine is ticking"),
    }


def _risk_block(handle: EngineHandle, asset_class: AssetClass,
                equity: Optional[Decimal],
                problems: list[dict[str, str]]) -> dict[str, Any]:
    """Read the circuit. A circuit that raises is reported as unreadable, never
    as clear."""
    try:
        decision = handle.circuit.read(asset_class, equity)
    except Exception as exc:
        problems.append(_problem(asset_class.value, f"risk circuit raised: {exc!r}"))
        return {
            "action": None,
            "entries_allowed": False,
            "must_flatten": False,
            "drawdown_fraction": None,
            "daily_loss_fraction": None,
            "equity_known": equity is not None,
            "state_known": False,
            "size_multiplier": None,
            "reason": f"circuit unreadable: {exc!r}",
        }
    return _decision_json(decision)


def _apply_control(handle: EngineHandle, settings: ApiSettings,
                   body: _ControlBody, action: str,
                   apply: Callable[[ControlState, dt.datetime], ControlState],
                   guard: Optional[Callable[[ControlState], Optional[str]]] = None,
                   extra: Optional[Mapping[str, Any]] = None) -> JSONResponse:
    """Run one control action over the requested scopes, transactionally.

    Read-modify-write happens inside the store's `mutate`, so two operators
    hitting pause at the same moment cannot lose one of the pauses.
    """
    now = handle.now()
    targets = ([body.asset_class] if body.asset_class is not None
               else list(handle.asset_classes))
    for asset_class in targets:
        if asset_class not in handle.asset_classes:
            return JSONResponse(
                {"detail": f"{asset_class.value} is not configured on this engine"},
                status_code=400)

    # Guards are evaluated across every target BEFORE anything is written, so a
    # multi-class action cannot half-apply and leave the classes disagreeing.
    if guard is not None:
        for asset_class in targets:
            try:
                current = _read_control(handle, asset_class.value)
            except ControlUnavailable as exc:
                return JSONResponse({"detail": str(exc)}, status_code=503)
            blocked = guard(current)
            if blocked is not None:
                return JSONResponse(
                    {"detail": f"{action} refused for {asset_class.value}: {blocked}",
                     "asset_class": asset_class.value},
                    status_code=409)

    results: dict[str, Any] = {}
    for asset_class in targets:
        def mutation(row: Optional[Mapping[str, str]]) -> Mapping[str, str]:
            current = ControlState() if row is None else ControlState.from_row(row)
            return apply(current, now).as_row()

        try:
            written = handle.control.mutate(asset_class.value, mutation)
        except Exception as exc:
            # Partial application is possible here if a later class fails. Say
            # so rather than reporting a clean failure: the operator must know
            # which classes actually took the action.
            return JSONResponse(
                {"detail": f"{action} failed for {asset_class.value}: {exc!r}",
                 "applied": sorted(results),
                 "not_applied": [a.value for a in targets if a.value not in results]},
                status_code=503)
        results[asset_class.value] = _control_json(ControlState.from_row(written))
        handle.bus.publish(Event(
            kind=f"control.{action}",
            at=now,
            data={"asset_class": asset_class.value, "reason": body.reason,
                  "actor_claimed": body.actor, "mode": handle.mode.value},
        ))

    payload: dict[str, Any] = {
        "action": action,
        "at": now.astimezone(UTC).isoformat(),
        "applied": results,
    }
    if extra:
        payload.update(extra)
        if extra.get("executed") is False:
            payload["note"] = (
                "recorded as a request. This process places no orders; the "
                "engine executes it through the gate on its next tick. Watch "
                "flatten_acked_id in /api/state to confirm it was taken up."
            )
    return JSONResponse(payload)


def _last_event_id(raw: Optional[str]) -> Optional[int]:
    """Parse the reconnect cursor. An unparseable value is treated as absent —
    the stream then starts from now, which loses history but never fabricates
    a position in it."""
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _sse_frame(seq: int, event: Event) -> bytes:
    return (f"id: {seq}\n"
            f"event: {event.kind}\n"
            f"data: {event.payload()}\n\n").encode()


def _sse_gap(missed: int) -> bytes:
    payload = json.dumps({"kind": "gap", "missed": missed},
                         sort_keys=True, separators=(",", ":"))
    return f"event: gap\ndata: {payload}\n\n".encode()
