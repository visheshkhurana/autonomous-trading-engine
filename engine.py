"""
The supervisor loop — the thing that makes this system autonomous.

Everything below the gate already refuses to be dangerous on its own. This file
is what runs them on a clock, and its whole job is to make sure that when
something breaks, the break is VISIBLE rather than quiet. A trading loop that
keeps ticking while its strategies throw, its ledger rejects writes, or its
quotes have gone stale is worse than a loop that stops: the dashboard stays
green while the book goes unmanaged. Every failure path here therefore ends in
one of exactly three places — an ERROR event, a disabled strategy, or a halted
asset class — and never in a substituted value that reads as "fine".

Five rules that are load-bearing, in the order they bite:

1.  **Nothing here arms anything.** The engine reads the arming flags (the same
    ones core.gate reads, imported from it so the two cannot drift) and it
    reads them once, at construction. It never sets them, never defaults a
    class to live, and never promotes an order from paper to live.

2.  **Approval.mode decides where an order executes, and the adapter must
    agree.** An approval whose mode differs from the mode this class's tick was
    measured in is refused outright: equity, positions and the caps were all
    evaluated against one book, and placing into the other one means every
    number that authorised the order described a different account.

3.  **A failed read is never an empty read.** `store.positions()` raising skips
    the whole tick — treating it as "flat" would let the engine re-enter
    positions it already holds. `adapter.equity()` failing yields None, which
    the circuit and the gate both refuse on. A quote that cannot be fetched is
    absent from `ctx.quotes`, never zero.

4.  **A strategy that raises is disabled for the session.** Not retried, not
    rate-limited, not silently swallowed — disabled, logged with its traceback,
    and reported in `status()` forever after. If it was holding positions when
    it died, those positions are now unmanaged and get their own alarm, because
    "strategy stopped proposing" and "strategy has nothing to do" are
    indistinguishable from the outside and only one of them is an emergency.

5.  **A ledger write that fails halts the class.** If a real fill cannot be
    recorded, every downstream cap — exposure, open positions, the circuit's
    drawdown — is being computed against a book we know is wrong. The engine
    stops placing orders for that class and screams. This is deliberately
    stricter than "log it and carry on", and it is deliberately not a flatten:
    liquidating on numbers we just admitted we cannot write is its own
    disaster.

Contract friction worked around here, rather than by editing the frozen files:

*   `AssetClassProfile` exposes `is_open()` but no session-end instant, and the
    mandatory intraday square-off needs one. Rather than hardcoding 15:30 IST
    in a second place (profiles.py explicitly owns that fact), the close is
    read back out of the profile's own `is_open` by search. See
    `_session_close`, and note why a plain binary search is wrong there.
*   `AssetClassProfile.quantize_price` rounds HALF_UP, which can round a
    marketable exit back inside the spread, where it rests instead of filling —
    the exact failure mode a mandatory square-off exists to prevent. Exit limits
    are snapped directionally here (see `_marketable_limit`), the same
    workaround adapters/dhan.py chose.
*   `OrderGate` takes a single `RiskCircuitProto`, but a circuit is bound to one
    mode. If one class is armed and another is not, the gate's circuit can only
    carry one of the two modes. The engine keeps its own per-class circuit and
    validates its mode; the gate's remains the integrator's problem to wire.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import Enum
from types import MappingProxyType
from typing import (
    Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, TypeVar,
)

# Imported from core.gate ON PURPOSE, private names included: "which env var
# arms this class" and "what counts as true" must have exactly ONE definition.
# A second copy here would let the engine route to a live adapter for an order
# the gate coerced to paper. If core.gate renames these, this import fails at
# start-up — loudly — instead of quietly diverging.
from core.gate import _ARM_ENV as ARM_ENV
from core.gate import _parse_arm_flag as _parse_arm_flag
from core.profiles import AssetClassProfile, profile_for
from core.safety import (
    REASON_STOP, BlackoutWindow, ManualReservedError, StopPlaceResult,
    base_asset, blackout_active, entry_block_reason, fetch_status_windows,
    key_freeze_active, load_configured_windows, non_reserved_free,
    overlap_stops_allowed, reserved_for, stop_qty_for_ledger,
)
from core.protocols import (
    BrokerAdapter, OrderGateProto, PositionStoreProto, RiskCircuitProto,
    Strategy,
)
from core.types import (
    Approval, AssetClass, Fill, Instrument, Intent, MarketContext, Mode,
    OrderRef, OrderStatus, OrderType, Position, Side,
)

__all__ = [
    "ArmReport",
    "ClassStatus",
    "ClassWiring",
    "Engine",
    "EngineConfig",
    "EngineConfigError",
    "EngineStatus",
    "Event",
    "EventBus",
    "EventKind",
    "IntentSource",
    "Subscription",
]

LOG = logging.getLogger("trader.engine")

ZERO = Decimal("0")
_T = TypeVar("_T")


UTC = dt.timezone.utc


def _utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def _persist_native_stop(fn: Callable[..., Any], mode: Mode, symbol: str,
                         strategy_id: str, txid: Optional[str],
                         price: Optional[Decimal], qty: Optional[Decimal],
                         status: str) -> None:
    """Adapter for set_native_stop's keyword-only arguments.

    ``_offload`` only forwards positional args, and a keyword slipped
    through a thread pool is how a txid write silently targets the wrong
    parameter.
    """
    fn(mode, symbol, strategy_id, txid=txid, stop_price=price,
       stop_qty=qty, status=status)


class ControlRowGone(RuntimeError):
    """The control row changed under an acknowledgement. Never swallowed: it
    means the operator's request and what the engine did have diverged."""


class EngineConfigError(ValueError):
    """Raised at construction for a wiring that could trade the wrong account.

    Every one of these is fatal on purpose. A misrouted adapter or a duplicated
    strategy id is not something to warn about and continue from — it is a
    reason not to start.
    """


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


class EventKind(str, Enum):
    """What happened. The API streams these verbatim.

    The set is deliberately small and closed: a dashboard that has to guess
    what an event means is a dashboard nobody trusts during an incident.
    """
    BOOT = "boot"                 # resolved wiring, once at start
    SESSION = "session"           # market opened / closed for a class
    TICK = "tick"                 # one full pass over a class
    INTENT = "intent"             # a strategy (or the square-off) wants to trade
    APPROVAL = "approval"         # gate said yes
    REJECTION = "rejection"       # gate said no, or the engine refused to place
    ORDER = "order"               # terminal OrderRef came back from an adapter
    FILL = "fill"                 # a fill was netted into the store
    RISK = "risk"                 # circuit verdict for a class
    ERROR = "error"               # anything that failed
    STRATEGY_DISABLED = "strategy_disabled"
    HALT = "halt"                 # a class stopped placing orders
    CONTROL = "control"           # an operator pause/resume/flatten took effect
    SHUTDOWN = "shutdown"


class IntentSource(str, Enum):
    STRATEGY = "strategy"
    SQUARE_OFF = "square_off"     # the engine's mandatory intraday flatten
    OPERATOR = "operator"         # an operator asked for a flatten through the API


# Scope key for the engine-wide control row. Must match api.server.ENGINE_SCOPE:
# the API reads the heartbeat this engine writes there to decide whether the
# process is alive at all.
CONTROL_ENGINE_SCOPE: str = "__engine__"


class ControlStoreProto(Protocol):
    """Durable operator intent, shared with the API process.

    Declared structurally rather than imported from `api.server` so the
    dependency keeps pointing one way: the API reads the engine's world, the
    engine must not import the API's. The shape is `api.server.ControlStateStore`
    and `SqliteControlStore` satisfies it as written.

    `read` returns None ONLY for a scope that was never written, and RAISES when
    it cannot read. The engine relies on those staying distinguishable: a failed
    read reported as "nothing here" is a pause that silently did not happen.
    """

    def read(self, scope: str) -> Optional[Mapping[str, str]]: ...
    def mutate(self, scope: str,
               fn: Callable[[Optional[Mapping[str, str]]], Mapping[str, str]],
               ) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class ControlRead:
    """What one tick made of the operator's control row.

    `known` is the load-bearing field. An unreadable row is NOT an unpaused
    engine: `paused=False` and `known=False` mean completely different things
    and the tick treats the second as "no new entries", because the operator may
    have pressed pause and been told 200 OK.
    """
    known: bool
    paused: bool = False
    paused_reason: Optional[str] = None
    flatten_request_id: Optional[str] = None
    flatten_acked_id: Optional[str] = None
    flatten_reason: Optional[str] = None
    detail: Optional[str] = None

    @property
    def entries_allowed(self) -> bool:
        return self.known and not self.paused

    @property
    def flatten_pending(self) -> bool:
        """An outstanding operator flatten. Compared against the id stored in
        the ROW, not against anything this process remembers, so a restarted
        engine still carries out a request made while it was down."""
        return (self.known
                and self.flatten_request_id is not None
                and self.flatten_acked_id != self.flatten_request_id)

    @staticmethod
    def unreadable(detail: str) -> "ControlRead":
        return ControlRead(known=False, detail=detail)


def _control_flag(row: Mapping[str, str], key: str) -> bool:
    """Parse a stored boolean, refusing anything that is not exactly 0 or 1.

    A row written by a newer API version, or a partially-written one, must not
    be read as False. False here un-pauses a paused engine.
    """
    raw = row.get(key, "0")
    if raw not in {"0", "1"}:
        raise ValueError(f"control row has non-boolean {key}={raw!r}")
    return raw == "1"


def _json_safe(value: Any) -> Any:
    """Render a value for the API stream without losing exactness.

    Decimals become strings rather than floats: these numbers are prices,
    quantities and drawdowns, and a float round-trip in the display layer is
    how a P&L stops reconciling with the ledger.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass(frozen=True)
class Event:
    """One thing that happened, with a monotonic sequence number.

    `seq` is the whole point of the ring buffer: a client that reconnects asks
    for everything after the last seq it saw, and a client whose queue
    overflowed can see the gap rather than believing it saw everything.
    """
    seq: int
    at: dt.datetime
    kind: EventKind
    message: str
    asset_class: Optional[AssetClass] = None
    mode: Optional[Mode] = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at": self.at.isoformat(),
            "kind": self.kind.value,
            "message": self.message,
            "asset_class": self.asset_class.value if self.asset_class else None,
            "mode": self.mode.value if self.mode else None,
            "data": _json_safe(dict(self.data)),
        }


class Subscription:
    """A live feed of events for one client.

    Bounded, and honest about it: when a slow consumer fills the queue the
    OLDEST event is dropped and `dropped` is incremented. The client can see
    both the counter and the jump in `Event.seq`, so a gap is detectable rather
    than being a stream that silently stops being complete.
    """

    __slots__ = ("_queue", "_loop", "_closed", "dropped")

    def __init__(self, maxsize: int, loop: asyncio.AbstractEventLoop) -> None:
        self._queue: asyncio.Queue[Optional[Event]] = asyncio.Queue(maxsize=maxsize)
        self._loop = loop
        self._closed = False
        self.dropped = 0

    # Called from whichever thread published. asyncio.Queue is not thread-safe,
    # so a foreign thread hops back onto the subscription's own loop.
    def _offer(self, event: Event) -> None:
        if self._closed:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._push(event)
        else:
            self._loop.call_soon_threadsafe(self._push, event)

    def _push(self, event: Optional[Event]) -> None:
        if self._closed and event is not None:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:      # pragma: no cover - race with a reader
                pass
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:           # pragma: no cover - race with a writer
            self.dropped += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._push(None)
        else:
            self._loop.call_soon_threadsafe(self._push, None)

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> Event:
        event = await self._queue.get()
        if event is None:
            raise StopAsyncIteration
        return event


class EventBus:
    """In-process pub/sub with a bounded ring buffer of recent events.

    The ring exists so a client that connects late still sees state. It is
    bounded so a long session cannot exhaust memory; `recent()` reports the
    oldest seq it still holds, so a client asking for something older learns it
    missed events instead of receiving a truncated history that looks complete.
    """

    def __init__(self, buffer_size: int = 2000,
                 clock: Callable[[], dt.datetime] = _utc_now) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        self._buffer: deque[Event] = deque(maxlen=buffer_size)
        self._subs: list[Subscription] = []
        self._lock = threading.RLock()
        self._seq = 0
        self._clock = clock

    @property
    def buffer_size(self) -> int:
        return self._buffer.maxlen or 0

    def publish(self, kind: EventKind, message: str, *,
                asset_class: Optional[AssetClass] = None,
                mode: Optional[Mode] = None,
                data: Optional[Mapping[str, Any]] = None) -> Event:
        with self._lock:
            self._seq += 1
            event = Event(
                seq=self._seq, at=self._clock(), kind=kind, message=message,
                asset_class=asset_class, mode=mode,
                data=MappingProxyType(dict(data or {})),
            )
            self._buffer.append(event)
            subs = tuple(self._subs)
        for sub in subs:
            sub._offer(event)
        return event

    def recent(self, after_seq: Optional[int] = None,
               limit: Optional[int] = None,
               kinds: Optional[Iterable[EventKind]] = None) -> tuple[Event, ...]:
        wanted = frozenset(kinds) if kinds is not None else None
        with self._lock:
            events = [
                e for e in self._buffer
                if (after_seq is None or e.seq > after_seq)
                and (wanted is None or e.kind in wanted)
            ]
        if limit is not None:
            events = events[-limit:]
        return tuple(events)

    def oldest_seq(self) -> Optional[int]:
        """Seq of the oldest event still buffered, or None when empty.

        A client whose `after_seq` is below this has missed events. Returning
        the buffer contents without saying so would present a hole as history.
        """
        with self._lock:
            return self._buffer[0].seq if self._buffer else None

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def subscribe(self, maxsize: int = 512) -> Subscription:
        if maxsize < 1:
            raise ValueError("subscription maxsize must be >= 1")
        loop = asyncio.get_running_loop()   # subscriptions belong to a loop
        sub = Subscription(maxsize, loop)
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)
        sub.close()

    def close(self) -> None:
        with self._lock:
            subs = tuple(self._subs)
            self._subs.clear()
        for sub in subs:
            sub.close()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineConfig:
    """Cadence and policy. Contains no credentials and no arming."""

    # What the engine ASKS for. Must match GateConfig.requested_mode; if it
    # does not, the engine and the gate disagree about which book a tick
    # measured, and every approval is refused with a loud MODE MISMATCH rather
    # than executed against the other account.
    requested_mode: Mode = Mode.PAPER
    event_buffer: int = 2000
    place_timeout_s: float = 30.0
    # How long before the session close the mandatory intraday flatten starts.
    flatten_lead_s: float = 900.0
    # Ticks the exit limit through the mark so it crosses instead of resting.
    square_off_cross_ticks: int = 2
    # Square-off intents reuse one idempotency key for this long, so a
    # rejection retries on the next bucket instead of on every 5s tick.
    square_off_retry_bucket_s: float = 60.0
    max_consecutive_tick_errors: int = 10
    shutdown_drain_s: float = 60.0
    # The engine owns the store in the normal deployment and closes it on stop.
    close_store_on_stop: bool = True
    unmanaged_alarm_interval_s: float = 300.0
    # How often the engine stamps the heartbeat the API reads. Must stay well
    # under the entry-block threshold or a healthy engine reads as stale.
    heartbeat_interval_s: float = 10.0
    # Loop beat older than this blocks new entries. Exits still run.
    entry_heartbeat_max_age_s: float = 60.0
    # Older than this, /health and /live fail and the watchdog exits.
    dead_heartbeat_s: float = 180.0
    # Native stop must be visible on open orders inside this window.
    stop_confirm_s: float = 10.0
    # Off in tests. The runner turns it on outside pytest.
    watchdog_enabled: bool = False
    # Best-effort Kraken status page. Off unless the runner enables it, so
    # a unit test tick never opens a socket.
    blackout_status_fetch: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.requested_mode, Mode):
            raise EngineConfigError("requested_mode must be a Mode")
        if self.event_buffer < 1:
            raise EngineConfigError("event_buffer must be >= 1")
        for name in ("place_timeout_s", "flatten_lead_s",
                     "square_off_retry_bucket_s", "shutdown_drain_s",
                     "unmanaged_alarm_interval_s", "heartbeat_interval_s",
                     "entry_heartbeat_max_age_s", "dead_heartbeat_s",
                     "stop_confirm_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise EngineConfigError(f"{name} must be a number")
            if value <= 0:
                raise EngineConfigError(f"{name} must be > 0")
        if not isinstance(self.square_off_cross_ticks, int) or isinstance(
                self.square_off_cross_ticks, bool):
            raise EngineConfigError("square_off_cross_ticks must be an int")
        if self.square_off_cross_ticks < 0:
            raise EngineConfigError("square_off_cross_ticks must be >= 0")
        if not isinstance(self.max_consecutive_tick_errors, int) or isinstance(
                self.max_consecutive_tick_errors, bool):
            raise EngineConfigError("max_consecutive_tick_errors must be an int")
        if self.max_consecutive_tick_errors < 1:
            raise EngineConfigError("max_consecutive_tick_errors must be >= 1")
        if self.dead_heartbeat_s < self.entry_heartbeat_max_age_s:
            raise EngineConfigError(
                "dead_heartbeat_s must be >= entry_heartbeat_max_age_s"
            )
        if not isinstance(self.watchdog_enabled, bool) or not isinstance(
                self.blackout_status_fetch, bool):
            raise EngineConfigError(
                "watchdog_enabled and blackout_status_fetch must be bools"
            )


@dataclass(frozen=True)
class ClassWiring:
    """Everything one asset class needs, supplied by the caller.

    `live_adapter` being present is NOT arming. It is only ever used when the
    class's env flag is set AND the engine was asked for live; otherwise it is
    constructed, reported at boot, and never called.
    """

    asset_class: AssetClass
    paper_adapter: BrokerAdapter
    risk: RiskCircuitProto
    strategies: tuple[Strategy, ...] = ()
    live_adapter: Optional[BrokerAdapter] = None
    # Symbols to quote every tick, on top of whatever is already held.
    instruments: tuple[Instrument, ...] = ()
    tick_interval_s: float = 5.0
    # Supplies ctx.extra (bars, regime, option chain). The engine deliberately
    # fetches no market history itself: that belongs to a data layer, and a
    # supervisor that also owns a feed cannot be reasoned about.
    extra_provider: Optional[
        Callable[[AssetClass, dt.datetime], Mapping[str, Any]]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategies", tuple(self.strategies))
        object.__setattr__(self, "instruments", tuple(self.instruments))
        if not isinstance(self.asset_class, AssetClass):
            raise EngineConfigError("asset_class must be an AssetClass")
        if not isinstance(self.tick_interval_s, (int, float)) or isinstance(
                self.tick_interval_s, bool):
            raise EngineConfigError("tick_interval_s must be a number")
        if self.tick_interval_s <= 0:
            raise EngineConfigError("tick_interval_s must be > 0")

        self._check_adapter(self.paper_adapter, Mode.PAPER, "paper_adapter")
        if self.live_adapter is not None:
            self._check_adapter(self.live_adapter, Mode.LIVE, "live_adapter")

        seen: set[str] = set()
        for strategy in self.strategies:
            sid = getattr(strategy, "id", "")
            if not isinstance(sid, str) or not sid.strip():
                raise EngineConfigError(
                    f"strategy {strategy!r} has no id; positions are keyed by "
                    "strategy_id and an unnamed strategy cannot own one")
            if sid in seen:
                raise EngineConfigError(
                    f"duplicate strategy id {sid!r} in {self.asset_class.value}: "
                    "two strategies sharing an id net into one position book")
            seen.add(sid)
            if getattr(strategy, "asset_class", None) is not self.asset_class:
                raise EngineConfigError(
                    f"strategy {sid!r} trades "
                    f"{getattr(strategy, 'asset_class', None)} but is wired into "
                    f"{self.asset_class.value}")
        for instrument in self.instruments:
            if instrument.asset_class is not self.asset_class:
                raise EngineConfigError(
                    f"{instrument.symbol} is {instrument.asset_class.value} but is "
                    f"wired into {self.asset_class.value}")

    def _check_adapter(self, adapter: Any, mode: Mode, label: str) -> None:
        if getattr(adapter, "mode", None) is not mode:
            raise EngineConfigError(
                f"{label} for {self.asset_class.value} reports mode "
                f"{getattr(adapter, 'mode', None)!r}, expected {mode.value}")
        if getattr(adapter, "asset_class", None) is not self.asset_class:
            raise EngineConfigError(
                f"{label} trades {getattr(adapter, 'asset_class', None)!r} but is "
                f"wired into {self.asset_class.value}")


@dataclass(frozen=True)
class ArmReport:
    """What an operator sees at boot. Names and modes only — never a secret."""
    asset_class: AssetClass
    mode: Mode
    adapter: str
    arm_var: str
    armed: bool
    strategies: tuple[str, ...]
    tick_interval_s: float

    def line(self) -> str:
        arm = "ARMED" if self.armed else "disarmed"
        return (f"{self.asset_class.value:<7} mode={self.mode.value:<5} "
                f"adapter={self.adapter:<16} {self.arm_var}={arm} "
                f"strategies={','.join(self.strategies) or '-'}")


@dataclass(frozen=True)
class ClassStatus:
    asset_class: AssetClass
    mode: Mode
    adapter: str
    market_open: bool
    ticks: int
    last_tick_at: Optional[dt.datetime]
    consecutive_errors: int
    halted: bool
    halt_reason: Optional[str]
    disabled_strategies: tuple[str, ...]
    unmanaged_positions: tuple[str, ...]
    last_risk_action: Optional[str]
    orders_placed: int
    fills_applied: int


@dataclass(frozen=True)
class EngineStatus:
    running: bool
    stopping: bool
    started_at: Optional[dt.datetime]
    classes: tuple[ClassStatus, ...]

    @property
    def degraded(self) -> bool:
        """True when something needs a human.

        A disabled strategy or a halted class means part of the book is no
        longer being managed. Reporting that as healthy is the specific lie
        this engine is built to avoid.
        """
        return any(
            c.halted or c.disabled_strategies or c.unmanaged_positions
            or c.consecutive_errors
            for c in self.classes
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


async def _offload(fn: Callable[..., _T], *args: Any) -> _T:
    """Run a blocking collaborator off the event loop.

    Every adapter call and every store call is synchronous and may block for
    seconds (place() blocks until terminal by contract). Running them inline
    would stall the other asset class's loop, so a slow equity broker would
    stop crypto from managing its stops.
    """
    return await asyncio.to_thread(fn, *args)


def _session_close(profile: AssetClassProfile, now: dt.datetime, *,
                   horizon_s: int = 86_400,
                   coarse_s: int = 300) -> Optional[dt.datetime]:
    """Last instant of the CURRENT open window, or None.

    Read out of the profile's own `is_open` rather than from a session-end
    constant: profiles.py is the single place allowed to know when a market
    closes, and a second copy of 15:30 IST here would eventually disagree with
    the first.

    A plain binary search would be wrong. `is_open` is not monotonic over a
    multi-day horizon — tomorrow's session is open again — so a search bounded
    by "open here, closed there" can converge on TOMORROW's close and the
    square-off window would never open today. Hence a forward scan for the
    FIRST close, then a binary search inside that one coarse step for the exact
    second. `coarse_s` only has to be shorter than the market's shortest closed
    gap, and NSE's is ~17 hours.

    None means "not currently open" or "no close within the horizon" (crypto).
    """
    if not profile.is_open(now):
        return None
    step = dt.timedelta(seconds=coarse_s)
    end = now + dt.timedelta(seconds=horizon_s)
    lo, hi = now, None
    probe = now + step
    while probe <= end:
        if not profile.is_open(probe):
            hi = probe
            break
        lo = probe
        probe += step
    if hi is None:
        return None
    while (hi - lo) > dt.timedelta(seconds=1):
        mid = lo + (hi - lo) / 2
        if profile.is_open(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _marketable_limit(instrument: Instrument, side: Side, mark: Decimal,
                      cross_ticks: int) -> Optional[Decimal]:
    """An exit limit that crosses the mark and STAYS crossed after snapping.

    `AssetClassProfile.quantize_price` rounds HALF_UP, which can round a buy
    back down (or a sell back up) into the spread — where the order rests
    instead of filling. A resting square-off order is the whole failure this
    function exists to prevent, so the snap is directional. None when no
    positive on-tick price exists.
    """
    tick = instrument.tick_size
    offset = tick * cross_ticks
    raw = mark + offset if side is Side.BUY else mark - offset
    if raw <= 0:
        return None
    rounding = ROUND_CEILING if side is Side.BUY else ROUND_FLOOR
    snapped = (raw / tick).to_integral_value(rounding=rounding) * tick
    return snapped if snapped > 0 else None


def _reduces(intent: Intent, book: Sequence[Position]) -> bool:
    """True when this order can only shrink an existing position.

    Same rule as the gate's private `_is_reducing`, duplicated deliberately
    rather than imported: the engine needs it BEFORE the gate runs, to refuse
    new entries once the square-off window has opened. An order bigger than the
    position is not a reduction — it closes and re-opens on the other side.
    """
    for position in book:
        if position.instrument.symbol != intent.instrument.symbol:
            continue
        if position.qty > 0 and intent.side is Side.SELL:
            return intent.qty <= position.qty
        if position.qty < 0 and intent.side is Side.BUY:
            return intent.qty <= -position.qty
    return False


def _exception_is_ambiguous(exc: BaseException) -> bool:
    """Whether a failed place() may have left a live order behind.

    adapters/dhan.py raises OrderSubmitFailed(ambiguous=True) when the submit
    died in flight: an order may exist under an id we never received. Read
    structurally so this module does not import an adapter.
    """
    return bool(getattr(exc, "ambiguous", False))


def _intent_data(intent: Intent, source: IntentSource) -> dict[str, Any]:
    return {
        "source": source.value,
        "strategy_id": intent.strategy_id,
        "symbol": intent.instrument.symbol,
        "side": intent.side.value,
        "qty": intent.qty,
        "order_type": intent.order_type.value,
        "limit_price": intent.limit_price,
        "reason": intent.reason,
        "structure_id": intent.structure_id,
        "idempotency_key": intent.idempotency_key,
    }


def _checks_data(approval: Approval) -> list[dict[str, Any]]:
    return [{"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in approval.checks]


# --------------------------------------------------------------------------
# Per-asset-class runner
# --------------------------------------------------------------------------


class _ClassRunner:
    """Owns one asset class's loop and its mutable session state."""

    def __init__(self, engine: "Engine", wiring: ClassWiring, mode: Mode) -> None:
        self.engine = engine
        self.wiring = wiring
        self.asset_class = wiring.asset_class
        self.profile = profile_for(wiring.asset_class)
        self.mode = mode
        self.adapters: Mapping[Mode, BrokerAdapter] = MappingProxyType(
            {Mode.PAPER: wiring.paper_adapter}
            if wiring.live_adapter is None else
            {Mode.PAPER: wiring.paper_adapter, Mode.LIVE: wiring.live_adapter}
        )
        self.adapter = self.adapters[mode]

        self.ticks = 0
        self.last_tick_at: Optional[dt.datetime] = None
        self.consecutive_errors = 0
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.disabled: dict[str, str] = {}
        self.last_risk_action: Optional[str] = None
        self.orders_placed = 0
        self.fills_applied = 0
        self._market_open: Optional[bool] = None
        self._close_cache: dict[str, Optional[dt.datetime]] = {}
        self._last_unmanaged_alarm: Optional[dt.datetime] = None
        self._unmanaged: tuple[str, ...] = ()
        # Last control row this class acted on, so a steady state is not
        # re-announced every 5s and the day's real events survive in the ring.
        self._control_paused: Optional[bool] = None
        self._control_known: Optional[bool] = None
        self._flatten_acked: Optional[str] = None
        # Set when a native stop could not be confirmed. Entries stay closed
        # until a later tick sees every open crypto position protected.
        self._entry_halt_reason: Optional[str] = None
        self._entry_halt_loaded = False
        # True only when THIS process latched the halt. A reason loaded from
        # the control row on restart may be cleared once the book is flat or
        # every open position is protected again.
        self._entry_halt_latched = False

    # -- plumbing -------------------------------------------------------

    @property
    def bus(self) -> EventBus:
        return self.engine.events

    def _emit(self, kind: EventKind, message: str,
              data: Optional[Mapping[str, Any]] = None) -> Event:
        return self.bus.publish(kind, message, asset_class=self.asset_class,
                                mode=self.mode, data=data)

    def _error(self, where: str, message: str, exc: Optional[BaseException] = None,
               extra: Optional[Mapping[str, Any]] = None) -> Event:
        data: dict[str, Any] = {"where": where}
        if exc is not None:
            data["error"] = f"{type(exc).__name__}: {exc}"
            data["traceback"] = traceback.format_exc()
        if extra:
            data.update(extra)
        LOG.error("[%s/%s] %s: %s", self.asset_class.value, self.mode.value,
                  where, message, exc_info=exc is not None)
        return self._emit(EventKind.ERROR, message, data)

    def halt(self, reason: str) -> None:
        """Stop placing orders for this class for the rest of the session.

        Not a flatten. We reach here because we cannot trust our own record of
        the book; liquidating on that basis would be acting on the same broken
        information. An operator has to look.
        """
        if self.halted:
            return
        self.halted = True
        self.halt_reason = reason
        LOG.critical("[%s] HALTED: %s", self.asset_class.value, reason)
        self._emit(EventKind.HALT,
                   f"{self.asset_class.value} halted: {reason}",
                   {"reason": reason, "orders_placed": self.orders_placed,
                    "fills_applied": self.fills_applied})

    def disable_strategy(self, strategy_id: str, exc: BaseException,
                         positions: Sequence[Position]) -> None:
        """Take a strategy out of the session and say so at the top of my voice.

        Never retried. A strategy that throws on one tick will throw on the
        next, and a loop that keeps calling it produces exactly the shape this
        engine exists to prevent: a green dashboard over an unmanaged book.
        """
        if strategy_id in self.disabled:
            return
        detail = f"{type(exc).__name__}: {exc}"
        self.disabled[strategy_id] = detail
        held = tuple(sorted(
            p.instrument.symbol for p in positions
            if p.strategy_id == strategy_id and p.qty != 0
        ))
        LOG.critical("[%s] strategy %s DISABLED for the session: %s",
                     self.asset_class.value, strategy_id, detail, exc_info=exc)
        self._emit(
            EventKind.STRATEGY_DISABLED,
            f"strategy {strategy_id} disabled for the session: {detail}",
            {"strategy_id": strategy_id, "error": detail,
             "traceback": traceback.format_exc(),
             "open_positions": list(held),
             "unmanaged": bool(held)},
        )
        if held:
            self._unmanaged = tuple(sorted(set(self._unmanaged) | set(held)))
            self._alarm_unmanaged(held, force=True)

    def note_unmanaged(self, positions: Sequence[Position]) -> tuple[str, ...]:
        """Symbols held by strategies that are no longer running.

        Nobody is watching these. They do not appear in any strategy's exit
        logic and, outside the equity square-off window, nothing will close
        them — so the answer is CACHED for `status()`, which must stay
        non-blocking. A status call that quietly reported () because it had no
        position list to hand is the "dashboard looks healthy" bug in
        miniature; it is refreshed on every tick and on every disable.
        """
        found = tuple(sorted(
            p.instrument.symbol for p in positions
            if p.qty != 0 and p.strategy_id in self.disabled
        ))
        self._unmanaged = found
        return found

    def _alarm_unmanaged(self, symbols: Sequence[str], *, force: bool = False) -> None:
        now = self.engine.now()
        interval = dt.timedelta(seconds=self.engine.config.unmanaged_alarm_interval_s)
        if not force and self._last_unmanaged_alarm is not None:
            if now - self._last_unmanaged_alarm < interval:
                return
        self._last_unmanaged_alarm = now
        LOG.critical("[%s] UNMANAGED positions from disabled strategies: %s",
                     self.asset_class.value, ", ".join(symbols))
        self._emit(
            EventKind.ERROR,
            f"{len(symbols)} position(s) held by disabled strategies are "
            f"unmanaged: {', '.join(symbols)}",
            {"where": "unmanaged_positions", "symbols": list(symbols),
             "disabled_strategies": sorted(self.disabled)},
        )

    # -- operator control -------------------------------------------------

    def read_control(self) -> ControlRead:
        """The operator's standing instruction for this class.

        Returns `known=False` when a control store IS wired but its row could
        not be read or parsed, which closes entries for the tick. That is the
        asymmetry the whole feature turns on: an operator who hit pause got a
        200 OK back, so the engine may not treat "I could not check" as "nobody
        asked me to stop".

        Three states, not two, and the third is the one worth naming: a row that
        has genuinely never been written means nobody has ever paused this
        class, and reads as known-and-unpaused. A store that is not wired at all
        means there is no operator surface in this deployment — announced once
        at boot, and not a reason to stop trading on every tick thereafter.
        """
        control = self.engine.control
        if control is None:
            return ControlRead(known=True, detail="no control store is wired")
        scope = self.asset_class.value
        try:
            row = control.read(scope)
        except Exception as exc:
            return ControlRead.unreadable(f"control row {scope!r} unreadable: {exc!r}")
        if row is None:
            return ControlRead(known=True)
        try:
            paused = _control_flag(row, "paused")
            request_id = row.get("flatten_request_id")
            acked_id = row.get("flatten_acked_id")
            if paused and not row.get("paused_at"):
                raise ValueError("control row is paused with no paused_at timestamp")
        except Exception as exc:
            # Corrupt is not clear. Reported, and entries stay closed.
            return ControlRead.unreadable(f"control row {scope!r} is corrupt: {exc!r}")
        return ControlRead(
            known=True, paused=paused,
            paused_reason=row.get("paused_reason"),
            flatten_request_id=request_id,
            flatten_acked_id=acked_id,
            flatten_reason=row.get("flatten_reason"),
        )

    def announce_control(self, control: ControlRead) -> None:
        """Emit only on a CHANGE. A pause is a state, and re-announcing it on
        every tick would push the events that explain the incident out of the
        ring buffer."""
        if control.known != self._control_known or control.paused != self._control_paused:
            first = self._control_known is None
            self._control_known = control.known
            self._control_paused = control.paused
            if control.known and not control.paused and first:
                # The opening observation is a STATE, not a transition. Saying
                # "resumed by operator" for a class nobody ever paused invents
                # an operator action in the audit trail.
                self._emit(EventKind.CONTROL,
                           f"{self.asset_class.value}: no operator pause in "
                           f"effect; entries are open",
                           {"known": True, "paused": False, "initial": True,
                            "entries_allowed": True})
                return
            if not control.known:
                LOG.error("[%s] control state unknown: %s",
                          self.asset_class.value, control.detail)
                self._emit(EventKind.CONTROL,
                           f"{self.asset_class.value}: operator control state is "
                           f"UNKNOWN; entries closed until it can be read",
                           {"known": False, "detail": control.detail,
                            "entries_allowed": False})
            else:
                self._emit(
                    EventKind.CONTROL,
                    f"{self.asset_class.value} "
                    f"{'PAUSED by operator' if control.paused else 'resumed by operator'}"
                    + (f": {control.paused_reason}" if control.paused_reason else ""),
                    {"known": True, "paused": control.paused,
                     "reason": control.paused_reason,
                     "entries_allowed": control.entries_allowed})

    def ack_flatten(self, request_id: str) -> None:
        """Record that this engine has carried out a flatten pass.

        Written AFTER the exit intents have been routed, never before. The
        acknowledgement is what the operator's screen turns into "done"; writing
        it on receipt would show a closed book while the orders were still being
        placed, and the whole reason this request goes through the engine rather
        than the API is that only one of those two can honestly say it happened.
        """
        control = self.engine.control
        if control is None:                    # pragma: no cover - guarded by caller
            return
        stamp = self.engine.now().astimezone(UTC).isoformat()

        def mutation(row: Optional[Mapping[str, str]]) -> Mapping[str, str]:
            if row is None:
                # The request vanished under us. Do not invent a row: an ack
                # with no request would let a later resume through its guard.
                raise ControlRowGone(
                    f"control row {self.asset_class.value!r} disappeared before "
                    "the flatten could be acknowledged")
            current = dict(row)
            if current.get("flatten_request_id") != request_id:
                raise ControlRowGone(
                    f"flatten request changed from {request_id!r} to "
                    f"{current.get('flatten_request_id')!r} while it was being "
                    "carried out; not acknowledging the newer request")
            # Copy the row through verbatim and touch only the two ack keys, so
            # a field this engine does not know about survives the round trip.
            current["flatten_acked_id"] = request_id
            current["flatten_acked_at"] = stamp
            return current

        try:
            self.engine.control.mutate(self.asset_class.value, mutation)
        except Exception as exc:
            # Not fatal to the tick: the exits were already placed. It IS loud,
            # because the operator's screen will keep saying "pending" and they
            # must not conclude from that that nothing happened.
            self._error("control",
                        f"flatten {request_id} was carried out but could not be "
                        "acknowledged; the API will keep reporting it pending",
                        exc, {"request_id": request_id})
            return
        self._emit(EventKind.CONTROL,
                   f"{self.asset_class.value}: operator flatten {request_id} "
                   "carried out",
                   {"request_id": request_id, "acked_at": stamp})

    def status(self) -> ClassStatus:
        return ClassStatus(
            asset_class=self.asset_class,
            mode=self.mode,
            adapter=type(self.adapter).__name__,
            market_open=self.profile.is_open(self.engine.now()),
            ticks=self.ticks,
            last_tick_at=self.last_tick_at,
            consecutive_errors=self.consecutive_errors,
            halted=self.halted,
            halt_reason=self.halt_reason,
            disabled_strategies=tuple(sorted(self.disabled)),
            unmanaged_positions=self._unmanaged,
            last_risk_action=self.last_risk_action,
            orders_placed=self.orders_placed,
            fills_applied=self.fills_applied,
        )

    # -- the loop -------------------------------------------------------

    async def loop(self, stop: asyncio.Event) -> None:
        interval = self.wiring.tick_interval_s
        while not stop.is_set():
            try:
                await self.tick()
                self.consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.consecutive_errors += 1
                self._error("tick",
                            f"tick failed ({self.consecutive_errors} consecutive)",
                            exc)
                if self.consecutive_errors >= self.engine.config.max_consecutive_tick_errors:
                    # Retrying forever at 5s intervals is indistinguishable
                    # from working, from the outside.
                    self.halt(f"{self.consecutive_errors} consecutive tick "
                              f"failures; last: {type(exc).__name__}: {exc}")
                    return
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def tick(self) -> None:
        now = self.engine.now()
        # Age is read BEFORE this tick's stamp. A loop that was wedged comes
        # back, refuses new entries for this pass, and only then looks alive.
        heartbeat_age = self.engine.heartbeat_age_s(now)
        self.engine.mark_loop_heartbeat(self.asset_class, now)
        # Before the market-open early return: a closed market is not a dead
        # process, and the API decides "engine alive" purely from this beat.
        await self.engine.beat()
        open_now = self.profile.is_open(now)
        if self._market_open is None or self._market_open != open_now:
            # Only on the transition: a "market closed" event every 5s all
            # night pushes the day's real events out of the ring buffer.
            self._market_open = open_now
            self._emit(EventKind.SESSION,
                       f"{self.asset_class.value} market "
                       f"{'open' if open_now else 'closed'}",
                       {"open": open_now, "at": now})
        if not open_now:
            return

        self.ticks += 1
        self.last_tick_at = now

        equity = await self._read_equity()
        try:
            await self._trade(now, equity, heartbeat_age)
        finally:
            # The circuit advances even when the trading half failed. A tick
            # that could not read positions still moved the account's value,
            # and a high-water mark that stops updating during an outage
            # under-reports the drawdown once it comes back.
            await self._advance_risk(equity)

    async def _read_equity(self) -> Optional[Decimal]:
        """None means UNKNOWN, and every consumer treats it as such.

        `equity()` returning None is contractual. It raising is not, so that is
        reported — but it still resolves to None here, never to a number.
        """
        try:
            equity = await _offload(self.adapter.equity)
        except Exception as exc:
            self._error("equity", "adapter.equity() raised; equity is UNKNOWN "
                                  "for this tick", exc)
            return None
        if equity is None:
            self._emit(EventKind.ERROR,
                       "adapter reports equity is unavailable; entries will be "
                       "refused this tick",
                       {"where": "equity", "equity": None})
            return None
        if not isinstance(equity, Decimal):
            self._error("equity",
                        f"adapter.equity() returned {type(equity).__name__}, "
                        "expected Decimal; treating as UNKNOWN", None,
                        {"value": repr(equity)})
            return None
        return equity

    async def _trade(self, now: dt.datetime, equity: Optional[Decimal],
                     heartbeat_age: Optional[float] = None) -> None:
        try:
            positions = tuple(await _offload(
                self.engine.store.positions, self.mode, self.asset_class))
        except Exception as exc:
            # An empty book and an unreadable book look identical downstream,
            # and reading "flat" here means re-entering positions we hold.
            self._error("positions",
                        "position store unreadable; skipping this tick rather "
                        "than trading against an unknown book", exc)
            raise

        unmanaged = self.note_unmanaged(positions)
        if unmanaged:
            self._alarm_unmanaged(unmanaged)

        control = await _offload(self.read_control)
        self.announce_control(control)

        if self.halted or self.engine.stopping:
            # Keep ticking — the circuit still needs to see the account, and
            # the operator still needs a heartbeat — but stop calling the venue
            # for marks nothing is going to act on.
            self._emit(EventKind.TICK, f"tick {self.ticks} (not trading)", {
                "tick": self.ticks, "at": now, "positions": len(positions),
                "quotes": 0, "equity": equity, "halted": self.halted,
                "stopping": self.engine.stopping,
            })
            return

        # The provider runs BEFORE the marks are read, and the order is
        # load-bearing rather than incidental. A provider is where the feed
        # refreshes the quote source (see build_stack), so reading marks first
        # builds a context carrying THIS tick's bars beside LAST tick's prices
        # — the very thing the wiring claims it prevents. Worse, on the tick
        # where a feed decides it can no longer vouch for a mark and removes
        # it, a snapshot taken first still carries the retired price into the
        # strategy's stop and the gate's notional caps.
        #
        # The cost of this order is that a provider which raises produces no
        # TICK event. That is the honest report: the tick did not happen.
        try:
            extra = dict(self.wiring.extra_provider(self.asset_class, now)
                         if self.wiring.extra_provider is not None else {})
        except Exception as exc:
            self._error("extra_provider",
                        "market data provider raised; no strategy can be "
                        "evaluated this tick", exc)
            raise

        quotes = await self._quotes(positions)
        self._emit(EventKind.TICK, f"tick {self.ticks}", {
            "tick": self.ticks, "at": now, "positions": len(positions),
            "quotes": len(quotes), "equity": equity,
            "halted": False, "stopping": False,
        })

        ctx = MarketContext(now=now, quotes=quotes, positions=positions,
                            equity=equity, extra=extra)

        if self.asset_class is AssetClass.CRYPTO:
            await _offload(self.engine.refresh_blackout)
            await self._maintain_native_stops(ctx)
            # Maintain may have flattened a position whose stop could not be
            # confirmed. The strategy has to see that book, not the one from
            # the start of the tick, or it will sell a qty the ledger no
            # longer holds.
            try:
                positions = tuple(await _offload(
                    self.engine.store.positions, self.mode, self.asset_class))
            except Exception as exc:
                self._error("positions",
                            "position store unreadable after stop maintenance; "
                            "skipping this tick", exc)
                raise
            ctx = MarketContext(now=now, quotes=quotes, positions=positions,
                                equity=equity, extra=extra)

        flatten_from = self._square_off_starts(now)
        session_flatten = flatten_from is not None and now >= flatten_from
        operator_flatten = control.flatten_pending

        # Three separate reasons to stop opening positions, all of which only
        # ever subtract. Exits stay open in every one of them: refusing to close
        # a position because the operator paused, or because a control row could
        # not be read, would trap the book in exactly the state they reached for
        # the button to get out of.
        #
        # Safety reasons are checked first and use the stable tokens the API
        # reports (`stale_heartbeat`, `blackout`, `key_freeze`,
        # `native_stop_unconfirmed`). Key freeze also refuses exits, in
        # `_route`, because that one forbids every new private order.
        safety_block = entry_block_reason(
            key_freeze=self._orders_frozen(),
            blackout=(self.asset_class is AssetClass.CRYPTO
                      and self.engine.blackout_is_active(now)),
            heartbeat_age=heartbeat_age,
            entry_max_age_s=self.engine.config.entry_heartbeat_max_age_s,
            stop_halt=self._entry_halt_reason is not None,
        )
        if safety_block is not None:
            entry_block = safety_block
        elif session_flatten:
            entry_block = "square-off window: entries are closed"
        elif operator_flatten:
            entry_block = f"operator flatten requested: {control.flatten_reason or 'no reason given'}"
        elif control.paused:
            entry_block = f"paused by operator: {control.paused_reason or 'no reason given'}"
        elif not control.known:
            entry_block = f"operator control state is unknown ({control.detail})"
        else:
            entry_block = None

        proposals = self._collect(ctx, entry_block)
        if session_flatten or operator_flatten:
            source = (IntentSource.SQUARE_OFF if session_flatten
                      else IntentSource.OPERATOR)
            proposals = self._square_off(ctx, proposals, source) + proposals

        for intent, source in proposals:
            if self.engine.stopping:
                self._emit(EventKind.REJECTION,
                           f"shutting down; {intent.instrument.symbol} intent "
                           "not placed",
                           {**_intent_data(intent, source),
                            "reason": "engine is shutting down"})
                continue
            if self.halted:
                break
            await self._route(intent, ctx, source)

        if operator_flatten and not self.halted and not self.engine.stopping:
            # Acknowledged only once the exits above have actually been routed,
            # and only if this tick got that far. A class that halted mid-pass
            # leaves the request pending, which is what the operator needs to
            # see: the book did not necessarily close.
            assert control.flatten_request_id is not None   # implied by flatten_pending
            await _offload(self.ack_flatten, control.flatten_request_id)

    def _collect(self, ctx: MarketContext,
                 entry_block: Optional[str]) -> list[tuple[Intent, IntentSource]]:
        """Strategy proposals for this tick.

        `entry_block` is None when entries are open, otherwise the operator-
        readable reason they are closed. Only intents that REDUCE an existing
        position survive a block — a strategy's own exit logic must keep working
        while the class is paused or squaring off.
        """
        out: list[tuple[Intent, IntentSource]] = []
        for strategy in self.wiring.strategies:
            if strategy.id in self.disabled:
                continue
            try:
                proposed = strategy.propose(ctx)
                intents = list(proposed)
            except Exception as exc:
                self.disable_strategy(strategy.id, exc, ctx.positions)
                continue
            bad = next((i for i in intents if not isinstance(i, Intent)), None)
            if bad is not None:
                self.disable_strategy(
                    strategy.id,
                    TypeError(f"propose() returned {type(bad).__name__}, "
                              "expected Intent"),
                    ctx.positions)
                continue
            for intent in intents:
                if intent.strategy_id != strategy.id:
                    # Otherwise a strategy can write into another strategy's
                    # position book, and the store's per-strategy netting stops
                    # meaning anything.
                    self.disable_strategy(
                        strategy.id,
                        ValueError(f"intent claims strategy_id "
                                   f"{intent.strategy_id!r}"),
                        ctx.positions)
                    break
                if entry_block is not None and not _reduces(intent, ctx.positions):
                    self._emit(
                        EventKind.REJECTION,
                        f"entries are closed; refusing new "
                        f"{intent.instrument.symbol} entry",
                        {**_intent_data(intent, IntentSource.STRATEGY),
                         "reason": entry_block})
                    continue
                out.append((intent, IntentSource.STRATEGY))
        return out

    def _square_off_starts(self, now: dt.datetime) -> Optional[dt.datetime]:
        if not self.profile.must_flatten_intraday:
            return None
        day = self.profile.day_key(now)
        if day not in self._close_cache:
            self._close_cache[day] = _session_close(self.profile, now)
        close = self._close_cache[day]
        if close is None:
            return None
        return close - dt.timedelta(seconds=self.engine.config.flatten_lead_s)

    def _square_off(self, ctx: MarketContext,
                    already: Sequence[tuple[Intent, IntentSource]],
                    source: IntentSource = IntentSource.SQUARE_OFF,
                    ) -> list[tuple[Intent, IntentSource]]:
        """Exit intents for everything still open.

        Used for both the mandatory intraday flatten near the session close and
        an operator-requested one, which differ only in `source` and in the
        reason text an operator reads afterwards. The exit mechanics must not
        differ: a hand-rolled second path is how one of them ends up without the
        shorts-first ordering.

        Ordered SHORTS-COVERED-FIRST. Closing the long leg of a spread before
        buying back the short one leaves a naked short in the window where
        there is least time to fix it — the single most expensive thing that
        can happen to an F&O book.
        """
        operator = source is IntentSource.OPERATOR
        covered = {i.instrument.symbol for i, _ in already}
        bucket = int(ctx.now.timestamp() // self.engine.config.square_off_retry_bucket_s)
        buys: list[tuple[Intent, IntentSource]] = []
        sells: list[tuple[Intent, IntentSource]] = []

        for position in sorted(ctx.positions, key=lambda p: p.instrument.symbol):
            if position.qty == 0 or position.instrument.symbol in covered:
                continue
            side = Side.SELL if position.qty > 0 else Side.BUY
            # Book qty. OkxAdapter.place clips a crypto spot SELL to venue
            # free base so a fee-in-base leftover cannot 51008 the flatten.
            qty = abs(position.qty)
            instrument = position.instrument
            mark = ctx.quote(instrument.symbol)
            usable = (mark if isinstance(mark, Decimal)
                      and mark.is_finite() and mark > 0 else None)
            order_type = OrderType.MARKET
            limit: Optional[Decimal] = None
            if usable is None:
                # No invented price. A MARKET without a reference is the
                # unbounded-order refuse the OKX drift guard exists to stop;
                # a LIMIT derived from a stale average can rest or fill badly
                # and still look like a successful square-off in the log.
                why = ("cannot build a marketable exit and this venue "
                       "rejects MARKET orders"
                       if not self.profile.market_orders_allowed else
                       "MARKET exit would be unbounded")
                self._error(
                    "square_off",
                    f"no quote for {instrument.symbol}; {why}",
                    None,
                    {"symbol": instrument.symbol, "qty": qty,
                     "strategy_id": position.strategy_id})
                continue
            if not self.profile.market_orders_allowed:
                limit = _marketable_limit(
                    instrument, side, usable,
                    self.engine.config.square_off_cross_ticks)
                if limit is None:
                    self._error(
                        "square_off",
                        f"no positive on-tick exit price for "
                        f"{instrument.symbol} at mark {usable}", None,
                        {"symbol": instrument.symbol, "mark": usable})
                    continue
                order_type = OrderType.LIMIT
            else:
                # Stamp the last known quote so the drift guard can run.
                limit = usable

            # The source is part of the key: an operator flatten landing in the
            # same 60s bucket as the session square-off would otherwise reuse
            # its key, and the gate would dedup the operator's request away.
            key = (f"{source.value}:{self.mode.value}:{self.asset_class.value}:"
                   f"{instrument.symbol}:{position.strategy_id}:{bucket}")
            intent = Intent(
                instrument=instrument, side=side, qty=qty,
                order_type=order_type,
                strategy_id=position.strategy_id,
                reason=("operator-requested flatten through the control API"
                        if operator else
                        f"mandatory intraday square-off before the "
                        f"{self.asset_class.value} close"),
                limit_price=limit,
                structure_id=position.structure_id,
                idempotency_key=key,
            )
            (buys if side is Side.BUY else sells).append((intent, source))
        return buys + sells

    async def _quotes(self, positions: Sequence[Position]) -> dict[str, Decimal]:
        """Marks for the watchlist and everything held.

        A symbol whose quote cannot be fetched is ABSENT from the result. It is
        never zero and never a stale carry-over: strategies read a missing
        quote as "skip this name", and the gate refuses to size an order it
        cannot price.
        """
        wanted: dict[str, Instrument] = {}
        for instrument in self.wiring.instruments:
            wanted[instrument.symbol] = instrument
        for position in positions:
            wanted.setdefault(position.instrument.symbol, position.instrument)

        quotes: dict[str, Decimal] = {}
        for symbol, instrument in wanted.items():
            try:
                price = await _offload(self.adapter.quote, instrument)
            except Exception as exc:
                self._error("quote", f"quote for {symbol} failed", exc,
                            {"symbol": symbol})
                continue
            if price is None:
                continue
            if not isinstance(price, Decimal) or price <= 0:
                self._error("quote",
                            f"quote for {symbol} is {price!r}; discarding "
                            "rather than pricing an order against it", None,
                            {"symbol": symbol})
                continue
            quotes[symbol] = price
        return quotes

    # -- native exchange stops ------------------------------------------

    def _manages_stops(self) -> bool:
        return bool(getattr(self.adapter, "manages_native_stops", False))

    def _orders_frozen(self) -> bool:
        """Key freeze: no new crypto orders. Equity is a different venue."""
        return (self.asset_class is AssetClass.CRYPTO
                and key_freeze_active(self.engine.safety_env))

    def _ledger_position(self, symbol: str, strategy_id: str) -> Optional[Position]:
        try:
            rows = self.engine.store.positions(self.mode, self.asset_class)
        except Exception:
            return None
        for position in rows:
            if position.instrument.symbol == symbol and position.strategy_id == strategy_id:
                return position
        return None

    def _strategy_for(self, strategy_id: str) -> Optional[Strategy]:
        for strategy in self.wiring.strategies:
            if strategy.id == strategy_id:
                return strategy
        return None

    def _protective_price(self, position: Position,
                          ctx: MarketContext) -> Optional[Decimal]:
        strategy = self._strategy_for(position.strategy_id)
        if strategy is None:
            return position.stop_price
        fn = getattr(strategy, "protective_stop", None)
        if fn is None:
            return position.stop_price
        try:
            price = fn(ctx, position)
        except Exception as exc:
            self._error("native_stop",
                        f"protective_stop raised for {position.instrument.symbol}",
                        exc)
            return position.stop_price
        if price is None:
            return position.stop_price
        if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
            return position.stop_price
        return price

    def _stop_required(self, position: Position) -> bool:
        """Live crypto must not sit without a stop. Paper only when one was
        already asked for (a stored price or txid), so a pre-existing paper
        book is not flattened for lacking a feature it never had."""
        if not self._manages_stops():
            return False
        if position.qty <= 0:
            return False
        if self.mode is Mode.LIVE and self.asset_class is AssetClass.CRYPTO:
            return True
        return bool(position.stop_txid or position.stop_price)

    def _load_entry_halt_once(self) -> None:
        if self._entry_halt_loaded:
            return
        self._entry_halt_loaded = True
        control = self.engine.control
        if control is None:
            return
        try:
            row = control.read(CONTROL_ENGINE_SCOPE)
        except Exception as exc:
            self._error("native_stop",
                        "could not read the persisted entry halt; entries stay "
                        "closed until it can be read", exc)
            self._entry_halt_reason = REASON_STOP
            return
        if row and row.get("entry_halt_reason"):
            self._entry_halt_reason = row["entry_halt_reason"]

    def _set_entry_halt(self, reason: Optional[str]) -> None:
        self._entry_halt_reason = reason
        if reason:
            self._entry_halt_latched = True
        else:
            self._entry_halt_latched = False
        control = self.engine.control
        if control is None:
            return

        def mutation(row: Optional[Mapping[str, str]]) -> Mapping[str, str]:
            current = dict(row or {})
            current.setdefault("paused", "0")
            if reason:
                current["entry_halt_reason"] = reason
            else:
                current.pop("entry_halt_reason", None)
            return current

        try:
            control.mutate(CONTROL_ENGINE_SCOPE, mutation)
        except Exception as exc:
            self._error("native_stop",
                        f"could not persist entry halt {reason!r}", exc)

    async def _maintain_native_stops(self, ctx: MarketContext) -> None:
        """Confirm, replace, or re-place the stop on each open long.

        Runs before entries are considered. A failure flattens that ledger
        qty and latches the entry halt. Key freeze only looks: it does not
        place, cancel, or flatten.
        """
        self._load_entry_halt_once()
        if not self._manages_stops():
            return
        frozen = self._orders_frozen()
        failed = False
        protected_required = False
        for position in ctx.positions:
            if position.qty <= 0:
                continue
            if frozen:
                await self._note_stop_readonly(position)
                continue
            if position.stop_txid:
                visible = await self._stop_is_open(position)
                if visible is None:
                    continue
                if visible:
                    protected_required = True
                    if await self._maybe_replace_stop(position, ctx):
                        failed = True
                    continue
                # Not on the open book. That can be a fill, a cancel, or a
                # read we cannot trust. Reconcile the txid before any new sell.
                reconciled = await self._reconcile_absent_stop(position)
                if reconciled != "place":
                    if reconciled == "filled":
                        protected_required = False
                    continue
            if not self._stop_required(position):
                continue
            price = self._protective_price(position, ctx)
            if price is None:
                await self._fail_unprotected(
                    position, ctx, "open position has no strategy stop to rest")
                failed = True
                continue
            result = await self._submit_stop(position, price, write=False)
            if result is not None and result.ok and result.txid:
                try:
                    await self._write_stop(position, result.txid, result.stop_price,
                                           result.qty, "open")
                    protected_required = True
                except Exception as exc:
                    self._error("native_stop", "could not store stop txid", exc)
                    failed = True
                continue
            if result is not None and result.txid:
                cancel = await self._cancel_stop(result.txid, position)
                if cancel is not None and cancel.outcome == "still_open":
                    await self._write_stop(position, result.txid, result.stop_price,
                                           result.qty, "open")
                    continue
                if cancel is not None and cancel.outcome == "filled" and cancel.fill is not None:
                    await self._apply_external_fill(position, cancel.fill)
                    failed = True
                    continue
            await self._fail_unprotected(
                position, ctx,
                result.reason if result is not None else "native stop was not confirmed")
            failed = True
        if failed:
            self._set_entry_halt(REASON_STOP)
        elif self._entry_halt_reason == REASON_STOP and (
                protected_required
                or (not self._entry_halt_latched
                    and not any(p.qty > 0 for p in ctx.positions))):
            # A halt this process just latched stays up while the book is
            # flat, so the next tick cannot re-enter. A halt only inherited
            # from the control row lifts once nothing is open, and either
            # kind lifts once a required position is protected again.
            self._set_entry_halt(None)

    async def _note_stop_readonly(self, position: Position) -> None:
        if not position.stop_txid:
            return
        visible = await self._stop_is_open(position)
        if visible is not False:
            return
        # Keep the txid. An open-order miss is not proof the stop is gone,
        # and key freeze must not forget an order it is not allowed to replace.
        await self._write_stop(position, position.stop_txid, position.stop_price,
                               position.stop_qty, "not_open")

    async def _reconcile_absent_stop(self, position: Position) -> str:
        """What to do with a stop that is not on the open-order book.

        ``filled`` — the fill was booked. ``keep`` — outcome unknown or still
        working; the txid stays. ``place`` — the venue confirmed the order is
        gone with no fill, so a new stop may be rested.
        """
        described = await self._describe_stop(position)
        outcome = None if described is None else described.outcome
        if outcome == "filled" and described is not None and described.fill is not None:
            await self._apply_external_fill(position, described.fill)
            return "filled"
        if outcome == "cancelled":
            try:
                await self._write_stop(position, None, None, None, "cancelled")
            except Exception as exc:
                self._error("native_stop",
                            f"could not clear cancelled stop {position.stop_txid}", exc)
                return "keep"
            return "place"
        self._error(
            "native_stop",
            f"stop {position.stop_txid} on {position.instrument.symbol} is "
            f"not open and its outcome is {outcome or 'unknown'}; keeping the txid",
            None)
        return "keep"

    async def _describe_stop(self, position: Position):
        fn = getattr(self.adapter, "describe_native_stop", None)
        if fn is None or not position.stop_txid:
            return None
        try:
            return await _offload(fn, position.stop_txid, position.instrument)
        except Exception as exc:
            self._error("native_stop",
                        f"could not read stop {position.stop_txid}", exc)
            return None

    async def _stop_is_open(self, position: Position) -> Optional[bool]:
        fn = getattr(self.adapter, "native_stop_open", None)
        if fn is None or not position.stop_txid:
            return None
        try:
            return await _offload(fn, position.stop_txid, position.instrument)
        except Exception as exc:
            self._error("native_stop",
                        f"could not read open orders for {position.instrument.symbol}",
                        exc)
            return None

    async def _maybe_replace_stop(self, position: Position,
                                  ctx: MarketContext) -> bool:
        """Replace a resting stop. True when the position was left unprotected."""
        if self.engine.blackout_is_active(self.engine.now()):
            # Leave the resting stop. A replace cancels or overlaps an order
            # during the window the blackout exists to avoid.
            return False
        new_price = self._protective_price(position, ctx)
        if new_price is None:
            return False
        tighter = position.stop_price is None or new_price > position.stop_price
        resized = (position.stop_qty is None
                   or abs(position.qty) != position.stop_qty)
        # Never loosen a resting stop. A wider ATR is the strategy's floor
        # problem; the exchange stop stays at the tighter price. Whether a
        # fee clip is required is decided after free balance is read.
        price = new_price if tighter else position.stop_price
        if price is None:
            return False
        ledger = abs(position.qty)
        try:
            reserved = self._reserved_base(position)
        except ManualReservedError as exc:
            self._error("native_stop",
                        f"MANUAL_RESERVED_BASE is unreadable: {exc}", exc)
            self._set_entry_halt(REASON_STOP)
            return True
        free = await self._free_base(position)
        available = non_reserved_free(free, reserved)
        target, decision = stop_qty_for_ledger(ledger, available)
        if decision == "halt":
            await self._fail_unprotected(
                position, ctx,
                f"non-reserved free {available} is more than 2% under "
                f"ledger qty {ledger}")
            return True
        needs_clip = decision == "clip" and position.stop_qty != target
        if not tighter and not resized and not needs_clip:
            return False
        old_qty = position.stop_qty if position.stop_qty is not None else ledger
        old_txid = position.stop_txid
        can_overlap = bool(old_txid) and overlap_stops_allowed(
            free=free, reserved=reserved, old_qty=old_qty, new_qty=target,
            ledger_qty=ledger,
        )
        if can_overlap and old_txid:
            result = await self._submit_stop(
                position, price, write=False, qty=target)
            if result is None or not result.ok or not result.txid:
                self._error(
                    "native_stop",
                    f"{position.instrument.symbol}: replacement stop was not "
                    "confirmed; the previous stop is still resting",
                    None)
                return False
            cancel = await self._cancel_stop(old_txid, position)
            if cancel is not None and cancel.outcome == "filled" and cancel.fill is not None:
                if result.txid:
                    await self._cancel_stop(result.txid, position)
                await self._apply_external_fill(position, cancel.fill)
                return False
            if cancel is None or cancel.outcome != "cancelled":
                # The new stop may be live and the old one may be too. Do not
                # forget the old txid. Roll the new order back; if that also
                # fails, remember its id on the status without replacing the
                # stored txid.
                rolled = None
                if result.txid:
                    rolled = await self._cancel_stop(result.txid, position)
                if rolled is not None and rolled.outcome == "filled" and rolled.fill is not None:
                    await self._apply_external_fill(position, rolled.fill)
                    return False
                remembered = (rolled is not None and rolled.outcome == "cancelled")
                if not remembered and result.txid:
                    try:
                        await self._write_stop(
                            position, old_txid, position.stop_price,
                            position.stop_qty, f"overlap:{result.txid}")
                    except Exception as exc:
                        self._error("native_stop",
                                    "could not remember the unretired replacement stop",
                                    exc)
                self._error(
                    "native_stop",
                    f"{position.instrument.symbol}: previous stop {old_txid} "
                    f"was not confirmed cancelled "
                    f"({None if cancel is None else cancel.outcome}); "
                    "the stored txid is unchanged",
                    None, {"txid": old_txid, "replacement": result.txid})
                return False
            await self._write_stop(position, result.txid, result.stop_price,
                                   result.qty, "open")
            return False
        if old_txid:
            cancel = await self._cancel_stop(old_txid, position)
            if cancel is not None and cancel.outcome == "filled" and cancel.fill is not None:
                await self._apply_external_fill(position, cancel.fill)
                return False
            if cancel is None or cancel.outcome != "cancelled":
                self._error(
                    "native_stop",
                    f"{position.instrument.symbol}: could not cancel the resting "
                    "stop to replace it; leaving the existing stop",
                    None, {"txid": old_txid})
                return False
            # The venue stop is gone. Drop the txid before the new place so a
            # failed replacement can flatten without trying to cancel an id
            # that no longer exists.
            try:
                await self._write_stop(position, None, None, None, "cancelled")
            except Exception as exc:
                self._error("native_stop",
                            "cancelled stop txid could not be cleared", exc)
        result = await self._submit_stop(position, price, write=True, qty=target)
        if result is None or not result.ok:
            await self._fail_unprotected(
                position, ctx,
                "stop replace left the position without a confirmed stop")
            return True
        return False

    def _reserved_base(self, position: Position) -> Decimal:
        return reserved_for(
            self.engine.safety_env, base_asset(position.instrument.symbol))

    async def _sync_ledger_qty(self, position: Position,
                               target: Decimal) -> Optional[Position]:
        """Write the fee-in-base gap off the book so qty matches the stop.

        The write-off is a sell at the position's own average, so realised
        P&L on the gap is zero. It is not a venue order and it does not
        touch the reserved manual balance.
        """
        gap = abs(position.qty) - target
        if gap <= 0:
            return position
        if position.avg_price <= 0:
            self._error("native_stop",
                        f"{position.instrument.symbol} has no average price; "
                        "cannot sync the ledger to the clipped stop", None)
            return None
        self._writeoff_seq = getattr(self, "_writeoff_seq", 0) + 1
        fill = Fill(
            order_id=(f"fee-base-writeoff:{self._writeoff_seq}:"
                      f"{position.instrument.symbol}"),
            instrument=position.instrument,
            side=Side.SELL if position.qty > 0 else Side.BUY,
            qty=gap, price=position.avg_price, fees=ZERO,
            at=self.engine.now(), mode=self.mode,
        )
        await self._apply_external_fill(
            position, fill,
            reason="fee-in-base write-off; ledger synced to the stop qty")
        synced = self._ledger_position(position.instrument.symbol, position.strategy_id)
        if synced is None or abs(synced.qty) != target:
            # apply() nets exactly; a miss means the write did not land.
            self._error("native_stop",
                        f"ledger for {position.instrument.symbol} did not "
                        f"sync to {target}", None)
            return None
        return synced

    async def _free_base(self, position: Position) -> Optional[Decimal]:
        fn = getattr(self.adapter, "free_base", None)
        if fn is None:
            return None
        try:
            free = await _offload(fn, position.instrument)
        except Exception as exc:
            self._error("native_stop",
                        f"could not read free {position.instrument.symbol}", exc)
            return None
        if free is None:
            return None
        if not isinstance(free, Decimal):
            return None
        return free

    async def _submit_stop(self, position: Position, price: Decimal,
                           *, write: bool = True,
                           qty: Optional[Decimal] = None) -> Optional[StopPlaceResult]:
        fn = getattr(self.adapter, "place_native_stop", None)
        if fn is None:
            return None
        ledger = abs(position.qty)
        if qty is None:
            try:
                reserved = self._reserved_base(position)
            except ManualReservedError as exc:
                return StopPlaceResult(
                    False, None, ledger, price,
                    f"MANUAL_RESERVED_BASE unreadable: {exc}")
            free = await self._free_base(position)
            available = non_reserved_free(free, reserved)
            qty, decision = stop_qty_for_ledger(ledger, available)
            if decision == "halt":
                return StopPlaceResult(
                    False, None, qty, price,
                    "non-reserved free short of ledger by more than 2%")
        if qty < ledger:
            synced = await self._sync_ledger_qty(position, qty)
            if synced is None:
                return StopPlaceResult(
                    False, None, qty, price,
                    "could not sync the ledger down to the clipped stop qty")
            position = synced
        try:
            result = await _offload(
                fn, position.instrument, qty, price,
                self.engine.config.stop_confirm_s)
        except Exception as exc:
            self._error("native_stop",
                        f"place_native_stop raised for {position.instrument.symbol}",
                        exc)
            return None
        if not isinstance(result, StopPlaceResult):
            self._error("native_stop",
                        f"place_native_stop returned {type(result).__name__}", None)
            return None
        if result.ok and result.txid and write:
            await self._write_stop(position, result.txid, result.stop_price,
                                   result.qty, "open")
        return result

    async def _cancel_stop(self, txid: str, position: Position):
        fn = getattr(self.adapter, "cancel_native_stop", None)
        if fn is None:
            return None
        try:
            return await _offload(fn, txid, position.instrument)
        except Exception as exc:
            self._error("native_stop",
                        f"cancel of stop {txid} raised", exc)
            return None

    async def _write_stop(self, position: Position, txid: Optional[str],
                          price: Optional[Decimal], qty: Optional[Decimal],
                          status: str) -> None:
        fn = getattr(self.engine.store, "set_native_stop", None)
        if fn is None:
            raise RuntimeError("position store cannot persist a native stop txid")
        await _offload(
            _persist_native_stop, fn, self.mode, position.instrument.symbol,
            position.strategy_id, txid, price, qty, status)

    async def _apply_external_fill(self, position: Position, fill: Fill,
                                  *, reason: str = "native exchange stop filled",
                                  ) -> None:
        """Book a stop that filled at the venue before our own exit did.

        Qty is capped at the ledger position. A fill larger than the book
        would be the manual balance, and this ledger does not own it.
        """
        qty = fill.qty
        if qty > abs(position.qty):
            self._error(
                "native_stop",
                f"stop fill {qty} on {position.instrument.symbol} exceeds "
                f"ledger qty {abs(position.qty)}; booking the ledger qty only",
                None)
            qty = abs(position.qty)
        if qty <= 0:
            return
        if qty != fill.qty or fill.mode is not self.mode:
            fill = replace(fill, qty=qty, mode=self.mode)
        intent = Intent(
            instrument=position.instrument, side=fill.side, qty=qty,
            order_type=OrderType.MARKET, strategy_id=position.strategy_id,
            reason=reason,
            limit_price=fill.price, structure_id=position.structure_id,
            idempotency_key=f"native-stop-fill:{fill.order_id}",
        )
        ref = OrderRef(
            order_id=fill.order_id, status=OrderStatus.FILLED, intent=intent,
            mode=self.mode, filled_qty=qty, avg_price=fill.price, fills=(fill,),
        )
        try:
            await _offload(self.engine.store.record_order, ref)
            await _offload(self.engine.store.apply, fill, position.strategy_id,
                           position.structure_id, 0)
        except Exception as exc:
            self._error("native_stop",
                        f"stop fill on {fill.order_id} could not be booked", exc)
            self.halt(f"ledger write failed for native stop fill {fill.order_id}")
            return
        self.fills_applied += 1
        self._emit(EventKind.FILL,
                   f"native stop filled {fill.side.value} {qty} "
                   f"{fill.instrument.symbol} @ {fill.price}",
                   {"order_id": fill.order_id, "symbol": fill.instrument.symbol,
                    "qty": qty, "price": fill.price,
                    "strategy_id": position.strategy_id, "source": "native_stop"})

    async def _fail_unprotected(self, position: Position, ctx: MarketContext,
                                reason: str) -> None:
        self._error(
            "native_stop",
            f"{position.instrument.symbol}: {reason}; market-flattening the "
            f"ledger qty and halting new entries",
            None, {"qty": abs(position.qty), "reason": reason})
        self._set_entry_halt(REASON_STOP)
        await self._flatten_ledger(position, ctx, reason)

    async def _flatten_ledger(self, position: Position, ctx: MarketContext,
                              reason: str) -> None:
        """Sell (or cover) exactly the ledger qty. Not the wallet."""
        mark = ctx.quote(position.instrument.symbol)
        if mark is None or not isinstance(mark, Decimal) or mark <= 0:
            self._error(
                "native_stop",
                f"no quote for {position.instrument.symbol}; cannot flatten "
                "without inventing a price",
                None)
            return
        side = Side.SELL if position.qty > 0 else Side.BUY
        order_type = (OrderType.MARKET if self.profile.market_orders_allowed
                      else OrderType.LIMIT)
        limit = mark
        if order_type is OrderType.LIMIT:
            snapped = _marketable_limit(
                position.instrument, side, mark,
                self.engine.config.square_off_cross_ticks)
            if snapped is None:
                self._error("native_stop",
                            f"no marketable flatten price for "
                            f"{position.instrument.symbol}", None)
                return
            limit = snapped
        intent = Intent(
            instrument=position.instrument, side=side, qty=abs(position.qty),
            order_type=order_type, strategy_id=position.strategy_id,
            reason=f"native stop unconfirmed: {reason}",
            limit_price=limit, structure_id=position.structure_id,
            idempotency_key=(
                f"native-stop-flatten:{self.mode.value}:"
                f"{position.instrument.symbol}:{position.strategy_id}:"
                f"{self.engine.now().astimezone(UTC).isoformat()}"
            ),
        )
        await self._route(intent, ctx, IntentSource.OPERATOR)

    async def _arm_stop_after_entry(self, intent: Intent,
                                    ctx: MarketContext) -> None:
        if not self._manages_stops():
            return
        if intent.side is not Side.BUY:
            return
        live_crypto = (self.mode is Mode.LIVE
                       and self.asset_class is AssetClass.CRYPTO)
        if intent.stop_price is None and not live_crypto:
            return
        if self._orders_frozen():
            return
        position = self._ledger_position(intent.instrument.symbol, intent.strategy_id)
        if position is None or position.qty <= 0:
            return
        price = intent.stop_price or self._protective_price(position, ctx)
        if price is None:
            await self._fail_unprotected(
                position, ctx, "entry fill has no strategy stop")
            return
        result = await self._submit_stop(position, price, write=False)
        if result is not None and result.ok and result.txid:
            try:
                await self._write_stop(position, result.txid, result.stop_price,
                                       result.qty, "open")
            except Exception as exc:
                self._error("native_stop",
                            "stop is live but its txid could not be stored", exc)
                await self._fail_unprotected(
                    position, ctx, "stop txid could not be stored on the position")
            return
        if result is not None and result.txid:
            cancel = await self._cancel_stop(result.txid, position)
            if cancel is not None and cancel.outcome == "filled" and cancel.fill is not None:
                await self._apply_external_fill(position, cancel.fill)
                self._set_entry_halt(REASON_STOP)
                return
            if cancel is not None and cancel.outcome == "still_open":
                try:
                    await self._write_stop(
                        position, result.txid, result.stop_price, result.qty, "open")
                except Exception as exc:
                    self._error("native_stop", "could not adopt a visible stop", exc)
                return
        detail = result.reason if result is not None else "stop placement failed"
        await self._fail_unprotected(position, ctx, detail)

    async def _release_stop_before_exit(self, intent: Intent,
                                        ctx: MarketContext) -> bool:
        """Cancel the resting stop before an engine exit.

        Returns True when the exit order itself should still be sent.
        A stop that already filled closes the ledger and the exit is skipped.
        """
        if not self._manages_stops():
            return True
        if not _reduces(intent, ctx.positions):
            return True
        position = self._ledger_position(intent.instrument.symbol, intent.strategy_id)
        if position is None or not position.stop_txid:
            held = next((p for p in ctx.positions
                         if p.instrument.symbol == intent.instrument.symbol
                         and p.strategy_id == intent.strategy_id
                         and p.stop_txid), None)
            position = position or held
        if position is None or not position.stop_txid:
            return True
        cancel = await self._cancel_stop(position.stop_txid, position)
        if cancel is None:
            self._error("native_stop",
                        f"stop {position.stop_txid} could not be cancelled; "
                        "not sending a second sell", None)
            self._emit(EventKind.REJECTION,
                       f"native stop on {intent.instrument.symbol} was not "
                       "cancelled cleanly; exit not sent",
                       {"symbol": intent.instrument.symbol,
                        "reason": "native_stop", "txid": position.stop_txid})
            return False
        if cancel.outcome == "filled" and cancel.fill is not None:
            await self._apply_external_fill(position, cancel.fill)
            return False
        if cancel.outcome == "cancelled":
            return True
        if cancel.outcome == "blocked":
            self._emit(EventKind.REJECTION,
                       f"key_freeze; not cancelling the stop on "
                       f"{intent.instrument.symbol}",
                       {"symbol": intent.instrument.symbol,
                        "reason": "key_freeze"})
            return False
        self._error(
            "native_stop",
            f"stop {position.stop_txid} is {cancel.outcome}; not sending an "
            f"exit that could sell twice ({cancel.reason})",
            None)
        self._emit(EventKind.REJECTION,
                   f"native stop on {intent.instrument.symbol} is "
                   f"{cancel.outcome}; exit not sent",
                   {"symbol": intent.instrument.symbol,
                    "reason": "native_stop", "outcome": cancel.outcome})
        return False

    # -- one intent, end to end -----------------------------------------

    async def _route(self, intent: Intent, ctx: MarketContext,
                     source: IntentSource) -> None:
        if self._orders_frozen():
            self._emit(EventKind.REJECTION,
                       f"key_freeze; refusing {intent.side.value} "
                       f"{intent.instrument.symbol}",
                       {**_intent_data(intent, source),
                        "reason": "key_freeze"})
            return
        self._emit(EventKind.INTENT,
                   f"{intent.strategy_id}: {intent.side.value} {intent.qty} "
                   f"{intent.instrument.symbol}",
                   _intent_data(intent, source))

        try:
            approval = await _offload(self.engine.gate.check, intent, ctx)
        except Exception as exc:
            self._error("gate", f"gate.check raised for "
                                f"{intent.instrument.symbol}; not placing", exc,
                        _intent_data(intent, source))
            return
        if not isinstance(approval, Approval):
            self._error("gate",
                        f"gate returned {type(approval).__name__}, expected "
                        "Approval; not placing", None,
                        _intent_data(intent, source))
            return

        if not approval.approved:
            self._emit(EventKind.REJECTION,
                       f"rejected {intent.instrument.symbol}: {approval.reason}",
                       {**_intent_data(intent, source),
                        "reason": approval.reason,
                        "checks": _checks_data(approval)})
            return

        # The gate's intent is authoritative: its qty is snapped and possibly
        # throttled. Placing the original makes both decorative.
        effective = approval.intent
        self._emit(EventKind.APPROVAL,
                   f"approved {effective.side.value} {effective.qty} "
                   f"{effective.instrument.symbol} in {approval.mode.value}",
                   {**_intent_data(effective, source),
                    "approved_mode": approval.mode,
                    "checks": _checks_data(approval)})

        adapter = self._adapter_for(approval, source, effective)
        if adapter is None:
            return
        if not await self._release_stop_before_exit(effective, ctx):
            return
        await self._place(adapter, approval, effective, source, ctx)

    def _adapter_for(self, approval: Approval, source: IntentSource,
                     effective: Intent) -> Optional[BrokerAdapter]:
        """The one place an order is routed to a venue.

        Two guards, both fatal to the order rather than to the tick:

        * the approval's mode must be the mode THIS tick measured. Equity,
          positions, exposure and the circuit were all read for one book; an
          approval for the other one was authorised by numbers that describe a
          different account. This can only happen when EngineConfig and
          GateConfig disagree about `requested_mode`, which is a wiring bug
          worth stopping for.
        * the adapter must itself report that mode. It is the last check
          between a live adapter and an order the gate coerced to paper.
        """
        if approval.mode is not self.mode:
            self._error(
                "mode_mismatch",
                f"gate approved {effective.instrument.symbol} in "
                f"{approval.mode.value} but this tick was measured in "
                f"{self.mode.value}; refusing to place",
                None,
                {**_intent_data(effective, source),
                 "approved_mode": approval.mode, "tick_mode": self.mode,
                 "hint": "EngineConfig.requested_mode must match "
                         "GateConfig.requested_mode"})
            return None
        adapter = self.adapters.get(approval.mode)
        if adapter is None:
            self._error("routing",
                        f"no {approval.mode.value} adapter wired for "
                        f"{self.asset_class.value}; refusing to place", None,
                        _intent_data(effective, source))
            return None
        if adapter.mode is not approval.mode:
            self._error("routing",
                        f"adapter {type(adapter).__name__} reports mode "
                        f"{adapter.mode!r}, approval says "
                        f"{approval.mode.value}; refusing to place", None,
                        _intent_data(effective, source))
            return None
        return adapter

    async def _place(self, adapter: BrokerAdapter, approval: Approval,
                     effective: Intent, source: IntentSource,
                     ctx: Optional[MarketContext] = None) -> None:
        try:
            ref = await _offload(adapter.place, effective,
                                 self.engine.config.place_timeout_s)
        except Exception as exc:
            ambiguous = _exception_is_ambiguous(exc)
            self._error("place",
                        f"place() raised for {effective.instrument.symbol}"
                        + (" and the outcome is AMBIGUOUS: an order may be live "
                           "at the venue under an id we never received"
                           if ambiguous else ""),
                        exc,
                        {**_intent_data(effective, source),
                         "ambiguous": ambiguous})
            if ambiguous:
                self.halt(
                    f"ambiguous submit for {effective.instrument.symbol}: an "
                    "untracked order may be live at the venue")
            return

        if not isinstance(ref, OrderRef):
            self._error("place",
                        f"adapter returned {type(ref).__name__}, expected "
                        "OrderRef", None, _intent_data(effective, source))
            self.halt("adapter returned a non-OrderRef from place(); the "
                      "outcome of a submitted order is unknown")
            return

        self.orders_placed += 1
        self._emit(EventKind.ORDER,
                   f"{ref.order_id} {ref.status.value} "
                   f"{ref.filled_qty}/{effective.qty} "
                   f"{effective.instrument.symbol}",
                   {**_intent_data(effective, source),
                    "order_id": ref.order_id, "status": ref.status.value,
                    "filled_qty": ref.filled_qty, "avg_price": ref.avg_price,
                    "error": ref.error, "fills": len(ref.fills),
                    "order_mode": ref.mode})

        await self._record(ref, effective, source, ctx)

    async def _record(self, ref: OrderRef, effective: Intent,
                      source: IntentSource,
                      ctx: Optional[MarketContext] = None) -> None:
        """Write the order and net every fill.

        record_order() stores what the venue said; apply() is what moves the
        position. Doing only the first leaves the book flat while real
        contracts are open, so both run and BOTH failing is a halt: from that
        point on every cap in the system is being measured against a ledger we
        know is wrong.
        """
        try:
            await _offload(self.engine.store.record_order, ref)
        except Exception as exc:
            self._error("record_order",
                        f"could not record order {ref.order_id}; the ledger no "
                        "longer matches the venue", exc,
                        {"order_id": ref.order_id, "status": ref.status.value,
                         "filled_qty": ref.filled_qty})
            self.halt(f"ledger write failed for order {ref.order_id}")

        for index, fill in enumerate(ref.fills):
            try:
                # fill_index is passed explicitly because the store's default
                # is content-addressed: two byte-identical fills on one order
                # (same qty, price, fees AND timestamp — which a fast venue or
                # a coarse clock produces) would be treated as a replay and the
                # second one silently dropped, understating the position.
                await _offload(self.engine.store.apply, fill,
                               effective.strategy_id, effective.structure_id,
                               index)
            except Exception as exc:
                self._error("apply_fill",
                            f"could not apply fill {index} of {ref.order_id}; "
                            f"{fill.qty} {fill.instrument.symbol} is filled at "
                            "the venue but missing from the book", exc,
                            {"order_id": ref.order_id, "fill_index": index,
                             "symbol": fill.instrument.symbol,
                             "side": fill.side.value, "qty": fill.qty,
                             "price": fill.price, "fees": fill.fees})
                self.halt(f"ledger write failed for a fill on {ref.order_id}")
                continue
            self.fills_applied += 1
            self._emit(EventKind.FILL,
                       f"{fill.side.value} {fill.qty} "
                       f"{fill.instrument.symbol} @ {fill.price}",
                       {"order_id": ref.order_id, "fill_index": index,
                        "symbol": fill.instrument.symbol,
                        "side": fill.side.value, "qty": fill.qty,
                        "price": fill.price, "fees": fill.fees,
                        "strategy_id": effective.strategy_id,
                        "structure_id": effective.structure_id,
                        "source": source.value, "fill_mode": fill.mode})

        if ctx is not None and any(f.side is Side.BUY and f.qty > 0 for f in ref.fills):
            await self._arm_stop_after_entry(effective, ctx)

    # -- risk -----------------------------------------------------------

    async def _advance_risk(self, equity: Optional[Decimal]) -> None:
        """Once per class per tick.

        advance() is the only writer of the circuit's baselines and refuses a
        non-positive equity, so an unknown or wiped-out account goes through
        read() instead — which reports it as blind or as a full drawdown. Never
        substituting a number is the entire point: a 0 passed to advance()
        would be persisted as a baseline.
        """
        try:
            if equity is not None and equity > ZERO:
                decision = await _offload(self.wiring.risk.advance,
                                          self.asset_class, equity)
            else:
                decision = await _offload(self.wiring.risk.read,
                                          self.asset_class, equity)
        except Exception as exc:
            self.last_risk_action = None
            self._error("risk",
                        "risk circuit failed; the gate will refuse entries "
                        "until it reads again", exc)
            return

        action = getattr(getattr(decision, "action", None), "value", None)
        self.last_risk_action = action
        self._emit(EventKind.RISK,
                   f"circuit: {action or 'unknown'} — "
                   f"{getattr(decision, 'reason', '')}",
                   {"action": action,
                    "reason": getattr(decision, "reason", None),
                    "drawdown_pct": getattr(decision, "drawdown_pct", None),
                    "daily_loss_pct": getattr(decision, "daily_loss_pct", None),
                    "equity": equity,
                    "equity_known": getattr(decision, "equity_known", None),
                    "state_known": getattr(decision, "state_known", None),
                    "size_multiplier": getattr(decision, "size_multiplier", None),
                    "advanced": equity is not None and equity > ZERO})


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class Engine:
    """One always-on asyncio supervisor over N asset classes.

    Construct it, `await run()`, and `await stop()`. It owns no credentials, no
    arming and no market data: adapters, the gate, the store and the circuits
    are all injected, which is what makes a full tick testable against fakes.
    """

    def __init__(
        self,
        *,
        wirings: Sequence[ClassWiring],
        store: PositionStoreProto,
        gate: OrderGateProto,
        config: Optional[EngineConfig] = None,
        bus: Optional[EventBus] = None,
        clock: Callable[[], dt.datetime] = _utc_now,
        env: Optional[Mapping[str, str]] = None,
        control: Optional[ControlStoreProto] = None,
    ) -> None:
        self.config = config or EngineConfig()
        if not wirings:
            raise EngineConfigError("engine needs at least one ClassWiring")
        self.store = store
        self.gate = gate
        # Optional, because a backtest or a test harness has no operator to
        # obey. Its absence is announced at boot rather than passed over: if an
        # API IS running against this ledger, its pause and flatten buttons are
        # writing rows nothing here will ever read, and the operator would be
        # told 200 OK by a process that cannot stop anything.
        self.control = control
        self._clock = clock
        self.events = bus or EventBus(self.config.event_buffer, clock)
        self._last_beat: Optional[dt.datetime] = None
        self._beat_lock = threading.Lock()
        self._class_heartbeats: dict[AssetClass, dt.datetime] = {}
        self._class_heartbeat_mono: dict[AssetClass, float] = {}
        self._watchdog_stop = threading.Event()
        self._watchdog: Optional[threading.Thread] = None
        self._status_windows: list[BlackoutWindow] = []
        self._status_fetched_mono: Optional[float] = None
        self._blackout_config_error: Optional[str] = None
        self._blackout_status_error: Optional[str] = None
        self._status_fetcher: Optional[Callable[..., Any]] = None

        source: Mapping[str, str] = os.environ if env is None else env
        self.safety_env: Mapping[str, str] = source
        seen_classes: set[AssetClass] = set()
        seen_strategies: dict[str, AssetClass] = {}
        runners: dict[AssetClass, _ClassRunner] = {}

        for wiring in wirings:
            if not isinstance(wiring, ClassWiring):
                raise EngineConfigError("wirings must contain ClassWiring objects")
            if wiring.asset_class in seen_classes:
                raise EngineConfigError(
                    f"two wirings for {wiring.asset_class.value}")
            seen_classes.add(wiring.asset_class)
            for strategy in wiring.strategies:
                if strategy.id in seen_strategies:
                    raise EngineConfigError(
                        f"strategy id {strategy.id!r} is used by "
                        f"{seen_strategies[strategy.id].value} and "
                        f"{wiring.asset_class.value}; positions are keyed by "
                        "strategy_id and the two books would merge")
                seen_strategies[strategy.id] = wiring.asset_class

            mode = self._resolve_mode(wiring.asset_class, source)
            if mode is Mode.LIVE and wiring.live_adapter is None:
                raise EngineConfigError(
                    f"{wiring.asset_class.value} is armed "
                    f"({ARM_ENV[wiring.asset_class]}) and the engine was asked "
                    "for live, but no live_adapter is wired; refusing to start "
                    "rather than trading a live-approved order on paper")
            self._check_circuit_mode(wiring, mode)
            runners[wiring.asset_class] = _ClassRunner(self, wiring, mode)

        self._runners: Mapping[AssetClass, _ClassRunner] = MappingProxyType(runners)
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._stopping = False
        self._started_at: Optional[dt.datetime] = None
        self._booted = False

    # -- construction helpers -------------------------------------------

    def _resolve_mode(self, asset_class: AssetClass,
                      env: Mapping[str, str]) -> Mode:
        """LIVE only when the engine was asked for it AND the class is armed.

        This method READS the arming flags. It does not set them, and there is
        no branch anywhere in this file that produces LIVE from anything else.
        """
        var = ARM_ENV[asset_class]
        armed = _parse_arm_flag(var, env.get(var))
        if self.config.requested_mode is Mode.LIVE and armed:
            return Mode.LIVE
        return Mode.PAPER

    @staticmethod
    def _check_circuit_mode(wiring: ClassWiring, mode: Mode) -> None:
        """A circuit is bound to one mode; the wrong one judges the wrong book.

        RiskCircuitProto does not declare `mode`, so a circuit that does not
        expose it cannot be checked. That is reported at boot rather than
        assumed correct.
        """
        circuit_mode = getattr(wiring.risk, "mode", None)
        if circuit_mode is None:
            return
        if circuit_mode is not mode:
            raise EngineConfigError(
                f"{wiring.asset_class.value} resolves to {mode.value} but its "
                f"risk circuit is bound to {getattr(circuit_mode, 'value', circuit_mode)}; "
                "a circuit reading the other mode's baselines would let paper "
                "P&L judge a live book")

    # -- introspection ---------------------------------------------------

    def now(self) -> dt.datetime:
        return self._clock()

    @property
    def stopping(self) -> bool:
        return self._stopping

    @property
    def running(self) -> bool:
        return self._running

    def arm_report(self) -> tuple[ArmReport, ...]:
        """What is armed right now, in plain words and with no secrets in it."""
        out: list[ArmReport] = []
        for asset_class in sorted(self._runners, key=lambda a: a.value):
            runner = self._runners[asset_class]
            out.append(ArmReport(
                asset_class=asset_class,
                mode=runner.mode,
                adapter=type(runner.adapter).__name__,
                arm_var=ARM_ENV[asset_class],
                armed=runner.mode is Mode.LIVE,
                strategies=tuple(s.id for s in runner.wiring.strategies),
                tick_interval_s=runner.wiring.tick_interval_s,
            ))
        return tuple(out)

    def arm_report_text(self) -> str:
        return "\n".join(r.line() for r in self.arm_report())

    def status(self) -> EngineStatus:
        """Cheap and synchronous. Reads no store — a status call must not be
        able to block the loop it is reporting on."""
        return EngineStatus(
            running=self._running,
            stopping=self._stopping,
            started_at=self._started_at,
            classes=tuple(self._runners[a].status()
                          for a in sorted(self._runners, key=lambda x: x.value)),
        )

    def runner_modes(self) -> Mapping[AssetClass, Mode]:
        return MappingProxyType({a: r.mode for a, r in self._runners.items()})

    # -- lifecycle -------------------------------------------------------

    def heartbeat_age_s(self, now: dt.datetime) -> Optional[float]:
        """Age of the oldest class loop beat, or None if none has ticked.

        The oldest, not the newest: one wedged class must not be hidden by
        a sibling that is still looping.
        """
        with self._beat_lock:
            if not self._class_heartbeats:
                return None
            oldest = min(self._class_heartbeats.values())
        return (now - oldest).total_seconds()

    def mark_loop_heartbeat(self, asset_class: AssetClass, now: dt.datetime) -> None:
        with self._beat_lock:
            self._class_heartbeats[asset_class] = now
            self._class_heartbeat_mono[asset_class] = time.monotonic()

    def heartbeat_is_stale(self, now: dt.datetime) -> bool:
        age = self.heartbeat_age_s(now)
        if age is None:
            return False
        return age > self.config.entry_heartbeat_max_age_s

    def blackout_windows(self) -> list[BlackoutWindow]:
        configured, error = load_configured_windows(self.safety_env)
        self._blackout_config_error = error
        return list(configured) + list(self._status_windows)

    def blackout_is_active(self, now: Optional[dt.datetime] = None) -> bool:
        moment = self.now() if now is None else now
        return blackout_active(moment, self.blackout_windows())

    def refresh_blackout(self) -> None:
        """Reload configured windows and, on a timer, the status page.

        A failed fetch keeps the previous status windows and records the
        error. It does not clear configured windows and it does not invent
        a blackout.
        """
        configured, error = load_configured_windows(self.safety_env)
        self._blackout_config_error = error
        if not self.config.blackout_status_fetch:
            self._persist_blackout_cache(configured)
            return
        now_m = time.monotonic()
        due = (self._status_fetched_mono is None
               or now_m - self._status_fetched_mono >= 60.0)
        if due:
            self._status_fetched_mono = now_m
            try:
                fetched, fetch_error = fetch_status_windows(self._status_fetcher)
            except Exception as exc:
                fetched, fetch_error = [], repr(exc)
            if fetch_error and not fetched:
                self._blackout_status_error = fetch_error
            else:
                self._status_windows = fetched
                self._blackout_status_error = None
        self._persist_blackout_cache(configured)

    def _persist_blackout_cache(self, configured: Sequence[BlackoutWindow]) -> None:
        control = self.control
        if control is None:
            return
        payload = json.dumps(
            [w.to_json() for w in self._status_windows],
            separators=(",", ":"),
        )

        def mutation(row: Optional[Mapping[str, str]]) -> Mapping[str, str]:
            current = dict(row or {})
            current.setdefault("paused", "0")
            current["blackout_status_json"] = payload
            if self._blackout_config_error:
                current["blackout_config_error"] = self._blackout_config_error
            else:
                current.pop("blackout_config_error", None)
            return current

        try:
            control.mutate(CONTROL_ENGINE_SCOPE, mutation)
        except Exception as exc:
            LOG.error("blackout cache write failed: %r", exc)

    async def beat(self) -> None:
        """Stamp the engine-wide heartbeat the API reads to decide "alive".

        The stamp is the oldest class beat, so a wedged sibling is not
        painted over by a class that is still looping. Throttled to
        `heartbeat_interval_s` because it is a database write on the tick
        path. A failure is logged and NOT raised: a heartbeat that cannot
        be written must not take down the trading loop it exists to
        describe — the API will report the beat as stale, which is the truth.
        """
        if self.control is None:
            return
        now = self.now()
        oldest = None
        with self._beat_lock:
            last = self._last_beat
            if last is not None and (now - last).total_seconds() < self.config.heartbeat_interval_s:
                return
            self._last_beat = now
            if self._class_heartbeats:
                oldest = min(self._class_heartbeats.values())
        stamp = (oldest or now).astimezone(UTC).isoformat()

        def mutation(row: Optional[Mapping[str, str]]) -> Mapping[str, str]:
            # Copied through so an engine-wide control row written by anything
            # else survives a heartbeat.
            current = dict(row or {})
            current.setdefault("paused", "0")
            current["heartbeat_at"] = stamp
            return current

        try:
            await _offload(self.control.mutate, CONTROL_ENGINE_SCOPE, mutation)
        except Exception as exc:
            LOG.error("heartbeat write failed: %r", exc)
            self.events.publish(
                EventKind.ERROR,
                f"heartbeat could not be written: {exc!r}; the API will report "
                "this engine as stale",
                data={"where": "heartbeat"})

    def _boot(self) -> None:
        if self._booted:
            return
        self._booted = True
        if self.control is None:
            # Loud, once. Silence here is how an operator ends up pressing a
            # pause button wired to nothing.
            LOG.warning("boot: no control store wired — operator pause/flatten "
                        "and the engine heartbeat are NOT available")
            self.events.publish(
                EventKind.CONTROL,
                "no control store is wired: operator pause and flatten cannot "
                "reach this engine, and it publishes no heartbeat",
                data={"control_store": False})
        report = self.arm_report()
        for line in self.arm_report_text().splitlines():
            LOG.info("boot: %s", line)
        self.events.publish(
            EventKind.BOOT, "engine boot", data={
                "requested_mode": self.config.requested_mode,
                "classes": [
                    {"asset_class": r.asset_class, "mode": r.mode,
                     "adapter": r.adapter, "arm_var": r.arm_var,
                     "armed": r.armed, "strategies": list(r.strategies),
                     "tick_interval_s": r.tick_interval_s}
                    for r in report
                ],
            })

    def _start_watchdog(self) -> None:
        """Kill the process if the loop stops stamping beats.

        A thread, not a task: a wedged event loop cannot serve /health, so
        the thing that notices has to be off that loop. ``os._exit`` is
        deliberate — ``sys.exit`` can be swallowed by a stuck interpreter
        frame, and Fly only restarts a process that has actually died.
        """
        if not self.config.watchdog_enabled or self._watchdog is not None:
            return
        self._watchdog_stop.clear()
        thread = threading.Thread(
            target=self._watchdog_loop, name="heartbeat-watchdog", daemon=True)
        thread.start()
        self._watchdog = thread

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread = self._watchdog
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._watchdog = None

    def _watchdog_loop(self) -> None:
        dead = self.config.dead_heartbeat_s
        started = time.monotonic()
        while not self._watchdog_stop.wait(min(5.0, max(dead / 4.0, 0.5))):
            if not self._running:
                continue
            with self._beat_lock:
                stamps = list(self._class_heartbeat_mono.values())
            now_m = time.monotonic()
            if not stamps:
                if now_m - started > dead:
                    LOG.critical(
                        "watchdog: engine running %.0fs with no loop heartbeat; exiting",
                        dead)
                    os._exit(3)
                continue
            if now_m - min(stamps) > dead:
                LOG.critical(
                    "watchdog: loop heartbeat older than %.0fs; exiting so the "
                    "supervisor restarts the machine", dead)
                os._exit(3)

    async def start(self) -> None:
        """Spawn one loop per asset class. Returns immediately."""
        if self._running:
            raise RuntimeError("engine is already running")
        self._boot()
        self._stop.clear()
        self._stopping = False
        self._running = True
        self._started_at = self.now()
        self._start_watchdog()
        self._tasks = [
            asyncio.create_task(runner.loop(self._stop),
                                name=f"trader-{runner.asset_class.value}")
            for runner in self._runners.values()
        ]

    async def run(self) -> None:
        """Start and block until stop() is called or every loop has halted."""
        await self.start()
        try:
            await asyncio.gather(*self._tasks)
        finally:
            self._running = False

    async def stop(self, drain_timeout_s: Optional[float] = None) -> None:
        """Stop accepting new intents, drain in-flight orders, checkpoint.

        `stopping` is set FIRST so a tick already in progress stops issuing new
        orders immediately, then the loops are awaited so any place() already
        in flight can return and be recorded. A place() that has reached a
        venue must not be abandoned: an order we stopped waiting for is an
        order nobody owns.
        """
        if not self._running and not self._tasks:
            return
        timeout = (self.config.shutdown_drain_s if drain_timeout_s is None
                   else drain_timeout_s)
        self._stopping = True
        self._stop.set()
        self._stop_watchdog()
        self.events.publish(EventKind.SHUTDOWN, "shutdown requested; draining",
                            data={"drain_timeout_s": timeout})

        pending = [t for t in self._tasks if not t.done()]
        if pending:
            done, still = await asyncio.wait(pending, timeout=timeout)
            for task in still:
                # The coroutine is cancelled, but a blocking place() already
                # handed to a worker thread cannot be. Say so.
                LOG.critical("shutdown: %s did not drain in %ss; an in-flight "
                             "order may be unaccounted", task.get_name(), timeout)
                self.events.publish(
                    EventKind.ERROR,
                    f"{task.get_name()} did not drain within {timeout}s; an "
                    "in-flight order may be unaccounted for",
                    data={"where": "shutdown", "task": task.get_name()})
                task.cancel()
            if still:
                await asyncio.gather(*still, return_exceptions=True)

        for task in self._tasks:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    LOG.critical("shutdown: %s ended with %r",
                                 task.get_name(), exc)
                    self.events.publish(
                        EventKind.ERROR,
                        f"{task.get_name()} ended with {type(exc).__name__}: {exc}",
                        data={"where": "loop", "task": task.get_name()})
        self._tasks = []
        self._running = False
        self._checkpoint()
        self.events.publish(EventKind.SHUTDOWN, "engine stopped", data={
            "classes": [
                {"asset_class": c.asset_class, "ticks": c.ticks,
                 "orders_placed": c.orders_placed,
                 "fills_applied": c.fills_applied,
                 "halted": c.halted, "halt_reason": c.halt_reason,
                 "disabled_strategies": list(c.disabled_strategies)}
                for c in self.status().classes
            ]})

    def _checkpoint(self) -> None:
        """Flush the ledger. Loud on failure — an unflushed ledger on shutdown
        is exactly the state you cannot reconstruct afterwards."""
        if not self.config.close_store_on_stop:
            return
        close = getattr(self.store, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception as exc:
            LOG.critical("shutdown: store checkpoint failed: %r", exc)
            self.events.publish(
                EventKind.ERROR, f"store checkpoint failed: {exc!r}",
                data={"where": "checkpoint"})

    # -- manual drive ----------------------------------------------------

    async def tick_once(self, asset_class: AssetClass) -> None:
        """Run exactly one tick for one class.

        For tests and for an operator stepping the engine by hand. It takes the
        identical path a scheduled tick takes — same gate, same adapters, same
        ledger writes — because a manual path that skipped any of them would
        not be testing the thing that runs.
        """
        self._boot()
        runner = self._runners.get(asset_class)
        if runner is None:
            raise KeyError(f"no wiring for {asset_class}")
        await runner.tick()
