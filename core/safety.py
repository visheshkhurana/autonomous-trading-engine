"""
Live-safety policy shared by the engine and the API.

Three independent reasons to refuse a new entry, each visible as a reason
string rather than as a silent skip:

* ``stale_heartbeat`` — the trading loop has not stamped a beat inside the
  entry threshold. Exits and stop maintenance stay allowed; a dead loop must
  not open risk, and it also must not trap a position the operator is trying
  to close.
* ``blackout`` — a configured UTC window, or a Kraken status-page maintenance
  that affects trading, padded 15 minutes before and 30 minutes after.
  Delistings, funding, deposits and withdrawals, specific assets or
  networks, and any status-page window longer than
  ``KRAKEN_STATUS_MAX_WINDOW_H`` (default 12 hours) are not blackouts.
  An ``in_progress`` incident Kraken never marks complete still ends at
  ``scheduled_until`` plus the 30 minute pad. Configured windows always
  apply. A failed status-page fetch adds nothing (fail open) and does not
  drop windows that were configured.
* ``key_freeze`` — ``KEY_FREEZE`` is truthy, or the freeze file exists. No
  private call that places or modifies an order. Stops already resting at
  the venue stay there.

An unreadable blackout file is not a blackout. Treating "I could not read
the calendar" as "the market is closed" would halt a book because a path
was wrong. The error is returned beside the windows so an operator can see
it; it does not flip ``blackout_active``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional, Sequence

from core.types import Fill

UTC = dt.timezone.utc

ENTRY_HEARTBEAT_MAX_AGE_S = 60.0
DEAD_HEARTBEAT_S = 180.0
STOP_CONFIRM_S = 10.0

# Status-page maintenances are widened so a clock skew or an early drain
# does not leave an entry in flight as the matching engine goes down.
STATUS_PAD_BEFORE = dt.timedelta(minutes=15)
STATUS_PAD_AFTER = dt.timedelta(minutes=30)
# Real API maintenance is short. The Sep–Dec 2026 asset-delisting notice
# is a multi-month status entry; it must not become an entry blackout.
STATUS_MAX_WINDOW = dt.timedelta(hours=12)
STATUS_MAX_WINDOW_ENV = "KRAKEN_STATUS_MAX_WINDOW_H"

# Component `name` values on status.kraken.com/api/v2/components.json are
# short labels: "REST", "WebSocket", "FIX", "Unified WebSocket", "REST API",
# "Websocket API", "FIX API", "Order Placement", "Prop Trading". Asset and
# network rails look like "API3 (API3) - Ethereum" or "Euro (EUR) - SEPA".
# Match the trading labels. A parenthetical ticker is not an API.
_TRADING_MARK = re.compile(
    r"("
    r"\brest\b"
    r"|websocket"
    r"|\bfix\b"
    r"|\bapi\b"
    r"|order\s+entr(?:y|ies)"
    r"|order\s+placement"
    r"|\btrading\b"
    r"|website[\s-]+and[\s-]+api"
    r"|platform[\s-]+wide"
    r")",
    re.IGNORECASE,
)
# Funding rails and per-asset notices. "API3" must not match \bapi\b; the
# ticker pattern is what drops "API3 (API3) - Ethereum" if a looser rule
# is added later.
_ASSET_OR_FUNDING_COMPONENT = re.compile(
    r"\([A-Z0-9]{2,}\)|deposit|withdraw|funding",
    re.IGNORECASE,
)
_NON_TRADING_NOTICE = re.compile(
    r"delist|funding|deposit|withdraw|banking|\bnetwork\b|\([A-Z0-9]{2,}\)",
    re.IGNORECASE,
)

STATUS_URLS: tuple[str, ...] = (
    "https://status.kraken.com/api/v2/scheduled-maintenances/upcoming.json",
    "https://status.kraken.com/api/v2/scheduled-maintenances/active.json",
)

KEY_FREEZE_PATH = "/data/KEY_FREEZE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

REASON_STALE = "stale_heartbeat"
REASON_BLACKOUT = "blackout"
REASON_KEY_FREEZE = "key_freeze"
REASON_STOP = "native_stop_unconfirmed"

# A fee taken in the base asset leaves free balance a hair under the ledger
# qty. Clip the stop down to that hair. Anything wider is not a fee.
STOP_FEE_CLIP_LIMIT = Decimal("0.02")


class ManualReservedError(ValueError):
    """``MANUAL_RESERVED_BASE`` is set but cannot be read.

    An unreadable reserve must not be treated as zero: zero would let an
    order spend the manual balance the variable exists to protect.
    """


def base_asset(symbol: str) -> str:
    """``BTC/USDT`` and ``BTC/USDT:USDT`` both yield ``BTC``."""
    head = symbol.split("/", 1)[0]
    return head.split(":", 1)[0].strip().upper()


def parse_manual_reserved(raw: Optional[str]) -> dict[str, Decimal]:
    """Parse ``MANUAL_RESERVED_BASE`` (``BTC:0.0880257350``, comma-separated).

    Empty means nothing is reserved. A malformed entry raises rather than
    becoming zero.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    out: dict[str, Decimal] = {}
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ManualReservedError(
                f"MANUAL_RESERVED_BASE entry {piece!r} must be ASSET:QTY"
            )
        asset, amount = piece.split(":", 1)
        asset = asset.strip().upper()
        if not asset:
            raise ManualReservedError(
                f"MANUAL_RESERVED_BASE entry {piece!r} has no asset"
            )
        try:
            qty = Decimal(amount.strip())
        except Exception as exc:
            raise ManualReservedError(
                f"MANUAL_RESERVED_BASE amount {amount!r} is not a number"
            ) from exc
        if not qty.is_finite() or qty < 0:
            raise ManualReservedError(
                f"MANUAL_RESERVED_BASE {asset} qty must be >= 0, got {qty}"
            )
        out[asset] = out.get(asset, Decimal("0")) + qty
    return out


def reserved_for(env: Optional[Mapping[str, str]], asset: str) -> Decimal:
    """Qty of ``asset`` that is not available to the engine.

    Default is 0 when the variable is unset. A bad value raises
    ``ManualReservedError``.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    table = parse_manual_reserved(source.get("MANUAL_RESERVED_BASE"))
    return table.get(asset.upper(), Decimal("0"))


def non_reserved_free(free: Optional[Decimal], reserved: Decimal) -> Optional[Decimal]:
    """Venue free minus the manual reserve. None stays None (unreadable).

    Negative results are 0: the reserve is larger than free, so the engine
    has nothing it may spend.
    """
    if free is None:
        return None
    available = free - reserved
    if available < 0:
        return Decimal("0")
    return available


def overlap_stops_allowed(
    *,
    free: Optional[Decimal],
    reserved: Decimal,
    old_qty: Decimal,
    new_qty: Decimal,
    ledger_qty: Decimal,
) -> bool:
    """True only when a second stop can rest without touching reserved coins.

    Both stops together must fit in free balance after the reserve is
    removed, and that combined qty must not exceed the engine ledger. Two
    full-size stops are ``2 * ledger`` and are therefore refused: the
    second order would be funded by the manual balance.
    """
    available = non_reserved_free(free, reserved)
    if available is None or ledger_qty <= 0:
        return False
    both = old_qty + new_qty
    if both <= 0:
        return False
    return available >= both and both <= ledger_qty


def stop_qty_for_ledger(
    ledger: Decimal,
    available: Optional[Decimal],
    *,
    limit: Decimal = STOP_FEE_CLIP_LIMIT,
) -> tuple[Decimal, str]:
    """How large a stop may be, and what to do about the gap.

    ``full`` — available covers the ledger (or free could not be read, so
    the caller sends the ledger qty and the venue is the backstop).
    ``clip`` — shortfall is within ``limit`` (a fee in base). Use
    ``available`` and sync the ledger down to it.
    ``halt`` — shortfall is wider than ``limit``, or nothing is available.
    """
    if ledger <= 0:
        return Decimal("0"), "halt"
    if available is None:
        return ledger, "full"
    if available >= ledger:
        return ledger, "full"
    if available <= 0:
        return Decimal("0"), "halt"
    shortfall = (ledger - available) / ledger
    if shortfall <= limit:
        return available, "clip"
    return available, "halt"


@dataclass(frozen=True)
class StopPlaceResult:
    """Outcome of asking a venue to rest a stop sized to the ledger qty.

    ``ok`` is true only when the stop was seen on the open-order book inside
    the confirm window. A txid with ``ok`` false means an order may exist
    and the caller must not pretend it does not.
    """

    ok: bool
    txid: Optional[str]
    qty: Decimal
    stop_price: Decimal
    reason: str


@dataclass(frozen=True)
class StopCancelResult:
    """What a cancel of a native stop actually did.

    ``filled`` carries the venue fill so the ledger can close the position
    without sending a second sell. ``blocked`` means key freeze refused the
    private call; the stop is untouched.
    """

    outcome: str  # cancelled | filled | still_open | unknown | blocked
    fill: Optional[Fill] = None
    reason: str = ""

    def __post_init__(self) -> None:
        allowed = {"cancelled", "filled", "still_open", "unknown", "blocked"}
        if self.outcome not in allowed:
            raise ValueError(f"stop cancel outcome {self.outcome!r} is not one of {allowed}")
        if self.outcome == "filled" and self.fill is None:
            raise ValueError("a filled native stop must carry the fill")


@dataclass(frozen=True)
class BlackoutWindow:
    """An interval during which new entries are refused.

    ``start`` and ``end`` are already the effective bounds: configured
    windows include ``cancel_before``, status-page windows include padding.
    """

    start: dt.datetime
    end: dt.datetime
    source: str
    name: str = ""

    def __post_init__(self) -> None:
        for label, moment in (("start", self.start), ("end", self.end)):
            if moment.tzinfo is None or moment.utcoffset() is None:
                raise ValueError(f"blackout {label} must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("blackout end must be after start")

    def contains(self, now: dt.datetime) -> bool:
        return self.start <= now < self.end

    def to_json(self) -> dict[str, str]:
        return {
            "start": self.start.astimezone(UTC).isoformat(),
            "end": self.end.astimezone(UTC).isoformat(),
            "source": self.source,
            "name": self.name,
        }


def env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def env_float(source: Mapping[str, str], name: str, default: float) -> float:
    raw = source.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def env_bool(source: Mapping[str, str], name: str, default: bool) -> bool:
    raw = source.get(name)
    if raw is None or not str(raw).strip():
        return default
    text = str(raw).strip().lower()
    if text in _TRUTHY:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean")


def key_freeze_active(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when the operator has frozen the key.

    The env mapping is read live (pass ``os.environ`` in production) and the
    freeze file is stat'd on every call, so dropping ``/data/KEY_FREEZE``
    onto the volume takes effect on the next tick without a restart.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    if env_truthy(source.get("KEY_FREEZE")):
        return True
    path = (source.get("KEY_FREEZE_PATH") or KEY_FREEZE_PATH).strip()
    if not path:
        return False
    try:
        return os.path.exists(path)
    except OSError:
        return False


def _aware(moment: dt.datetime, label: str) -> dt.datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware, got {moment!r}")
    return moment.astimezone(UTC)


def _parse_instant(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an ISO-8601 string")
    parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return _aware(parsed, label)


def _parse_duration_s(value: Any) -> dt.timedelta:
    """Seconds, or a compact '15m' / '30s' / '1h' duration."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"cancel_before must be a duration, got {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower()
        if not text:
            return dt.timedelta(0)
        unit = text[-1]
        if unit in {"s", "m", "h"} and text[:-1].strip():
            try:
                magnitude = float(text[:-1].strip())
            except ValueError as exc:
                raise ValueError(f"cancel_before {value!r} is not a duration") from exc
            scale = {"s": 1, "m": 60, "h": 3600}[unit]
            seconds = magnitude * scale
        else:
            try:
                seconds = float(text)
            except ValueError as exc:
                raise ValueError(f"cancel_before {value!r} is not a duration") from exc
    if seconds < 0:
        raise ValueError("cancel_before must be >= 0")
    return dt.timedelta(seconds=seconds)


def windows_from_payload(payload: Any, *, source: str) -> list[BlackoutWindow]:
    """Parse a JSON list or ``{\"windows\": [...]}`` of UTC intervals."""
    if isinstance(payload, dict):
        rows = payload.get("windows", payload.get("blackouts"))
    else:
        rows = payload
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError("blackout windows must be a list")
    out: list[BlackoutWindow] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"blackout window {index} must be an object")
        start = _parse_instant(row.get("start"), f"window {index} start")
        end = _parse_instant(row.get("end"), f"window {index} end")
        early = row.get("cancel_before_s", row.get("cancel_before"))
        if early is not None and str(early).strip() != "":
            start = start - _parse_duration_s(early)
        name = str(row.get("name") or "")
        out.append(BlackoutWindow(start=start, end=end, source=source, name=name))
    return out


def load_configured_windows(
    env: Optional[Mapping[str, str]] = None,
) -> tuple[list[BlackoutWindow], Optional[str]]:
    """Windows from ``BLACKOUT_WINDOWS`` and ``BLACKOUT_WINDOWS_FILE``.

    A missing file path means "no file configured", not an error. A path that
    is set but cannot be read, or JSON that cannot be parsed, is reported
    and contributes no windows. Other successfully loaded windows still apply.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    windows: list[BlackoutWindow] = []
    errors: list[str] = []

    raw = source.get("BLACKOUT_WINDOWS")
    if raw is not None and raw.strip():
        try:
            windows.extend(windows_from_payload(json.loads(raw), source="config"))
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"BLACKOUT_WINDOWS is unreadable: {exc}")

    path = (source.get("BLACKOUT_WINDOWS_FILE") or "").strip()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            windows.extend(windows_from_payload(payload, source="config"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"BLACKOUT_WINDOWS_FILE {path!r} is unreadable: {exc}")

    return windows, ("; ".join(errors) if errors else None)


def status_max_window(source: Mapping[str, str]) -> dt.timedelta:
    """``KRAKEN_STATUS_MAX_WINDOW_H`` (default 12). Longer status windows drop.

    A value that is not a positive number raises, same as the other safety
    env parsers. The engine treats that as a failed status fetch (previous
    status windows stay, configured windows stay). The API falls back to
    12 hours and reports the error instead of failing the request.
    """
    hours = env_float(
        source, STATUS_MAX_WINDOW_ENV, STATUS_MAX_WINDOW.total_seconds() / 3600.0)
    return dt.timedelta(hours=hours)


def _component_names(row: Mapping[str, Any]) -> list[str]:
    """Short component labels from a scheduled-maintenance object.

    statuspage.io puts the group's display prefix on incident-update
    ``affected_components`` ("Kraken API - Embed"). That prefix contains
    "API", so Embed-only work would look like an API outage. The
    ``components`` array carries the real label ("Embed", "REST", "FIX").
    """
    components = row.get("components")
    if not isinstance(components, list):
        return []
    names: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        name = component.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _maintenance_affects_trading(row: Mapping[str, Any]) -> bool:
    """True when this status entry is about placing or reaching orders.

    Delistings never qualify, even if a component label happens to match:
    the Sep 2026 delisting notice ran for months with an empty component
    list and was blocking every new entry. Funding, deposits, withdrawals,
    and a single asset or network do not qualify either. A trading
    component still counts when the title is vague ("Derivatives Platform
    Maintenance" whose components are REST API / Websocket API / FIX API).
    """
    name = str(row.get("name") or "")
    if re.search(r"delist", name, re.IGNORECASE):
        return False
    for component in _component_names(row):
        if _ASSET_OR_FUNDING_COMPONENT.search(component):
            continue
        if _TRADING_MARK.search(component):
            return True
    if _NON_TRADING_NOTICE.search(name):
        return False
    return _TRADING_MARK.search(name) is not None


def windows_from_status_payload(
    payload: Any,
    *,
    pad_before: dt.timedelta = STATUS_PAD_BEFORE,
    pad_after: dt.timedelta = STATUS_PAD_AFTER,
    max_window: dt.timedelta = STATUS_MAX_WINDOW,
) -> list[BlackoutWindow]:
    """Kraken / statuspage.io scheduled-maintenance documents.

    Each trading maintenance is widened by the padding. The length cap is
    applied to the published ``scheduled_for`` / ``scheduled_until`` span,
    before padding. An item without both timestamps is skipped rather than
    given an invented end.

    ``in_progress`` items whose ``scheduled_until`` has passed stay ended
    at ``scheduled_until`` plus ``pad_after``. Kraken leaves API maintenance
    marked in progress after the window; stretching those to "now" would
    block entries until somebody edits the status page.
    """
    if not isinstance(payload, dict):
        return []
    rows = payload.get("scheduled_maintenances")
    if not isinstance(rows, list):
        return []
    out: list[BlackoutWindow] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            start = _parse_instant(row.get("scheduled_for"), "scheduled_for")
            # in_progress does not extend past scheduled_until. The pad below
            # is the only grace after the published end.
            end = _parse_instant(row.get("scheduled_until"), "scheduled_until")
        except ValueError:
            continue
        if end <= start or (end - start) > max_window:
            continue
        if not _maintenance_affects_trading(row):
            continue
        name = str(row.get("name") or row.get("id") or "")
        try:
            out.append(BlackoutWindow(
                start=start - pad_before,
                end=end + pad_after,
                source="kraken_status",
                name=name,
            ))
        except ValueError:
            continue
    return out


def cached_status_window_applies(
    window: BlackoutWindow,
    *,
    max_window: dt.timedelta = STATUS_MAX_WINDOW,
) -> bool:
    """Whether a status window already stored by the engine still counts.

    The cache has no component list. Drop what the parser would no longer
    write: a span longer than the cap plus padding, a delisting, or a
    funding / deposit / asset notice whose title is not a trading maintenance.
    A short window the engine stored because of its components (the title
    need not repeat "API") is kept.
    """
    limit = max_window + STATUS_PAD_BEFORE + STATUS_PAD_AFTER
    if window.end - window.start > limit:
        return False
    if re.search(r"delist", window.name, re.IGNORECASE):
        return False
    if _NON_TRADING_NOTICE.search(window.name) and not _TRADING_MARK.search(window.name):
        return False
    return True


def fetch_status_windows(
    opener: Optional[Callable[[str], Any]] = None,
    *,
    urls: Sequence[str] = STATUS_URLS,
    timeout_s: float = 3.0,
    max_window: Optional[dt.timedelta] = None,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[list[BlackoutWindow], Optional[str]]:
    """Best-effort status page read. Any failure returns no windows.

    ``opener`` is the test seam: it receives a URL and returns parsed JSON.
    The default opener is HTTP GET. A failure here must not become a blackout.
    ``max_window`` wins when given; otherwise ``env`` supplies
    ``KRAKEN_STATUS_MAX_WINDOW_H``. Neither falls through to ``os.environ``,
    so a test that does not pass them stays on the 12 hour default.
    """
    if max_window is None:
        max_window = status_max_window(env) if env is not None else STATUS_MAX_WINDOW
    fetch = opener or _http_json
    windows: list[BlackoutWindow] = []
    errors: list[str] = []
    for url in urls:
        try:
            payload = fetch(url) if opener is not None else fetch(url, timeout_s)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        windows.extend(windows_from_status_payload(payload, max_window=max_window))
    # Duplicate incidents listed on both feeds collapse to one window.
    unique: list[BlackoutWindow] = []
    seen: set[tuple[str, str, str]] = set()
    for window in windows:
        key = (window.start.isoformat(), window.end.isoformat(), window.name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)
    if unique:
        return unique, None
    return [], ("; ".join(errors) if errors else None)


def _http_json(url: str, timeout_s: float = 3.0) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "kite-trader-safety"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except urllib.error.URLError:
        raise
    return json.loads(body.decode("utf-8"))


def active_windows(now: dt.datetime, windows: Sequence[BlackoutWindow]) -> list[BlackoutWindow]:
    moment = _aware(now, "now")
    return [w for w in windows if w.contains(moment)]


def next_window(now: dt.datetime,
                windows: Sequence[BlackoutWindow]) -> Optional[BlackoutWindow]:
    """The open blackout if one contains ``now``, else the soonest future one."""
    moment = _aware(now, "now")
    current = active_windows(moment, windows)
    if current:
        return min(current, key=lambda w: w.start)
    upcoming = [w for w in windows if w.start > moment]
    if not upcoming:
        return None
    return min(upcoming, key=lambda w: w.start)


def blackout_active(now: dt.datetime, windows: Sequence[BlackoutWindow]) -> bool:
    return bool(active_windows(now, windows))


def blackout_causes(now: dt.datetime,
                    windows: Sequence[BlackoutWindow]) -> list[dict[str, str]]:
    """Why entries are blocked right now, one object per active window.

    ``source`` is ``configured`` for ``BLACKOUT_WINDOWS`` and ``status_page``
    for a Kraken status incident. ``name`` is the incident or window name.
    Empty when nothing contains ``now`` — a future ``next_window`` is not a
    cause.
    """
    moment = _aware(now, "now")
    causes: list[dict[str, str]] = []
    active = sorted(active_windows(moment, windows),
                    key=lambda w: (w.start, w.name, w.source))
    for window in active:
        kind = "status_page" if window.source == "kraken_status" else "configured"
        causes.append({"source": kind, "name": window.name})
    return causes


def heartbeat_age_s(now: dt.datetime, beat: Optional[dt.datetime]) -> Optional[float]:
    if beat is None:
        return None
    if beat.tzinfo is None or beat.utcoffset() is None:
        return None
    return (now - beat.astimezone(UTC)).total_seconds()


def entry_block_reason(
    *,
    key_freeze: bool,
    blackout: bool,
    heartbeat_age: Optional[float],
    entry_max_age_s: float,
    stop_halt: bool,
) -> Optional[str]:
    """The single reason new entries are closed, highest severity first.

    Key freeze outranks a missing stop, which outranks a calendar blackout,
    which outranks a stale loop. An unknown heartbeat (no beat yet) is not
    stale: the first tick has not had a chance to stamp one.
    """
    if key_freeze:
        return REASON_KEY_FREEZE
    if stop_halt:
        return REASON_STOP
    if blackout:
        return REASON_BLACKOUT
    if heartbeat_age is not None and heartbeat_age > entry_max_age_s:
        return REASON_STALE
    return None


def entry_block_reasons(
    *,
    key_freeze: bool,
    blackout: bool,
    heartbeat_age: Optional[float],
    entry_max_age_s: float,
    stop_halt: bool,
) -> list[str]:
    reasons: list[str] = []
    if key_freeze:
        reasons.append(REASON_KEY_FREEZE)
    if stop_halt:
        reasons.append(REASON_STOP)
    if blackout:
        reasons.append(REASON_BLACKOUT)
    if heartbeat_age is not None and heartbeat_age > entry_max_age_s:
        reasons.append(REASON_STALE)
    return reasons
