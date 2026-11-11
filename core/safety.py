"""
Live-safety policy shared by the engine and the API.

Three independent reasons to refuse a new entry, each visible as a reason
string rather than as a silent skip:

* ``stale_heartbeat`` — the trading loop has not stamped a beat inside the
  entry threshold. Exits and stop maintenance stay allowed; a dead loop must
  not open risk, and it also must not trap a position the operator is trying
  to close.
* ``blackout`` — a configured UTC window, or a Kraken status-page maintenance
  padded 15 minutes before and 30 minutes after. Configured windows always
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


def windows_from_status_payload(
    payload: Any,
    *,
    pad_before: dt.timedelta = STATUS_PAD_BEFORE,
    pad_after: dt.timedelta = STATUS_PAD_AFTER,
) -> list[BlackoutWindow]:
    """Kraken / statuspage.io scheduled-maintenance documents.

    Each maintenance is widened by the padding. An item without both
    timestamps is skipped rather than given an invented end.
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
            end = _parse_instant(row.get("scheduled_until"), "scheduled_until")
        except ValueError:
            continue
        name = str(row.get("name") or row.get("id") or "")
        out.append(BlackoutWindow(
            start=start - pad_before,
            end=end + pad_after,
            source="kraken_status",
            name=name,
        ))
    return out


def fetch_status_windows(
    opener: Optional[Callable[[str], Any]] = None,
    *,
    urls: Sequence[str] = STATUS_URLS,
    timeout_s: float = 3.0,
) -> tuple[list[BlackoutWindow], Optional[str]]:
    """Best-effort status page read. Any failure returns no windows.

    ``opener`` is the test seam: it receives a URL and returns parsed JSON.
    The default opener is HTTP GET. A failure here must not become a blackout.
    """
    fetch = opener or _http_json
    windows: list[BlackoutWindow] = []
    errors: list[str] = []
    for url in urls:
        try:
            payload = fetch(url) if opener is not None else fetch(url, timeout_s)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        windows.extend(windows_from_status_payload(payload))
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
