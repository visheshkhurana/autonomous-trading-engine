"""
The risk circuit: a drawdown / daily-loss ladder that decides how much freedom
the rest of the engine has right now.

Two rules shape every line below.

1. FAIL CLOSED. Two things must be true before the circuit is willing to judge
   anything: equity must be a real number, and the stored high-water state must
   have been read back intact. If either is missing the circuit says so
   explicitly (`equity_known` / `state_known` are False, the fractions are None)
   and blocks new entries. It never substitutes a plausible-looking number for
   a missing one. Unknown equity is not zero equity; a failed state read is not
   a fresh account.

2. read() IS PURE, advance() IS THE ONLY WRITER. Reading the circuit happens on
   every tick, from the UI, from health checks. If reading also persisted the
   current equity as the new baseline, a single bad equity read would be written
   over the real high-water mark and the drawdown would vanish for good — the
   circuit would then be permanently disarmed while reporting "ok". That is the
   worst failure this module has, so the write path is separated out and is
   never reached from read().

   The same reasoning is why advance() refuses to write when it could not first
   read the existing state: overwriting an unverified high-water mark erases
   drawdown history silently.

UNITS: every threshold and every reported figure in this module is a FRACTION
(Decimal("0.05") == 5%). The dataclass fields are named `*_pct` because that is
the name the rest of the engine uses; they are still fractions. Nothing here
multiplies by 100 except the human-readable `reason` string.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Callable, Mapping, Optional, Protocol, runtime_checkable

from .profiles import AssetClassProfile, profile_for
from .protocols import PositionStoreProto
from .types import AssetClass, Mode

ZERO = Decimal("0")
ONE = Decimal("1")
LOG = logging.getLogger("trader.risk")
_BASIS_FULL = "full"
_BASIS_ENGINE = "engine_only"


def equity_for_drawdown(
    equity: Decimal,
    reserved_value: Decimal = ZERO,
    *,
    exclude_reserved: bool = False,
) -> Decimal:
    """Equity the drawdown ladder should see.

    ``exclude_reserved`` drops the marked value of ``MANUAL_RESERVED_BASE``.
    The high-water mark and the ladder thresholds are not modified here;
    callers pass the returned number into ``advance`` / ``read`` as the
    account value. The default leaves equity unchanged.
    """
    if not exclude_reserved:
        return equity
    if not reserved_value.is_finite() or reserved_value < ZERO:
        raise ValueError(f"reserved_value must be finite and >= 0, got {reserved_value}")
    net = equity - reserved_value
    if net < ZERO:
        return ZERO
    return net


class CircuitAction(str, Enum):
    """What the rest of the engine is allowed to do.

    ok       — normal operation.
    reduce   — new entries allowed, but sized down by `size_multiplier`.
    pause    — no new entries. Existing positions may still be managed/exited;
               pausing is deliberately NOT flattening, because the most common
               reason to reach pause is that we cannot see clearly, and dumping
               a book on unverified numbers is its own disaster.
    flatten  — close everything, and open nothing.
    """
    OK = "ok"
    REDUCE = "reduce"
    PAUSE = "pause"
    FLATTEN = "flatten"


# Ordered worst-last. Used to combine the drawdown ladder with the daily-loss
# ladder: the stricter of the two always wins.
_SEVERITY: dict[CircuitAction, int] = {
    CircuitAction.OK: 0,
    CircuitAction.REDUCE: 1,
    CircuitAction.PAUSE: 2,
    CircuitAction.FLATTEN: 3,
}

_ENTRY_ACTIONS = frozenset({CircuitAction.OK, CircuitAction.REDUCE})


def _worse(a: CircuitAction, b: CircuitAction) -> CircuitAction:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


def _as_pct(fraction: Decimal) -> str:
    return f"{fraction * 100:.2f}%"


@dataclass(frozen=True)
class CircuitDecision:
    """The circuit's verdict, plus how much of it is actually known.

    `equity_known` / `state_known` are not diagnostics — they are part of the
    verdict. A caller that ignores them and looks only at `drawdown_pct` would
    read None, and None is not a small number. `entries_allowed` is the property
    every caller should use.
    """
    action: CircuitAction
    drawdown_pct: Optional[Decimal]     # fraction below the high-water mark
    daily_loss_pct: Optional[Decimal]   # fraction below the day's opening equity
    reason: str
    equity_known: bool
    state_known: bool
    # Multiplier the gate should apply to position size. Defaults to 0 so that
    # a decision constructed carelessly denies rather than permits.
    size_multiplier: Decimal = ZERO

    def __post_init__(self) -> None:
        # These invariants make the fail-OPEN shape unconstructible: no object
        # can claim tradable status while admitting it could not read equity or
        # state, and no object can report a drawdown number it did not compute.
        known = self.equity_known and self.state_known
        if not known:
            if self.action in _ENTRY_ACTIONS:
                raise ValueError(
                    f"cannot return {self.action.value} with "
                    f"equity_known={self.equity_known} state_known={self.state_known}: "
                    "unknown inputs must block new entries"
                )
            if self.drawdown_pct is not None or self.daily_loss_pct is not None:
                raise ValueError(
                    "drawdown/daily-loss must be None when equity or state is unknown; "
                    "a number here would be read as a measured value"
                )
            if self.size_multiplier != ZERO:
                raise ValueError("size_multiplier must be 0 when inputs are unknown")
        if self.size_multiplier < ZERO or self.size_multiplier > ONE:
            raise ValueError(f"size_multiplier out of range: {self.size_multiplier}")
        if self.action not in _ENTRY_ACTIONS and self.size_multiplier != ZERO:
            raise ValueError(f"{self.action.value} must carry size_multiplier 0")
        if known:
            # The gate applies size_multiplier verbatim, so these two shapes are
            # controls that report themselves as active and change nothing: a
            # `reduce` that does not reduce, and an `ok` that silently throttles
            # a book nobody asked to throttle. Unconstructible, same as the
            # fail-open shapes above.
            if self.action is CircuitAction.REDUCE and not (
                    ZERO < self.size_multiplier < ONE):
                raise ValueError(
                    f"reduce must carry a size_multiplier in (0, 1), got "
                    f"{self.size_multiplier}: a reduce that does not reduce is "
                    "a decorative control"
                )
            if self.action is CircuitAction.OK and self.size_multiplier != ONE:
                raise ValueError(
                    f"ok must carry size_multiplier 1, got {self.size_multiplier}"
                )
        if not self.reason.strip():
            raise ValueError("every CircuitDecision must carry a reason")

    @property
    def entries_allowed(self) -> bool:
        return (
            self.action in _ENTRY_ACTIONS
            and self.equity_known
            and self.state_known
        )

    @property
    def must_flatten(self) -> bool:
        return self.action is CircuitAction.FLATTEN


@dataclass(frozen=True)
class RiskLimits:
    """Ladder thresholds as FRACTIONS. Configured per asset class because a 5%
    day is a catastrophe in an index book and a Tuesday in crypto."""
    reduce_drawdown: Decimal
    pause_drawdown: Decimal
    flatten_drawdown: Decimal
    reduce_daily_loss: Decimal
    pause_daily_loss: Decimal
    flatten_daily_loss: Decimal
    reduce_size_multiplier: Decimal = Decimal("0.5")

    def __post_init__(self) -> None:
        rungs = (
            ("drawdown", self.reduce_drawdown, self.pause_drawdown, self.flatten_drawdown),
            ("daily_loss", self.reduce_daily_loss, self.pause_daily_loss, self.flatten_daily_loss),
        )
        for name, reduce_at, pause_at, flatten_at in rungs:
            for label, value in (("reduce", reduce_at), ("pause", pause_at), ("flatten", flatten_at)):
                if not value.is_finite() or value <= ZERO or value > ONE:
                    raise ValueError(
                        f"{name}.{label} must be a fraction in (0, 1], got {value}. "
                        "These are fractions, not percentages: 5% is Decimal('0.05')."
                    )
            # An out-of-order ladder silently skips a rung — e.g. a pause set
            # below reduce means the book never reduces, it jumps to pause.
            if not (reduce_at <= pause_at <= flatten_at):
                raise ValueError(
                    f"{name} ladder must be non-decreasing, got "
                    f"reduce={reduce_at} pause={pause_at} flatten={flatten_at}"
                )
        if not (ZERO < self.reduce_size_multiplier < ONE):
            raise ValueError(
                f"reduce_size_multiplier must be in (0, 1), got {self.reduce_size_multiplier}"
            )


DEFAULT_LIMITS: dict[AssetClass, RiskLimits] = {
    AssetClass.EQUITY: RiskLimits(
        reduce_drawdown=Decimal("0.05"),
        pause_drawdown=Decimal("0.10"),
        flatten_drawdown=Decimal("0.15"),
        reduce_daily_loss=Decimal("0.02"),
        pause_daily_loss=Decimal("0.03"),
        flatten_daily_loss=Decimal("0.05"),
    ),
    AssetClass.CRYPTO: RiskLimits(
        reduce_drawdown=Decimal("0.08"),
        pause_drawdown=Decimal("0.15"),
        flatten_drawdown=Decimal("0.25"),
        reduce_daily_loss=Decimal("0.03"),
        pause_daily_loss=Decimal("0.05"),
        flatten_daily_loss=Decimal("0.08"),
    ),
}


@dataclass(frozen=True)
class RiskState:
    """One row of the store's risk_state table: (mode, asset_class) -> baselines.

    Validated on construction, so a corrupt row cannot become a usable baseline.
    A store that builds these from SQL will raise inside its own read, which the
    circuit reports as state_known=False — exactly the intended outcome.
    """
    mode: Mode
    asset_class: AssetClass
    high_water: Decimal
    day_key: str
    day_open_equity: Decimal
    updated_at: dt.datetime
    # ``full`` is account equity including the manual reserve. ``engine_only``
    # is the basis after RISK_EXCLUDE_RESERVED_BASE has been applied once.
    equity_basis: str = "full"
    hwm_rebase_json: str = ""

    def __post_init__(self) -> None:
        for label, value in (("high_water", self.high_water),
                             ("day_open_equity", self.day_open_equity)):
            # Both are denominators. A zero or negative baseline cannot produce
            # a meaningful fraction, and silently treating it as "no drawdown"
            # is how a circuit gets disarmed.
            if not value.is_finite() or value <= ZERO:
                raise ValueError(f"{label} must be finite and > 0, got {value}")
        if not self.day_key.strip():
            raise ValueError("day_key must be set; it is the daily-loss boundary")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")

    def as_row(self) -> dict[str, str]:
        """The persisted form. Every value is a string: Decimals are stringified
        exactly here so nothing round-trips through float on its way back into a
        drawdown comparison."""
        return {
            "mode": self.mode.value,
            "asset_class": self.asset_class.value,
            "high_water": str(self.high_water),
            "day_key": self.day_key,
            "day_open_equity": str(self.day_open_equity),
            "updated_at": self.updated_at.isoformat(),
            "equity_basis": self.equity_basis or "full",
            "hwm_rebase_json": self.hwm_rebase_json or "",
        }

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "RiskState":
        """Raises on a corrupt row. The circuit reports that as state_known=False
        rather than guessing at the missing baselines."""
        return cls(
            mode=Mode(row["mode"]),
            asset_class=AssetClass(row["asset_class"]),
            high_water=Decimal(str(row["high_water"])),
            day_key=str(row["day_key"]),
            day_open_equity=Decimal(str(row["day_open_equity"])),
            updated_at=dt.datetime.fromisoformat(str(row["updated_at"])),
            equity_basis=str(row.get("equity_basis") or "full"),
            hwm_rebase_json=str(row.get("hwm_rebase_json") or ""),
        )


@runtime_checkable
class RiskStateStore(PositionStoreProto, Protocol):
    """PositionStoreProto is frozen and does not declare risk-state access, so
    the circuit depends on this narrow extension of it instead. The concrete
    store satisfies both; the circuit only ever calls the two methods below.

    read_risk_state MUST raise on a failed read and MUST return None only when
    the row genuinely does not exist. Returning None for a failed read would
    make a database outage look like a brand-new account, which is precisely the
    confusion this module exists to prevent.

    Rows are flat str->str maps (RiskState.as_row / from_row) because that is
    what the ledger stores; the circuit owns their interpretation.
    """

    def read_risk_state(self, mode: Mode,
                        asset_class: AssetClass) -> Optional[Mapping[str, str]]: ...
    def write_risk_state(self, mode: Mode, asset_class: AssetClass,
                         state: Mapping[str, str]) -> None: ...


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class RiskCircuit:
    """Implements RiskCircuitProto.

    `mode` is fixed at construction: paper and live keep separate baselines, the
    same way every position read filters on mode. A shared high-water mark would
    let paper losses pause a live book.
    """

    def __init__(
        self,
        store: RiskStateStore,
        mode: Mode,
        limits: Optional[Mapping[AssetClass, RiskLimits]] = None,
        clock: Callable[[], dt.datetime] = _utc_now,
    ) -> None:
        self._store = store
        self._mode = mode
        self._limits: Mapping[AssetClass, RiskLimits] = dict(limits or DEFAULT_LIMITS)
        self._clock = clock

    @property
    def mode(self) -> Mode:
        return self._mode

    def limits_for(self, asset_class: AssetClass) -> RiskLimits:
        limits = self._limits.get(asset_class)
        if limits is None:
            # No default-to-permissive. An unconfigured asset class is a
            # configuration bug, and guessing thresholds for real money is worse
            # than refusing to start.
            raise KeyError(f"no RiskLimits configured for {asset_class.value}")
        return limits

    # ---------------------------------------------------------------- read

    def read(self, asset_class: AssetClass,
             equity: Optional[Decimal]) -> CircuitDecision:
        """Judge the current state. PURE: touches no writer, persists nothing.

        `equity` is total account value and MAY include unrealised P&L on open
        positions — the ladder is a mark-to-market ladder, so an unrealised
        drawdown must trip it. What it must never be is a fallback: pass None
        when the value could not be established.
        """
        now = self._now()
        limits = self.limits_for(asset_class)
        profile = profile_for(asset_class)

        equity_ok, equity_detail = _check_equity(equity)
        state, state_detail = self._load(asset_class)
        if state_detail is not None:
            return _blind(equity_known=equity_ok, state_known=False, detail=state_detail)
        if not equity_ok:
            return _blind(equity_known=False, state_known=True, detail=equity_detail or "")

        assert equity is not None  # narrowed by _check_equity
        return _judge(profile, limits, state, equity, now)

    # ------------------------------------------------------------- advance

    def advance(self, asset_class: AssetClass, equity: Decimal,
                equity_basis: str = _BASIS_FULL) -> CircuitDecision:
        """Roll the high-water mark and the daily baseline forward, then judge.

        The ONLY writer. Belongs to the order path (post-fill / post-settlement),
        not to any polling or display path.

        ``equity_basis="engine_only"`` means this equity already excludes
        ``MANUAL_RESERVED_BASE``. The first time that basis is seen against a
        high-water mark built from full equity, the mark and the day's open
        are rebased once to this equity and an audit row is stored. Later
        ticks use the normal "high-water only rises" rule. The default
        ``full`` never rebases.
        """
        now = self._now()
        limits = self.limits_for(asset_class)
        profile = profile_for(asset_class)

        # Loud, not lenient. A None or NaN equity reaching the writer is a bug
        # upstream, and the one thing we must not do is persist a baseline
        # derived from it.
        if equity is None:
            raise ValueError("advance() requires a known equity; pass None only to read()")
        if not isinstance(equity, Decimal):
            raise TypeError(f"equity must be Decimal, got {type(equity).__name__}")
        if not equity.is_finite():
            raise ValueError(f"equity must be finite, got {equity}")
        if equity <= ZERO:
            raise ValueError(
                f"refusing to advance state on equity {equity}: a zero or negative "
                "account value is a circuit event, not a baseline. read() will "
                "report it as a full drawdown."
            )

        state, state_detail = self._load(asset_class)
        if state_detail is not None:
            # THE fail-open bug, prevented: without the previous high-water mark
            # we would write today's equity as the new one, permanently erasing
            # the drawdown. Refuse to judge and refuse to persist.
            return _blind(equity_known=True, state_known=False, detail=state_detail)

        day_key = profile.day_key(now)
        basis = equity_basis if equity_basis in {_BASIS_FULL, _BASIS_ENGINE} else _BASIS_FULL
        rebase_json = ""
        if state is None:
            high_water = equity
            day_open = equity
        elif state.day_key > day_key:
            # The stored day is AHEAD of our clock. Either the clock stepped
            # backwards or this row was written by a host that disagrees about
            # the boundary; either way the daily boundary is not established,
            # and the branch below would read that disagreement as "a new day"
            # and re-baseline day_open at whatever the equity is right now —
            # erasing the day's loss at the exact moment it is largest. Refuse
            # to judge and refuse to persist, same as an unreadable row.
            return _blind(
                equity_known=True, state_known=False,
                detail=(f"stored day_key {state.day_key} is ahead of {day_key}; "
                        "the daily-loss boundary cannot be established"),
            )
        elif basis == _BASIS_ENGINE and state.equity_basis != _BASIS_ENGINE:
            # One time only. The stored mark was built from equity that
            # included the manual reserve. Replacing the input with
            # engine-only equity must not look like a drawdown. The day's
            # open is moved with the mark so the daily-loss rung does not
            # flatten on the same fictional loss. Thresholds are untouched.
            high_water = equity
            day_open = equity
            rebase_json = json.dumps({
                "at": now.isoformat(),
                "old_high_water": str(state.high_water),
                "new_high_water": str(high_water),
                "old_day_open_equity": str(state.day_open_equity),
                "new_day_open_equity": str(day_open),
                "reason": "RISK_EXCLUDE_RESERVED_BASE enabled; one-time rebase "
                          "to engine-only equity",
            })
            LOG.info(
                "risk HWM rebased once to engine-only equity "
                "old_high_water=%s new_high_water=%s",
                state.high_water, high_water,
            )
        else:
            # max(), never assignment: the high-water mark only ever goes up.
            high_water = state.high_water if state.high_water > equity else equity
            # state.day_key <= day_key here, so this only ever rolls forward.
            day_open = equity if state.day_key != day_key else state.day_open_equity
            rebase_json = state.hwm_rebase_json
            if basis != _BASIS_ENGINE:
                basis = state.equity_basis or _BASIS_FULL

        new_state = RiskState(
            mode=self._mode,
            asset_class=asset_class,
            high_water=high_water,
            day_key=day_key,
            day_open_equity=day_open,
            updated_at=now,
            equity_basis=basis if state is None or basis == _BASIS_ENGINE else (
                state.equity_basis or basis),
            hwm_rebase_json=rebase_json,
        )
        # Deliberately unguarded: a failed write must reach the order path, not
        # be smoothed into a decision that looks persisted but is not.
        self._store.write_risk_state(self._mode, asset_class, new_state.as_row())
        return _judge(profile, limits, new_state, equity, now)

    # -------------------------------------------------------------- internals

    def _now(self) -> dt.datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            # profile.day_key() calls astimezone(), which would silently read a
            # naive stamp as machine-local time and roll the daily baseline on
            # the wrong boundary.
            raise ValueError("clock must return a timezone-aware datetime")
        return now

    def _load(self, asset_class: AssetClass) -> tuple[Optional[RiskState], Optional[str]]:
        """Returns (state, failure_detail). Exactly one is meaningful.

        (None, None) means the row genuinely does not exist yet — a legitimate
        empty, distinct from a failed read, which returns (None, "...").
        """
        try:
            row = self._store.read_risk_state(self._mode, asset_class)
            # Parsing is inside the same guard on purpose: a row that cannot be
            # interpreted is no more usable than a row that could not be
            # fetched, and both must fail closed. The exception text is carried
            # into the decision's reason, so nothing is swallowed.
            state = None if row is None else RiskState.from_row(row)
        except Exception as exc:  # noqa: BLE001 - re-reported, never swallowed
            return None, f"risk_state unreadable: {type(exc).__name__}: {exc}"
        if state is None:
            return None, None
        if state.mode is not self._mode or state.asset_class is not asset_class:
            # A row for another book is worse than no row: judging a live account
            # against paper baselines is exactly the contamination the store's
            # mode filtering exists to stop.
            return None, (
                f"risk_state mismatched: asked ({self._mode.value},{asset_class.value}), "
                f"got ({state.mode.value},{state.asset_class.value})"
            )
        return state, None


def _check_equity(equity: Optional[Decimal]) -> tuple[bool, Optional[str]]:
    if equity is None:
        return False, "equity unknown (broker returned None)"
    if not isinstance(equity, Decimal):
        raise TypeError(f"equity must be Decimal or None, got {type(equity).__name__}")
    if not equity.is_finite():
        return False, f"equity not a finite number ({equity})"
    return True, None


def _blind(equity_known: bool, state_known: bool, detail: str) -> CircuitDecision:
    """The fail-closed verdict: cannot judge, so nothing new opens."""
    return CircuitDecision(
        action=CircuitAction.PAUSE,
        drawdown_pct=None,
        daily_loss_pct=None,
        reason=f"circuit blind, new entries blocked - {detail}",
        equity_known=equity_known,
        state_known=state_known,
        size_multiplier=ZERO,
    )


def _rung(value: Decimal, reduce_at: Decimal, pause_at: Decimal,
          flatten_at: Decimal) -> CircuitAction:
    if value >= flatten_at:
        return CircuitAction.FLATTEN
    if value >= pause_at:
        return CircuitAction.PAUSE
    if value >= reduce_at:
        return CircuitAction.REDUCE
    return CircuitAction.OK


def _judge(profile: AssetClassProfile, limits: RiskLimits,
           state: Optional[RiskState], equity: Decimal,
           now: dt.datetime) -> CircuitDecision:
    """Pure arithmetic over verified inputs. No I/O, no writes, no clock reads."""
    if state is None:
        # First run for this (mode, asset_class): with no history, current
        # equity IS the high-water mark. Nothing is being overwritten here —
        # this branch is only reachable when the store confirmed the row is
        # absent, never when the read failed.
        if equity <= ZERO:
            return _blind(
                equity_known=True, state_known=True,
                detail=f"no stored baseline and equity is {equity}; cannot form a fraction",
            )
        high_water = equity
        day_open = equity
        origin = "no stored baseline, using current equity"
    else:
        high_water = state.high_water
        today = profile.day_key(now)
        if state.day_key > today:
            # Same reasoning as advance(): a stored day ahead of our clock means
            # the boundary is not established, and rolling here would reset the
            # daily baseline to the current (post-loss) equity. Refusing to
            # judge is the only reading that cannot hide a loss.
            return _blind(
                equity_known=True, state_known=False,
                detail=(f"stored day_key {state.day_key} is ahead of {today}; "
                        "the daily-loss boundary cannot be established"),
            )
        # A wiped account must not reseed the daily baseline at zero: keeping
        # yesterday's baseline reports the loss instead of hiding it behind an
        # unusable denominator. read() is pure, so this roll is computed, not
        # persisted; advance() persists the same roll on the order path.
        rolled = state.day_key != today and equity > ZERO
        day_open = equity if rolled else state.day_open_equity
        origin = (
            f"day rolled to {today}, baseline reset"
            if rolled else
            f"day {state.day_key} opened at {state.day_open_equity}"
        )

    # high_water > 0 is guaranteed: RiskState validates it, and the stateless
    # branch above rejects non-positive equity.
    drawdown = (high_water - equity) / high_water
    if drawdown < ZERO:
        drawdown = ZERO   # new high; a negative drawdown is not a thing
    daily_loss = (day_open - equity) / day_open
    if daily_loss < ZERO:
        daily_loss = ZERO

    dd_action = _rung(drawdown, limits.reduce_drawdown,
                      limits.pause_drawdown, limits.flatten_drawdown)
    dl_action = _rung(daily_loss, limits.reduce_daily_loss,
                      limits.pause_daily_loss, limits.flatten_daily_loss)
    action = _worse(dd_action, dl_action)

    driver = "drawdown" if _SEVERITY[dd_action] >= _SEVERITY[dl_action] else "daily loss"
    reason = (
        f"{action.value}: drawdown {_as_pct(drawdown)} of high-water {high_water}, "
        f"daily loss {_as_pct(daily_loss)} ({origin}); driven by {driver}"
    )
    multiplier = (
        ONE if action is CircuitAction.OK
        else limits.reduce_size_multiplier if action is CircuitAction.REDUCE
        else ZERO
    )
    return CircuitDecision(
        action=action,
        drawdown_pct=drawdown,
        daily_loss_pct=daily_loss,
        reason=reason,
        equity_known=True,
        state_known=True,
        size_multiplier=multiplier,
    )
