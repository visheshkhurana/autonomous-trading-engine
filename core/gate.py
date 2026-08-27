"""
OrderGate — the single chokepoint every order passes through.

Contract for everyone downstream: **no adapter method that touches a venue may
be called without an Approval from this gate, and the Approval's `mode` is the
only authority on where the order executes.** A caller that reads
`intent`/`config`/its own flags to decide paper-vs-live has reintroduced the
bug this class exists to prevent.

Two behaviours here are the direct scar tissue of the previous system:

* The daily order counter is incremented ONLY after an order is approved.
  Both old gates burned the daily quota on rejected orders, so a strategy that
  spammed rejects locked out the orders that would have been fine.
* No state is ever reconstructed from memory when a store cannot be read. A
  store that fails makes the gate REFUSE, never approve on assumed numbers.
  The old gates degraded to per-process dictionaries, which meant every cap —
  dedup, daily count, exposure — was decorative in production.

The same rule applies to every input: equity that cannot be established, a
circuit state that cannot be interpreted, an order-count read that comes back
as something other than an int — all of these refuse. A failure is never
allowed to look like a permissive zero.

Two things the Approval carries that callers must actually use, rather than
reading the original Intent: `Approval.mode`, and `Approval.intent`, whose qty
is the SNAPPED and (where the circuit says so) THROTTLED size. Placing
`intent.qty` instead makes quantization and the circuit's size_multiplier
decorative.

Ordering note: the eleven checks run in the documented order and stop at the
FIRST failure, because later checks depend on facts established by earlier ones
(a notional cap needs a quantized qty and a price). A twelfth — the structure
leg rule — sits between them and is reported only for an order that carries a
structure_id, since it has nothing to say about an outright. The mode coercion is
reported last, as the closing statement of the check list, but the *effective*
mode is computed before any store read — position and exposure caps have to be
evaluated against the book the order will actually land in, and asking the
store for the live book while the class is disarmed would measure the wrong
account. `Approval.mode` therefore carries the coerced mode even on rejection.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field, replace
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Final, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .profiles import AssetClassProfile, profile_for
from .protocols import PositionStoreProto, RiskCircuitProto
from .types import (
    Approval, AssetClass, CheckResult, Intent, MarketContext, Mode,
    OrderType, Position, Side,
)

__all__ = [
    "ClassLimits",
    "GateConfig",
    "GateStateStoreProto",
    "GateStateUnavailable",
    "OrderGate",
]


class GateStateUnavailable(RuntimeError):
    """Raised by a GateStateStore that cannot answer.

    Stores MUST raise rather than return a placeholder: `False` from a dedup
    read means "this key is new" and `0` from a count read means "nothing
    traded today". Both are legitimate values, so neither can double as an
    error signal.
    """


@runtime_checkable
class GateStateStoreProto(Protocol):
    """Durable state that only the gate owns: dedup keys and the daily order
    count. Separate from PositionStoreProto because it must survive a process
    restart independently of fills — a gate that forgets its counters on deploy
    has no caps at all.

    Every method raises (ideally GateStateUnavailable) when it cannot read or
    write. The gate turns any raise into a refusal.

    An in-process dict satisfies this Protocol and MUST NOT be used outside
    tests. It looks like it works, and then each worker keeps its own counters:
    N workers means N times the daily cap, and a retry that lands on a
    different worker places a second real order. That is precisely how the
    previous system's caps became decorative.
    """

    def seen_recently(self, idempotency_key: str, window_s: float,
                      now: dt.datetime) -> bool:
        """True if this key was already submitted within the window."""
        ...

    def order_count(self, asset_class: AssetClass, day_key: str) -> int:
        """Approved orders recorded for this class on this trading day."""
        ...

    def record_submission(self, idempotency_key: str, asset_class: AssetClass,
                          day_key: str, now: dt.datetime) -> None:
        """Claim the key and increment the day's count. Called by the gate
        ONLY after every check has passed."""
        ...


@dataclass(frozen=True)
class ClassLimits:
    """Per-asset-class caps. Fractions are of LIVE equity, never of a constant:
    a cap denominated in rupees stops being a risk limit the moment the account
    halves."""

    # Largest single order, as a fraction of equity.
    max_order_notional_fraction: Decimal = Decimal("0.10")
    # Largest total exposure for the class, as a fraction of equity.
    max_allocation_fraction: Decimal = Decimal("0.30")
    max_open_positions: int = 5
    max_orders_per_day: int = 20
    # Floor applied on top of Instrument.min_notional, in the class's base
    # currency. Dust orders pay fees and move nothing.
    min_notional_floor: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for name in ("max_order_notional_fraction", "max_allocation_fraction"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise TypeError(f"{name} must be Decimal, got {type(value).__name__}")
            if not (Decimal("0") < value <= Decimal("1")):
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if self.max_order_notional_fraction > self.max_allocation_fraction:
            raise ValueError(
                "max_order_notional_fraction exceeds max_allocation_fraction: "
                "a single order could never be placed without breaching the "
                "class allocation cap"
            )
        for name in ("max_open_positions", "max_orders_per_day"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not isinstance(self.min_notional_floor, Decimal):
            raise TypeError("min_notional_floor must be Decimal")
        if self.min_notional_floor < 0:
            raise ValueError("min_notional_floor must be >= 0")


@dataclass(frozen=True)
class GateConfig:
    """Caps and the mode the engine *wants*. Arming is NOT here — it comes from
    the environment so that going live is an operator action, not a code
    change."""

    # What the engine asks for. PAPER by default: a config that forgets to say
    # anything must not trade real money.
    requested_mode: Mode = Mode.PAPER
    dedup_window_s: float = 300.0
    default_limits: ClassLimits = ClassLimits()
    limits: Mapping[AssetClass, ClassLimits] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.requested_mode, Mode):
            raise TypeError("requested_mode must be a Mode")
        if not isinstance(self.dedup_window_s, (int, float)) or isinstance(
                self.dedup_window_s, bool):
            raise TypeError("dedup_window_s must be a number")
        if self.dedup_window_s <= 0:
            raise ValueError(
                "dedup_window_s must be > 0; a zero window disables dedup and "
                "lets a retried decision place a second real order"
            )
        if not isinstance(self.default_limits, ClassLimits):
            raise TypeError("default_limits must be a ClassLimits")
        for key, value in self.limits.items():
            if not isinstance(key, AssetClass):
                raise TypeError(f"limits key must be an AssetClass, got {key!r}")
            if not isinstance(value, ClassLimits):
                raise TypeError(f"limits[{key}] must be a ClassLimits")
        # Copy + freeze so caps cannot be widened after the gate is built.
        object.__setattr__(self, "limits", MappingProxyType(dict(self.limits)))

    def limits_for(self, asset_class: AssetClass) -> ClassLimits:
        return self.limits.get(asset_class, self.default_limits)


# Arming is per asset class so a crypto experiment can never arm the equity
# book by accident.
_ARM_ENV: Final[Mapping[AssetClass, str]] = MappingProxyType({
    AssetClass.EQUITY: "EQUITY_LIVE",
    AssetClass.CRYPTO: "CRYPTO_LIVE",
})

_ENV_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSE: Final[frozenset[str]] = frozenset({"", "0", "false", "no", "off"})

# Circuit states, read structurally so this module does not import the risk
# module — the two are built and tested independently, and the gate must keep
# failing closed even if the circuit is swapped for another implementation.
# Anything not listed refuses: a state the gate cannot interpret is not
# evidence that trading is safe.
_CIRCUIT_BLOCKING: Final[frozenset[str]] = frozenset({
    "flatten", "flattening", "pause", "paused", "halt", "halted", "stop",
    "stopped", "locked", "tripped", "breached",
})
_CIRCUIT_REDUCE: Final[frozenset[str]] = frozenset({
    "reduce", "reduce_only", "reducing", "derisk", "throttle", "throttled",
})
_CIRCUIT_NORMAL: Final[frozenset[str]] = frozenset({
    "normal", "ok", "clear", "healthy", "active", "armed", "running", "none",
})

_CHECK_SANITY: Final[str] = "sanity"
_CHECK_MARKET: Final[str] = "market_open"
_CHECK_QUANT: Final[str] = "quantization"
_CHECK_MIN_NOTIONAL: Final[str] = "min_notional"
_CHECK_DEDUP: Final[str] = "dedup"
_CHECK_NOTIONAL_CAP: Final[str] = "notional_cap"
_CHECK_ALLOCATION: Final[str] = "allocation_cap"
_CHECK_OPEN_POSITIONS: Final[str] = "max_open_positions"
_CHECK_STRUCTURE: Final[str] = "structure_legs"
_CHECK_DAILY_ORDERS: Final[str] = "daily_order_cap"
_CHECK_CIRCUIT: Final[str] = "risk_circuit"
_CHECK_COERCION: Final[str] = "mode_coercion"


def _parse_arm_flag(var: str, raw: Optional[str]) -> bool:
    """Unset or empty means disarmed. An unrecognised value RAISES instead of
    falling back to disarmed: silently reading `EQUITY_LIVE=ture` as False
    would look identical to a correctly disarmed engine, and the operator would
    never learn the flag never took effect."""
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    raise ValueError(
        f"{var}={raw!r} is not a boolean. Use one of "
        f"{sorted(_ENV_TRUE)} or {sorted(_ENV_FALSE)}."
    )


def _circuit_token(decision: Any) -> Optional[str]:
    """Normalised state word from a CircuitDecision, or None if it cannot be
    read. None means refuse; it never means normal."""
    for attr in ("action", "state", "level", "status", "verdict"):
        raw = getattr(decision, attr, None)
        if raw is None:
            continue
        value = getattr(raw, "value", raw)   # unwrap Enum
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _circuit_allows_entries(decision: Any, token: Optional[str]) -> Optional[bool]:
    """Whether the circuit permits a NEW entry. None means unreadable.

    Two independent sources of NO, and either one is enough:

    * a blocking state word. It is checked FIRST and is not overridable. A
      decision that says "flatten" while also setting entries_allowed=True is
      internally inconsistent, and the safe reading of an inconsistent circuit
      is the strict one. core.risk cannot construct that shape, but this module
      reads the circuit structurally precisely so it keeps failing closed when
      the circuit is swapped for another implementation — trusting the flag
      over the word was a hole straight through that promise.
    * `entries_allowed=False`, which carries what the word cannot: core.risk
      folds "I could not read equity" into that property, and a gate that only
      pattern-matched the word would trade on unverified numbers.
    """
    if token in _CIRCUIT_BLOCKING:
        return False
    explicit = getattr(decision, "entries_allowed", None)
    if isinstance(explicit, bool):
        return explicit
    if token is None:
        return None
    if token in _CIRCUIT_REDUCE or token in _CIRCUIT_NORMAL:
        return True
    return None


def _circuit_multiplier(decision: Any) -> tuple[Optional[Decimal], Optional[str]]:
    """(multiplier, failure reason). A decision that carries a size_multiplier
    means it, so the gate applies it — an advertised throttle that nobody
    applies is exactly the kind of decorative control this engine was rebuilt
    to remove. Absent is fine (no throttle); present-but-nonsense refuses."""
    raw = getattr(decision, "size_multiplier", None)
    if raw is None:
        return None, None
    if not isinstance(raw, Decimal) or not raw.is_finite():
        return None, f"size_multiplier {raw!r} is not a finite Decimal"
    if not (Decimal("0") <= raw <= Decimal("1")):
        return None, f"size_multiplier {raw} is outside [0, 1]"
    return raw, None


def _circuit_reason(decision: Any) -> str:
    for attr in ("reason", "detail", "message", "note"):
        raw = getattr(decision, attr, None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _positive_decimal(value: Any) -> Optional[Decimal]:
    """Decimal > 0, or None. Rejects bool/int/float deliberately: a float
    price silently loses precision against Decimal quantities, and the callers
    of this system all speak Decimal.

    is_finite() is tested BEFORE the comparison: `Decimal("NaN") > 0` raises
    InvalidOperation, which would escape check() as an exception instead of the
    verdict this class promises to return.
    """
    if isinstance(value, Decimal) and value.is_finite() and value > 0:
        return value
    return None


class OrderGate:
    """The chokepoint. Construct once with its collaborators; the only public
    method is check().

    Collaborators are private and the instance is sealed after construction, so
    a caller cannot swap in a permissive store or widen a cap at runtime. That
    is not paranoia about colleagues — it is what makes "every order passed the
    gate" a statement you can trust when reading an incident timeline.
    """

    __slots__ = ("__positions", "__state", "__circuit", "__config", "__armed",
                 "__sealed")

    def __init__(
        self,
        *,
        position_store: PositionStoreProto,
        gate_state: GateStateStoreProto,
        risk_circuit: RiskCircuitProto,
        config: GateConfig,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not isinstance(config, GateConfig):
            raise TypeError("config must be a GateConfig")
        source = os.environ if env is None else env
        # Read once, at construction. A gate whose arming can change under it
        # mid-session cannot be reasoned about from a log line.
        armed: dict[AssetClass, bool] = {}
        for asset_class, var in _ARM_ENV.items():
            armed[asset_class] = _parse_arm_flag(var, source.get(var))
        self.__positions = position_store
        self.__state = gate_state
        self.__circuit = risk_circuit
        self.__config = config
        self.__armed = MappingProxyType(armed)
        self.__sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_OrderGate__sealed", False):
            raise AttributeError(
                "OrderGate is sealed after construction; caps, stores and "
                "arming cannot be replaced at runtime"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("OrderGate is sealed after construction")

    # ---------------------------------------------------------------- public

    def check(self, intent: Intent, ctx: MarketContext) -> Approval:
        """Run every gate check in order and return the verdict.

        Nothing here places, modifies or cancels anything. On rejection the
        dedup key is NOT claimed and the daily counter is NOT incremented, so a
        rejected order costs the strategy nothing but the log line.
        """
        asset_class = intent.instrument.asset_class
        limits = self.__config.limits_for(asset_class)
        # Effective mode first: every store read below must query the book this
        # order would really land in. Reported as the last CheckResult.
        mode = self.__effective_mode(asset_class)
        checks: list[CheckResult] = []

        def reject(name: str, detail: str, effective: Intent = intent) -> Approval:
            checks.append(CheckResult(name, False, detail))
            return Approval(False, mode, effective, tuple(checks))

        def passed(name: str, detail: str) -> None:
            checks.append(CheckResult(name, True, detail))

        try:
            profile = profile_for(asset_class)
        except KeyError:
            return reject(_CHECK_SANITY, f"no profile for asset class {asset_class!r}")

        # 1. sanity ---------------------------------------------------------
        sanity = self.__sanity(intent, ctx, profile)
        if sanity is not None:
            return reject(_CHECK_SANITY, sanity)
        passed(_CHECK_SANITY, f"{intent.side.value} {intent.qty} "
                              f"{intent.instrument.symbol} as {intent.order_type.value}")

        # 2. market open ----------------------------------------------------
        if not profile.is_open(ctx.now):
            return reject(_CHECK_MARKET,
                          f"{asset_class.value} market closed at {ctx.now.isoformat()}")
        passed(_CHECK_MARKET, f"open at {ctx.now.isoformat()}")

        # 3. quantization ---------------------------------------------------
        qty = profile.quantize_qty(intent.instrument, intent.qty)
        if qty <= 0:
            return reject(
                _CHECK_QUANT,
                f"qty {intent.qty} snaps to {qty} for lot_size="
                f"{intent.instrument.lot_size} precision="
                f"{intent.instrument.qty_precision}; nothing to place",
            )
        if profile.quantize_qty(intent.instrument, qty) != qty:
            # Quantization must be idempotent. If it is not, the instrument's
            # lot_size/precision disagrees with the profile and the qty that
            # reaches the venue is not the qty we sized.
            return reject(_CHECK_QUANT,
                          f"qty {qty} is not stable under quantization for "
                          f"{intent.instrument.symbol}")
        # The Approval carries the SNAPPED intent. Handing back the original
        # would make this check decorative — the executor would place the
        # unsnapped size.
        effective = intent if qty == intent.qty else replace(intent, qty=qty)
        passed(_CHECK_QUANT,
               f"qty {intent.qty} -> {qty}" if qty != intent.qty else f"qty {qty} exact")

        price = self.__reference_price(effective, ctx)
        if price is None:
            return reject(
                _CHECK_MIN_NOTIONAL,
                f"no usable price for {effective.instrument.symbol}: a "
                f"{effective.order_type.value} order cannot be size-checked "
                "without one",
                effective,
            )
        notional = qty * price

        # 4. min notional / min lot ----------------------------------------
        floor = max(effective.instrument.min_notional, limits.min_notional_floor)
        if notional < floor:
            return reject(_CHECK_MIN_NOTIONAL,
                          f"notional {notional} below minimum {floor}", effective)
        passed(_CHECK_MIN_NOTIONAL, f"notional {notional} >= {floor}")

        # 5. dedup ----------------------------------------------------------
        key = effective.idempotency_key.strip()
        if not key:
            # Without a key the gate cannot tell a retry from a new decision,
            # so a retried order becomes a second real position.
            return reject(_CHECK_DEDUP,
                          "intent has no idempotency_key; the gate cannot "
                          "distinguish a retry from a new decision", effective)
        try:
            seen = self.__state.seen_recently(key, self.__config.dedup_window_s, ctx.now)
        except Exception as exc:                      # fail CLOSED, never to memory
            return reject(_CHECK_DEDUP,
                          f"dedup store unavailable: {exc!r}", effective)
        if not isinstance(seen, bool):
            return reject(_CHECK_DEDUP,
                          f"dedup store returned {seen!r}, expected bool", effective)
        if seen:
            return reject(_CHECK_DEDUP,
                          f"idempotency_key {key!r} already submitted within "
                          f"{self.__config.dedup_window_s}s", effective)
        passed(_CHECK_DEDUP, f"{key!r} not seen in {self.__config.dedup_window_s}s")

        # Equity underpins checks 6 and 7 and is passed to the circuit. None is
        # "could not be established" and must refuse — treating it as 0 reads
        # as a 100% drawdown, treating it as a default reads as unlimited room.
        equity = _positive_decimal(ctx.equity)
        if equity is None:
            return reject(
                _CHECK_NOTIONAL_CAP,
                f"live equity unavailable (got {ctx.equity!r}); caps are a "
                "fraction of equity and cannot be evaluated",
                effective,
            )

        try:
            book = tuple(self.__positions.positions(mode, asset_class))
        except Exception as exc:                      # fail CLOSED
            return reject(_CHECK_ALLOCATION,
                          f"position store unavailable: {exc!r}", effective)

        reducing = _is_reducing(effective, book)

        # 6. notional cap ---------------------------------------------------
        notional_cap = equity * limits.max_order_notional_fraction
        if reducing:
            passed(_CHECK_NOTIONAL_CAP,
                   "reducing an open position; order-size cap not applied")
        elif notional > notional_cap:
            return reject(
                _CHECK_NOTIONAL_CAP,
                f"notional {notional} exceeds {limits.max_order_notional_fraction} "
                f"of equity {equity} ({notional_cap})",
                effective,
            )
        else:
            passed(_CHECK_NOTIONAL_CAP, f"notional {notional} <= {notional_cap}")

        # 7. per-asset-class allocation cap ---------------------------------
        allocation_cap = equity * limits.max_allocation_fraction
        exposure, unmarkable = _exposure(book, ctx)
        if reducing:
            passed(_CHECK_ALLOCATION,
                   f"reducing; class exposure "
                   f"{'unknown' if exposure is None else exposure} will fall")
        elif exposure is None:
            # An exposure that cannot be measured cannot be added to. Only
            # entries are refused; the branch above keeps every exit open.
            return reject(_CHECK_ALLOCATION, unmarkable or "exposure unknown",
                          effective)
        elif exposure + notional > allocation_cap:
            return reject(
                _CHECK_ALLOCATION,
                f"{asset_class.value} exposure {exposure} + {notional} exceeds "
                f"{limits.max_allocation_fraction} of equity {equity} "
                f"({allocation_cap})",
                effective,
            )
        else:
            passed(_CHECK_ALLOCATION,
                   f"exposure {exposure} + {notional} <= {allocation_cap}")

        # 8. max open positions ---------------------------------------------
        held = {p.instrument.symbol for p in book if p.qty != 0}
        if effective.instrument.symbol in held:
            passed(_CHECK_OPEN_POSITIONS,
                   f"{effective.instrument.symbol} already open; no new slot needed")
        elif len(held) >= limits.max_open_positions:
            return reject(
                _CHECK_OPEN_POSITIONS,
                f"{len(held)} open {asset_class.value} positions in {mode.value}, "
                f"cap is {limits.max_open_positions}",
                effective,
            )
        else:
            passed(_CHECK_OPEN_POSITIONS,
                   f"{len(held)}/{limits.max_open_positions} slots used")

        # 9. structure legs --------------------------------------------------
        # A multi-leg structure is placed leg by leg and each leg passes this
        # gate on its own, so any check can refuse one leg after another has
        # already reached the venue. What is left behind decides how bad that
        # is: a lone long leg is a paid debit with a known maximum loss, a lone
        # short leg is unbounded. So the rule is that a short leg may never
        # exceed the long protection its own structure already has on.
        #
        # That covers both halves of the failure. The short leg cannot go on
        # FIRST (no protection yet, so nothing to lean on), and it cannot go on
        # in full against a PARTIALLY filled protective leg.
        #
        # This is not hypothetical ordering pedantry. On a debit spread the
        # long leg is always the more expensive one, so the order-notional cap
        # rejects the protective leg and approves the naked one every time the
        # spread is slightly too large for the cap.
        #
        # Structures the rule deliberately refuses: anything short by
        # construction (a short strangle, a 1x2 front spread), and anything
        # whose hedge is held outside the structure_id (a call written against
        # an outright long). Tag the hedge with the same structure_id, or leave
        # the structure_id off and let the outright caps judge the order.
        #
        # Reported only for an order that carries a structure_id: a "not
        # applicable" line on every outright order is noise in the one list an
        # operator reads during an incident.
        structure = (effective.structure_id or "").strip()
        if structure:
            if effective.side is Side.SELL and not reducing:
                protection, already_short, legs = _structure_sides(book, structure)
                if already_short + qty > protection:
                    return reject(
                        _CHECK_STRUCTURE,
                        f"{effective.instrument.symbol} would take structure "
                        f"{structure!r} to {already_short + qty} short against "
                        f"{protection} long in {mode.value} ({legs} leg(s) open); "
                        "a half-placed structure must not be left as the naked "
                        "half",
                        effective,
                    )
                passed(_CHECK_STRUCTURE,
                       f"structure {structure!r}: {already_short + qty} short "
                       f"covered by {protection} long across {legs} open leg(s)")
            else:
                passed(_CHECK_STRUCTURE,
                       f"structure {structure!r}: this leg does not open a short")

        # 10. daily order count ----------------------------------------------
        day_key = profile.day_key(ctx.now)
        try:
            count = self.__state.order_count(asset_class, day_key)
        except Exception as exc:                      # fail CLOSED
            return reject(_CHECK_DAILY_ORDERS,
                          f"order-count store unavailable: {exc!r}", effective)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return reject(
                _CHECK_DAILY_ORDERS,
                f"order-count store returned {count!r}, expected a "
                "non-negative int",
                effective,
            )
        if count >= limits.max_orders_per_day:
            return reject(
                _CHECK_DAILY_ORDERS,
                f"{count} orders already placed for {asset_class.value} on "
                f"{day_key}, cap is {limits.max_orders_per_day}",
                effective,
            )
        passed(_CHECK_DAILY_ORDERS,
               f"{count}/{limits.max_orders_per_day} used on {day_key}")

        # 11. risk circuit ---------------------------------------------------
        try:
            decision = self.__circuit.read(asset_class, equity)
        except Exception as exc:                      # fail CLOSED
            return reject(_CHECK_CIRCUIT,
                          f"risk circuit unreadable: {exc!r}", effective)
        token = _circuit_token(decision)
        allows_entries = _circuit_allows_entries(decision, token)
        if allows_entries is None:
            return reject(
                _CHECK_CIRCUIT,
                f"risk circuit returned {decision!r} with no state this gate "
                "knows how to honour; refusing rather than assuming it is clear",
                effective,
            )
        note = _circuit_reason(decision)
        suffix = f" ({note})" if note else ""
        state = token or "unnamed state"
        if reducing:
            # Exits stay open in every circuit state. A flatten instruction
            # that blocks its own exits traps the book behind its own safety
            # rule, and a throttle applied to an exit leaves a tail on.
            passed(_CHECK_CIRCUIT,
                   f"circuit is {state}; reducing orders are always allowed{suffix}")
        elif not allows_entries:
            return reject(
                _CHECK_CIRCUIT,
                f"circuit is {state}; new {asset_class.value} entries are "
                f"blocked{suffix}",
                effective,
            )
        else:
            multiplier, bad = _circuit_multiplier(decision)
            if bad is not None:
                return reject(_CHECK_CIRCUIT, f"circuit is {state} but {bad}",
                              effective)
            if multiplier is None or multiplier == Decimal("1"):
                passed(_CHECK_CIRCUIT, f"circuit is {state}{suffix}")
            else:
                throttled = profile.quantize_qty(
                    effective.instrument, qty * multiplier)
                if throttled <= 0:
                    return reject(
                        _CHECK_CIRCUIT,
                        f"circuit is {state} with size_multiplier {multiplier}; "
                        f"{qty} throttles to {throttled} and there is nothing "
                        "left to place",
                        effective,
                    )
                floor_after = max(effective.instrument.min_notional,
                                  limits.min_notional_floor)
                if throttled * price < floor_after:
                    return reject(
                        _CHECK_MIN_NOTIONAL,
                        f"throttled notional {throttled * price} below minimum "
                        f"{floor_after}",
                        effective,
                    )
                effective = replace(effective, qty=throttled)
                passed(_CHECK_CIRCUIT,
                       f"circuit is {state}; qty {qty} -> {throttled} "
                       f"(x{multiplier}){suffix}")

        # 12. mode coercion ---------------------------------------------------
        passed(_CHECK_COERCION, self.__coercion_detail(asset_class, mode))

        # Only now — after everything passed — is the quota spent and the key
        # claimed. Doing this earlier is what made the old daily cap useless.
        try:
            self.__state.record_submission(key, asset_class, day_key, ctx.now)
        except Exception as exc:
            return reject(
                _CHECK_DEDUP,
                f"could not record submission, refusing rather than placing an "
                f"unaccounted order: {exc!r}",
                effective,
            )
        return Approval(True, mode, effective, tuple(checks))

    # --------------------------------------------------------------- private

    def __effective_mode(self, asset_class: AssetClass) -> Mode:
        """LIVE only when the engine asked for LIVE *and* the class is armed.
        Everything else is PAPER."""
        if self.__config.requested_mode is Mode.LIVE and self.__armed.get(
                asset_class, False):
            return Mode.LIVE
        return Mode.PAPER

    def __coercion_detail(self, asset_class: AssetClass, mode: Mode) -> str:
        var = _ARM_ENV[asset_class]
        if self.__config.requested_mode is Mode.LIVE and mode is Mode.PAPER:
            return (f"requested live but {var} is not set; coerced to paper")
        if mode is Mode.LIVE:
            return f"live: {var} is armed"
        return "paper (requested)"

    @staticmethod
    def __sanity(intent: Intent, ctx: MarketContext,
                 profile: AssetClassProfile) -> Optional[str]:
        """Re-validate what Intent's own __post_init__ enforces. Intent is
        frozen, but objects reach the gate after serialisation round-trips and
        replace() calls; the chokepoint does not take construction on trust.
        Returns a failure reason, or None when clean."""
        instrument = intent.instrument
        if not instrument.symbol.strip():
            return "instrument has no symbol"
        if not instrument.broker_id.strip():
            return f"{instrument.symbol} has no broker_id; it cannot be routed"
        if instrument.lot_size < 1:
            return f"{instrument.symbol} has lot_size {instrument.lot_size}"
        if instrument.tick_size <= 0:
            return f"{instrument.symbol} has tick_size {instrument.tick_size}"
        if instrument.qty_precision < 0:
            return f"{instrument.symbol} has qty_precision {instrument.qty_precision}"
        if not isinstance(intent.qty, Decimal):
            return f"qty must be Decimal, got {type(intent.qty).__name__}"
        if not intent.qty.is_finite() or intent.qty <= 0:
            return f"qty must be > 0, got {intent.qty}"
        if intent.order_type is OrderType.LIMIT:
            if _positive_decimal(intent.limit_price) is None:
                return f"LIMIT order needs a positive limit_price, got {intent.limit_price!r}"
        elif not profile.market_orders_allowed:
            return (f"{profile.asset_class.value} venue rejects MARKET orders; "
                    f"use a marketable {profile.default_order_type.value}")
        if not intent.strategy_id.strip():
            return "intent has no strategy_id"
        if not intent.reason.strip():
            return "intent has no reason"
        if ctx.now.tzinfo is None or ctx.now.utcoffset() is None:
            # profile.is_open()/day_key() call astimezone(), which reads a naive
            # datetime as the host's local time. On a server in the wrong region
            # that silently shifts the whole trading session.
            return f"ctx.now {ctx.now!r} is not timezone-aware"
        return None

    @staticmethod
    def __reference_price(intent: Intent, ctx: MarketContext) -> Optional[Decimal]:
        """Price used for every notional test. None when it cannot be
        established — the caps are meaningless without it, so the gate refuses
        rather than sizing against a guess."""
        if intent.order_type is OrderType.LIMIT:
            return _positive_decimal(intent.limit_price)
        return _positive_decimal(ctx.quote(intent.instrument.symbol))


def _is_reducing(intent: Intent, book: Sequence[Position]) -> bool:
    """True when this order can only shrink an existing position.

    Derived from the position store, never from a caller-supplied flag — a flag
    would be a one-line bypass of every exposure cap and of the circuit's
    entry block. An order larger than the position is not a reduction: it
    closes and re-opens on the other side, which is a new entry.

    Matched on symbol AND strategy_id, because that is the key the store nets
    on. Matching the symbol alone let one strategy sell against ANOTHER
    strategy's long and be waved through as a reduction: the caps and the
    circuit's entry block were both skipped, and since the other strategy's row
    never went away the same trick worked again on the next order, and the next.
    "Reducing" here means reducing the position this order will actually be
    netted into — nobody else's.
    """
    for position in book:
        if position.instrument.symbol != intent.instrument.symbol:
            continue
        if position.strategy_id != intent.strategy_id:
            continue
        if position.qty > 0 and intent.side is Side.SELL:
            return intent.qty <= position.qty
        if position.qty < 0 and intent.side is Side.BUY:
            return intent.qty <= -position.qty
    return False


def _structure_sides(book: Sequence[Position],
                     structure_id: str) -> tuple[Decimal, Decimal, int]:
    """(long qty, short qty, open leg count) already on for this structure, in
    the book this order will land in.

    The long total is the protection a further short leg is allowed to lean on.
    Counting it in quantity rather than in legs is what makes a PARTIAL fill on
    the protective leg safe: 25 of an intended 75 bought is 25 of protection,
    not a licence to sell 75.
    """
    long_qty = Decimal("0")
    short_qty = Decimal("0")
    legs = 0
    for position in book:
        if (position.structure_id or "").strip() != structure_id:
            continue
        if position.qty > 0:
            long_qty += position.qty
        elif position.qty < 0:
            short_qty += -position.qty
        else:
            continue
        legs += 1
    return long_qty, short_qty, legs


def _exposure(book: Sequence[Position],
              ctx: MarketContext) -> tuple[Optional[Decimal], Optional[str]]:
    """Absolute notional of the open book, or (None, why) when a row cannot be
    marked at all.

    Shorts add exposure, so magnitudes are summed — netting them would let a
    large paired book look flat. A live quote is preferred; the row's own
    avg_price is the fallback because it is a real recorded price for that
    position rather than an invented one.

    When NEITHER is a positive number the row's notional is unknown, and an
    unknown contributes None — never the 0 it used to contribute, which made a
    real open position invisible to the allocation cap. That is reachable: a
    fill recorded at price 0 produces a position whose cost basis is 0.
    """
    total = Decimal("0")
    for position in book:
        mark = _positive_decimal(ctx.quote(position.instrument.symbol))
        if mark is None:
            mark = _positive_decimal(position.avg_price)
        if mark is None:
            return None, (
                f"{position.instrument.symbol} ({position.strategy_id}) has no "
                f"quote and avg_price {position.avg_price}; the class exposure "
                "cannot be measured"
            )
        total += abs(position.qty) * mark
    return total, None
