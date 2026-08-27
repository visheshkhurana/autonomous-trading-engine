"""
Index-option directional debit spread, driven by a regime read the runner
publishes in ctx.extra["regime"].

Why a debit spread and not a naked long option: the loss on a debit spread is
capped at the premium paid, and it is capped by the STRUCTURE rather than by a
stop that has to fill. On an index gap the stop does not fill; the spread still
cannot lose more than the debit.

Two ordering rules in here exist purely to stop a half-placed spread from
becoming a naked short option, which is the single most expensive failure this
system can produce:

  * entering, the LONG leg is proposed first  — if only one fills we are long
    an option (defined loss);
  * exiting, the SHORT leg is bought back first — if only one fills we are
    still long an option, not naked short.

The executor must respect intent order for a shared structure_id. That is a
contract this strategy cannot enforce on its own, so it is stated here.

This strategy does not read bars. Its decision input is the regime read, and
the idempotency key is stamped with the regime's `as_of` so that re-proposing
on the same read is the same decision.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional, Sequence

from core.profiles import IST
from core.types import AssetClass, Instrument, Intent, MarketContext, Position, Side

from .base import (
    MissingDataError, StrategyBase, StrategyInputError, floor_ts, structure_key,
    to_decimal,
)

REGIME_KEY = "regime"
CHAIN_KEY = "option_chain"


class Right(str, Enum):
    CALL = "CE"
    PUT = "PE"


class RegimeState(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"


@dataclass(frozen=True)
class RegimeRead:
    """What the regime engine published, and WHEN. `as_of` is not decoration:
    a regime read with no timestamp cannot be told apart from one computed
    before the gap that just happened."""
    state: RegimeState
    confidence: Decimal
    as_of: dt.datetime

    def __post_init__(self) -> None:
        if not isinstance(self.state, RegimeState):
            raise StrategyInputError(f"regime.state must be a RegimeState, got {self.state!r}")
        if not isinstance(self.confidence, Decimal):
            raise StrategyInputError("regime.confidence must be a Decimal")
        if not (Decimal(0) <= self.confidence <= Decimal(1)):
            raise StrategyInputError(f"regime.confidence out of [0,1]: {self.confidence}")
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise StrategyInputError("regime.as_of must be timezone-aware")

    @classmethod
    def coerce(cls, obj: object) -> "RegimeRead":
        if isinstance(obj, RegimeRead):
            return obj
        if not isinstance(obj, MappingABC):
            raise StrategyInputError(
                f"regime read must be a RegimeRead or a mapping, got {type(obj).__name__}"
            )
        missing = {"state", "confidence", "as_of"} - set(obj.keys())
        if missing:
            raise StrategyInputError(f"regime read is missing {sorted(missing)}")
        raw_state = obj["state"]
        try:
            state = raw_state if isinstance(raw_state, RegimeState) else RegimeState(raw_state)
        except ValueError as exc:
            # An unknown label is NOT quietly mapped to RANGE. "I do not know
            # what this means" and "the market is ranging" are different facts.
            raise StrategyInputError(
                f"unknown regime state {raw_state!r}; expected one of "
                f"{[s.value for s in RegimeState]}"
            ) from exc
        return cls(
            state=state,
            confidence=to_decimal(obj["confidence"], field="regime.confidence"),
            as_of=obj["as_of"],
        )


@dataclass(frozen=True)
class OptionContract:
    """core.types.Instrument is frozen and carries no strike or right, so the
    chain is supplied as this wrapper. Deriving strike/right by parsing the
    trading symbol was the alternative and it is not acceptable for something
    that decides which leg is sold."""
    instrument: Instrument
    underlying: str
    strike: Decimal
    right: Right

    def __post_init__(self) -> None:
        if not isinstance(self.strike, Decimal):
            raise StrategyInputError("OptionContract.strike must be a Decimal")
        if self.strike <= 0:
            raise StrategyInputError("OptionContract.strike must be > 0")
        if not isinstance(self.right, Right):
            raise StrategyInputError("OptionContract.right must be a Right")
        if self.instrument.expiry is None:
            raise StrategyInputError(
                f"OptionContract {self.instrument.symbol} has no expiry on its Instrument"
            )
        if self.instrument.asset_class is not AssetClass.EQUITY:
            raise StrategyInputError(
                f"OptionContract {self.instrument.symbol} is not an EQUITY instrument"
            )

    @property
    def expiry(self) -> dt.date:
        assert self.instrument.expiry is not None  # enforced in __post_init__
        return self.instrument.expiry


class IndexOptionTrendStrategy(StrategyBase):
    """One instance per underlying index."""

    def __init__(self, *, underlying: str, strategy_id: Optional[str] = None,
                 width_strikes: int = 2,
                 min_confidence: Decimal = Decimal("0.60"),
                 min_days_to_expiry: int = 3,
                 close_before_expiry_days: int = 1,
                 regime_max_age: dt.timedelta = dt.timedelta(minutes=30),
                 orphan_grace: dt.timedelta = dt.timedelta(minutes=5),
                 decision_interval: dt.timedelta = dt.timedelta(minutes=5),
                 slippage_bps: Decimal = Decimal("25")) -> None:
        if not underlying.strip():
            raise StrategyInputError("underlying must be non-empty")
        if width_strikes < 1:
            raise StrategyInputError("width_strikes must be >= 1")
        if not (Decimal(0) < min_confidence <= Decimal(1)):
            raise StrategyInputError("min_confidence must be in (0,1]")
        if min_days_to_expiry <= close_before_expiry_days:
            # Otherwise the strategy opens a structure it is already obliged to
            # close, and churns the spread every session.
            raise StrategyInputError(
                "min_days_to_expiry must exceed close_before_expiry_days"
            )
        super().__init__(
            strategy_id=strategy_id or f"equity_trend_{underlying.lower()}",
            asset_class=AssetClass.EQUITY,
            bar_interval=decision_interval,
            slippage_bps=slippage_bps,
        )
        self.underlying = underlying
        self.width_strikes = width_strikes
        self.min_confidence = min_confidence
        self.min_days_to_expiry = min_days_to_expiry
        self.close_before_expiry_days = close_before_expiry_days
        self.regime_max_age = regime_max_age
        self.orphan_grace = orphan_grace

    # -- inputs ------------------------------------------------------------
    def _regime(self, ctx: MarketContext) -> Optional[RegimeRead]:
        raw = ctx.extra.get(REGIME_KEY)
        if raw is None:
            return None
        if not isinstance(raw, MappingABC):
            raise StrategyInputError(
                f"ctx.extra['{REGIME_KEY}'] must be a mapping keyed by underlying, "
                f"got {type(raw).__name__}"
            )
        entry = raw.get(self.underlying)
        if entry is None:
            return None
        read = RegimeRead.coerce(entry)
        if read.as_of > ctx.now:
            raise StrategyInputError(
                f"{self.id}: regime read is stamped {read.as_of}, ahead of ctx.now {ctx.now}"
            )
        if ctx.now - read.as_of > self.regime_max_age:
            # A stale read is treated as NO read, for entries and for exits
            # alike. Flipping a live structure on a regime computed an hour ago
            # is worse than doing nothing.
            return None
        return read

    def _chain(self, ctx: MarketContext) -> Optional[list[OptionContract]]:
        raw = ctx.extra.get(CHAIN_KEY)
        if raw is None:
            return None
        if not isinstance(raw, MappingABC):
            raise StrategyInputError(
                f"ctx.extra['{CHAIN_KEY}'] must be a mapping keyed by underlying, "
                f"got {type(raw).__name__}"
            )
        entry = raw.get(self.underlying)
        if entry is None:
            return None
        if isinstance(entry, (str, bytes)) or not isinstance(entry, SequenceABC):
            raise StrategyInputError(
                f"option_chain[{self.underlying}] must be a sequence of OptionContract"
            )
        seen: set[tuple[str, dt.date, Decimal]] = set()
        chain: list[OptionContract] = []
        for c in entry:
            if not isinstance(c, OptionContract):
                raise StrategyInputError(
                    f"option_chain[{self.underlying}] holds {type(c).__name__}, "
                    "expected OptionContract"
                )
            if c.underlying != self.underlying:
                raise StrategyInputError(
                    f"option_chain[{self.underlying}] contains a {c.underlying} contract"
                )
            key = (c.right.value, c.expiry, c.strike)
            if key in seen:
                # Two contracts claiming the same strike means the chain merged
                # two sources; picking either one silently is a coin flip on
                # which instrument gets sold.
                raise StrategyInputError(
                    f"option_chain[{self.underlying}] has duplicate {c.right.value} "
                    f"{c.strike} {c.expiry}"
                )
            seen.add(key)
            chain.append(c)
        return chain

    # -- main --------------------------------------------------------------
    def propose(self, ctx: MarketContext) -> Sequence[Intent]:
        if not self.profile.is_open(ctx.now):
            return []

        intents: list[Intent] = []
        regime = self._regime(ctx)
        decision_ts = floor_ts(ctx.now, self.bar_interval)

        structures = self._group_structures(ctx)
        for sid, legs in structures:
            reason = self._exit_reason(ctx, sid, legs, regime)
            if reason is not None:
                intents.extend(self._close_structure(ctx, sid, legs, reason, decision_ts))

        if structures:
            # Never stack a second structure on top of an open one, including
            # on the tick where we are unwinding it.
            return self.finalize(intents)
        if regime is None:
            return self.finalize(intents)

        entry = self._entry(ctx, regime)
        intents.extend(entry)
        return self.finalize(intents)

    # -- exits -------------------------------------------------------------
    def _group_structures(self, ctx: MarketContext) -> list[tuple[Optional[str], list[Position]]]:
        groups: dict[Optional[str], list[Position]] = {}
        for p in self.own_positions(ctx):
            groups.setdefault(p.structure_id, []).append(p)
        return sorted(groups.items(), key=lambda kv: (kv[0] is not None, kv[0] or ""))

    def _exit_reason(self, ctx: MarketContext, sid: Optional[str], legs: list[Position],
                     regime: Optional[RegimeRead]) -> Optional[str]:
        symbols = ", ".join(sorted(p.instrument.symbol for p in legs))
        expiries = {p.instrument.expiry for p in legs}

        if sid is None or len(legs) != 2 or len(expiries) != 1 or None in expiries:
            # No structure_id, one leg, three legs, or legs on different
            # expiries: whatever this is, it is not the defined-risk structure
            # we intended to hold.
            if sid is not None and len(legs) == 1 and self._within_grace(ctx, legs[0]):
                # A structure is placed leg by leg; a single leg seen inside the
                # grace window is most likely the other leg still in flight.
                return None
            naked = " (NAKED SHORT)" if any(p.qty < 0 for p in legs) else ""
            grouped = "ungrouped legs" if sid is None else f"{len(legs)} leg(s)"
            return (
                f"{self.underlying}: closing malformed structure{naked} — {grouped} "
                f"[{symbols}] on expiries {sorted(str(e) for e in expiries)}; a partial spread "
                "is not defined risk"
            )

        expiry = expiries.pop()
        assert expiry is not None
        days_left = (expiry - ctx.now.astimezone(IST).date()).days
        if days_left <= self.close_before_expiry_days:
            return (
                f"{self.underlying}: unwinding {expiry} spread [{symbols}] — {days_left} day(s) "
                f"to expiry, at or inside the {self.close_before_expiry_days}-day cutoff"
            )

        if regime is None:
            return None
        held_right = self._structure_right(legs)
        wanted = self._right_for(regime.state)
        if wanted is not held_right:
            state_txt = regime.state.value
            return (
                f"{self.underlying}: regime flipped to {state_txt} (confidence "
                f"{regime.confidence}) against a {held_right.value} spread [{symbols}] — "
                "closing the structure"
            )
        return None

    def _within_grace(self, ctx: MarketContext, pos: Position) -> bool:
        if pos.opened_at is None:
            # Age cannot be established, so the leg is NOT assumed to be in
            # flight. That errs towards closing a leg that was about to be
            # paired (a visible round-trip of slippage) rather than sitting on
            # what may be a naked short (an unbounded loss).
            return False
        if pos.opened_at.tzinfo is None or pos.opened_at.utcoffset() is None:
            raise StrategyInputError(
                f"{self.id}: position {pos.instrument.symbol} has a naive opened_at"
            )
        return (ctx.now - pos.opened_at) <= self.orphan_grace

    def _structure_right(self, legs: list[Position]) -> Right:
        rights = {self._right_of(p.instrument) for p in legs}
        if len(rights) != 1:
            raise StrategyInputError(
                f"{self.id}: structure mixes rights {sorted(r.value for r in rights)}"
            )
        return rights.pop()

    @staticmethod
    def _right_of(instrument: Instrument) -> Right:
        sym = instrument.symbol.upper()
        if sym.endswith("CE"):
            return Right.CALL
        if sym.endswith("PE"):
            return Right.PUT
        raise StrategyInputError(
            f"cannot tell CE from PE for {instrument.symbol}; option symbols must end CE or PE"
        )

    @staticmethod
    def _right_for(state: RegimeState) -> Optional[Right]:
        if state is RegimeState.TREND_UP:
            return Right.CALL
        if state is RegimeState.TREND_DOWN:
            return Right.PUT
        return None

    def _close_structure(self, ctx: MarketContext, sid: Optional[str],
                         legs: list[Position], reason: str,
                         decision_ts: dt.datetime) -> list[Intent]:
        # Short legs first: if only the first order fills we are left long an
        # option, never short one.
        ordered = sorted(legs, key=lambda p: (p.qty > 0, p.instrument.symbol))
        out: list[Intent] = []
        for p in ordered:
            side = Side.BUY if p.qty < 0 else Side.SELL
            price = ctx.quote(p.instrument.symbol)
            if price is None:
                raise MissingDataError(
                    f"{self.id}: no quote for open leg {p.instrument.symbol}; cannot price its "
                    "exit and will not send an unpriced order for a live position"
                )
            out.append(self.make_intent(
                instrument=p.instrument,
                side=side,
                # Exactly what is held. Never quantized — rounding an exit down
                # leaves a residue of the position open.
                qty=abs(p.qty),
                reason=reason,
                decision_ts=decision_ts,
                tag="exit",
                ref_price=price,
                structure_id=sid,
            ))
        return out

    # -- entry -------------------------------------------------------------
    def _entry(self, ctx: MarketContext, regime: RegimeRead) -> list[Intent]:
        right = self._right_for(regime.state)
        if right is None:
            return []
        if regime.confidence < self.min_confidence:
            return []

        spot = ctx.quote(self.underlying)
        if spot is None:
            return []
        chain = self._chain(ctx)
        if not chain:
            return []

        today = ctx.now.astimezone(IST).date()
        candidates = [c for c in chain
                      if c.right is right and (c.expiry - today).days >= self.min_days_to_expiry]
        if not candidates:
            return []
        expiry = min(c.expiry for c in candidates)
        by_strike = {c.strike: c for c in candidates if c.expiry == expiry}
        strikes = sorted(by_strike)

        # Deterministic ATM: nearest strike, lower one wins a tie, so the same
        # spot never picks a different leg on two ticks.
        atm = min(strikes, key=lambda s: (abs(s - spot), s))
        idx = strikes.index(atm)
        short_idx = idx + self.width_strikes if right is Right.CALL else idx - self.width_strikes
        if not (0 <= short_idx < len(strikes)):
            return []

        long_c = by_strike[atm]
        short_c = by_strike[strikes[short_idx]]
        if self.held_by_anyone(ctx, long_c.instrument.symbol) or \
                self.held_by_anyone(ctx, short_c.instrument.symbol):
            return []

        long_px = ctx.quote(long_c.instrument.symbol)
        short_px = ctx.quote(short_c.instrument.symbol)
        if long_px is None or short_px is None:
            # Half a spread priced is not a spread. Skip the tick.
            return []
        if long_px <= short_px:
            # A "debit" spread taken for a credit is a different structure with
            # different risk. Something is wrong with the quotes; do not trade.
            return []

        debit = long_px - short_px
        width = abs(short_c.strike - long_c.strike)
        days_left = (expiry - today).days
        sid = structure_key(self.id, self.underlying, regime.as_of,
                            tag=f"{right.value}{expiry.isoformat()}")
        qty = self.entry_unit_qty(long_c.instrument)
        kind = "bull call" if right is Right.CALL else "bear put"
        base = (
            f"{self.underlying} {kind} debit spread: regime {regime.state.value} "
            f"(confidence {regime.confidence}), spot {spot}, long {long_c.strike}"
            f"{right.value} @ {long_px} / short {short_c.strike}{right.value} @ {short_px}, "
            f"net debit {debit}, width {width}, expiry {expiry} ({days_left}d). "
            f"Max loss is the {debit} debit."
        )
        # Long leg first: a lone fill leaves defined risk, not a naked short.
        return [
            self.make_intent(
                instrument=long_c.instrument, side=Side.BUY, qty=qty,
                reason=(f"{base} Leg 1/2: buy {long_c.instrument.symbol} "
                        f"({long_c.strike}{right.value})."),
                decision_ts=regime.as_of, tag="entry_long",
                ref_price=long_px, structure_id=sid,
            ),
            self.make_intent(
                instrument=short_c.instrument, side=Side.SELL, qty=qty,
                reason=(f"{base} Leg 2/2: sell {short_c.instrument.symbol} "
                        f"({short_c.strike}{right.value})."),
                decision_ts=regime.as_of, tag="entry_short",
                ref_price=short_px, structure_id=sid,
            ),
        ]
