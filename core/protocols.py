"""
The seams of the system. Everything below is an interface with exactly one
purpose, so each can be tested and replaced on its own.

Dependency direction is strictly one way:

    Strategy -> Intent -> OrderGate -> RiskCircuit
                              |
                              v
                        BrokerAdapter -> Fill -> PositionStore

Nothing below the gate calls anything above it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional, Protocol, Sequence, runtime_checkable

from .types import (
    Approval, AssetClass, Fill, Instrument, Intent, MarketContext,
    Mode, OrderRef, Position,
)


@runtime_checkable
class Strategy(Protocol):
    """Pure. No I/O, no clock, no broker. Given a context, say what you want.

    Called on every tick. Returning [] is the normal case and must be cheap.
    A strategy that raises is disabled for the session, not retried blindly —
    a strategy silently throwing every tick is how a book goes unmanaged.
    """

    id: str
    asset_class: AssetClass

    def propose(self, ctx: MarketContext) -> Sequence[Intent]: ...


@runtime_checkable
class BrokerAdapter(Protocol):
    """The only thing that talks to a venue.

    place() MUST block until the order is terminal or the timeout fires, and
    MUST return what actually happened. Fire-and-forget exits are how a
    position gets marked closed while still being open at the broker.
    """

    mode: Mode
    asset_class: AssetClass

    def place(self, intent: Intent, timeout_s: float = 30.0) -> OrderRef: ...
    def cancel(self, order_id: str) -> bool: ...
    def positions(self) -> Sequence[Position]: ...
    def quote(self, instrument: Instrument) -> Optional[Decimal]: ...
    def equity(self) -> Optional[Decimal]:
        """Account value in the class's base currency, or None if it cannot be
        established. NEVER 0 on failure — zero reads as a 100% drawdown and
        latches the circuit into flatten."""
        ...


@runtime_checkable
class PositionStoreProto(Protocol):
    """Single writer. Every read filters on mode, so paper rows can never
    contaminate a live decision — that mixing is what let paper P&L
    force-flatten a live book in the previous system."""

    def apply(self, fill: Fill, strategy_id: str,
              structure_id: Optional[str] = None) -> None: ...
    def positions(self, mode: Mode,
                  asset_class: Optional[AssetClass] = None) -> Sequence[Position]: ...
    def realized_pnl(self, mode: Mode, asset_class: AssetClass,
                     since_iso: str) -> Decimal: ...
    def record_order(self, ref: OrderRef) -> None: ...


@runtime_checkable
class RiskCircuitProto(Protocol):
    """Drawdown and daily-loss ladder. Fails CLOSED: if equity or stored state
    cannot be read, new entries are refused rather than allowed on unverified
    numbers."""

    def read(self, asset_class: AssetClass,
             equity: Optional[Decimal]) -> "CircuitDecision": ...
    def advance(self, asset_class: AssetClass, equity: Decimal) -> "CircuitDecision": ...


@runtime_checkable
class OrderGateProto(Protocol):
    """The single chokepoint. Every order in the system passes through here."""

    def check(self, intent: Intent, ctx: MarketContext) -> Approval: ...
