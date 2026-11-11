"""
Core contracts for the trading engine.

Everything in this system is expressed in these types. The rule that makes the
design work: a Strategy produces Intents and nothing else — it never touches a
broker, a database, or a clock. That keeps strategies pure, unit-testable, and
identical in paper and live.

The one chokepoint is OrderGate.check(). No code path may reach a broker
without an Approval, and an Approval carries the mode the order will actually
execute in — not the mode the caller wanted.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class AssetClass(str, Enum):
    EQUITY = "equity"   # NSE cash + NFO derivatives, via Dhan
    CRYPTO = "crypto"   # OKX spot


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class Mode(str, Enum):
    """Paper vs live is a property of the ADAPTER, never a flag a strategy
    reads. Strategies are written once and cannot tell the difference."""
    PAPER = "paper"
    LIVE = "live"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"   # terminal for us: we stopped waiting, broker may differ

    @property
    def terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.TIMEOUT,
        }


@dataclass(frozen=True)
class Instrument:
    """Canonical instrument. `broker_id` is the only broker-specific field —
    Dhan security_id, or a ccxt market symbol."""
    symbol: str
    asset_class: AssetClass
    broker_id: str
    lot_size: int = 1
    tick_size: Decimal = Decimal("0.05")
    # Crypto needs fractional amounts; equity does not.
    qty_precision: int = 0
    min_notional: Decimal = Decimal("0")
    expiry: Optional[dt.date] = None

    def __post_init__(self) -> None:
        if self.lot_size < 1:
            raise ValueError(f"{self.symbol}: lot_size must be >= 1")
        if self.tick_size <= 0:
            raise ValueError(f"{self.symbol}: tick_size must be > 0")


@dataclass(frozen=True)
class Intent:
    """A strategy's desire to trade. Pure data — no side effects, no I/O.

    `reason` is mandatory and is surfaced verbatim in the UI. An autonomous
    trader that cannot say why it did something is not operable.
    """
    instrument: Instrument
    side: Side
    qty: Decimal
    order_type: OrderType
    strategy_id: str
    reason: str
    limit_price: Optional[Decimal] = None
    # Groups the legs of a multi-leg structure so they are placed, tracked and
    # unwound together. A naked leg from a half-placed spread is the single
    # most expensive failure mode in F&O.
    structure_id: Optional[str] = None
    # Stable across retries of the SAME decision, different across new ones.
    # The gate dedups on this.
    idempotency_key: str = ""
    # Protective stop the strategy already computed for an entry. The engine
    # places the native exchange stop at this price after the fill. Absent on
    # exits and on strategies that do not name a stop.
    stop_price: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be > 0")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT intent requires limit_price")
        if not self.reason.strip():
            raise ValueError("every Intent must carry a reason")
        if self.stop_price is not None and (
                not self.stop_price.is_finite() or self.stop_price <= 0):
            raise ValueError("stop_price must be a finite Decimal > 0")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Approval:
    """The gate's verdict. `mode` is authoritative: the gate COERCES live to
    paper when the asset class is not armed, so a stale live-flagged object can
    never place a real order on its own."""
    approved: bool
    mode: Mode
    intent: Intent
    checks: tuple[CheckResult, ...] = ()

    @property
    def rejections(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def reason(self) -> str:
        rej = self.rejections
        return "; ".join(f"{c.name}: {c.detail}" for c in rej) if rej else "approved"


@dataclass(frozen=True)
class Fill:
    order_id: str
    instrument: Instrument
    side: Side
    qty: Decimal
    price: Decimal
    fees: Decimal
    at: dt.datetime
    mode: Mode


@dataclass(frozen=True)
class OrderRef:
    """Result of place(). `status` is always terminal — place() does not return
    until the order resolves or the timeout fires. A caller that gets an
    OrderRef back knows what happened; there is no fire-and-forget."""
    order_id: str
    status: OrderStatus
    intent: Intent
    mode: Mode
    filled_qty: Decimal = Decimal("0")
    avg_price: Optional[Decimal] = None
    fills: tuple[Fill, ...] = ()
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.status.terminal:
            raise ValueError(
                f"OrderRef must carry a terminal status, got {self.status}. "
                "place() blocks until the order resolves."
            )

    @property
    def complete(self) -> bool:
        return self.status is OrderStatus.FILLED


@dataclass(frozen=True)
class Position:
    instrument: Instrument
    qty: Decimal            # signed: negative is short
    avg_price: Decimal
    mode: Mode
    strategy_id: str
    structure_id: Optional[str] = None
    opened_at: Optional[dt.datetime] = None
    # Native exchange stop resting against THIS ledger qty. txid is the
    # venue order id (Kraken) or the paper stop id. None means no stop has
    # been confirmed. The wallet balance is never stored here.
    stop_txid: Optional[str] = None
    stop_price: Optional[Decimal] = None
    stop_qty: Optional[Decimal] = None
    stop_status: Optional[str] = None


@dataclass
class MarketContext:
    """What a strategy is allowed to see. Passed in, never fetched — this is
    what keeps strategies pure and replayable in backtests."""
    now: dt.datetime
    quotes: dict[str, Decimal] = field(default_factory=dict)
    positions: tuple[Position, ...] = ()
    equity: Optional[Decimal] = None
    extra: dict = field(default_factory=dict)

    def quote(self, symbol: str) -> Optional[Decimal]:
        return self.quotes.get(symbol)
