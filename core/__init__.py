"""Core contracts and the three things every order passes through.

`types` and `protocols` are frozen: they are the vocabulary the rest of the
system is written in, and `profiles` is the ONE place an asset-class difference
is allowed to live. Re-exported here so a caller writes `from core import
Intent` rather than reaching into a module path that may move.

Deliberately NOT re-exported: nothing from `gate`, `risk` or `store` is
importable as `core.X` without naming its module. `OrderGate`, `RiskCircuit` and
`SQLitePositionStore` are collaborators you construct once, in the composition
root, and a short import path is exactly the wrong kind of convenient for them.
"""

from .profiles import (
    IST, PROFILES, UTC, AssetClassProfile, CryptoProfile, EquityProfile,
    profile_for,
)
from .protocols import (
    BrokerAdapter, OrderGateProto, PositionStoreProto, RiskCircuitProto,
    Strategy,
)
from .types import (
    Approval, AssetClass, CheckResult, Fill, Instrument, Intent, MarketContext,
    Mode, OrderRef, OrderStatus, OrderType, Position, Side,
)

__all__ = [
    # types
    "Approval", "AssetClass", "CheckResult", "Fill", "Instrument", "Intent",
    "MarketContext", "Mode", "OrderRef", "OrderStatus", "OrderType", "Position",
    "Side",
    # protocols
    "BrokerAdapter", "OrderGateProto", "PositionStoreProto", "RiskCircuitProto",
    "Strategy",
    # profiles
    "IST", "PROFILES", "UTC", "AssetClassProfile", "CryptoProfile",
    "EquityProfile", "profile_for",
]
