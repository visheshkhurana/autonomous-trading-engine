"""Frozen vocabulary for the paper mandate engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Mapping, Optional

from core.types import Instrument, Intent, Side

from .constants import (
    OPTIONS_PREMIUM_FRACTION, PERPS_MARGIN_FRACTION, RESERVE_FRACTION,
    SLEEVE_START, SPOT_FRACTION,
)


class Regime(str, Enum):
    STRONG_BULL = "A_STRONG_BULL"
    WEAK_BULL = "B_WEAK_BULL"
    RANGE_NEUTRAL = "C_RANGE_NEUTRAL"
    BEAR = "D_BEAR"

    @property
    def bullish(self) -> bool:
        return self in {Regime.STRONG_BULL, Regime.WEAK_BULL}


class Verdict(str, Enum):
    EXECUTE = "EXECUTE"
    WAIT = "WAIT"
    REJECT = "REJECT"


class InstrumentKind(str, Enum):
    SPOT = "spot"
    PERP = "perp"
    OPTION = "option"
    CASH = "usdt"


@dataclass(frozen=True)
class AllocationCaps:
    """Maximum sleeve fractions for one regime. Not fill-targets."""

    reserve: Decimal
    spot: Decimal
    perps: Decimal
    options: Decimal

    def __post_init__(self) -> None:
        total = self.reserve + self.spot + self.perps + self.options
        if total != Decimal("1"):
            raise ValueError(f"allocation caps must sum to 1, got {total}")


# Portfolio MUST change with regime. These are caps on new risk, not
# automatic deployments into idle USDT.
REGIME_CAPS: Mapping[Regime, AllocationCaps] = {
    Regime.STRONG_BULL: AllocationCaps(
        RESERVE_FRACTION, SPOT_FRACTION,
        PERPS_MARGIN_FRACTION, OPTIONS_PREMIUM_FRACTION,
    ),
    Regime.WEAK_BULL: AllocationCaps(
        Decimal("0.45"), Decimal("0.30"),
        Decimal("0.15"), Decimal("0.10"),
    ),
    Regime.RANGE_NEUTRAL: AllocationCaps(
        Decimal("0.70"), Decimal("0.20"),
        Decimal("0.00"), Decimal("0.10"),
    ),
    Regime.BEAR: AllocationCaps(
        Decimal("0.85"), Decimal("0.00"),
        Decimal("0.05"), Decimal("0.10"),
    ),
}


@dataclass(frozen=True)
class MandateState:
    """Published on ctx.extra[STATE_KEY]. Defaults are a fresh 1000 sleeve."""

    sleeve_nav: Decimal = SLEEVE_START
    equity_high: Decimal = SLEEVE_START
    consecutive_losses: int = 0
    daily_loss: Decimal = Decimal("0")
    trades_24h: int = 0
    open_risk: Decimal = Decimal("0")
    open_cluster: str = ""
    paper_derivatives: bool = True
    open_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sleeve_nav <= 0:
            raise ValueError("sleeve_nav must be > 0")
        if self.equity_high <= 0:
            raise ValueError("equity_high must be > 0")
        if self.consecutive_losses < 0 or self.trades_24h < 0:
            raise ValueError("counts must be >= 0")
        if self.daily_loss < 0 or self.open_risk < 0:
            raise ValueError("loss/risk must be >= 0")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "MandateState":
        def dec(key: str, default: Decimal) -> Decimal:
            value = raw.get(key, default)
            if isinstance(value, Decimal):
                return value
            if isinstance(value, bool) or isinstance(value, float):
                raise TypeError(f"mandate_state.{key} must be Decimal, not float")
            return Decimal(str(value))

        symbols = raw.get("open_symbols", ())
        return cls(
            sleeve_nav=dec("sleeve_nav", SLEEVE_START),
            equity_high=dec("equity_high", SLEEVE_START),
            consecutive_losses=int(raw.get("consecutive_losses", 0)),
            daily_loss=dec("daily_loss", Decimal("0")),
            trades_24h=int(raw.get("trades_24h", 0)),
            open_risk=dec("open_risk", Decimal("0")),
            open_cluster=str(raw.get("open_cluster", "") or ""),
            paper_derivatives=bool(raw.get("paper_derivatives", True)),
            open_symbols=tuple(str(s) for s in symbols),
        )

    @property
    def drawdown(self) -> Decimal:
        if self.sleeve_nav >= self.equity_high:
            return Decimal("0")
        return (self.equity_high - self.sleeve_nav) / self.equity_high


@dataclass(frozen=True)
class ScoreBreakdown:
    trend: int
    structure: int
    momentum: int
    volume: int
    reward_risk: int
    funding: int
    oi_liq: int
    cross_market: int
    options: int

    def __post_init__(self) -> None:
        pairs = (
            ("trend", self.trend, 20),
            ("structure", self.structure, 15),
            ("momentum", self.momentum, 10),
            ("volume", self.volume, 10),
            ("reward_risk", self.reward_risk, 15),
            ("funding", self.funding, 10),
            ("oi_liq", self.oi_liq, 10),
            ("cross_market", self.cross_market, 5),
            ("options", self.options, 5),
        )
        for name, value, cap in pairs:
            if not isinstance(value, int) or value < 0 or value > cap:
                raise ValueError(f"score.{name} must be in 0..{cap}, got {value}")

    @property
    def total(self) -> int:
        return (
            self.trend + self.structure + self.momentum + self.volume
            + self.reward_risk + self.funding + self.oi_liq
            + self.cross_market + self.options
        )


@dataclass(frozen=True)
class DecisionCard:
    """The card that must exist before any proposed intent."""

    regime: Regime
    asset: str
    instrument: str
    kind: InstrumentKind
    direction: str
    entry: Optional[Decimal]
    stop: Optional[Decimal]
    target_1: Optional[Decimal]
    target_2: Optional[Decimal]
    position_notional: Decimal
    margin_required: Decimal
    leverage: Decimal
    maximum_loss: Decimal
    expected_profit: Decimal
    reward_risk: Optional[Decimal]
    funding: str
    oi_signal: str
    vol_condition: str
    signal_score: int
    portfolio_correlation: str
    current_nav: Decimal
    total_open_risk: Decimal
    trade_thesis: str
    invalidation: str
    confidence: str
    verdict: Verdict
    tag: str = "entry"
    closed_pnl: Optional[Decimal] = None

    def reason_text(self) -> str:
        """Operator-facing; shown verbatim on the Intent."""
        def fmt(value: Optional[Decimal]) -> str:
            if value is None:
                return "n/a"
            return str(value.quantize(Decimal("0.01")) if value >= 1
                       else value)

        return (
            f"{self.instrument} mandate {self.verdict.value}: {self.trade_thesis} "
            f"REGIME={self.regime.value} ASSET={self.asset} "
            f"INSTRUMENT={self.kind.value} DIRECTION={self.direction} "
            f"ENTRY={fmt(self.entry)} STOP={fmt(self.stop)} "
            f"TARGET1={fmt(self.target_1)} TARGET2={fmt(self.target_2)} "
            f"NOTIONAL={fmt(self.position_notional)} "
            f"MARGIN={fmt(self.margin_required)} LEVERAGE={self.leverage} "
            f"MAX_LOSS={fmt(self.maximum_loss)} "
            f"EXPECTED_PROFIT={fmt(self.expected_profit)} "
            f"RR={fmt(self.reward_risk)} FUNDING={self.funding} "
            f"OI={self.oi_signal} VOL={self.vol_condition} "
            f"SCORE={self.signal_score} CORRELATION={self.portfolio_correlation} "
            f"NAV={fmt(self.current_nav)} OPEN_RISK={fmt(self.total_open_risk)} "
            f"INVALIDATION={self.invalidation} CONFIDENCE={self.confidence} "
            f"VERDICT={self.verdict.value}."
        )


@dataclass(frozen=True)
class MandateTick:
    regime: Regime
    cards: tuple[DecisionCard, ...]
    intents: tuple[Intent, ...]
    state: MandateState


@dataclass(frozen=True)
class TradePlan:
    instrument: Instrument
    kind: InstrumentKind
    side: Side
    entry: Decimal
    stop: Decimal
    target_1: Decimal
    target_2: Decimal
    reward_risk: Decimal
    atr: Decimal
    score: ScoreBreakdown
    thesis: str
    invalidation: str
    funding: str
    oi_signal: str
    vol_condition: str
    leverage: Decimal = Decimal("1")
    decision_ts: Optional[object] = None
    extras: Mapping[str, object] = field(default_factory=dict)

    @property
    def stop_distance_pct(self) -> Decimal:
        return abs(self.entry - self.stop) / self.entry

    @property
    def asset(self) -> str:
        return self.instrument.symbol.split("/")[0].split("-")[0]
