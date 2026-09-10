"""Mandate risk: all fractions of CURRENT sleeve NAV."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from .constants import (
    CONSECUTIVE_LOSS_STOP, DAILY_LOSS_HALT_FRACTION, DRAWDOWN_DELEVERAGE,
    DRAWDOWN_REDUCE_25, DRAWDOWN_REDUCE_50, DRAWDOWN_STOP_LEVERED,
    HARD_STOP_FRACTION, HIGH_CONVICTION_RISK_FRACTION, MAX_DIRECTIONAL_PER_24H,
    MAX_OPEN_RISK_FRACTION, MIN_RR, NORMAL_RISK_FRACTION,
    PREFER_DIRECTIONAL_PER_24H, SCORE_HALF, SCORE_NO_TRADE, SCORE_NORMAL,
)
from .types import InstrumentKind, MandateState, Regime, TradePlan


CRYPTO_BETA = "crypto_long"


def cluster_of(asset: str, kind: InstrumentKind, side_long: bool) -> str:
    """BTC/ETH/SOL longs — spot, perp, or long call — are one thesis."""
    base = asset.upper()
    if base in {"BTC", "ETH", "SOL"} and side_long:
        if kind is InstrumentKind.OPTION:
            return CRYPTO_BETA
        return CRYPTO_BETA
    if kind is InstrumentKind.CASH:
        return ""
    return f"{base}_{kind.value}_{'long' if side_long else 'short'}"


def new_directional_blocked(state: MandateState) -> Optional[str]:
    if state.consecutive_losses >= CONSECUTIVE_LOSS_STOP:
        return (
            f"consecutive-loss stop: {state.consecutive_losses} losers "
            f"(halt at {CONSECUTIVE_LOSS_STOP})"
        )
    if state.daily_loss >= state.sleeve_nav * DAILY_LOSS_HALT_FRACTION:
        return (
            f"daily loss halt: {state.daily_loss} >= "
            f"{DAILY_LOSS_HALT_FRACTION * 100}% of NAV {state.sleeve_nav}"
        )
    if state.trades_24h >= MAX_DIRECTIONAL_PER_24H:
        return (
            f"max {MAX_DIRECTIONAL_PER_24H} new directional trades / 24h"
        )
    if state.drawdown >= DRAWDOWN_DELEVERAGE:
        return (
            f"drawdown {state.drawdown} from equity high {state.equity_high}: "
            "liquidation-prevention, majority USDT, no new directional"
        )
    return None


def risk_haircut(state: MandateState) -> Decimal:
    """Multiplier on new risk after an equity-high drawdown."""
    dd = state.drawdown
    if dd >= DRAWDOWN_STOP_LEVERED:
        return Decimal("0")   # no new discretionary leveraged; spot still
    if dd >= DRAWDOWN_REDUCE_50:
        return Decimal("0.50")
    if dd >= DRAWDOWN_REDUCE_25:
        return Decimal("0.75")
    return Decimal("1")


def thesis_risk_fraction(score: int) -> Decimal:
    if score < SCORE_NO_TRADE:
        return Decimal("0")
    if score >= SCORE_NORMAL:
        return HIGH_CONVICTION_RISK_FRACTION
    return NORMAL_RISK_FRACTION


def size_mult(score: int) -> Decimal:
    if score < SCORE_NO_TRADE:
        return Decimal("0")
    if score < SCORE_HALF:
        return Decimal("0.5")
    return Decimal("1")


def size_notional(
    state: MandateState,
    plan: TradePlan,
    *,
    levered: bool,
) -> tuple[Optional[Decimal], Decimal, str]:
    """SIZE = MAX LOSS / STOP DISTANCE %. Returns (notional, max_loss, why)."""
    if plan.reward_risk < MIN_RR:
        return None, Decimal("0"), f"R:R {plan.reward_risk} < {MIN_RR}"
    if plan.stop_distance_pct <= 0:
        return None, Decimal("0"), "stop distance is zero"
    score = plan.score.total
    if score < SCORE_NO_TRADE:
        return None, Decimal("0"), f"score {score} < {SCORE_NO_TRADE}"
    if state.trades_24h >= PREFER_DIRECTIONAL_PER_24H and score < SCORE_NORMAL:
        return None, Decimal("0"), (
            f"{state.trades_24h} trades in 24h; prefer 0-3 unless score "
            f">={SCORE_NORMAL}"
        )

    haircut = risk_haircut(state)
    if levered and state.drawdown >= DRAWDOWN_STOP_LEVERED:
        return None, Decimal("0"), (
            f"drawdown {state.drawdown}: stop new discretionary leveraged"
        )
    if haircut <= 0 and levered:
        return None, Decimal("0"), "drawdown halt on leveraged risk"

    risk_frac = thesis_risk_fraction(score) * size_mult(score) * (
        haircut if haircut > 0 else Decimal("1")
    )
    max_loss = (state.sleeve_nav * risk_frac).quantize(Decimal("0.0001"))
    # Never more than the high-conviction cap of CURRENT NAV on one thesis.
    cap = state.sleeve_nav * HIGH_CONVICTION_RISK_FRACTION
    if max_loss > cap:
        max_loss = cap
    if max_loss <= 0:
        return None, Decimal("0"), "sized max loss is 0"

    notional = max_loss / plan.stop_distance_pct
    return notional, max_loss, "ok"


def cluster_allows(
    state: MandateState,
    plan: TradePlan,
    added_risk: Decimal,
) -> Optional[str]:
    side_long = plan.entry > plan.stop
    cluster = cluster_of(plan.asset, plan.kind, side_long)
    if not cluster:
        return None
    if state.open_cluster and state.open_cluster != cluster:
        # A different cluster is fine if combined risk still fits; crypto-beta
        # is the one the mandate names as a single thesis.
        pass
    combined = state.open_risk + added_risk
    limit = state.sleeve_nav * MAX_OPEN_RISK_FRACTION
    same = (not state.open_cluster) or state.open_cluster == cluster
    if same and combined > limit:
        return (
            f"correlated cluster {cluster}: open risk {state.open_risk} + "
            f"{added_risk} exceeds {MAX_OPEN_RISK_FRACTION * 100}% NAV "
            f"({limit})"
        )
    return None


def hard_stop(entry: Decimal) -> Decimal:
    return entry * (Decimal("1") - HARD_STOP_FRACTION)


def confidence_label(score: int, regime: Regime) -> str:
    if score >= SCORE_NORMAL:
        return "high"
    if score >= SCORE_HALF:
        return "normal"
    if score >= SCORE_NO_TRADE:
        return "half"
    return f"below-threshold ({regime.value})"
