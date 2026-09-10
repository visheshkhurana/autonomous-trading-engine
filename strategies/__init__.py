"""Trading strategies. Every one of them is pure: propose(ctx) -> [Intent],
no I/O, no clock, no broker, and no sizing — the gate sizes and caps."""

from .base import (
    Bar, DonchianChannel, MissingDataError, StrategyBase, StrategyInputError,
    atr, donchian, ema, ema_series, floor_ts, idempotency_key, rsi, sma,
    structure_key, true_range, validate_bars,
)
from .crypto_breakout import CryptoBreakoutStrategy
from .crypto_mandate import CryptoMandateStrategy
from .crypto_trend import CryptoEmaTrendStrategy
from .equity_meanrev import EquityMeanReversionStrategy
from .equity_trend import (
    IndexOptionTrendStrategy, OptionContract, RegimeRead, RegimeState, Right,
)

__all__ = [
    "Bar", "DonchianChannel", "MissingDataError", "StrategyBase", "StrategyInputError",
    "atr", "donchian", "ema", "ema_series", "floor_ts", "idempotency_key", "rsi",
    "sma", "structure_key", "true_range", "validate_bars",
    "CryptoBreakoutStrategy", "CryptoMandateStrategy", "CryptoEmaTrendStrategy",
    "EquityMeanReversionStrategy",
    "IndexOptionTrendStrategy", "OptionContract", "RegimeRead", "RegimeState", "Right",
]
