"""Vishesh's 1,000 USDT AI crypto trading mandate — numbers only.

Every risk percentage is of CURRENT sleeve NAV, not a frozen 1000.
The 1000 figure is the paper sleeve start; the rest of any OKX balance
is out of scope. Engine ClassLimits (50 / 250 / max 1) still bind on
top of these and are not raised here.
"""

from __future__ import annotations

from decimal import Decimal

SLEEVE_START = Decimal("1000")

# Capital structure caps of CURRENT NAV. Caps, not targets — holding USDT
# is a valid position and idle capital is not a reason to trade.
RESERVE_FRACTION = Decimal("0.30")
SPOT_FRACTION = Decimal("0.40")
PERPS_MARGIN_FRACTION = Decimal("0.20")
OPTIONS_PREMIUM_FRACTION = Decimal("0.10")

# Spot mix applies only when the regime is bullish AND a score-passing
# setup exists for that asset. Not an automatic rebalance.
SPOT_WEIGHTS = {
    "BTC/USDT": Decimal("0.50"),
    "ETH/USDT": Decimal("0.30"),
    "SOL/USDT": Decimal("0.20"),
}

NORMAL_RISK_FRACTION = Decimal("0.005")       # 0.5% NAV
HIGH_CONVICTION_RISK_FRACTION = Decimal("0.01")  # 1% NAV; never more on one thesis
MAX_OPEN_RISK_FRACTION = Decimal("0.025")     # 2.5% NAV simultaneous
DAILY_LOSS_HALT_FRACTION = Decimal("0.02")    # 2% NAV then no new directional
CONSECUTIVE_LOSS_STOP = 3

DRAWDOWN_REDUCE_25 = Decimal("0.03")          # -3% from equity high
DRAWDOWN_REDUCE_50 = Decimal("0.05")          # -5%
DRAWDOWN_STOP_LEVERED = Decimal("0.075")      # -7.5%
DRAWDOWN_DELEVERAGE = Decimal("0.10")         # -10%

MAX_DIRECTIONAL_PER_24H = 5
PREFER_DIRECTIONAL_PER_24H = 3

SCORE_NO_TRADE = 65
SCORE_HALF = 75
SCORE_NORMAL = 85

MIN_RR = Decimal("2")
PREFER_RR = Decimal("2.5")

# Perps leverage caps (paper/sim only). Isolated where practical.
PERP_LEVERAGE = {
    "BTC": 3,
    "ETH": 3,
    "SOL": 2,
    "OTHER": 2,
}

# Hard stop floor on a held spot, as a fraction of entry. Matches the
# high-conviction thesis cap so a live-looking leftover cannot run to zero.
HARD_STOP_FRACTION = Decimal("0.01")

ATR_STOP_MULT = Decimal("1.2")
ATR_PERIOD = 14
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

STATE_KEY = "mandate_state"
MARKET_KEY = "mandate_market"
CARDS_KEY = "mandate_cards"
