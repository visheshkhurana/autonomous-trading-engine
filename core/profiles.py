"""
What differs between asset classes — deliberately small, and deliberately in
ONE place. Everything else in the engine is asset-class agnostic.

If you find yourself writing `if asset_class == EQUITY` anywhere else, the
distinction belongs here instead.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from .types import AssetClass, Instrument, OrderType

IST = ZoneInfo("Asia/Kolkata")
UTC = dt.timezone.utc


@dataclass(frozen=True)
class AssetClassProfile:
    asset_class: AssetClass
    base_currency: str
    # Equity must be flat by the session close; crypto carries overnight.
    must_flatten_intraday: bool
    # F&O rejects MARKET orders outright, so entries are marketable LIMITs.
    market_orders_allowed: bool
    default_order_type: OrderType

    def is_open(self, now: dt.datetime) -> bool:
        raise NotImplementedError

    def day_key(self, now: dt.datetime) -> str:
        """The boundary the daily-loss baseline rolls on. Getting this wrong
        means the daily cap either never resets or resets mid-session."""
        raise NotImplementedError

    def quantize_qty(self, instrument: Instrument, qty: Decimal) -> Decimal:
        raise NotImplementedError

    def quantize_price(self, instrument: Instrument, price: Decimal) -> Decimal:
        step = instrument.tick_size
        return (price / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step


class EquityProfile(AssetClassProfile):
    def __init__(self) -> None:
        super().__init__(
            asset_class=AssetClass.EQUITY, base_currency="INR",
            must_flatten_intraday=True, market_orders_allowed=False,
            default_order_type=OrderType.LIMIT,
        )

    def is_open(self, now: dt.datetime) -> bool:
        t = now.astimezone(IST)
        if t.weekday() > 4:          # NSE is closed at weekends
            return False
        return dt.time(9, 15) <= t.time() <= dt.time(15, 30)

    def day_key(self, now: dt.datetime) -> str:
        return now.astimezone(IST).strftime("%Y-%m-%d")

    def quantize_qty(self, instrument: Instrument, qty: Decimal) -> Decimal:
        """Whole lots only. Round DOWN — rounding up silently buys more than
        the sizing allowed."""
        lots = (qty / instrument.lot_size).to_integral_value(rounding=ROUND_DOWN)
        return lots * instrument.lot_size


class CryptoProfile(AssetClassProfile):
    def __init__(self) -> None:
        super().__init__(
            asset_class=AssetClass.CRYPTO, base_currency="USDT",
            must_flatten_intraday=False, market_orders_allowed=True,
            default_order_type=OrderType.MARKET,
        )

    def is_open(self, now: dt.datetime) -> bool:
        return True

    def day_key(self, now: dt.datetime) -> str:
        return now.astimezone(UTC).strftime("%Y-%m-%d")

    def quantize_qty(self, instrument: Instrument, qty: Decimal) -> Decimal:
        step = Decimal(1).scaleb(-instrument.qty_precision)
        return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


PROFILES: dict[AssetClass, AssetClassProfile] = {
    AssetClass.EQUITY: EquityProfile(),
    AssetClass.CRYPTO: CryptoProfile(),
}


def profile_for(asset_class: AssetClass) -> AssetClassProfile:
    return PROFILES[asset_class]
