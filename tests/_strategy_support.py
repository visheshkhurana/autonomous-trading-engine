"""Builders shared by the strategy tests. Not collected by pytest (leading _)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Iterable, Mapping, Optional, Sequence

from core.profiles import IST
from core.types import (
    AssetClass, Instrument, MarketContext, Mode, Position,
)
from strategies.base import Bar

UTC = dt.timezone.utc


def d(v: object) -> Decimal:
    return v if isinstance(v, Decimal) else Decimal(str(v))


def ist(year: int, month: int, day: int, hour: int = 10, minute: int = 0) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=IST)


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=UTC)


def cash_equity(symbol: str = "RELIANCE", lot_size: int = 1) -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.EQUITY,
                      broker_id=f"dhan-{symbol}", lot_size=lot_size,
                      tick_size=Decimal("0.05"))


def option(symbol: str, expiry: dt.date, lot_size: int = 75) -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.EQUITY,
                      broker_id=f"dhan-{symbol}", lot_size=lot_size,
                      tick_size=Decimal("0.05"), expiry=expiry)


def coin(symbol: str = "BTC/USDT") -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.CRYPTO,
                      broker_id=symbol, lot_size=1, tick_size=Decimal("0.01"),
                      qty_precision=6, min_notional=Decimal("10"))


def make_bars(values: Sequence[object], start: dt.datetime,
              interval: dt.timedelta = dt.timedelta(minutes=15),
              spread: object = 1) -> list[Bar]:
    """Bars whose closes are exactly `values`; highs/lows bracket them."""
    out: list[Bar] = []
    prev: Optional[Decimal] = None
    sp = d(spread)
    for i, v in enumerate(values):
        close = d(v)
        open_ = prev if prev is not None else close
        out.append(Bar.of(
            start + i * interval,
            open_,
            max(open_, close) + sp,
            min(open_, close) - sp,
            close,
        ))
        prev = close
    return out


def ohlc_bars(rows: Sequence[tuple[object, object, object, object]],
              start: dt.datetime,
              interval: dt.timedelta = dt.timedelta(minutes=15)) -> list[Bar]:
    return [Bar.of(start + i * interval, d(o), d(h), d(l), d(c))
            for i, (o, h, l, c) in enumerate(rows)]


def position(instrument: Instrument, qty: object, avg_price: object,
             strategy_id: str, structure_id: Optional[str] = None,
             opened_at: Optional[dt.datetime] = None,
             mode: Mode = Mode.PAPER) -> Position:
    return Position(instrument=instrument, qty=d(qty), avg_price=d(avg_price),
                    mode=mode, strategy_id=strategy_id, structure_id=structure_id,
                    opened_at=opened_at)


def context(now: dt.datetime, *,
            bars: Optional[Mapping[str, Sequence[Bar]]] = None,
            quotes: Optional[Mapping[str, object]] = None,
            positions: Iterable[Position] = (),
            extra: Optional[Mapping[str, object]] = None) -> MarketContext:
    payload: dict = {"bars": dict(bars or {})}
    payload.update(dict(extra or {}))
    return MarketContext(
        now=now,
        quotes={k: d(v) for k, v in (quotes or {}).items()},
        positions=tuple(positions),
        extra=payload,
    )


def close(a: Decimal, b: object, tol: str = "1e-9") -> bool:
    return abs(a - d(b)) <= Decimal(tol)
