"""
Dhan adapter — NSE cash and NFO derivatives.

Two things about Dhan drive the shape of this file.

1. Dhan addresses instruments by a numeric `security_id`, never by a trading
   symbol. `DhanSecurityMap` is the ONLY place that translation happens; the
   rest of the engine speaks `Instrument`. An instrument that cannot be
   resolved raises — a fabricated id would route a real order at the wrong
   contract.

2. NFO rejects MARKET orders outright, and `EquityProfile.market_orders_allowed`
   is False for the whole class. So every order this adapter sends is a LIMIT,
   priced marketable: through the LTP by `limit_ticks`, rounded in the direction
   that keeps it crossing.

The contract this module holds itself to, and the reason it is written this way:

    place() RETURNS  =>  an order reached the venue and the OrderRef is its
                         real, confirmed terminal state.
    place() RAISES   =>  no order reached the venue, EXCEPT
                         OrderSubmitFailed(ambiguous=True), which means the
                         submit itself failed in-flight and the order may or
                         may not be live. That one needs reconciliation, and it
                         says so rather than guessing.

Nothing here converts a failed read into a plausible number. Fill confirmation
is never inferred from acceptance, equity is None rather than 0 when it cannot
be established, and a position whose average price cannot be parsed raises
instead of being reported at 0 (which reads as infinite profit).

Live use: construct the `dhanhq` client elsewhere and inject it. This module
performs no network I/O of its own and imports no HTTP library at module level.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import time
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from core.profiles import profile_for
from core.types import (
    AssetClass,
    Fill,
    Instrument,
    Intent,
    Mode,
    OrderRef,
    OrderStatus,
    OrderType,
    Position,
    Side,
)

log = logging.getLogger(__name__)

# Broker-reported positions carry no strategy attribution — Dhan does not know
# which strategy opened what. The authoritative mapping lives in PositionStore;
# this is the venue's own view, used for reconciliation.
BROKER_POSITION_STRATEGY_ID = "broker"

PRODUCT_TYPES = frozenset({"INTRADAY", "CNC", "MARGIN", "MTF", "CO", "BO"})

# Dhan order-status vocabulary -> ours. An UNRECOGNISED status maps to OPEN, not
# to a fill: the worst an unknown status can then do is run the clock out and
# report TIMEOUT. It can never manufacture a confirmation.
_DHAN_STATUS: dict[str, OrderStatus] = {
    "TRADED": OrderStatus.FILLED,
    "EXECUTED": OrderStatus.FILLED,
    "FILLED": OrderStatus.FILLED,
    "COMPLETE": OrderStatus.FILLED,
    "PART_TRADED": OrderStatus.PARTIAL,
    "PARTIALLY_FILLED": OrderStatus.PARTIAL,
    "TRANSIT": OrderStatus.PENDING,
    "PENDING": OrderStatus.OPEN,
    "OPEN": OrderStatus.OPEN,
    "CONFIRM": OrderStatus.OPEN,
    "TRIGGERED": OrderStatus.OPEN,
    "MODIFIED": OrderStatus.OPEN,
    "REJECTED": OrderStatus.REJECTED,
    "CANCELLED": OrderStatus.CANCELLED,
    "CANCELED": OrderStatus.CANCELLED,
    "EXPIRED": OrderStatus.CANCELLED,
}

# SEM_SEGMENT letter -> the suffix Dhan wants in `exchange_segment`.
_SEGMENT_SUFFIX = {"E": "EQ", "D": "FNO", "C": "CURRENCY", "M": "COMM", "I": "INDEX"}


# ── failures ────────────────────────────────────────────────────────────────

class DhanError(RuntimeError):
    """Base for every failure this adapter raises. Never caught internally and
    turned into a value."""


class DhanApiError(DhanError):
    """The broker returned an explicit failure payload."""


class SecurityIdUnknown(DhanError):
    """No Dhan security_id for this instrument. Nothing was sent."""


class QuoteUnavailable(DhanError):
    """No usable last price, so a marketable limit cannot be priced. Nothing
    was sent — an unpriced order is not sent at a guessed price."""


class InvalidOrderSize(DhanError):
    """Quantity does not round to at least one lot. Nothing was sent."""


class OrderSubmitFailed(DhanError):
    """The submit call did not yield an order id.

    `ambiguous` is the part that matters: False means the broker explicitly
    refused and nothing is live; True means the call failed in flight and an
    order may be resting at the venue under an id we never received. The
    caller must reconcile rather than assume either way.
    """

    def __init__(self, message: str, *, ambiguous: bool) -> None:
        super().__init__(message)
        self.ambiguous = ambiguous


# ── payload helpers ─────────────────────────────────────────────────────────

def _num(value: Any) -> Optional[Decimal]:
    """Decimal, or None when the field is absent/unparseable.

    Deliberately no `or 0` anywhere near this. A missing balance and a zero
    balance are different facts and the risk circuit reads them differently.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None
    return None


def unwrap(resp: Any) -> Any:
    """DhanHQ returns {'status': 'success'|'failure', 'data': ...}.

    Success yields the payload; failure raises with the broker's own remarks
    attached, so the caller sees the venue's reason rather than a shrug.
    """
    if not isinstance(resp, Mapping):
        return resp
    status = str(resp.get("status", "")).lower()
    if status in ("success", "ok"):
        return resp.get("data", resp)
    if "data" in resp and status not in ("failure", "error"):
        return resp["data"]
    remarks = resp.get("remarks") or resp.get("errorMessage") or resp.get("data") or resp
    raise DhanApiError(f"Dhan API error: {remarks}")


def _extract_ltp(resp: Any, segment: str, security_id: str) -> Optional[Decimal]:
    """Dig the last price out of Dhan's sometimes double-nested ticker payload.

    Shape is {status, data: {data: {SEG: {sid: {last_price}}}}} — descend
    through any nested 'data' keys until the segment map appears. Returns None
    when the payload simply carries no price for this id; that is a real
    market-data answer, not a transport failure (those raise out of the client).
    """
    node: Any = resp
    for _ in range(4):
        if not isinstance(node, Mapping):
            return None
        if segment in node:
            seg_map = node[segment]
            if not isinstance(seg_map, Mapping):
                return None
            rec = seg_map.get(str(security_id))
            if rec is None and str(security_id).isdigit():
                rec = seg_map.get(int(security_id))
            if not isinstance(rec, Mapping):
                return None
            px = _num(rec.get("last_price"))
            if px is None:
                px = _num(rec.get("ltp"))
            # A zero or negative print is not a price.
            return px if (px is not None and px > 0) else None
        node = node.get("data")
    return None


def _quantize_to_tick(price: Decimal, tick: Decimal, rounding: str) -> Decimal:
    steps = (price / tick).quantize(Decimal("1"), rounding=rounding)
    return steps * tick


def marketable_limit(ltp: Decimal, side: Side, tick: Decimal, ticks: int) -> Decimal:
    """Price a LIMIT that crosses: through the LTP by `ticks`.

    Rounding direction is not cosmetic. Quantizing a BUY down (or a SELL up)
    can land the order back inside the spread, where it rests instead of
    filling — and a resting protective leg is the thing that leaves a short
    naked. So BUY rounds up and SELL rounds down: the give-up stays bounded by
    `ticks` and the order still crosses. Never returns <= 0.
    """
    if ltp <= 0:
        raise QuoteUnavailable(f"cannot price a marketable limit off ltp={ltp}")
    delta = tick * ticks
    raw = ltp + delta if side is Side.BUY else ltp - delta
    px = _quantize_to_tick(raw, tick, ROUND_CEILING if side is Side.BUY else ROUND_FLOOR)
    return px if px >= tick else tick


def _respect_limit(price: Decimal, side: Side, tick: Decimal) -> Decimal:
    """Tick-align a limit the strategy asked for, without making it more
    aggressive than requested: a BUY rounds down, a SELL rounds up."""
    px = _quantize_to_tick(price, tick, ROUND_FLOOR if side is Side.BUY else ROUND_CEILING)
    return px if px >= tick else tick


# ── symbol -> security_id ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ScripRecord:
    security_id: str
    symbol: str
    exchange: str
    segment: str            # Dhan exchange_segment, e.g. "NSE_EQ" / "NSE_FNO"
    series: str             # EQ / BE / BZ / SM … — surveillance scrips show here
    lot_size: int
    tick_size: Decimal
    instrument_kind: str    # EQUITY / OPTIDX / FUTIDX / OPTSTK / FUTSTK
    expiry: Optional[dt.date] = None


class DhanSecurityMap:
    """(exchange, symbol) -> Dhan security_id, and the instrument metadata that
    rides along with it.

    This is the single translation point in the engine. Everything above it
    speaks `Instrument`; everything below it speaks security_id.

    Source is Dhan's scrip-master CSV. The CSV is supplied as text, or by a
    `loader` callable the caller injects — this module never fetches it itself,
    which keeps the adapter offline-testable and keeps HTTP concerns out of the
    order path. A lookup miss raises; it never guesses an id.
    """

    def __init__(
        self,
        records: Optional[Iterable[ScripRecord]] = None,
        *,
        loader: Optional[Callable[[], str]] = None,
    ) -> None:
        self._loader = loader
        self._by_symbol: dict[tuple[str, str], ScripRecord] = {}
        self._by_id: dict[str, ScripRecord] = {}
        self._loaded = False
        if records is not None:
            self._index(records)
            self._loaded = True

    # -- construction ------------------------------------------------------

    @classmethod
    def from_csv(cls, text: str) -> "DhanSecurityMap":
        return cls(records=parse_scrip_master(text))

    def _index(self, records: Iterable[ScripRecord]) -> None:
        for rec in records:
            self._by_symbol[(rec.exchange, rec.symbol)] = rec
            self._by_id[rec.security_id] = rec

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._loader is None:
            raise SecurityIdUnknown(
                "DhanSecurityMap has no records and no loader — cannot resolve "
                "any symbol. Supply the scrip master."
            )
        self._index(parse_scrip_master(self._loader()))
        self._loaded = True

    # -- lookup ------------------------------------------------------------

    def record(self, symbol: str, exchange: str = "NSE") -> Optional[ScripRecord]:
        self._ensure_loaded()
        return self._by_symbol.get((exchange.upper(), symbol.upper()))

    def record_by_id(self, security_id: str) -> Optional[ScripRecord]:
        self._ensure_loaded()
        return self._by_id.get(str(security_id))

    def try_resolve(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        rec = self.record(symbol, exchange)
        return rec.security_id if rec else None

    def resolve(self, symbol: str, exchange: str = "NSE") -> str:
        sid = self.try_resolve(symbol, exchange)
        if sid is None:
            raise SecurityIdUnknown(
                f"no Dhan security_id for {exchange.upper()}:{symbol.upper()}"
            )
        return sid

    def series(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        rec = self.record(symbol, exchange)
        return rec.series if rec else None

    def instrument(self, symbol: str, exchange: str = "NSE") -> Instrument:
        """Build the canonical Instrument for a symbol. The one supported way
        to get a tradable Instrument for Dhan — lot size, tick size, expiry and
        broker_id all come from the same scrip row, so they cannot drift apart."""
        rec = self.record(symbol, exchange)
        if rec is None:
            raise SecurityIdUnknown(
                f"no Dhan security_id for {exchange.upper()}:{symbol.upper()}"
            )
        return Instrument(
            symbol=rec.symbol,
            asset_class=AssetClass.EQUITY,
            broker_id=rec.security_id,
            lot_size=rec.lot_size,
            tick_size=rec.tick_size,
            expiry=rec.expiry,
        )

    def __len__(self) -> int:
        return len(self._by_symbol)


def parse_scrip_master(text: str) -> list[ScripRecord]:
    """Parse Dhan's compact scrip-master CSV.

    Rows missing an exchange, symbol or security_id are skipped: they cannot be
    traded and indexing them under a partial key would let a later lookup
    return the wrong contract. Everything else is kept, derivatives included.
    """
    out: list[ScripRecord] = []
    for row in csv.DictReader(io.StringIO(text)):
        exch = (row.get("SEM_EXM_EXCH_ID") or "").strip().upper()
        symbol = (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
        sid = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
        if not (exch and symbol and sid):
            continue
        seg_letter = (row.get("SEM_SEGMENT") or "E").strip().upper()[:1]
        lot = _num(row.get("SEM_LOT_UNITS"))
        tick = _num(row.get("SEM_TICK_SIZE"))
        out.append(
            ScripRecord(
                security_id=sid,
                symbol=symbol,
                exchange=exch,
                segment=f"{exch}_{_SEGMENT_SUFFIX.get(seg_letter, 'EQ')}",
                series=(row.get("SEM_SERIES") or "").strip().upper(),
                lot_size=int(lot) if (lot is not None and lot >= 1) else 1,
                tick_size=tick if (tick is not None and tick > 0) else Decimal("0.05"),
                instrument_kind=(row.get("SEM_INSTRUMENT_NAME") or "").strip().upper(),
                expiry=_parse_expiry(row.get("SEM_EXPIRY_DATE")),
            )
        )
    return out


def _parse_expiry(value: Any) -> Optional[dt.date]:
    if not value:
        return None
    text = str(value).strip()
    if not text or text.startswith("0000"):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# ── results ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructureResult:
    """Outcome of a multi-leg placement.

    `manual_action_required` is the field an operator has to see. It means the
    adapter could not get the book back to a safe shape on its own — either an
    unwind order failed, or a submit failed ambiguously and there may be an
    order live under an id we never received.
    """
    ok: bool
    refs: tuple[OrderRef, ...] = ()
    unwound: tuple[OrderRef, ...] = ()
    failed_intent: Optional[Intent] = None
    error: Optional[str] = None
    manual_action_required: bool = False


@dataclass(frozen=True)
class _Snapshot:
    status: OrderStatus
    traded_qty: Optional[Decimal] = None
    avg_price: Optional[Decimal] = None
    polls_ok: int = 0
    last_error: Optional[str] = None


@dataclass(frozen=True)
class _Exposure:
    """Quantity actually confirmed on the book for one leg. Carries partial
    fills from legs that did NOT complete — a timed-out leg that traded half is
    still exposure and must be unwound with everything else."""
    intent: Intent
    qty: Decimal


# ── the adapter ─────────────────────────────────────────────────────────────

class DhanAdapter:
    """BrokerAdapter for Indian equity and F&O over DhanHQ v2.

    The `client` is injected, always. It must expose the DhanHQ surface:
    place_order / cancel_order / get_order_by_id / ticker_data / get_positions
    / get_fund_limits. Building a real one is the caller's job — this class
    never constructs a live session and never reads credentials.
    """

    asset_class: AssetClass = AssetClass.EQUITY

    def __init__(
        self,
        client: Any,
        *,
        mode: Mode = Mode.LIVE,
        security_map: Optional[DhanSecurityMap] = None,
        limit_ticks: int = 2,
        unwind_ticks: int = 10,
        poll_interval_s: float = 1.0,
        product: str = "INTRADAY",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        fee_estimator: Optional[
            Callable[[Instrument, Side, Decimal, Decimal], Decimal]
        ] = None,
    ) -> None:
        if client is None:
            raise ValueError("DhanAdapter requires an injected Dhan client")
        # A Dhan session sends real orders. Labelling one PAPER and letting it
        # run would put live fills into a paper book, which is the exact class
        # of mix-up the mode field exists to prevent. Paper belongs to a paper
        # adapter; this one refuses rather than quietly no-op'ing.
        if mode is not Mode.LIVE:
            raise ValueError(
                "DhanAdapter is a live venue adapter and cannot run in "
                f"{mode.value} mode — use the paper adapter instead"
            )
        if product.upper() not in PRODUCT_TYPES:
            raise ValueError(f"unsupported Dhan product type: {product}")
        if limit_ticks < 0 or unwind_ticks < 0:
            raise ValueError("tick offsets must be >= 0")

        self._client = client
        self.mode = mode
        self.securities = security_map
        self.limit_ticks = limit_ticks
        self.unwind_ticks = unwind_ticks
        self.poll_interval_s = max(float(poll_interval_s), 0.0)
        self.product = product.upper()
        self._clock = clock
        self._sleep = sleep
        self._fee_estimator = fee_estimator
        self._profile = profile_for(AssetClass.EQUITY)

    # -- instrument plumbing ----------------------------------------------

    def security_id(self, instrument: Instrument) -> str:
        """Dhan's numeric id for this instrument.

        `Instrument.broker_id` is authoritative when set — it came from the
        scrip master. Otherwise the symbol is resolved through the map. There
        is no third branch: an unresolvable instrument raises, because the
        alternative is routing an order at whatever contract a guessed id
        happens to name.
        """
        raw = (instrument.broker_id or "").strip()
        if not raw:
            if self.securities is None:
                raise SecurityIdUnknown(
                    f"{instrument.symbol}: no broker_id and no security map"
                )
            raw = self.securities.resolve(instrument.symbol)
        if not raw.isdigit():
            raise SecurityIdUnknown(
                f"{instrument.symbol}: Dhan security_id must be numeric, got {raw!r}"
            )
        return raw

    def exchange_segment(self, instrument: Instrument) -> str:
        """Dhan's exchange_segment for this instrument.

        Preference order: the scrip row (authoritative), then the presence of
        an expiry, which distinguishes a derivative from cash.
        """
        if self.securities is not None:
            rec = None
            if (instrument.broker_id or "").strip():
                rec = self.securities.record_by_id(instrument.broker_id.strip())
            if rec is None:
                rec = self.securities.record(instrument.symbol)
            if rec is not None:
                return rec.segment
        return "NSE_FNO" if instrument.expiry is not None else "NSE_EQ"

    # -- market data -------------------------------------------------------

    def quote(self, instrument: Instrument) -> Optional[Decimal]:
        """Last traded price, or None when the venue carries no price for this
        contract right now.

        None means "the feed had nothing", which is a legitimate market answer.
        A broken feed raises out of the client instead, so the two stay
        distinguishable — a transport failure must never read as "no trades".
        """
        sid = self.security_id(instrument)
        segment = self.exchange_segment(instrument)
        resp = self._client.ticker_data({segment: [int(sid)]})
        return _extract_ltp(resp, segment, sid)

    def _require_quote(self, instrument: Instrument) -> Decimal:
        px = self.quote(instrument)
        if px is None or px <= 0:
            raise QuoteUnavailable(
                f"{instrument.symbol}: no last price, cannot price a marketable limit"
            )
        return px

    # -- account -----------------------------------------------------------

    def equity(self) -> Optional[Decimal]:
        """Total margin base in INR, or None when it cannot be established.

        None on every failure path, deliberately, and never 0: a zero equity
        reading is a 100% drawdown to the risk circuit, which latches it into
        flatten. The circuit fails closed on None instead, which is the correct
        response to "I could not read the account".

        A genuine zero balance still returns Decimal(0) — that is a fact, not a
        failure, and the two must stay distinguishable.

        Both components are required. Reporting available cash alone while
        utilised margin is unreadable would understate equity and read as a
        drawdown that did not happen.
        """
        try:
            data = unwrap(self._client.get_fund_limits())
        except Exception as exc:  # noqa: BLE001 - None is the designated failure signal
            log.error("dhan: equity read failed: %s", exc)
            return None
        if not isinstance(data, Mapping):
            log.error("dhan: equity read returned %s, not a mapping", type(data).__name__)
            return None
        # "availabelBalance" is Dhan's own spelling; both are checked.
        available = _num(data.get("availabelBalance"))
        if available is None:
            available = _num(data.get("availableBalance"))
        utilised = _num(data.get("utilizedAmount"))
        if available is None or utilised is None:
            log.error(
                "dhan: equity fields missing (available=%r utilised=%r) — "
                "reporting None rather than a partial number",
                available, utilised,
            )
            return None
        return available + utilised

    def positions(self) -> Sequence[Position]:
        """The venue's own view of open positions.

        An empty tuple means the broker reported a flat book. It never means
        "the read failed" — a failed read raises, because a book silently
        reported as flat is how ten live positions become invisible.
        """
        data = unwrap(self._client.get_positions())
        if data is None:
            return ()
        if not isinstance(data, (list, tuple)):
            raise DhanApiError(f"dhan: positions payload was {type(data).__name__}, not a list")

        out: list[Position] = []
        for row in data:
            if not isinstance(row, Mapping):
                raise DhanApiError(f"dhan: position row was {type(row).__name__}, not a mapping")
            pos_type = str(row.get("positionType") or "").strip().upper()
            if pos_type == "CLOSED":
                continue
            net = _num(row.get("netQty"))
            if net is None:
                raise DhanApiError(
                    f"dhan: position row for {row.get('tradingSymbol')!r} has no "
                    "readable netQty — refusing to report a size we cannot read"
                )
            if pos_type == "SHORT":
                net = -abs(net)
            elif pos_type == "LONG":
                net = abs(net)
            if net == 0:
                continue

            sid = str(row.get("securityId") or "").strip()
            if not sid:
                raise DhanApiError(
                    f"dhan: position row {row.get('tradingSymbol')!r} carries no "
                    "securityId — it cannot be identified, let alone unwound"
                )
            avg = _num(row.get("sellAvg") if net < 0 else row.get("buyAvg"))
            if avg is None or avg <= 0:
                avg = _num(row.get("costPrice"))
            if avg is None or avg <= 0:
                raise DhanApiError(
                    f"dhan: position {row.get('tradingSymbol')!r} has no readable "
                    "average price — 0 would read as infinite P&L"
                )

            rec = self.securities.record_by_id(sid) if self.securities else None
            symbol = str(row.get("tradingSymbol") or "").strip()
            if not symbol:
                symbol = rec.symbol if rec else sid
            out.append(
                Position(
                    instrument=Instrument(
                        symbol=symbol,
                        asset_class=AssetClass.EQUITY,
                        broker_id=sid,
                        # lot_size/tick_size on a read-back are informational —
                        # qty below is already in contracts.
                        lot_size=rec.lot_size if rec else 1,
                        tick_size=rec.tick_size if rec else Decimal("0.05"),
                        expiry=rec.expiry if rec else None,
                    ),
                    qty=net,
                    avg_price=avg,
                    mode=self.mode,
                    strategy_id=BROKER_POSITION_STRATEGY_ID,
                )
            )
        return tuple(out)

    # -- orders ------------------------------------------------------------

    def cancel(self, order_id: str) -> bool:
        """True only when the broker confirmed the cancel. False means the
        order may still be resting — the caller must not treat it as gone."""
        try:
            unwrap(self._client.cancel_order(order_id))
            return True
        except Exception as exc:  # noqa: BLE001 - False is the designated failure signal
            log.error("dhan: cancel failed for %s: %s", order_id, exc)
            return False

    def place(self, intent: Intent, timeout_s: float = 30.0) -> OrderRef:
        """Submit one order and block until it is terminal.

        Returns the real terminal state. On timeout the resting order is
        cancelled and the result says TIMEOUT — acceptance is not a fill, and
        a leg reported as filled while it is still resting is what leaves a
        short naked.

        Raises (nothing sent) on an unresolvable instrument, a missing quote,
        or a sub-lot size. Raises OrderSubmitFailed(ambiguous=True) when the
        submit call itself failed in flight; that one may have reached the
        venue and needs reconciliation.
        """
        return self._place(intent, timeout_s=timeout_s, ticks=self.limit_ticks)

    def _place(self, intent: Intent, *, timeout_s: float, ticks: int) -> OrderRef:
        instrument = intent.instrument
        sid = self.security_id(instrument)
        segment = self.exchange_segment(instrument)

        # Whole lots, rounded DOWN by the profile. Rounding up would silently
        # trade more than the sizing allowed.
        qty = self._profile.quantize_qty(instrument, intent.qty)
        if qty <= 0:
            raise InvalidOrderSize(
                f"{instrument.symbol}: qty {intent.qty} rounds to zero lots "
                f"(lot_size={instrument.lot_size}) — nothing sent"
            )

        price = self._order_price(intent, ticks=ticks)

        payload: dict[str, Any] = {
            "security_id": sid,
            "exchange_segment": segment,
            "transaction_type": "BUY" if intent.side is Side.BUY else "SELL",
            "quantity": int(qty),
            # Every order is a LIMIT. NFO rejects MARKET outright and
            # EquityProfile.market_orders_allowed is False for the class.
            "order_type": "LIMIT",
            "product_type": self.product,
            "price": float(price),
        }
        if intent.idempotency_key:
            # Dhan's correlation field is `tag` in the SDK and caps at 25
            # characters; a longer one is rejected outright, which would turn
            # a good order into a failed submit.
            payload["tag"] = intent.idempotency_key[:25]

        try:
            raw = self._client.place_order(**payload)
        except Exception as exc:  # noqa: BLE001 - re-raised as an explicit ambiguity
            raise OrderSubmitFailed(
                f"{instrument.symbol}: submit failed in flight ({exc}) — the order "
                "may or may not be live at the venue; reconcile before retrying",
                ambiguous=True,
            ) from exc

        try:
            data = unwrap(raw)
        except DhanApiError as exc:
            # An explicit failure payload means the venue refused it. Nothing
            # is live, and that is worth saying precisely.
            raise OrderSubmitFailed(str(exc), ambiguous=False) from exc

        order_id = ""
        if isinstance(data, Mapping):
            order_id = str(data.get("orderId") or data.get("order_id") or "").strip()
        if not order_id:
            raise OrderSubmitFailed(
                f"{instrument.symbol}: broker accepted the submit but returned no "
                f"order id ({data!r}) — an order may be live and untrackable",
                ambiguous=True,
            )

        snap = self._await_terminal(order_id, timeout_s=timeout_s, expected_qty=qty)

        if snap.status is OrderStatus.TIMEOUT:
            snap = self._cancel_and_reread(order_id, snap, expected_qty=qty)

        return self._to_ref(intent, order_id, qty, snap)

    def _order_price(self, intent: Intent, *, ticks: int) -> Decimal:
        tick = intent.instrument.tick_size
        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None  # enforced by Intent.__post_init__
            return _respect_limit(intent.limit_price, intent.side, tick)
        # MARKET asked for, marketable LIMIT sent. Needs a live LTP; without
        # one we do not send at a guessed price.
        return marketable_limit(self._require_quote(intent.instrument),
                                intent.side, tick, ticks)

    # -- fill confirmation -------------------------------------------------

    def _poll_once(self, order_id: str) -> _Snapshot:
        data = unwrap(self._client.get_order_by_id(order_id))
        rec = data[0] if isinstance(data, (list, tuple)) and data else data
        if not isinstance(rec, Mapping):
            raise DhanApiError(f"dhan: order status for {order_id} was {rec!r}")
        raw_status = str(rec.get("orderStatus") or rec.get("status") or "").strip().upper()
        status = _DHAN_STATUS.get(raw_status)
        if status is None:
            log.warning(
                "dhan: unrecognised order status %r for %s — treating as OPEN, "
                "which can only end in TIMEOUT, never in a fill",
                raw_status, order_id,
            )
            status = OrderStatus.OPEN
        traded = _num(rec.get("filledQty"))
        if traded is None:
            traded = _num(rec.get("filled_qty"))
        if traded is None:
            traded = _num(rec.get("tradedQuantity"))
        avg = _num(rec.get("averageTradedPrice"))
        if avg is None:
            avg = _num(rec.get("avgPrice"))
        if avg is None or avg <= 0:
            avg = None
        return _Snapshot(status=status, traded_qty=traded, avg_price=avg, polls_ok=1)

    @staticmethod
    def _reconcile_qty(snap: _Snapshot, expected_qty: Decimal,
                       order_id: str) -> _Snapshot:
        """A FILLED status that does not account for the whole order is not a
        fill.

        Dhan reports TRADED only when an order completes, so a terminal record
        carrying a short traded quantity disagrees with itself. Believing the
        status over the quantity is how a half-filled leg gets booked as done
        and its residue never unwound, so the quantity wins: the order goes
        back to PARTIAL and keeps being watched.
        """
        if (snap.status is OrderStatus.FILLED and snap.traded_qty is not None
                and snap.traded_qty < expected_qty):
            log.warning(
                "dhan: %s reported FILLED for only %s of %s — treating as partial",
                order_id, snap.traded_qty, expected_qty,
            )
            return replace(snap, status=OrderStatus.PARTIAL)
        return snap

    def _await_terminal(self, order_id: str, *, timeout_s: float,
                        expected_qty: Decimal) -> _Snapshot:
        """Poll until the order resolves or the budget runs out.

        A poll that raises is logged and retried — it tells us nothing about
        the order, and treating "I could not ask" as "it filled" is the failure
        this whole module is built to avoid. If the budget expires having never
        got an answer, the snapshot says TIMEOUT with `polls_ok == 0`, so the
        caller can tell "we watched it not fill" from "we never saw it".
        """
        deadline = self._clock() + max(float(timeout_s), 0.0)
        last = _Snapshot(status=OrderStatus.PENDING)
        while True:
            try:
                snap = self._reconcile_qty(self._poll_once(order_id),
                                           expected_qty, order_id)
                last = replace(snap, polls_ok=last.polls_ok + 1)
                if last.status.terminal:
                    return last
            except Exception as exc:  # noqa: BLE001 - recorded, never converted to a fill
                log.warning("dhan: order status poll failed for %s: %s", order_id, exc)
                last = replace(last, last_error=str(exc))

            now = self._clock()
            if now >= deadline:
                return replace(last, status=OrderStatus.TIMEOUT)
            self._sleep(min(self.poll_interval_s, max(deadline - now, 0.0)))

    def _cancel_and_reread(self, order_id: str, snap: _Snapshot, *,
                           expected_qty: Decimal) -> _Snapshot:
        """Cancel the residue, then read the order once more.

        The re-read is not optional. A fill can land between the last poll and
        the cancel, and an order that actually filled must not be reported as a
        no-fill — that direction of error leaves a real position nobody is
        tracking. A confirmed full fill here is a confirmed fill, so it is
        reported as one; anything else stays TIMEOUT.
        """
        cancelled = self.cancel(order_id)
        try:
            after = self._reconcile_qty(self._poll_once(order_id),
                                        expected_qty, order_id)
        except Exception as exc:  # noqa: BLE001 - keep the timeout verdict, note why
            log.warning("dhan: post-cancel re-read failed for %s: %s", order_id, exc)
            return replace(
                snap,
                last_error=(f"timeout; cancel {'accepted' if cancelled else 'FAILED'}; "
                            f"post-cancel re-read failed: {exc}"),
            )

        if after.status is OrderStatus.FILLED:
            return replace(after, polls_ok=snap.polls_ok + 1,
                           last_error="filled between the last poll and the cancel")
        return replace(
            snap,
            traded_qty=after.traded_qty if after.traded_qty is not None else snap.traded_qty,
            avg_price=after.avg_price or snap.avg_price,
            polls_ok=snap.polls_ok + 1,
            last_error=(f"timeout; cancel {'accepted' if cancelled else 'FAILED'}; "
                        f"broker status {after.status.value}"),
        )

    def _to_ref(self, intent: Intent, order_id: str, submitted_qty: Decimal,
                snap: _Snapshot) -> OrderRef:
        filled = snap.traded_qty
        if snap.status is OrderStatus.FILLED and (filled is None or filled <= 0):
            # TRADED means the whole order traded; the size field is sometimes
            # absent on Dhan's terminal record. Falling back to the submitted
            # quantity is the only reading consistent with the status.
            filled = submitted_qty
        if filled is None or filled < 0:
            filled = Decimal("0")

        error: Optional[str] = snap.last_error
        fills: tuple[Fill, ...] = ()
        if filled > 0 and snap.avg_price is not None:
            fills = (
                Fill(
                    order_id=order_id,
                    instrument=intent.instrument,
                    side=intent.side,
                    qty=filled,
                    price=snap.avg_price,
                    fees=self._fees(intent, filled, snap.avg_price),
                    at=dt.datetime.now(dt.timezone.utc),
                    mode=self.mode,
                ),
            )
        elif filled > 0:
            # Filled but the venue gave no average price. avg_price stays None
            # rather than being invented, and no Fill is emitted, because a Fill
            # at a made-up price poisons every P&L number downstream.
            note = "filled but no average traded price returned by the venue"
            error = f"{error}; {note}" if error else note

        return OrderRef(
            order_id=order_id,
            status=snap.status,
            intent=intent,
            mode=self.mode,
            filled_qty=filled,
            avg_price=snap.avg_price if filled > 0 else None,
            fills=fills,
            error=error,
        )

    def _fees(self, intent: Intent, qty: Decimal, price: Decimal) -> Decimal:
        """Dhan's order record carries no charges, so fees are 0 unless the
        integrator injects an estimator. 0 here means "not reported by the
        venue", and the contract note is the reconciling source of truth."""
        if self._fee_estimator is None:
            return Decimal("0")
        return self._fee_estimator(intent.instrument, intent.side, qty, price)

    # -- multi-leg ---------------------------------------------------------

    def place_structure(self, intents: Sequence[Intent],
                        timeout_s: float = 30.0) -> StructureResult:
        """Place the legs of one structure, protective side first.

        Ordering is the whole point. Every BUY leg is placed AND confirmed
        before any SELL leg is sent, so a failure part-way through can only
        ever leave a long — never a naked short.

        If any leg fails, everything already on the book is unwound, and the
        unwind itself runs shorts-first: buying a short back before selling the
        long that covers it. Unwinding in the other order re-creates the naked
        leg it is supposed to be removing. Partial fills on the failed leg are
        included in the unwind — a half-filled timed-out leg is still exposure.

        A leg that fails ambiguously (submit died in flight) stops the sequence
        and flags manual action: there is no id to cancel, so unwinding blind
        could double the position instead of closing it.
        """
        legs = list(intents)
        if not legs:
            raise ValueError("place_structure requires at least one intent")
        structure_ids = {leg.structure_id for leg in legs}
        if len(structure_ids) != 1 or not next(iter(structure_ids)):
            raise ValueError(
                "every leg of a structure must carry the same non-empty "
                f"structure_id, got {sorted(str(s) for s in structure_ids)}"
            )

        ordered = ([leg for leg in legs if leg.side is Side.BUY]
                   + [leg for leg in legs if leg.side is Side.SELL])

        refs: list[OrderRef] = []
        exposure: list[_Exposure] = []

        for leg in ordered:
            try:
                ref = self._place(leg, timeout_s=timeout_s, ticks=self.limit_ticks)
            except OrderSubmitFailed as exc:
                if exc.ambiguous:
                    log.error("dhan: ambiguous submit on %s: %s", leg.instrument.symbol, exc)
                    # Unwind the legs we DO have ids for. The ambiguous one is
                    # deliberately left alone: with no order id there is nothing
                    # to cancel, and a blind offsetting order could double the
                    # position instead of closing it. That is what the manual
                    # flag is for.
                    unwound, _ = self._unwind(exposure, timeout_s=timeout_s)
                    return StructureResult(
                        ok=False, refs=tuple(refs), unwound=unwound,
                        failed_intent=leg,
                        error=f"ambiguous submit: {exc}",
                        manual_action_required=True,
                    )
                return self._abort(refs, exposure, leg, str(exc), timeout_s)
            except DhanError as exc:
                return self._abort(refs, exposure, leg, str(exc), timeout_s)

            refs.append(ref)
            if ref.filled_qty > 0:
                exposure.append(_Exposure(intent=leg, qty=ref.filled_qty))
            if ref.status is not OrderStatus.FILLED:
                return self._abort(
                    refs, exposure, leg,
                    f"leg not filled: {ref.status.value}"
                    + (f" ({ref.error})" if ref.error else ""),
                    timeout_s,
                )

        return StructureResult(ok=True, refs=tuple(refs))

    def _abort(self, refs: list[OrderRef], exposure: list[_Exposure],
               failed: Intent, error: str, timeout_s: float) -> StructureResult:
        unwound, clean = self._unwind(exposure, timeout_s=timeout_s)
        return StructureResult(
            ok=False,
            refs=tuple(refs),
            unwound=unwound,
            failed_intent=failed,
            error=error if clean else f"{error}; UNWIND INCOMPLETE",
            manual_action_required=not clean,
        )

    def _unwind(self, exposure: Sequence[_Exposure],
                *, timeout_s: float) -> tuple[tuple[OrderRef, ...], bool]:
        """Flatten confirmed exposure, shorts first.

        Returns (orders, clean). `clean` is False if any leg could not be
        confirmed flat — that is an operator page, not a log line, and the
        caller surfaces it as manual_action_required.

        Unwind legs cross harder than entries (`unwind_ticks`): getting out is
        worth more than the last few ticks of price.
        """
        if not exposure:
            return (), True

        shorts = [e for e in exposure if e.intent.side is Side.SELL]
        longs = [e for e in exposure if e.intent.side is Side.BUY]

        out: list[OrderRef] = []
        clean = True
        for item in shorts + longs:
            closing = Side.BUY if item.intent.side is Side.SELL else Side.SELL
            unwind_intent = Intent(
                instrument=item.intent.instrument,
                side=closing,
                qty=item.qty,
                # MARKET here means "get out"; _place turns it into a limit that
                # crosses by unwind_ticks.
                order_type=OrderType.MARKET,
                strategy_id=item.intent.strategy_id,
                reason=f"unwind {item.intent.structure_id}: structure aborted mid-placement",
                structure_id=item.intent.structure_id,
                idempotency_key=(f"{item.intent.idempotency_key}:unwind"
                                 if item.intent.idempotency_key else ""),
            )
            try:
                ref = self._place(unwind_intent, timeout_s=timeout_s,
                                  ticks=self.unwind_ticks)
            except DhanError as exc:
                log.error(
                    "dhan: UNWIND FAILED for %s %s x%s: %s — MANUAL ACTION NEEDED",
                    closing.value, item.intent.instrument.symbol, item.qty, exc,
                )
                clean = False
                continue
            out.append(ref)
            if ref.status is not OrderStatus.FILLED or ref.filled_qty < item.qty:
                log.error(
                    "dhan: UNWIND INCOMPLETE for %s: %s of %s (%s) — MANUAL ACTION NEEDED",
                    item.intent.instrument.symbol, ref.filled_qty, item.qty,
                    ref.status.value,
                )
                clean = False
        return tuple(out), clean


def build_dhan_client(client_id: str, access_token: str) -> Any:  # pragma: no cover
    """Construct a real DhanHQ client for injection into DhanAdapter.

    Separated out and never called by this module or its tests: the adapter
    stays a pure, offline-testable object, and every live session is created
    explicitly at a wiring site you can grep for.
    """
    from dhanhq import DhanContext, dhanhq  # imported lazily, on purpose

    if not (client_id and access_token):
        raise ValueError("Dhan client needs both client_id and access_token")
    return dhanhq(DhanContext(client_id, access_token))
