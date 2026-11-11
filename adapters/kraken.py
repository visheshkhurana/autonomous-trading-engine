"""
Kraken spot adapter (ccxt).

PAPER-FIRST. This class can place real orders only when a signed exchange is
injected and the owner later arms it. Tests inject a fake. Nothing here reads
credentials.

Everything venue-specific about Kraken lives here: symbol shapes, precision,
minimum order sizes, fee reporting, and the fact that a spot account reports
BALANCES, not positions.

Five things in this file exist because their absence cost real money before:

1. Precision and minimums come from the loaded ccxt market, never from a
   constant in this file. The previous system carried hardcoded minimums and
   six of them had drifted away from the venue.
2. Symbol resolution walks a quote-currency fallback chain and is filtered
   against the loaded markets FIRST. ccxt raises BadSymbol on an unknown pair,
   and one delisted name would otherwise take down a whole batch of intents.
3. equity() prices hard-pegged fiat from a table. An AED balance with no crypto
   pair used to make the total unpriceable, which nulled equity and blocked
   every buy indefinitely.
4. A MARKET order re-checks the mark against the price the strategy decided on
   and refuses to send if the market has run away from it.
5. A spot SELL is sized to min(book qty, venue FREE base), never the engine
   book alone. Kraken can take a BUY fee in the coin bought, so free ETH is
   smaller than the fill the ledger recorded; sending book qty is insufficient
   funds and the flatten sticks. Buys are not shrunk this way.

The adapter never invents a number. Anything it cannot establish is None, an
explicit rejection carrying its reason, or an exception -- never a zero, never
an empty list, never a "close enough" price.

The exchange object is injected. Nothing here reads credentials, and nothing
here is executed against a real venue by its tests.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import time
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from uuid import uuid4

from core.safety import (
    STOP_FEE_CLIP_LIMIT, ManualReservedError, StopCancelResult, StopPlaceResult,
    base_asset, key_freeze_active, non_reserved_free, parse_manual_reserved,
    stop_qty_for_ledger,
)
from core.types import (
    AssetClass, Fill, Instrument, Intent, Mode, OrderRef, OrderStatus,
    OrderType, Position, Side,
)

UTC = dt.timezone.utc

__all__ = [
    "KrakenAdapter", "EquityBreakdown", "EquityComponent", "DriftCheck",
    "KrakenAdapterError", "MarketsUnavailable", "OrderSubmitFailed",
    "OrderStateUnknown", "UnknownOrder", "CostBasisUnavailable",
    "FundsMovementForbidden", "is_funds_movement", "clip_spot_sell_qty",
    "QUOTE_FALLBACK", "STABLE_FACE_VALUE", "FIAT_PER_USD",
]

# The production Kraken key has historically carried withdraw permission. This
# adapter is a trade sleeve: it may load markets, fetch ticks/balances, and
# place/cancel/fetch orders. It must never move funds off the account.
_FUNDS_MOVE_TOKENS: tuple[str, ...] = (
    "withdraw",
    "transfer",
    "earn",
    "staking",
    "savings",
)


def is_funds_movement(name: str) -> bool:
    """True for ccxt/Kraken methods that withdraw, transfer, or earn.

    Compared collapsed (no `_` / `-`) so `private_post_asset_withdrawal`
    and `withdrawCcxt` are the same kind of refusal as `withdraw`.
    """
    collapsed = name.lower().replace("-", "").replace("_", "")
    return any(token in collapsed for token in _FUNDS_MOVE_TOKENS)


def clip_spot_sell_qty(book_qty: Decimal, venue_free: Decimal) -> Decimal:
    """Qty a spot SELL may send: min(engine book qty, venue free).

    Venue free is a ceiling, never a size. A wallet that also holds coins
    the engine does not track (a manual balance on the same account) cannot
    increase the order above `book_qty`. Buys must not go through this.
    """
    if book_qty <= 0 or venue_free <= 0:
        return Decimal("0")
    return book_qty if book_qty <= venue_free else venue_free


# Tried in this order. Same chain as OkxAdapter so a USDT-named instrument
# still resolves; Kraken's deepest book is often USD, which is the second hop.
QUOTE_FALLBACK: tuple[str, ...] = ("USDT", "USD", "USDC")

# Counted at face value against a dollar base. This accepts depeg risk
# deliberately: pricing them through a ticker would make total equity depend on
# a network call for a balance already known to be ~1:1, and a failed call
# there is what nulls equity.
STABLE_FACE_VALUE: frozenset[str] = frozenset({
    "USDT", "USDC", "USD", "DAI", "TUSD", "USDP", "PYUSD", "FDUSD",
})

# HARD pegs only, expressed the way the peg is quoted: units per USD.
# A floating rate in a static table goes stale silently and shows up later as
# an equity drift nobody can explain, so EUR/INR/GBP are deliberately absent --
# a floating fiat balance is priced live or reported unpriceable.
# HKD is also absent on purpose: it is a 7.75-7.85 band, not a peg.
FIAT_PER_USD: Mapping[str, Decimal] = {
    "AED": Decimal("3.6725"),   # UAE dirham, unchanged since 1997
    "SAR": Decimal("3.75"),     # Saudi riyal
}



class KrakenAdapterError(RuntimeError):
    """Base class. Every subclass means 'I could not establish this', never
    'nothing happened'."""


class FundsMovementForbidden(KrakenAdapterError, AttributeError):
    """The injected exchange was asked to move funds. Always a bug.

    Also an AttributeError so `hasattr(exchange, "withdraw")` reports
    missing rather than leaking the exception past the proxy.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            f"KrakenAdapter refuses {name!r}: this process never withdraws, "
            "transfers, or earns. The key may have those permissions; the "
            "adapter does not use them."
        )
        self.name = name


class _FundsMoveGuard:
    """Proxy that raises on funds-movement method names and delegates the rest.

    Attribute *access* is the refusal, not a later call: `hasattr` then
    reports the method as missing, and `ex.withdraw(...)` never reaches
    the inner client.
    """

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        if is_funds_movement(name):
            raise FundsMovementForbidden(name)
        return getattr(object.__getattribute__(self, "_inner"), name)


class MarketsUnavailable(KrakenAdapterError):
    """Markets could not be loaded. Distinct from 'this pair is not listed' --
    with no market table we know nothing about any pair."""


class OrderSubmitFailed(KrakenAdapterError):
    """create_order did not return a usable order id. An order MAY exist at the
    venue. Deliberately not a REJECTED OrderRef: 'rejected' reads as 'no order
    exists' and would invite a retry that doubles the position."""


class OrderStateUnknown(KrakenAdapterError):
    """The order exists and we cannot read its state. Carries the order id so a
    caller can reconcile."""

    def __init__(self, order_id: str, message: str) -> None:
        super().__init__(f"{message} (order_id={order_id!r})")
        self.order_id = order_id


class UnknownOrder(KrakenAdapterError):
    """cancel() was handed an id this adapter never placed, so the symbol Kraken
    requires for a cancel is unknown."""


class CostBasisUnavailable(KrakenAdapterError):
    """A venue spot balance carries no entry price. See KrakenAdapter.positions."""


@dataclass(frozen=True)
class DriftCheck:
    reference: Decimal
    mark: Decimal
    drift_pct: Decimal
    limit_pct: Decimal

    @property
    def ok(self) -> bool:
        return self.drift_pct <= self.limit_pct


@dataclass(frozen=True)
class EquityComponent:
    currency: str
    amount: Decimal
    rate: Decimal            # value of ONE unit in the base currency
    value: Decimal
    source: str              # base | stable | fiat-peg | market:SYMBOL


@dataclass(frozen=True)
class EquityBreakdown:
    """Why this is not just a Decimal: a total that silently omits a balance is
    indistinguishable from a drawdown, and that reads as a loss the circuit
    acts on. So the total is None whenever anything was left out, and the
    reason travels with it."""

    base_currency: str
    total: Optional[Decimal]
    components: tuple[EquityComponent, ...] = ()
    unpriceable: tuple[tuple[str, Decimal], ...] = ()
    excluded: tuple[tuple[str, Decimal], ...] = ()
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        # The invariant that makes a short total unrepresentable.
        if self.unpriceable and self.total is not None:
            raise ValueError(
                "equity total must be None while any balance is unpriceable; "
                f"unpriced: {[c for c, _ in self.unpriceable]}"
            )
        if self.total is None and not self.reason:
            raise ValueError("an unknown equity must carry its reason")


def _dec_opt(value: Any) -> Optional[Decimal]:
    """Decimal, or None for a genuinely absent field. A value that is present
    but unparseable raises -- turning it into 0 here would fabricate a price or
    a size."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("refusing to read a bool as a number")
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"cannot read a number from {value!r}") from exc
    raise TypeError(f"cannot read a number from {type(value).__name__}")


def _client_order_id(idempotency_key: str) -> Optional[int]:
    """Kraken `userref` is a signed 32-bit int, not a free-form clOrdId.

    Hashing the idempotency key keeps retries correlated without inventing a
    second identity. None when there is nothing to hash.
    """
    raw = (idempotency_key or "").strip()
    if not raw:
        return None
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2 ** 31)


def _step_from_precision(value: Any) -> Optional[Decimal]:
    """A ccxt precision field as a step size.

    ccxt reports precision either as a step (0.00000001) or as a digit count
    (8), depending on the exchange's precisionMode, so both shapes are read.
    Order sizes are quantized through this step in Decimal, never via
    float() or via ccxt's amount_to_precision (which re-rounds a float).
    """
    number = _dec_opt(value)
    if number is None or number <= 0:
        return None
    if number >= 1 and number == number.to_integral_value():
        return Decimal(1).scaleb(-int(number))
    return number


def _precision_digits(value: Any) -> Optional[int]:
    step = _step_from_precision(value)
    if step is None:
        return None
    exponent = step.normalize().as_tuple().exponent
    return max(0, -int(exponent))


class KrakenAdapter:
    """BrokerAdapter for Kraken spot.

    ``manages_native_stops`` tells the engine this adapter can rest an
    exchange stop sized to the ledger quantity. The wallet is never the size.

    `exchange` is any object with the ccxt surface used here. It is injected so
    this class can be exercised without a venue and without credentials.
    """

    asset_class: AssetClass = AssetClass.CRYPTO
    manages_native_stops: bool = True

    def __init__(
        self,
        exchange: Any,
        *,
        mode: Mode,
        base_currency: str = "USDT",
        max_drift_pct: Decimal = Decimal("0.5"),
        quote_fallback: Sequence[str] = QUOTE_FALLBACK,
        require_reference_price: bool = True,
        cancel_on_timeout: bool = True,
        poll_interval_s: float = 0.5,
        ignore_currencies: Iterable[str] = (),
        cost_basis: Optional[Callable[[str], Optional[Decimal]]] = None,
        extra_params: Optional[Mapping[str, Any]] = None,
        sandbox: bool = False,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        key_freeze: Optional[Callable[[], bool]] = None,
        manual_reserved: Optional[str] = None,
    ) -> None:
        # This class reaches a real venue. A PAPER stamp on a real fill is how a
        # live book gets recorded as paper and then traded against twice, so
        # PAPER is only legal when the caller states the exchange is Kraken demo.
        if mode is Mode.PAPER and not sandbox:
            raise ValueError(
                "KrakenAdapter places real orders; mode=PAPER is only permitted "
                "with sandbox=True (an explicit demo/sandbox exchange). Use a paper "
                "adapter instead of mis-stamping live fills."
            )
        if max_drift_pct < 0:
            raise ValueError("max_drift_pct must be >= 0")
        if not quote_fallback:
            raise ValueError("quote_fallback must list at least one quote currency")

        self._ex = _FundsMoveGuard(exchange)
        self.mode = mode
        self.sandbox = sandbox
        self.base_currency = base_currency.upper()
        self.max_drift_pct = max_drift_pct
        self._quotes = tuple(q.upper() for q in quote_fallback)
        self._require_reference_price = require_reference_price
        self._cancel_on_timeout = cancel_on_timeout
        self._poll_interval_s = poll_interval_s
        self._ignore = frozenset(c.upper() for c in ignore_currencies)
        self._cost_basis = cost_basis
        self._extra_params = dict(extra_params or {})
        self._now = now
        self._monotonic = monotonic
        self._sleep = sleep
        # Default reads KEY_FREEZE / the freeze file on every call so a
        # file dropped on the volume takes effect without rebuilding this
        # object. Tests inject a lambda so they never touch the real path.
        self._key_freeze = key_freeze if key_freeze is not None else key_freeze_active
        # None reads MANUAL_RESERVED_BASE from the process environment on
        # every order, so the reserve can be set without rebuilding the
        # adapter. A string is an explicit override (tests).
        self._manual_reserved_raw = manual_reserved
        self._markets: Optional[dict[str, Any]] = None
        # Kraken needs the symbol to cancel, and the protocol's cancel() only gets
        # an id, so the mapping is kept here for orders this adapter placed.
        self._order_symbols: dict[str, str] = {}

    # ---------------------------------------------------------------- markets

    def load_markets(self, reload: bool = False) -> int:
        markets = self._ex.load_markets(reload) if reload else self._ex.load_markets()
        if not isinstance(markets, dict) or not markets:
            markets = getattr(self._ex, "markets", None)
        # An exchange with zero markets is a failed read, not an empty venue.
        # Accepting {} here would make every symbol resolve to "not listed" and
        # quietly stand the whole strategy down.
        if not isinstance(markets, dict) or not markets:
            raise MarketsUnavailable(
                "load_markets returned no markets; treating this as a failed "
                "read rather than an exchange that lists nothing"
            )
        self._markets = markets
        return len(markets)

    @property
    def markets(self) -> dict[str, Any]:
        if self._markets is None:
            self.load_markets()             # raises rather than leaving None
        if self._markets is None:
            raise MarketsUnavailable("market table is still unset after a load")
        return self._markets

    def _usable(self, symbol: str) -> bool:
        market = self.markets.get(symbol)
        if not isinstance(market, dict):
            return False
        # `spot is not True` also excludes Kraken swaps such as BTC/USDT:USDT.
        # Answering a spot intent with a perpetual is an unbounded-leverage bug.
        if market.get("spot") is not True:
            return False
        if market.get("active") is False:      # delisted; absent means unknown, allow
            return False
        return True

    def resolve_symbol(self, instrument: Instrument) -> Optional[str]:
        """The ccxt market symbol to trade, or None when the venue lists no
        active spot market for this instrument.

        None is a determinate answer ("not listed") so one dead name cannot
        drop a batch; an unreadable market table raises instead.
        """
        for candidate in self._candidates(instrument):
            if self._usable(candidate):
                return candidate
        return None

    def _candidates(self, instrument: Instrument) -> list[str]:
        broker_id = (instrument.broker_id or "").strip().upper()
        out: list[str] = []
        if broker_id:
            out.append(broker_id)
        if "/" in broker_id:
            base = broker_id.split("/", 1)[0]
        else:
            base = broker_id or instrument.symbol.strip().upper()
        base = base.split(":", 1)[0]           # tolerate a swap-shaped broker_id
        for quote in self._quotes:
            out.append(f"{base}/{quote}")
        seen: set[str] = set()
        return [s for s in out if s and not (s in seen or seen.add(s))]

    def instrument_for(self, currency: str, quote: Optional[str] = None) -> Optional[Instrument]:
        """An Instrument for a held currency, with precision read off the venue.

        Walks the same quote fallback chain as resolve_symbol, so a coin listed
        only against USD still resolves instead of looking unheld.
        """
        quotes = (quote.upper(),) if quote else (self.base_currency,) + self._quotes
        symbol = next(
            (s for s in (f"{currency.upper()}/{q}" for q in quotes) if self._usable(s)),
            None,
        )
        if symbol is None:
            return None
        market = self.markets[symbol]
        precision = market.get("precision") or {}
        limits = market.get("limits") or {}
        min_cost = _dec_opt((limits.get("cost") or {}).get("min")) or Decimal("0")
        return Instrument(
            symbol=currency.upper(),
            asset_class=AssetClass.CRYPTO,
            broker_id=symbol,
            qty_precision=_precision_digits(precision.get("amount")) or 8,
            tick_size=_step_from_precision(precision.get("price")) or Decimal("0.01"),
            min_notional=min_cost,
        )

    # ------------------------------------------------------------ quantization

    def _quantize_amount(self, symbol: str, qty: Decimal) -> Decimal:
        """Venue amount precision, in Decimal, never via float().

        `float(qty)` at this boundary is how a size that cleared the gate
        silently changes before it is sent. The step comes from the already-
        loaded market; if that information is missing we refuse rather than
        inventing a precision or asking ccxt to round a float.
        """
        return self._quantize_to_market(symbol, qty, "amount")

    def _quantize_price(self, symbol: str, price: Decimal) -> Decimal:
        return self._quantize_to_market(symbol, price, "price")

    def _quantize_to_market(self, symbol: str, value: Decimal,
                            field: str) -> Decimal:
        market = self.markets.get(symbol)
        if not isinstance(market, dict):
            raise MarketsUnavailable(
                f"{symbol}: no loaded market, cannot quantize {field} without "
                "inventing a precision"
            )
        step = _step_from_precision((market.get("precision") or {}).get(field))
        if step is None or step <= 0:
            raise MarketsUnavailable(
                f"{symbol}: market has no {field} precision; refusing to round "
                "via float() rather than send an unquantized size"
            )
        quantized = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
        return quantized.normalize() if quantized == 0 else quantized

    @staticmethod
    def _ccxt_decimal(value: Decimal) -> str:
        """String form of an already-quantized Decimal.

        ccxt's create_order historically took a float and re-precisioned it.
        Passing a string of a value we already snapped to the market step
        means a binary float cannot change the size between us and the wire.
        """
        return format(value, "f")

    def _min_amount(self, symbol: str) -> Optional[Decimal]:
        limits = (self.markets[symbol].get("limits") or {})
        return _dec_opt((limits.get("amount") or {}).get("min"))

    @staticmethod
    def _base_of(symbol: str) -> str:
        return symbol.split("/", 1)[0].upper()

    def _free_base(self, symbol: str) -> Optional[Decimal]:
        """Spendable units of the base asset, or None if that cannot be read.

        `total` is the wrong number: locked / frozen coins sit in total and
        cannot be sold. Sizing a sell from total is how a fee-in-base leftover
        becomes Kraken insufficient-funds. A missing `free` map is unreadable, not zero: zero
        would look like a flat wallet and invite a write-off.
        """
        raw = self._ex.fetch_balance()
        if not isinstance(raw, dict):
            return None
        free_map = raw.get("free")
        if not isinstance(free_map, dict):
            return None
        code = self._base_of(symbol)
        amount: Optional[Decimal] = None
        for key, value in free_map.items():
            if str(key).upper() == code:
                amount = _dec_opt(value)
                break
        if amount is None:
            return Decimal("0")
        if amount < 0:
            return None
        return amount

    def _reserved_base(self, symbol: str) -> Decimal:
        """Manual balance that must not fund an engine order in this base."""
        if self._manual_reserved_raw is None:
            raw = os.environ.get("MANUAL_RESERVED_BASE", "")
        else:
            raw = self._manual_reserved_raw
        table = parse_manual_reserved(raw)
        return table.get(base_asset(self._base_of(symbol)), Decimal("0"))

    def _spendable_base(self, symbol: str) -> Optional[Decimal]:
        """Free base after the manual reserve. None if free cannot be read."""
        free = self._free_base(symbol)
        if free is None:
            return None
        return non_reserved_free(free, self._reserved_base(symbol))

    def _is_unplaceable_spot_qty(
        self, symbol: str, qty: Decimal, price: Decimal, instrument: Instrument,
    ) -> bool:
        """True when this leftover cannot be sent: below venue min size or
        min notional. That leftover must be written off or flatten sticks."""
        if qty <= 0 or price <= 0:
            return True
        try:
            quantized = self._quantize_amount(symbol, qty)
        except MarketsUnavailable:
            quantized = qty
        if quantized <= 0:
            return True
        min_amount = self._min_amount(symbol)
        if min_amount is not None and quantized < min_amount:
            return True
        min_cost = self._min_cost(symbol, instrument)
        if min_cost is not None and quantized * price < min_cost:
            return True
        if instrument.min_notional > 0 and qty * price < instrument.min_notional:
            return True
        return False

    def _annotate_spot_sell(
        self, ref: OrderRef, symbol: str, clip_note: Optional[str],
    ) -> OrderRef:
        """Record a clip, and attach a dust write-off fill so the ledger can
        match remaining venue qty instead of holding an unsellable stub.

        The write-off is not a venue fill. filled_qty stays what Kraken did.
        Engine apply() nets every fill on the ref, which is how the leftover
        fee-in-base gap leaves the book.
        """
        if ref.intent.side is not Side.SELL:
            return ref
        notes: list[str] = []
        if clip_note:
            notes.append(clip_note)
        leftover = ref.intent.qty - ref.filled_qty
        price = ref.avg_price
        if price is None or price <= 0:
            for fill in reversed(ref.fills):
                if fill.price > 0:
                    price = fill.price
                    break
        fills = ref.fills
        if (leftover > 0 and ref.filled_qty > 0 and price is not None
                and self._is_unplaceable_spot_qty(
                    symbol, leftover, price, ref.intent.instrument)):
            fills = fills + (Fill(
                order_id=ref.order_id, instrument=ref.intent.instrument,
                side=Side.SELL, qty=leftover, price=price, fees=Decimal("0"),
                at=self._now(), mode=self.mode,
            ),)
            notes.append(
                f"wrote off dust leftover {leftover} {self._base_of(symbol)} "
                f"(below venue min size / min notional at {price}); ledger "
                "matches remaining venue qty"
            )
            ref = replace(ref, fills=fills)
        if not notes:
            return ref
        extra = "; ".join(notes)
        error = f"{ref.error}; {extra}" if ref.error else extra
        return replace(ref, error=error)

    def _min_cost(self, symbol: str, instrument: Instrument) -> Optional[Decimal]:
        """The venue's own minimum notional, tightened (never loosened) by any
        minimum the caller put on the Instrument."""
        limits = (self.markets[symbol].get("limits") or {})
        venue_min = _dec_opt((limits.get("cost") or {}).get("min"))
        caller_min = instrument.min_notional if instrument.min_notional > 0 else None
        candidates = [m for m in (venue_min, caller_min) if m is not None]
        return max(candidates) if candidates else None

    # ------------------------------------------------------------------ quotes

    def _ticker_price(self, symbol: str) -> Optional[Decimal]:
        ticker = self._ex.fetch_ticker(symbol)
        if not isinstance(ticker, dict):
            return None
        for key in ("last", "close"):
            price = _dec_opt(ticker.get(key))
            if price is not None and price > 0:
                return price
        bid, ask = _dec_opt(ticker.get("bid")), _dec_opt(ticker.get("ask"))
        if bid and ask and bid > 0 and ask > 0:
            return (bid + ask) / 2
        return None

    def quote(self, instrument: Instrument) -> Optional[Decimal]:
        """Last price in the quote currency, or None when the venue has no
        price for it.

        A transport failure propagates rather than becoming None: strategies
        read None as "skip this name", and a network outage reading as a calm
        market is how a book goes unmanaged while everything looks fine.
        """
        symbol = self.resolve_symbol(instrument)
        if symbol is None:
            return None
        return self._ticker_price(symbol)

    # ------------------------------------------------------------------ equity

    def _rate_to_base(self, currency: str) -> tuple[Optional[Decimal], str]:
        """Value of one unit of `currency` in the base currency, plus its
        source. (None, reason) when it cannot be established."""
        currency = currency.upper()
        if currency == self.base_currency:
            return Decimal("1"), "base"

        # Face value and the peg table are only meaningful against a dollar
        # base. Against a BTC base they would be nonsense, so fall through.
        dollar_base = self.base_currency in STABLE_FACE_VALUE
        if dollar_base and currency in STABLE_FACE_VALUE:
            return Decimal("1"), "stable"
        if dollar_base and currency in FIAT_PER_USD:
            return Decimal("1") / FIAT_PER_USD[currency], "fiat-peg"

        direct = f"{currency}/{self.base_currency}"
        if self._usable(direct):
            price = self._ticker_price(direct)
            if price is not None and price > 0:
                return price, f"market:{direct}"
            return None, f"{direct} returned no usable price"

        inverse = f"{self.base_currency}/{currency}"
        if self._usable(inverse):
            price = self._ticker_price(inverse)
            if price is not None and price > 0:
                return Decimal("1") / price, f"market:{inverse}"
            return None, f"{inverse} returned no usable price"

        # Last resort: one hop through a fallback quote whose own value in the
        # base currency is known for certain (a stable, or a hard peg). A coin
        # listed only against USD must not null the whole equity read.
        for quote in self._quotes:
            if quote == self.base_currency:
                continue
            if dollar_base and quote in STABLE_FACE_VALUE:
                quote_rate = Decimal("1")
            elif dollar_base and quote in FIAT_PER_USD:
                quote_rate = Decimal("1") / FIAT_PER_USD[quote]
            else:
                continue                      # an unpriceable bridge is no bridge
            bridge = f"{currency}/{quote}"
            if not self._usable(bridge):
                continue
            price = self._ticker_price(bridge)
            if price is not None and price > 0:
                return price * quote_rate, f"market:{bridge}"
            return None, f"{bridge} returned no usable price"

        return None, f"no active spot market against {self.base_currency}"

    def equity_breakdown(self) -> EquityBreakdown:
        try:
            raw = self._ex.fetch_balance()
        except Exception as exc:
            # Narrow, deliberate: the ONLY conversion allowed here is failure ->
            # None, which is the documented "cannot establish" signal that fails
            # the risk circuit closed. The reason travels with it.
            return EquityBreakdown(
                self.base_currency, None,
                reason=f"fetch_balance failed: {exc!r}",
            )

        totals = raw.get("total") if isinstance(raw, dict) else None
        if not isinstance(totals, dict):
            return EquityBreakdown(
                self.base_currency, None,
                reason="fetch_balance returned no 'total' balance map",
            )
        if not totals:
            # A wallet listing zero currencies is indistinguishable from a
            # key that lost its read permission. Zero equity reads as a 100%
            # drawdown, so this refuses to be that number.
            return EquityBreakdown(
                self.base_currency, None,
                reason="fetch_balance returned an empty balance map; a venue "
                       "listing no currencies at all is treated as a failed read",
            )

        components: list[EquityComponent] = []
        unpriceable: list[tuple[str, Decimal]] = []
        excluded: list[tuple[str, Decimal]] = []
        notes: list[str] = []

        for currency, value in sorted(totals.items()):
            amount = _dec_opt(value)
            if amount is None or amount == 0:
                continue
            code = str(currency).upper()
            if code in self._ignore:
                excluded.append((code, amount))
                continue
            try:
                rate, source = self._rate_to_base(code)
            except Exception as exc:
                rate, source = None, f"pricing raised {exc!r}"
            if rate is None:
                unpriceable.append((code, amount))
                notes.append(f"{code}: {source}")
                continue
            components.append(EquityComponent(code, amount, rate, amount * rate, source))

        if unpriceable:
            return EquityBreakdown(
                self.base_currency, None, tuple(components), tuple(unpriceable),
                tuple(excluded),
                reason="unpriceable balances (" + "; ".join(notes) + ")",
            )

        total = sum((c.value for c in components), Decimal("0"))
        return EquityBreakdown(
            self.base_currency, total, tuple(components), (), tuple(excluded),
        )

    def equity(self) -> Optional[Decimal]:
        """Account value in the base currency, or None if it cannot be
        established. Call equity_breakdown() for the reason behind a None."""
        return self.equity_breakdown().total

    # --------------------------------------------------------------- holdings

    def holdings(self) -> dict[str, Decimal]:
        """Non-zero venue balances. Raw truth, no pricing, no interpretation."""
        raw = self._ex.fetch_balance()
        totals = raw.get("total") if isinstance(raw, dict) else None
        if not isinstance(totals, dict) or not totals:
            raise KrakenAdapterError(
                "fetch_balance returned no balance map; refusing to report an "
                "empty wallet on an unreadable response"
            )
        out: dict[str, Decimal] = {}
        for currency, value in totals.items():
            amount = _dec_opt(value)
            if amount is not None and amount != 0:
                out[str(currency).upper()] = amount
        return out

    def positions(self) -> Sequence[Position]:
        """Venue holdings as Positions.

        Kraken spot reports balances, which carry a quantity but no entry price,
        and Position.avg_price is not optional. Rather than fill it with the
        current mark -- a number that looks like a cost basis and would make
        every P&L wrong -- this requires a `cost_basis` provider (normally
        backed by the PositionStore) and raises when one is missing. An empty
        list here would be read as "flat" while coins are actually held.
        """
        if self._cost_basis is None:
            raise CostBasisUnavailable(
                "KrakenAdapter.positions() needs a cost_basis provider: Kraken spot "
                "balances carry no entry price. Use holdings() for venue "
                "quantities, or wire cost_basis= from the PositionStore."
            )
        out: list[Position] = []
        for currency, amount in sorted(self.holdings().items()):
            # The base currency and stables are cash, not positions.
            if currency == self.base_currency or currency in STABLE_FACE_VALUE \
                    or currency in FIAT_PER_USD:
                continue
            instrument = self.instrument_for(currency)
            if instrument is None:
                raise CostBasisUnavailable(
                    f"holding {amount} {currency} has no active spot market "
                    f"against {self.base_currency}; it cannot be reported as a "
                    "position and must not be silently dropped"
                )
            basis = self._cost_basis(currency)
            if basis is None:
                raise CostBasisUnavailable(
                    f"no cost basis for {currency} ({amount} held); refusing to "
                    "report a position with a fabricated entry price"
                )
            out.append(Position(
                instrument=instrument, qty=amount, avg_price=basis,
                mode=self.mode, strategy_id="kraken:venue",
            ))
        return tuple(out)

    # ----------------------------------------------------------- drift guard

    def drift_check(self, symbol: str, reference: Decimal) -> DriftCheck:
        """How far the mark has moved from the price the strategy decided on."""
        if reference <= 0:
            raise ValueError("reference price must be > 0")
        mark = self._ticker_price(symbol)
        if mark is None:
            raise KrakenAdapterError(f"{symbol}: no usable mark price for the drift guard")
        drift = abs(mark - reference) / reference * Decimal("100")
        return DriftCheck(reference, mark, drift, self.max_drift_pct)

    # ---------------------------------------------------------------- placing

    def place(self, intent: Intent, timeout_s: float = 30.0) -> OrderRef:
        if self._key_freeze():
            # Before any private call, including the balance read a sell
            # would otherwise make. A rejection is honest: nothing was sent.
            return self._unplaced(
                intent,
                "key_freeze: private Kraken order calls are blocked; "
                "native stops already on the exchange are left in place",
            )
        if intent.instrument.asset_class is not AssetClass.CRYPTO:
            raise ValueError(
                f"KrakenAdapter cannot place {intent.instrument.asset_class} orders"
            )

        symbol = self.resolve_symbol(intent.instrument)
        if symbol is None:
            return self._unplaced(
                intent,
                f"{intent.instrument.symbol}: no active Kraken spot market against "
                f"any of {', '.join(self._quotes)}",
            )

        reference = intent.limit_price
        if intent.order_type is OrderType.MARKET:
            if reference is None:
                if self._require_reference_price:
                    return self._unplaced(
                        intent,
                        "MARKET intent carries no reference price, so the drift "
                        "guard cannot run; refusing to send an unbounded order",
                    )
                try:
                    mark = self._ticker_price(symbol)
                except Exception as exc:
                    return self._unplaced(intent, f"ticker fetch failed: {exc!r}")
                if mark is None or mark <= 0:
                    return self._unplaced(intent, f"{symbol}: no usable mark price")
                trade_price = mark
            else:
                try:
                    check = self.drift_check(symbol, reference)
                except Exception as exc:
                    # No order was created, so a rejection here states a fact.
                    return self._unplaced(
                        intent, f"drift guard could not price {symbol}: {exc!r}"
                    )
                if not check.ok:
                    return self._unplaced(
                        intent,
                        f"price drift {check.drift_pct:.4f}% from reference "
                        f"{check.reference} to mark {check.mark} exceeds "
                        f"{check.limit_pct}%",
                    )
                trade_price = check.mark
        else:
            # Intent.__post_init__ guarantees this; re-checked rather than
            # asserted because `python -O` strips asserts and a None here would
            # become a market order in all but name.
            if reference is None:
                raise ValueError("LIMIT intent reached the adapter with no limit price")
            # No drift check: the limit price is itself the bound.
            try:
                trade_price = self._quantize_price(symbol, reference)
            except MarketsUnavailable as exc:
                return self._unplaced(intent, str(exc))
            if trade_price <= 0:
                return self._unplaced(
                    intent, f"limit price {reference} quantized to {trade_price}"
                )

        try:
            amount = self._quantize_amount(symbol, intent.qty)
        except MarketsUnavailable as exc:
            return self._unplaced(intent, str(exc))
        if amount <= 0:
            return self._unplaced(
                intent,
                f"qty {intent.qty} quantized to {amount} at the venue's amount "
                "precision",
            )

        clip_note: Optional[str] = None
        if intent.side is Side.SELL:
            try:
                free = self._free_base(symbol)
                reserved = self._reserved_base(symbol)
            except ManualReservedError as exc:
                return self._unplaced(
                    intent,
                    f"MANUAL_RESERVED_BASE is unreadable ({exc}); refusing to "
                    "sell, the manual balance must not be spendable",
                )
            except Exception as exc:
                return self._unplaced(
                    intent,
                    f"cannot read venue free {self._base_of(symbol)} for a "
                    f"spot sell: {exc!r}; refusing to send book qty that may "
                    "exceed spendable balance (Kraken insufficient-funds)",
                )
            if free is None:
                return self._unplaced(
                    intent,
                    f"venue free {self._base_of(symbol)} is unreadable; "
                    "refusing to send a spot sell of book qty (Kraken insufficient-funds)",
                )
            spendable = non_reserved_free(free, reserved)
            if spendable is None:
                return self._unplaced(
                    intent,
                    f"spendable {self._base_of(symbol)} is unreadable after "
                    "the manual reserve",
                )
            try:
                sellable = self._quantize_amount(symbol, spendable)
            except MarketsUnavailable as exc:
                return self._unplaced(intent, str(exc))
            clipped = clip_spot_sell_qty(amount, sellable)
            if clipped < amount:
                clip_note = (
                    f"sell qty clipped from {amount} to {clipped} "
                    f"(venue free {self._base_of(symbol)}={free}"
                    f", reserved {reserved}, spendable {spendable})"
                )
                amount = clipped
            if amount <= 0:
                return self._unplaced(
                    intent,
                    f"venue free {self._base_of(symbol)}={free} quantizes to "
                    f"{amount}; nothing sellable",
                )

        min_amount = self._min_amount(symbol)
        if min_amount is not None and amount < min_amount:
            return self._unplaced(
                intent, f"amount {amount} below the venue minimum {min_amount}"
            )
        min_cost = self._min_cost(symbol, intent.instrument)
        notional = amount * trade_price
        if min_cost is not None and notional < min_cost:
            return self._unplaced(
                intent,
                f"notional {notional} below the minimum {min_cost} for {symbol}",
            )

        params: dict[str, Any] = dict(self._extra_params)
        # Kraken market orders size `amount` in base units. Do not send
        # OKX's tgtCcy — that parameter is not a Kraken field and must not
        # leak onto this venue.
        client_id = _client_order_id(intent.idempotency_key)
        if client_id is not None:
            params["userref"] = client_id

        price_arg = (self._ccxt_decimal(trade_price)
                     if intent.order_type is OrderType.LIMIT else None)
        try:
            created = self._ex.create_order(
                symbol, intent.order_type.value, intent.side.value,
                self._ccxt_decimal(amount), price_arg, params,
            )
        except Exception as exc:
            raise OrderSubmitFailed(
                f"create_order({symbol}, {intent.side.value}, {amount}) raised "
                f"{exc!r}; an order may or may not exist at Kraken. Reconcile "
                "before retrying -- this is deliberately not a rejection."
            ) from exc

        order_id = str(created.get("id") or "") if isinstance(created, dict) else ""
        if not order_id:
            raise OrderSubmitFailed(
                f"create_order({symbol}) returned no order id ({created!r}); the "
                "order cannot be tracked and may be live"
            )
        self._order_symbols[order_id] = symbol
        ref = self._await_terminal(intent, symbol, order_id, amount, timeout_s)
        return self._annotate_spot_sell(ref, symbol, clip_note)

    def _await_terminal(
        self, intent: Intent, symbol: str, order_id: str,
        requested: Decimal, timeout_s: float,
    ) -> OrderRef:
        deadline = self._monotonic() + timeout_s
        last_error: Optional[BaseException] = None

        while True:
            try:
                order = self._ex.fetch_order(order_id, symbol)
                last_error = None
            except Exception as exc:
                # Retried until the deadline: a blip should not become a
                # verdict. If it never clears, this raises rather than guessing.
                last_error, order = exc, None
            if isinstance(order, dict):
                status = self._map_status(order, requested)
                if status is not None:
                    return self._build_ref(intent, order, order_id, status)
            if self._monotonic() >= deadline:
                break
            self._sleep(self._poll_interval_s)

        notes = [f"still working after {timeout_s}s"]

        if last_error is not None:
            # One more read before escalating. A poll window lost to a transient
            # outage is not evidence about the order, and the venue's own view
            # is cheap; only a read that also fails is genuinely unknown.
            try:
                recovered = self._ex.fetch_order(order_id, symbol)
            except Exception as exc:
                raise OrderStateUnknown(
                    order_id,
                    f"could not read the state of a live {symbol} order within "
                    f"{timeout_s}s: {exc!r}",
                ) from exc
            unreadable = f"state was unreadable during the window: {last_error!r}"
            status = self._map_status(recovered, requested)
            if status is not None:
                return self._build_ref(intent, recovered, order_id, status, [unreadable])
            notes.append(unreadable)
        if self._cancel_on_timeout:
            try:
                self._ex.cancel_order(order_id, symbol)
            except Exception as exc:
                notes.append(f"cancel after timeout failed: {exc!r}")
        try:
            final = self._ex.fetch_order(order_id, symbol)
        except Exception as exc:
            raise OrderStateUnknown(
                order_id, f"timed out and could not confirm final state: {exc!r}"
            ) from exc

        status = self._map_status(final, requested)
        if status is None:
            # Genuinely still open at the venue. TIMEOUT says exactly that.
            notes.append("order is still open at Kraken")
            status = OrderStatus.TIMEOUT
        return self._build_ref(intent, final, order_id, status, notes)

    @staticmethod
    def _map_status(order: Mapping[str, Any], requested: Decimal) -> Optional[OrderStatus]:
        """Terminal status, or None while the order is still working.

        PARTIAL is never returned: OrderStatus.terminal excludes it, and
        OrderRef refuses a non-terminal status. A part-filled order that the
        venue has closed is CANCELLED carrying its real filled_qty.
        """
        filled = _dec_opt(order.get("filled")) or Decimal("0")
        raw = str(order.get("status") or "").strip().lower()

        if raw in ("rejected", "canceled", "cancelled", "expired"):
            return OrderStatus.REJECTED if raw == "rejected" else OrderStatus.CANCELLED
        if filled >= requested and requested > 0:
            return OrderStatus.FILLED
        if raw == "closed":
            # Closed without the full amount: the remainder will never fill.
            return OrderStatus.FILLED if filled >= requested else OrderStatus.CANCELLED
        return None

    def _build_ref(
        self, intent: Intent, order: Mapping[str, Any], order_id: str,
        status: OrderStatus, notes: Optional[list[str]] = None,
    ) -> OrderRef:
        notes = list(notes or [])
        filled = _dec_opt(order.get("filled")) or Decimal("0")

        avg = _dec_opt(order.get("average"))
        if avg is None:
            cost = _dec_opt(order.get("cost"))
            if cost is not None and filled > 0:
                avg = cost / filled
        # order["price"] is deliberately NOT a fallback: on a market order it is
        # the venue's reference, not a fill price, and reporting it as one
        # invents a P&L.
        if filled > 0 and avg is None:
            notes.append(
                f"venue reported {filled} filled with no average price; "
                "avg_price is unknown, not zero"
            )

        base, _, quote = str(order.get("symbol") or "").partition("/")
        base = base or intent.instrument.symbol.upper()
        quote = quote or self.base_currency

        fills: list[Fill] = []
        trades = order.get("trades")
        if isinstance(trades, list) and trades:
            for trade in trades:
                if not isinstance(trade, dict):
                    continue
                t_qty = _dec_opt(trade.get("amount"))
                t_price = _dec_opt(trade.get("price"))
                if t_qty is None or t_price is None or t_qty <= 0:
                    notes.append(f"unreadable trade on order {order_id}: {trade!r}")
                    continue
                t_fee, fee_note = self._fees_in_quote(
                    self._fee_entries(trade), base, quote, t_price
                )
                if fee_note:
                    notes.append(fee_note)
                fills.append(Fill(
                    order_id=order_id, instrument=intent.instrument,
                    side=intent.side, qty=t_qty, price=t_price, fees=t_fee,
                    at=self._timestamp(trade) or self._timestamp(order) or self._now(),
                    mode=self.mode,
                ))
        elif filled > 0 and avg is not None:
            fee, fee_note = self._fees_in_quote(
                self._fee_entries(order), base, quote, avg
            )
            if fee_note:
                notes.append(fee_note)
            fills.append(Fill(
                order_id=order_id, instrument=intent.instrument, side=intent.side,
                qty=filled, price=avg, fees=fee,
                at=self._timestamp(order) or self._now(), mode=self.mode,
            ))

        return OrderRef(
            order_id=order_id, status=status, intent=intent, mode=self.mode,
            filled_qty=filled, avg_price=avg, fills=tuple(fills),
            error="; ".join(notes) if notes else None,
        )

    @staticmethod
    def _fee_entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        entries = payload.get("fees")
        out = [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
        single = payload.get("fee")
        if isinstance(single, dict) and single not in out:
            out.append(single)
        return out

    def _fees_in_quote(
        self, entries: Sequence[Mapping[str, Any]], base: str, quote: str,
        price: Decimal,
    ) -> tuple[Decimal, Optional[str]]:
        """Fee cost expressed in the quote currency.

        Kraken charges a spot BUY fee in the coin bought, so summing raw costs
        across currencies would add coins to dollars. Anything that cannot be
        converted is reported in the note rather than dropped -- an
        understated fee is an overstated P&L.
        """
        total = Decimal("0")
        unconverted: list[str] = []
        for entry in entries:
            cost = _dec_opt(entry.get("cost"))
            if cost is None:
                continue
            # Kraken reports a fee as a negative deduction. abs() would also turn a
            # maker rebate into a cost, which errs toward under-reporting P&L.
            cost = abs(cost)
            currency = str(entry.get("currency") or "").upper()
            if not currency or currency == quote.upper():
                total += cost
            elif currency == base.upper():
                total += cost * price
            else:
                unconverted.append(f"{cost} {currency}")
        note = (
            "fees not converted to " + quote + ": " + ", ".join(unconverted)
            if unconverted else None
        )
        return total, note

    @staticmethod
    def _timestamp(payload: Mapping[str, Any]) -> Optional[dt.datetime]:
        millis = _dec_opt(payload.get("timestamp"))
        if millis is None or millis <= 0:
            return None
        return dt.datetime.fromtimestamp(float(millis) / 1000.0, tz=UTC)

    def _unplaced(self, intent: Intent, detail: str) -> OrderRef:
        """A rejection for an order that was never sent. Safe to report as
        REJECTED precisely because nothing exists at the venue."""
        ref = _client_order_id(intent.idempotency_key)
        key = str(ref) if ref is not None else uuid4().hex
        return OrderRef(
            order_id=f"unplaced:{key}", status=OrderStatus.REJECTED, intent=intent,
            mode=self.mode, filled_qty=Decimal("0"), avg_price=None, fills=(),
            error=detail,
        )

    # --------------------------------------------------------- native stops

    def free_base(self, instrument: Instrument) -> Optional[Decimal]:
        """Spendable base, used only as a capacity check for overlapping stops.

        The stop's size is never this number. A larger free balance (the
        manual BTC that shares the account) must not become the order qty.
        """
        symbol = self.resolve_symbol(instrument)
        if symbol is None:
            return None
        return self._free_base(symbol)

    def place_native_stop(
        self, instrument: Instrument, qty: Decimal, stop_price: Decimal,
        timeout_s: float = 10.0,
    ) -> StopPlaceResult:
        """Rest a sell stop-loss for ``qty`` at ``stop_price``.

        ``qty`` is the engine ledger size. This method does not read the
        wallet to choose it. It then polls open orders until the txid is
        visible or ``timeout_s`` elapses. Unconfirmed is not success: an
        order id may still be returned so the caller can try to cancel it.
        """
        if qty <= 0 or stop_price <= 0:
            return StopPlaceResult(False, None, qty, stop_price,
                                   "stop qty and price must be > 0")
        if self._key_freeze():
            return StopPlaceResult(
                False, None, qty, stop_price,
                "key_freeze: not placing a native stop; existing stops stay",
            )
        symbol = self.resolve_symbol(instrument)
        if symbol is None:
            return StopPlaceResult(
                False, None, qty, stop_price,
                f"{instrument.symbol}: no active Kraken spot market",
            )
        try:
            amount = self._quantize_amount(symbol, qty)
            price = self._quantize_price(symbol, stop_price)
        except Exception as exc:
            return StopPlaceResult(False, None, qty, stop_price,
                                   f"could not quantize stop: {exc!r}")
        if amount <= 0 or price <= 0:
            return StopPlaceResult(
                False, None, amount, price,
                f"stop quantized to amount {amount} price {price}",
            )
        # Free balance includes the manual reserve. A stop may use only what
        # is left after that reserve. A fee-sized gap is clipped; a wider
        # gap is not sent (the engine flattens). Unknown free keeps the
        # requested ledger qty — the wallet is still not the size.
        try:
            spendable = self._spendable_base(symbol)
        except ManualReservedError as exc:
            return StopPlaceResult(
                False, None, qty, stop_price,
                f"MANUAL_RESERVED_BASE unreadable: {exc}",
            )
        except Exception:
            spendable = None
        if spendable is not None:
            sized, decision = stop_qty_for_ledger(qty, spendable)
            if decision == "halt":
                return StopPlaceResult(
                    False, None, qty, stop_price,
                    f"non-reserved free {spendable} is more than "
                    f"{STOP_FEE_CLIP_LIMIT * 100}% under ledger qty {qty}",
                )
            if decision == "clip":
                try:
                    amount = self._quantize_amount(symbol, sized)
                except Exception as exc:
                    return StopPlaceResult(
                        False, None, sized, price,
                        f"could not quantize clipped stop: {exc!r}",
                    )
                if amount <= 0:
                    return StopPlaceResult(
                        False, None, amount, price,
                        "clipped stop quantized to 0",
                    )
        try:
            created = self._ex.create_order(
                symbol, "stop-loss", "sell",
                self._ccxt_decimal(amount), self._ccxt_decimal(price), {},
            )
        except Exception as exc:
            return StopPlaceResult(
                False, None, amount, price,
                f"stop-loss create_order raised {exc!r}; the stop may or may "
                "not exist. Not confirmed.",
            )
        txid = str(created.get("id") or "") if isinstance(created, dict) else ""
        if not txid:
            return StopPlaceResult(
                False, None, amount, price,
                f"stop-loss create_order returned no id ({created!r})",
            )
        self._order_symbols[txid] = symbol
        if self._confirm_stop(symbol, txid, timeout_s):
            return StopPlaceResult(True, txid, amount, price, "open")
        return StopPlaceResult(
            False, txid, amount, price,
            f"stop {txid} was not visible on open orders within {timeout_s}s",
        )

    def _confirm_stop(self, symbol: str, txid: str, timeout_s: float) -> bool:
        deadline = self._monotonic() + timeout_s
        while True:
            visible = self._stop_visible(symbol, txid)
            if visible is True:
                return True
            if self._monotonic() >= deadline:
                return False
            self._sleep(self._poll_interval_s)

    def _stop_visible(self, symbol: str, txid: str) -> Optional[bool]:
        """True if the stop is resting, False if the book was readable and
        the id was not on it, None if the book could not be read."""
        try:
            orders = self._ex.fetch_open_orders(symbol)
        except Exception:
            orders = None
        if isinstance(orders, list):
            for order in orders:
                if isinstance(order, dict) and str(order.get("id") or "") == txid:
                    return True
        try:
            order = self._ex.fetch_order(txid, symbol)
        except Exception:
            return None if not isinstance(orders, list) else False
        if not isinstance(order, dict):
            return None if not isinstance(orders, list) else False
        status = str(order.get("status") or "").strip().lower()
        if status in {"open", "pending", "new"}:
            return True
        if isinstance(orders, list):
            return False
        return None

    def native_stop_open(self, txid: str, instrument: Instrument) -> Optional[bool]:
        """True / False when the book answers, None when it does not.

        None is not "missing". A blip must not be read as a cancelled stop.
        """
        if not txid:
            return False
        symbol = self.resolve_symbol(instrument)
        if symbol is None:
            return None
        return self._stop_visible(symbol, txid)

    def cancel_native_stop(self, txid: str, instrument: Instrument) -> StopCancelResult:
        """Cancel a resting stop. If it already filled, return that fill.

        The fill quantity is whatever the venue reports for the order we
        placed (the ledger qty). Key freeze does not send the cancel.
        """
        if self._key_freeze():
            return StopCancelResult(
                "blocked",
                reason="key_freeze: not cancelling the native stop",
            )
        symbol = self.resolve_symbol(instrument)
        if symbol is None:
            return StopCancelResult(
                "unknown",
                reason=f"{instrument.symbol}: no market, cannot cancel {txid}",
            )
        self._order_symbols.setdefault(txid, symbol)
        cancel_error: Optional[BaseException] = None
        try:
            self._ex.cancel_order(txid, symbol)
        except Exception as exc:
            cancel_error = exc
        try:
            order = self._ex.fetch_order(txid, symbol)
        except Exception as exc:
            return StopCancelResult(
                "unknown",
                reason=(f"could not read stop {txid} after cancel: {exc!r}"
                        + (f" (cancel raised {cancel_error!r})" if cancel_error else "")),
            )
        if not isinstance(order, dict):
            return StopCancelResult("unknown", reason=f"stop {txid} read was not an order")
        requested = _dec_opt(order.get("amount")) or Decimal("0")
        status = self._map_status(order, requested)
        filled = _dec_opt(order.get("filled")) or Decimal("0")
        if status is OrderStatus.FILLED or (filled > 0 and status is OrderStatus.CANCELLED):
            fill = self._stop_fill(instrument, txid, order, filled)
            if fill is None:
                return StopCancelResult(
                    "unknown",
                    reason=f"stop {txid} filled but the fill price was unreadable",
                )
            return StopCancelResult("filled", fill=fill, reason="stop already filled")
        if status is OrderStatus.CANCELLED or status is OrderStatus.REJECTED:
            return StopCancelResult("cancelled", reason="cancelled")
        return StopCancelResult(
            "still_open",
            reason=f"stop {txid} still open after cancel"
            + (f" ({cancel_error!r})" if cancel_error else ""),
        )

    def _stop_fill(self, instrument: Instrument, txid: str,
                   order: Mapping[str, Any], filled: Decimal) -> Optional[Fill]:
        if filled <= 0:
            return None
        price = _dec_opt(order.get("average"))
        if price is None or price <= 0:
            cost = _dec_opt(order.get("cost"))
            if cost is not None and filled > 0:
                price = cost / filled
        if price is None or price <= 0:
            return None
        return Fill(
            order_id=txid, instrument=instrument, side=Side.SELL,
            qty=filled, price=price, fees=Decimal("0"),
            at=self._timestamp(order) or self._now(), mode=self.mode,
        )

    # --------------------------------------------------------------- cancelling

    def cancel(self, order_id: str) -> bool:
        """True only when Kraken confirms the order is cancelled.

        A key freeze returns False without calling the venue. False here
        means "not confirmed cancelled", which is also what a freeze is.

        False can mean 'still working' OR 'it filled before the cancel landed';
        those must not be conflated, so call cancel_detail() when the
        difference matters.
        """
        if self._key_freeze():
            return False
        return self.cancel_detail(order_id) is OrderStatus.CANCELLED

    def cancel_detail(self, order_id: str, symbol: Optional[str] = None) -> Optional[OrderStatus]:
        """The order's confirmed state after a cancel attempt. None means the
        venue still shows it working."""
        if self._key_freeze():
            return None
        symbol = symbol or self._order_symbols.get(order_id)
        if symbol is None:
            raise UnknownOrder(
                f"cancel({order_id!r}): this adapter did not place that order, so "
                "the symbol Kraken requires for a cancel is unknown; pass symbol="
            )
        cancel_error: Optional[BaseException] = None
        try:
            self._ex.cancel_order(order_id, symbol)
        except Exception as exc:
            # Not fatal on its own: ccxt raises OrderNotFound for an order that
            # already resolved. The venue's own view below is what decides.
            cancel_error = exc
        try:
            order = self._ex.fetch_order(order_id, symbol)
        except Exception as exc:
            raise OrderStateUnknown(
                order_id,
                f"cancel could not be confirmed: {exc!r}"
                + (f" (cancel_order raised {cancel_error!r})" if cancel_error else ""),
            ) from exc
        requested = _dec_opt(order.get("amount")) or Decimal("0")
        return self._map_status(order, requested)
