"""
Paper broker adapter.

Paper is an ADAPTER, not a flag: the engine, the gate and the strategies cannot
tell this apart from Dhan or OKX. That equivalence only holds if paper refuses
to be kinder than a venue. Three ways a paper fill lies, all refused here:

  * filling every LIMIT wherever the market actually is — teaches a strategy
    that its passive entries are free,
  * filling at the quoted price at no cost — turns a negative-edge strategy
    into a positive-edge backtest,
  * inventing a price when no quote is available.

So a non-marketable LIMIT sits OPEN and comes back TIMEOUT or CANCELLED, every
fill pays adverse slippage plus the real charge stack for its asset class, and
a missing quote is a REJECTED order rather than a guess.

Everything is deterministic under `seed`: the same seed and the same sequence
of intents produce identical fills, so a backtest is reproducible and a test
can assert on an exact price.
"""

from __future__ import annotations

import datetime as dt
import random
import threading
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from core.profiles import AssetClassProfile, profile_for
from core.safety import StopCancelResult, StopPlaceResult
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

UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Money hygiene
# ---------------------------------------------------------------------------

def _money(value: Decimal | int | str, what: str) -> Decimal:
    """Coerce to Decimal, but refuse float outright.

    A float that has been through a JSON round-trip is off by fractions of a
    paisa; accumulate that across a thousand fills and the simulated book stops
    reconciling against the real one for reasons nobody can find. Callers pass
    Decimal or str.
    """
    if isinstance(value, float):
        raise TypeError(f"{what} must not be a float — pass Decimal or str, got {value!r}")
    return value if isinstance(value, Decimal) else Decimal(value)


def _quantize(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quote:
    """A price, and optionally how much of it is there.

    `depth is None` means "unbounded" — the common case for a test that is not
    exercising partial fills. `depth == 0` means the price exists but nothing
    is available at it, which is a real state and must not be confused with
    "no quote".

    `live` is what stops a frozen option LTP from silently pricing a spread.
    A mark we cannot age (Dhan's bare LTP has no timestamp) is stored as
    `live=False` so it is distinguishable from a live print, and every path
    that would *act* on a quote — place, mark-to-market, engine snapshot —
    treats a frozen mark as absent, never as a number.
    """
    price: Decimal
    depth: Optional[Decimal] = None
    as_of: Optional[dt.datetime] = None
    live: bool = True

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"quote price must be > 0, got {self.price}")
        if self.depth is not None and self.depth < 0:
            raise ValueError(f"quote depth must be >= 0, got {self.depth}")
        if self.as_of is not None:
            if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
                raise ValueError(
                    f"quote as_of must be timezone-aware, got {self.as_of!r}"
                )


@runtime_checkable
class QuoteSource(Protocol):
    """Returns None — never a zero, never a stale last-known — when it has no
    quote for the instrument. The adapter turns None into a REJECTED order."""

    def quote(self, instrument: Instrument) -> Optional[Quote]: ...


class InMemoryQuoteSource:
    """Dict-backed quote source for tests and replay.

    Deliberately has no "default price" and no fallback: an unknown symbol
    returns None so that a wiring mistake shows up as rejected orders instead
    of a book priced off a made-up number.
    """

    def __init__(self, prices: Optional[dict[str, Decimal | int | str]] = None) -> None:
        self._quotes: dict[str, Quote] = {}
        for symbol, price in (prices or {}).items():
            self.set(symbol, price)

    def set(self, symbol: str, price: Decimal | int | str,
            depth: Optional[Decimal | int | str] = None, *,
            as_of: Optional[dt.datetime] = None, live: bool = True) -> None:
        px = _money(price, f"quote price for {symbol}")
        dp = None if depth is None else _money(depth, f"quote depth for {symbol}")
        self._quotes[symbol] = Quote(price=px, depth=dp, as_of=as_of, live=live)

    def remove(self, symbol: str) -> None:
        """Raises if the symbol was never there — a silent no-op would let a
        test believe it had removed a quote it had misspelled."""
        del self._quotes[symbol]

    def quote(self, instrument: Instrument) -> Optional[Quote]:
        return self._quotes.get(instrument.symbol)

    def __len__(self) -> int:
        return len(self._quotes)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

@runtime_checkable
class FeeModel(Protocol):
    def charge(self, instrument: Instrument, side: Side, qty: Decimal,
               price: Decimal, *, include_per_order_fee: bool) -> Decimal: ...


@dataclass(frozen=True)
class IndianChargeStack:
    """The Indian charge stack, which is not one fee but seven.

    Rates are fractions of turnover. STT and stamp duty are one-sided, which is
    why a round trip costs materially more in one direction than the other —
    a paper adapter charging a symmetric flat bps gets intraday scalping P&L
    wrong by enough to flip its sign.

    `include_per_order_fee` exists because brokerage is charged per ORDER, not
    per fill. Charging the flat leg on each partial fill would make a
    thin-liquidity fill look far more expensive than it is.
    """
    name: str
    brokerage_flat: Decimal
    brokerage_rate: Decimal
    exchange_txn_rate: Decimal
    stt_buy_rate: Decimal
    stt_sell_rate: Decimal
    sebi_rate: Decimal
    stamp_buy_rate: Decimal
    gst_rate: Decimal
    precision: int = 2

    def charge(self, instrument: Instrument, side: Side, qty: Decimal,
               price: Decimal, *, include_per_order_fee: bool) -> Decimal:
        turnover = qty * price
        if include_per_order_fee:
            brokerage = (min(self.brokerage_flat, turnover * self.brokerage_rate)
                         if self.brokerage_rate > 0 else self.brokerage_flat)
        else:
            brokerage = Decimal("0")
        stt = turnover * (self.stt_buy_rate if side is Side.BUY else self.stt_sell_rate)
        txn = turnover * self.exchange_txn_rate
        sebi = turnover * self.sebi_rate
        stamp = turnover * self.stamp_buy_rate if side is Side.BUY else Decimal("0")
        gst = (brokerage + txn + sebi) * self.gst_rate
        return _quantize(brokerage + stt + txn + sebi + stamp + gst, self.precision)


@dataclass(frozen=True)
class CryptoTakerFee:
    """Flat taker bps on notional. Paper assumes TAKER on every fill: we only
    fill marketable orders, and a marketable order crosses the spread."""
    bps: Decimal
    precision: int = 8

    def charge(self, instrument: Instrument, side: Side, qty: Decimal,
               price: Decimal, *, include_per_order_fee: bool) -> Decimal:
        return _quantize(qty * price * self.bps / Decimal("10000"), self.precision)


# Rates as of the 2024-25 NSE/SEBI schedules. They drift; they are constants in
# one place so that updating them is a one-line change, not an archaeology dig.
NSE_CASH_INTRADAY = IndianChargeStack(
    name="nse-cash-intraday",
    brokerage_flat=Decimal("20"), brokerage_rate=Decimal("0.0003"),
    exchange_txn_rate=Decimal("0.0000297"),
    stt_buy_rate=Decimal("0"), stt_sell_rate=Decimal("0.00025"),
    sebi_rate=Decimal("0.000001"), stamp_buy_rate=Decimal("0.00003"),
    gst_rate=Decimal("0.18"),
)
NFO_OPTIONS = IndianChargeStack(
    name="nfo-options",
    brokerage_flat=Decimal("20"), brokerage_rate=Decimal("0"),
    exchange_txn_rate=Decimal("0.0003503"),
    stt_buy_rate=Decimal("0"), stt_sell_rate=Decimal("0.001"),
    sebi_rate=Decimal("0.000001"), stamp_buy_rate=Decimal("0.00003"),
    gst_rate=Decimal("0.18"),
)
NFO_FUTURES = IndianChargeStack(
    name="nfo-futures",
    brokerage_flat=Decimal("20"), brokerage_rate=Decimal("0.0003"),
    exchange_txn_rate=Decimal("0.0000173"),
    stt_buy_rate=Decimal("0"), stt_sell_rate=Decimal("0.0002"),
    sebi_rate=Decimal("0.000001"), stamp_buy_rate=Decimal("0.00002"),
    gst_rate=Decimal("0.18"),
)
OKX_SPOT_TAKER = CryptoTakerFee(bps=Decimal("10"))


class DefaultFeeModel:
    """Picks the charge stack per instrument.

    Instrument carries no derivative-kind field, so an expiry is read as an
    OPTION — the dominant NFO case for this engine. A futures book must pass
    `fees=NFO_FUTURES` explicitly; guessing wrong the other way would
    understate STT by 5x on the sell leg.
    """

    def charge(self, instrument: Instrument, side: Side, qty: Decimal,
               price: Decimal, *, include_per_order_fee: bool) -> Decimal:
        stack = self.stack_for(instrument)
        return stack.charge(instrument, side, qty, price,
                            include_per_order_fee=include_per_order_fee)

    @staticmethod
    def stack_for(instrument: Instrument) -> FeeModel:
        if instrument.asset_class is AssetClass.CRYPTO:
            return OKX_SPOT_TAKER
        return NFO_OPTIONS if instrument.expiry is not None else NSE_CASH_INTRADAY


@dataclass(frozen=True)
class SlippageModel:
    """Uniform in [0, max_bps], and ALWAYS adverse.

    Real books sometimes hand you price improvement. Modelling it here would
    make the average paper fill better than the average live fill, and every
    strategy tuned on paper would then underperform on the day it goes live.
    Paper's job is to be pessimistic in the same shape the venue is expensive.
    """
    max_bps: Decimal

    def __post_init__(self) -> None:
        if self.max_bps < 0:
            raise ValueError("max_bps must be >= 0")

    def apply(self, price: Decimal, side: Side, rng: random.Random) -> Decimal:
        draw = Decimal(str(rng.random()))
        bps = self.max_bps * draw
        factor = (Decimal("1") + bps / Decimal("10000")) if side is Side.BUY \
            else (Decimal("1") - bps / Decimal("10000"))
        return price * factor


# Index options are wide; BTC/USDT top-of-book is not. Both are overridable.
DEFAULT_SLIPPAGE: dict[AssetClass, SlippageModel] = {
    AssetClass.EQUITY: SlippageModel(max_bps=Decimal("10")),
    AssetClass.CRYPTO: SlippageModel(max_bps=Decimal("5")),
}


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class PaperAdapter:
    """A BrokerAdapter that resolves orders against a QuoteSource.

    One adapter per asset class, matching the live adapters — an intent for the
    wrong class raises rather than being politely routed, because a crypto
    order landing on the equity adapter is a wiring bug and wiring bugs that
    return plausible results are the ones that reach production.
    """

    mode: Mode = Mode.PAPER
    # Same stop lifecycle as KrakenAdapter, against this process's book.
    # The resting stop qty is the qty the caller passes (the ledger), never
    # this adapter's cash or the sum of its simulated positions.
    manages_native_stops: bool = True

    def __init__(
        self,
        asset_class: AssetClass,
        quotes: QuoteSource,
        starting_cash: Decimal | int | str,
        *,
        seed: int = 0,
        fees: Optional[FeeModel] = None,
        slippage: Optional[SlippageModel] = None,
        poll_interval_s: float = 0.5,
        reject_off_tick: bool = True,
        now: Optional[Callable[[], dt.datetime]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        cash = _money(starting_cash, "starting_cash")
        if cash < 0:
            raise ValueError("starting_cash must be >= 0")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")

        self.asset_class = asset_class
        self._profile: AssetClassProfile = profile_for(asset_class)
        self._quotes = quotes
        self._cash = cash
        self._starting_cash = cash
        self._seed = seed
        self._fees: FeeModel = fees if fees is not None else DefaultFeeModel()
        self._slippage = slippage if slippage is not None else DEFAULT_SLIPPAGE[asset_class]
        self._poll_interval_s = poll_interval_s
        self._reject_off_tick = reject_off_tick
        self._now: Callable[[], dt.datetime] = now or (lambda: dt.datetime.now(UTC))
        self._sleep: Callable[[float], None] = sleep or time.sleep

        self._lock = threading.RLock()
        self._seq = 0
        self._positions: dict[tuple[str, str, Optional[str]], Position] = {}
        self._orders: dict[str, OrderRef] = {}
        self._working: set[str] = set()
        self._cancel_requested: set[str] = set()
        self._stops: dict[str, "_PaperStop"] = {}
        self._stop_seq = 0
        # Capacity for a second overlapping stop. Not a size. None means
        # "cannot tell", and the engine then cancels before replacing.
        self._stop_free: Optional[Decimal] = Decimal("1000000")
        self.fail_native_stop = False

    # -- introspection ------------------------------------------------------

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def starting_cash(self) -> Decimal:
        return self._starting_cash

    def order(self, order_id: str) -> Optional[OrderRef]:
        return self._orders.get(order_id)

    # -- BrokerAdapter ------------------------------------------------------

    def quote(self, instrument: Instrument) -> Optional[Decimal]:
        q = self._live_quote(instrument)
        return None if q is None else q.price

    def _live_quote(self, instrument: Instrument) -> Optional[Quote]:
        """The quote this adapter may act on. Frozen marks are present on the
        source (so they are distinguishable from a live print) and absent
        here (so they cannot silently price a fill or a book)."""
        q = self._quotes.quote(instrument)
        if q is None or not q.live:
            return None
        return q

    def positions(self) -> Sequence[Position]:
        with self._lock:
            return tuple(self._positions.values())

    def equity(self) -> Optional[Decimal]:
        """Cash plus the marked value of open positions.

        Returns None — not 0, not cash-only — when any open position cannot be
        marked. An unmarkable book has an UNKNOWN value; reporting cash alone
        would understate a long book as a loss and overstate a short one, and
        reporting 0 reads downstream as a 100% drawdown.
        """
        with self._lock:
            total = self._cash
            for pos in self._positions.values():
                q = self._live_quote(pos.instrument)
                if q is None:
                    return None
                total += pos.qty * q.price
            return total

    def cancel(self, order_id: str) -> bool:
        """True only if the order was still working and the cancel was taken.

        place() blocks to terminal, so this is only meaningful from another
        thread (or a test's clock hook) while an order sits OPEN. An unknown or
        already-terminal order returns False rather than pretending.
        """
        with self._lock:
            if order_id in self._working:
                self._cancel_requested.add(order_id)
                return True
            return False

    def place(self, intent: Intent, timeout_s: float = 30.0) -> OrderRef:
        instrument = intent.instrument
        if instrument.asset_class is not self.asset_class:
            raise ValueError(
                f"{instrument.symbol} is {instrument.asset_class.value}, but this "
                f"adapter trades {self.asset_class.value}"
            )

        with self._lock:
            self._seq += 1
            order_id = f"PAPER-{self.asset_class.value}-{self._seq:06d}"

        # Per-order RNG derived from the seed and the order id, NOT a shared
        # stream: quote() calls and rejected orders then cannot shift the
        # random draws of the orders around them, so a test that adds an
        # assertion in the middle still sees the same fills.
        rng = random.Random(f"{self._seed}|{order_id}")

        rejection = self._pre_trade_reject(order_id, intent)
        if rejection is not None:
            return self._record(rejection)

        qty = self._profile.quantize_qty(instrument, intent.qty)
        return self._record(self._work(order_id, intent, qty, timeout_s, rng))

    # -- pre-trade ----------------------------------------------------------

    def _pre_trade_reject(self, order_id: str, intent: Intent) -> Optional[OrderRef]:
        """Every check a venue would run before accepting the order.

        These are rejections, not exceptions: a venue refusing an order is a
        normal outcome the caller must see in the OrderRef. They run BEFORE any
        fill so that an order can never half-execute and then be refused,
        which would leave a strategy holding a leg it never chose to hold.
        """
        instrument = intent.instrument

        if intent.order_type is OrderType.MARKET and not self._profile.market_orders_allowed:
            return self._reject(order_id, intent,
                                f"{self.asset_class.value} venue rejects MARKET orders; "
                                "use a marketable LIMIT")

        qty = self._profile.quantize_qty(instrument, intent.qty)
        if qty <= 0:
            return self._reject(order_id, intent,
                                f"qty {intent.qty} quantizes to 0 for lot_size="
                                f"{instrument.lot_size}/precision={instrument.qty_precision}")

        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None  # enforced by Intent.__post_init__
            on_tick = self._profile.quantize_price(instrument, intent.limit_price)
            if self._reject_off_tick and on_tick != intent.limit_price:
                return self._reject(order_id, intent,
                                    f"limit {intent.limit_price} is off the "
                                    f"{instrument.tick_size} tick (nearest {on_tick}); "
                                    "quantize via profile_for(...).quantize_price")

        q = self._live_quote(instrument)
        if q is None:
            # The one thing a paper adapter must never do is make up a price.
            # A frozen mark is "no quote" for the order path: using it would
            # silently price a live or paper spread off a stale LTP.
            stored = self._quotes.quote(instrument)
            if stored is not None and not stored.live:
                return self._reject(
                    order_id, intent,
                    f"frozen (unageable) mark for {instrument.symbol}; "
                    "refusing to price an order off a stale LTP",
                )
            return self._reject(order_id, intent,
                                f"no quote for {instrument.symbol}")

        worst = self._worst_case_price(intent, q)
        notional = qty * worst
        if instrument.min_notional > 0 and notional < instrument.min_notional:
            return self._reject(order_id, intent,
                                f"notional {notional} below min_notional "
                                f"{instrument.min_notional}")

        if intent.side is Side.BUY:
            cost = notional + self._fees.charge(instrument, Side.BUY, qty, worst,
                                                include_per_order_fee=True)
            if cost > self._cash:
                return self._reject(order_id, intent,
                                    f"worst-case cost {cost} exceeds cash {self._cash}")
        return None

    def _worst_case_price(self, intent: Intent, q: Quote) -> Decimal:
        """Upper bound on what a BUY can pay / lower bound on a SELL's credit.

        Sizing checks use the bound, not the mid: a check that passes on the
        quote and then fills through slippage is a check that did not run.
        """
        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None
            return intent.limit_price
        edge = Decimal("1") + self._slippage.max_bps / Decimal("10000")
        return q.price * edge if intent.side is Side.BUY else q.price / edge

    # -- working the order --------------------------------------------------

    def _work(self, order_id: str, intent: Intent, qty: Decimal,
              timeout_s: float, rng: random.Random) -> OrderRef:
        instrument = intent.instrument
        start = self._now()
        remaining = qty
        fills: list[Fill] = []

        with self._lock:
            self._working.add(order_id)
        try:
            while True:
                q = self._live_quote(instrument)
                if q is not None:
                    take = self._fillable(intent, q, remaining)
                    if take > 0:
                        price = self._exec_price(intent, q, rng)
                        fee = self._fees.charge(
                            instrument, intent.side, take, price,
                            include_per_order_fee=not fills,
                        )
                        fill = Fill(
                            order_id=order_id, instrument=instrument, side=intent.side,
                            qty=take, price=price, fees=fee, at=self._now(),
                            mode=Mode.PAPER,
                        )
                        with self._lock:
                            self._apply_fill(fill, intent)
                        fills.append(fill)
                        remaining -= take
                        if remaining <= 0:
                            return self._terminal(order_id, intent, OrderStatus.FILLED, fills)

                with self._lock:
                    cancelled = order_id in self._cancel_requested
                if cancelled:
                    return self._terminal(order_id, intent, OrderStatus.CANCELLED, fills,
                                          error="cancelled while open")

                elapsed = (self._now() - start).total_seconds()
                if elapsed >= timeout_s:
                    # TIMEOUT, not CANCELLED: we stopped waiting. On a real
                    # venue the order may still be resting, and the caller has
                    # to reconcile. Paper reports the same shape.
                    return self._terminal(
                        order_id, intent, OrderStatus.TIMEOUT, fills,
                        error=f"not filled within {timeout_s}s "
                              f"({remaining} of {qty} unfilled)",
                    )
                self._sleep(min(self._poll_interval_s, max(timeout_s - elapsed, 0.0)))
        finally:
            with self._lock:
                self._working.discard(order_id)
                self._cancel_requested.discard(order_id)

    def _fillable(self, intent: Intent, q: Quote, remaining: Decimal) -> Decimal:
        """How much of `remaining` trades right now. 0 means the order rests.

        A LIMIT that is not marketable fills NOTHING. This is the whole reason
        paper is worth running: a strategy whose passive entries never get hit
        must find that out here, not in live.
        """
        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None
            marketable = (intent.limit_price >= q.price if intent.side is Side.BUY
                          else intent.limit_price <= q.price)
            if not marketable:
                return Decimal("0")

        available = remaining if q.depth is None else min(remaining, q.depth)
        if available <= 0:
            return Decimal("0")
        return self._profile.quantize_qty(intent.instrument, available)

    def _exec_price(self, intent: Intent, q: Quote, rng: random.Random) -> Decimal:
        instrument = intent.instrument
        raw = self._slippage.apply(q.price, intent.side, rng)
        price = self._profile.quantize_price(instrument, raw)
        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None
            # The limit is a hard bound, including against the tick rounding
            # above. Paper must never report a price the strategy did not
            # authorise — that is a fill the live venue would not have given.
            price = (min(price, intent.limit_price) if intent.side is Side.BUY
                     else max(price, intent.limit_price))
        return price

    # -- state --------------------------------------------------------------

    def _apply_fill(self, fill: Fill, intent: Intent) -> None:
        """Cash and position bookkeeping for one fill.

        Cash moves by the FULL notional in both directions rather than by
        margin. That is not how an F&O account is funded, but it makes
        equity == cash + Σ(qty × mark) exactly right for longs and shorts
        alike, which is the number the risk circuit reads.
        """
        notional = fill.qty * fill.price
        signed = fill.qty if fill.side is Side.BUY else -fill.qty
        self._cash += (-notional if fill.side is Side.BUY else notional) - fill.fees

        key = (fill.instrument.symbol, intent.strategy_id, intent.structure_id)
        old = self._positions.get(key)
        if old is None:
            self._positions[key] = Position(
                instrument=fill.instrument, qty=signed, avg_price=fill.price,
                mode=Mode.PAPER, strategy_id=intent.strategy_id,
                structure_id=intent.structure_id, opened_at=fill.at,
            )
            return

        new_qty = old.qty + signed
        if new_qty == 0:
            del self._positions[key]
            return

        if (old.qty > 0) == (signed > 0):
            new_avg = ((old.avg_price * abs(old.qty) + fill.price * fill.qty)
                       / abs(new_qty))
            opened = old.opened_at
        elif (new_qty > 0) == (old.qty > 0):
            new_avg = old.avg_price          # reduced; the basis does not move
            opened = old.opened_at
        else:
            new_avg = fill.price             # flipped through zero: new position
            opened = fill.at

        self._positions[key] = Position(
            instrument=old.instrument, qty=new_qty, avg_price=new_avg,
            mode=Mode.PAPER, strategy_id=old.strategy_id,
            structure_id=old.structure_id, opened_at=opened,
        )

    # -- OrderRef construction ---------------------------------------------

    def _reject(self, order_id: str, intent: Intent, error: str) -> OrderRef:
        return OrderRef(order_id=order_id, status=OrderStatus.REJECTED,
                        intent=intent, mode=Mode.PAPER, error=error)

    def _terminal(self, order_id: str, intent: Intent, status: OrderStatus,
                  fills: list[Fill], error: Optional[str] = None) -> OrderRef:
        filled = sum((f.qty for f in fills), Decimal("0"))
        avg = (sum((f.qty * f.price for f in fills), Decimal("0")) / filled
               if filled > 0 else None)
        # PARTIAL is not a terminal status in the contract, so a partly-filled
        # order that ran out of time is TIMEOUT carrying filled_qty > 0. The
        # caller must read filled_qty, not just the status.
        return OrderRef(order_id=order_id, status=status, intent=intent,
                        mode=Mode.PAPER, filled_qty=filled, avg_price=avg,
                        fills=tuple(fills), error=error)

    def _record(self, ref: OrderRef) -> OrderRef:
        with self._lock:
            self._orders[ref.order_id] = ref
        return ref

    # -- native stops (same contract as KrakenAdapter) ---------------------

    def free_base(self, instrument: Instrument) -> Optional[Decimal]:
        return self._stop_free

    def place_native_stop(
        self, instrument: Instrument, qty: Decimal, stop_price: Decimal,
        timeout_s: float = 10.0,
        fill_at: Optional[dt.datetime] = None,
        use_cached_balance: bool = False,
    ) -> StopPlaceResult:
        if qty <= 0 or stop_price <= 0:
            return StopPlaceResult(False, None, qty, stop_price,
                                   "stop qty and price must be > 0")
        if self.fail_native_stop:
            return StopPlaceResult(
                False, None, qty, stop_price,
                "paper native stop rejected before it rested",
            )
        _ = (fill_at, use_cached_balance)  # engine owns the fill budget
        with self._lock:
            self._stop_seq += 1
            txid = f"PAPER-STOP-{self.asset_class.value}-{self._stop_seq:06d}"
            self._stops[txid] = _PaperStop(
                txid=txid, instrument=instrument, qty=qty, stop_price=stop_price,
            )
        rested = self._now()
        return StopPlaceResult(
            True, txid, qty, stop_price, "open",
            placed_at=rested, confirmed_at=rested,
        )

    def native_stop_open(self, txid: str, instrument: Instrument) -> Optional[bool]:
        with self._lock:
            return txid in self._stops

    def describe_native_stop(self, txid: str, instrument: Instrument) -> StopCancelResult:
        """Read a paper stop without cancelling it.

        A trigger that already printed is a fill. An id this book does not
        hold is unknown, not a licence to sell again.
        """
        with self._lock:
            resting = self._stops.get(txid)
            if resting is None:
                return StopCancelResult("unknown", reason=f"no paper stop {txid}")
            quote = self._live_quote(resting.instrument)
            triggered = quote is not None and quote.price <= resting.stop_price
            if not triggered:
                return StopCancelResult("still_open", reason="resting")
            self._stops.pop(txid, None)
            fill = Fill(
                order_id=txid, instrument=resting.instrument, side=Side.SELL,
                qty=resting.qty, price=resting.stop_price, fees=Decimal("0"),
                at=self._now(), mode=Mode.PAPER,
            )
            return StopCancelResult("filled", fill=fill, reason="paper stop triggered")

    def cancel_native_stop(self, txid: str, instrument: Instrument) -> StopCancelResult:
        with self._lock:
            resting = self._stops.get(txid)
            if resting is None:
                return StopCancelResult("unknown", reason=f"no paper stop {txid}")
            quote = self._live_quote(resting.instrument)
            triggered = quote is not None and quote.price <= resting.stop_price
            self._stops.pop(txid, None)
            if triggered:
                fill = Fill(
                    order_id=txid, instrument=resting.instrument, side=Side.SELL,
                    qty=resting.qty, price=resting.stop_price, fees=Decimal("0"),
                    at=self._now(), mode=Mode.PAPER,
                )
                return StopCancelResult("filled", fill=fill,
                                        reason="paper stop already triggered")
        return StopCancelResult("cancelled", reason="cancelled")


@dataclass(frozen=True)
class _PaperStop:
    txid: str
    instrument: Instrument
    qty: Decimal
    stop_price: Decimal
