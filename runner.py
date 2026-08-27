"""
The composition root: the one place that knows how every part is wired.

Nothing else in this package constructs its own collaborators. Strategies do
not build adapters, the engine does not build a gate, the API does not build an
engine. That is what makes each of them testable against fakes — and it means
there has to be exactly one file that finally says which real object goes where.
This is it, and it is deliberately the only file allowed to read the
environment for anything other than the two arming flags.

Three wiring problems are solved here that no single module could solve alone,
because each of them lives in the seam BETWEEN two modules:

1.  **The gate needs a durable dedup and daily-count store.** `core.gate`
    declares `GateStateStoreProto` and warns that an in-process dict satisfies
    it and must never be used outside tests. `core.store.SQLiteGateStateStore`
    is that store, and it is pointed at the same file as the ledger here.

2.  **The gate takes ONE risk circuit, but a circuit is bound to one mode.**
    `engine.py` says so in its own header and leaves it to the integrator. With
    one class armed and another not, there is no single circuit that is correct
    for both. `_PerClassCircuit` below routes each `read()` to the circuit bound
    to the mode THAT class actually resolved to, which is the only answer that
    does not judge one book by the other's baselines.

3.  **The API's event stream and strategy registry are in-process only.**
    `api/server.py` states the caveat plainly: run the API in a different
    process from the engine and `/api/events` carries almost nothing while
    `/api/strategies` 503s. So this runner puts both in ONE process and bridges
    the engine's bus into the API's, which is what makes those endpoints tell
    the truth. The durable surfaces — positions, orders, P&L, risk and control
    state — go through SQLite and would have been correct either way.

ARMING IS NOT HERE. `EQUITY_LIVE` and `CRYPTO_LIVE` are read by `core.gate` and
by `engine.py`, from the environment, and this file never sets them, never
defaults them, and never passes a value for them. Building a live adapter is
also not arming: an adapter that is constructed but not armed is reported at
boot and never called. Setting those variables is the owner's action and
nobody else's.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import os
import sys
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

import engine as engine_mod
from adapters.paper import InMemoryQuoteSource, PaperAdapter
from api import server as api_mod
from core.gate import GateConfig, OrderGate
from core.risk import RiskCircuit
from core.store import SQLiteGateStateStore, SQLitePositionStore
from core.types import AssetClass, Instrument, Mode
from engine import ClassWiring, Engine, EngineConfig, EventKind
from feeds.base import FeedError
from feeds.dhan_feed import DhanFeed
from feeds.okx_feed import OkxFeed
from strategies import Bar, CryptoEmaTrendStrategy, EquityMeanReversionStrategy
from strategies.base import BARS_KEY

LOG = logging.getLogger("trader.runner")
UTC = dt.timezone.utc

__all__ = [
    "CompositeFeed",
    "FeedChoice",
    "FeedReport",
    "FeedUnconfigured",
    "RunnerConfig",
    "SyntheticBarFeed",
    "TraderStack",
    "build_stack",
    "create_asgi_app",
    "main",
    "select_feeds",
]

# Env vars this file reads. The two arming flags are pointedly NOT in this list;
# they belong to core.gate and engine.py, which read them themselves.
ENV_DB_PATH = "TRADER_DB"
ENV_MODE = "TRADER_MODE"
ENV_CLASSES = "TRADER_CLASSES"
ENV_FEED = "TRADER_FEED"
ENV_DHAN_IDS = "TRADER_DHAN_SECURITY_IDS"

DEFAULT_DB_PATH = "./var/trader.db"


class FeedChoice(str, Enum):
    """What `TRADER_FEED` may say.

    `real` is not a fourth feed. Crypto and equity are priced by different
    vendors — OKX publishes no NSE candle and Dhan publishes no BTC one — so
    "real" means *the real feed for each class*, routed per class by
    `CompositeFeed`. There is no single provider that could be a fourth option.
    """
    SYNTHETIC = "synthetic"     # default: generated prices, paper only
    OKX = "okx"                 # real OKX spot for crypto
    DHAN = "dhan"               # real NSE cash/NFO for equity
    REAL = "real"               # OKX for crypto AND Dhan for equity


# The one thing each real source can price. A source asked for a class it does
# not cover is UNCONFIGURED, never empty: paper falls back to synthetic with a
# warning, live refuses to boot.
SOURCE_COVERS: Mapping[str, AssetClass] = {
    FeedChoice.OKX.value: AssetClass.CRYPTO,
    FeedChoice.DHAN.value: AssetClass.EQUITY,
}


class FeedUnconfigured(RuntimeError):
    """A real feed was asked for and something it needs was never supplied.

    Deliberately distinct from a failure. Nothing is broken: there is no Dhan
    client, or no scrip id, or the chosen source does not price this class at
    all. That difference is what licenses the paper fallback — a BROKEN feed
    still raises, in paper and live alike, because a feed that should work and
    does not is a fault an operator has to see.
    """


def _parse_security_ids(raw: str) -> dict[str, str]:
    """`"RELIANCE=2885,INFY=1594"` -> `{"RELIANCE": "2885", ...}`.

    Every malformed entry raises. A skipped entry would produce a Dhan feed
    that prices some of its universe and silently omits the rest, which is the
    same class of failure as an empty read reported as a quiet market.
    """
    out: dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        symbol, sep, security_id = chunk.partition("=")
        symbol, security_id = symbol.strip(), security_id.strip()
        if not sep or not symbol or not security_id:
            raise ValueError(
                f"{ENV_DHAN_IDS} entry {chunk!r} is not SYMBOL=security_id")
        if not security_id.isdigit():
            raise ValueError(
                f"{ENV_DHAN_IDS} gives {symbol} the security_id {security_id!r}, "
                "which is not numeric")
        if symbol in out:
            raise ValueError(
                f"{ENV_DHAN_IDS} lists {symbol} twice ({out[symbol]!r} and "
                f"{security_id!r}); one of them would silently win")
        out[symbol] = security_id
    return out


# ==========================================================================
# configuration
# ==========================================================================

@dataclass(frozen=True)
class RunnerConfig:
    """What this process is. Contains no secrets and no arming."""

    db_path: str = DEFAULT_DB_PATH
    # What the process ASKS for. Live still requires the per-class arm flag, and
    # a class that is not armed is coerced to paper by the gate regardless.
    requested_mode: Mode = Mode.PAPER
    asset_classes: tuple[AssetClass, ...] = (AssetClass.CRYPTO,)
    # Which market data serves the book. Defaults to synthetic because the
    # default of a trading process must be the one that cannot lose money on a
    # misconfiguration, and because the real feeds need a network (OKX) or the
    # owner's credentials (Dhan).
    feed: FeedChoice = FeedChoice.SYNTHETIC
    # symbol -> Dhan security_id, for the equity universe. The paper universe
    # carries deliberately fake broker ids ("PAPER-RELIANCE") and DhanFeed
    # rejects them, so real Dhan data needs the real scrip-master ids and this
    # is where they come in. Empty means "no real equity universe configured".
    dhan_security_ids: Mapping[str, str] = field(default_factory=dict)
    tick_interval_s: float = 5.0
    starting_paper_cash: Mapping[AssetClass, Decimal] = field(
        default_factory=lambda: {
            # Sized so a one-unit crypto entry clears the gate's 10%-of-equity
            # single-order cap at realistic BTC prices. See the README: the gate
            # CAPS, it does not size down, so a paper float that is too small
            # produces an engine that ticks forever and never trades.
            AssetClass.CRYPTO: Decimal("2000000"),
            AssetClass.EQUITY: Decimal("1000000"),
        })

    def __post_init__(self) -> None:
        if not isinstance(self.requested_mode, Mode):
            raise ValueError("requested_mode must be a Mode")
        if not self.asset_classes:
            raise ValueError("at least one asset class must be configured")
        for asset_class in self.asset_classes:
            if not isinstance(asset_class, AssetClass):
                raise ValueError(f"not an AssetClass: {asset_class!r}")
            if asset_class not in self.starting_paper_cash:
                raise ValueError(
                    f"no starting paper cash for {asset_class.value}; a paper "
                    "adapter with no float reports equity 0, which the circuit "
                    "reads as a 100% drawdown")
        if self.tick_interval_s <= 0:
            raise ValueError("tick_interval_s must be > 0")
        if not isinstance(self.feed, FeedChoice):
            raise ValueError("feed must be a FeedChoice")
        for symbol, security_id in self.dhan_security_ids.items():
            if not str(security_id).strip().isdigit():
                # A non-numeric id would be rejected by Dhan or, worse,
                # resolved to something else. Caught here, at the composition
                # root, rather than at the first candle fetch.
                raise ValueError(
                    f"dhan_security_ids[{symbol!r}] = {security_id!r} is not a "
                    "numeric Dhan security_id")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None,
                 **overrides: Any) -> "RunnerConfig":
        """Read the process shape from the environment.

        `TRADER_MODE=live` is a REQUEST, not an arming. It means "use the live
        adapter for any class whose arm flag is set"; with no arm flag set it
        resolves to paper everywhere, loudly, in the boot report.
        """
        source = os.environ if env is None else env
        raw_mode = source.get(ENV_MODE, Mode.PAPER.value).strip().lower()
        try:
            mode = Mode(raw_mode)
        except ValueError:
            # Not defaulted to paper. "TRADER_MODE=papper" resolving silently to
            # paper is fine today and is the same silence that would resolve a
            # typo'd live to paper on the day it mattered.
            raise ValueError(
                f"{ENV_MODE}={raw_mode!r} is not a mode; use "
                f"{Mode.PAPER.value!r} or {Mode.LIVE.value!r}") from None

        raw_classes = source.get(ENV_CLASSES, AssetClass.CRYPTO.value)
        classes: list[AssetClass] = []
        for name in (c.strip().lower() for c in raw_classes.split(",")):
            if not name:
                continue
            try:
                classes.append(AssetClass(name))
            except ValueError:
                raise ValueError(
                    f"{ENV_CLASSES} contains {name!r}, which is not an asset "
                    f"class") from None
        if not classes:
            raise ValueError(f"{ENV_CLASSES} resolved to no asset classes")

        raw_feed = source.get(ENV_FEED, FeedChoice.SYNTHETIC.value).strip().lower()
        try:
            feed = FeedChoice(raw_feed)
        except ValueError:
            # Same reasoning as the mode above. `TRADER_FEED=okxx` quietly
            # resolving to synthetic is a live book on generated prices.
            raise ValueError(
                f"{ENV_FEED}={raw_feed!r} is not a feed; use one of "
                + ", ".join(repr(c.value) for c in FeedChoice)) from None

        return cls(
            db_path=source.get(ENV_DB_PATH, DEFAULT_DB_PATH),
            requested_mode=mode,
            asset_classes=tuple(dict.fromkeys(classes)),
            feed=feed,
            dhan_security_ids=_parse_security_ids(source.get(ENV_DHAN_IDS, "")),
            **overrides,
        )


# ==========================================================================
# the gate's circuit, per class
# ==========================================================================

class _PerClassCircuit:
    """One `RiskCircuitProto` over N circuits, routed by asset class.

    `engine.py` names this as the integrator's problem: `OrderGate` holds a
    single circuit, but each circuit is bound to one mode, and with equity armed
    and crypto not (or the reverse) no single circuit is right for both. Reading
    the wrong one would judge a live book against paper baselines — the exact
    contamination `PositionStoreProto` filters every read on `mode` to prevent.

    `advance()` raises. The circuit's only writer belongs to the order path and
    the engine calls it on its own per-class circuit; a second writer reachable
    through the gate would roll the high-water mark twice per tick.
    """

    def __init__(self, circuits: Mapping[AssetClass, RiskCircuit]) -> None:
        if not circuits:
            raise ValueError("_PerClassCircuit needs at least one circuit")
        self._circuits = dict(circuits)

    def circuit_for(self, asset_class: AssetClass) -> RiskCircuit:
        circuit = self._circuits.get(asset_class)
        if circuit is None:
            # Not a permissive default. An unconfigured class reaching the gate
            # is a wiring bug, and the gate turns this raise into a refusal.
            raise KeyError(
                f"no risk circuit wired for {asset_class.value}; refusing to "
                "judge an order against another class's circuit")
        return circuit

    def read(self, asset_class: AssetClass,
             equity: Optional[Decimal]) -> Any:
        return self.circuit_for(asset_class).read(asset_class, equity)

    def advance(self, asset_class: AssetClass, equity: Decimal) -> Any:
        raise RuntimeError(
            "the gate must not advance the risk circuit: advancing is the "
            "order path's job and the engine already does it per class")


# ==========================================================================
# local market data
# ==========================================================================

class SyntheticBarFeed:
    """Deterministic fake bars so the stack can be run end to end locally.

    THIS IS NOT MARKET DATA. It is a sine wave with a drift, seeded from the
    symbol, and it exists so that `python3 -m runner` produces a system an
    operator can watch tick, approve, fill and record without credentials, a
    network, or a data vendor. A real deployment replaces this with a feed.

    It REFUSES to serve a class the engine resolved to live. A live book trading
    on a generated series is the single worst outcome this file could produce,
    so the check is here rather than in a comment: `bind()` is given the modes
    the engine actually resolved, and raises if any of them is live.
    """

    def __init__(self, instruments: Sequence[Instrument],
                 bar_interval: dt.timedelta = dt.timedelta(minutes=15),
                 history: int = 120) -> None:
        if history < 2:
            raise ValueError("history must be >= 2 bars")
        self._instruments = tuple(instruments)
        self._interval = bar_interval
        self._history = history
        self._bound = False

    def bind(self, resolved_modes: Mapping[AssetClass, Mode]) -> "SyntheticBarFeed":
        live = sorted(a.value for a, m in resolved_modes.items() if m is Mode.LIVE)
        if live:
            raise RuntimeError(
                f"SyntheticBarFeed refuses to serve live class(es) {', '.join(live)}. "
                "This feed generates prices; it is for paper only. Wire a real "
                "market data provider before arming anything."
            )
        self._bound = True
        return self

    def _require_bound(self, what: str) -> None:
        """bind() is the ONLY place the live refusal lives, so anything that
        serves data before it has run has served a class nobody checked. This
        guard is on every method that hands out a price, not just one of
        them: the safety property must not depend on the caller's discipline
        in going through CompositeFeed."""
        if not self._bound:
            raise RuntimeError(
                f"SyntheticBarFeed.{what} called before bind(); the live-mode "
                "check has not run")

    # -- bars ------------------------------------------------------------

    def bars(self, symbol: str, now: dt.datetime) -> list[Bar]:
        self._require_bound("bars")
        base, amplitude = _series_shape(symbol)
        step = self._interval
        # Snap to the bar grid so the last bar is never stamped ahead of
        # ctx.now, which every strategy rejects outright.
        anchor = _floor_to(now, step)
        out: list[Bar] = []
        for i in range(self._history, 0, -1):
            ts = anchor - step * i
            price = _series_price(base, amplitude, ts)
            spread = amplitude / Decimal(50)
            out.append(Bar(ts=ts, open=price, high=price + spread,
                           low=price - spread, close=price,
                           volume=Decimal("100")))
        return out

    def mark(self, symbol: str, now: dt.datetime) -> Decimal:
        """The current mark, consistent with the last bar's close."""
        base, amplitude = _series_shape(symbol)
        return _series_price(base, amplitude, _floor_to(now, self._interval)
                             - self._interval)

    # -- engine seams ------------------------------------------------------

    def extra_provider(self, asset_class: AssetClass,
                       now: dt.datetime) -> Mapping[str, Any]:
        """`ctx.extra` for one class. Raises if bind() has not vouched for the
        modes: an unbound feed has not been checked against live."""
        self._require_bound("extra_provider")
        return {BARS_KEY: {i.symbol: self.bars(i.symbol, now)
                         for i in self._instruments
                         if i.asset_class is asset_class}}

    def refresh_quotes(self, quotes: InMemoryQuoteSource,
                       asset_class: AssetClass, now: dt.datetime) -> None:
        # The path that actually reaches an adapter: generated prices land in
        # the quote source a live adapter would size against.
        self._require_bound("refresh_quotes")
        for instrument in self._instruments:
            if instrument.asset_class is asset_class:
                quotes.set(instrument.symbol, self.mark(instrument.symbol, now))


def _floor_to(now: dt.datetime, step: dt.timedelta) -> dt.datetime:
    seconds = int(step.total_seconds())
    epoch = int(now.timestamp())
    return dt.datetime.fromtimestamp(epoch - (epoch % seconds), tz=UTC)


def _series_shape(symbol: str) -> tuple[Decimal, Decimal]:
    """A stable (base, amplitude) per symbol. Derived from the symbol itself so
    a restart replays the identical series and a paper run is reproducible."""
    seed = sum(ord(c) for c in symbol)
    base = Decimal(1000 + (seed % 900) * 100)
    return base, base / Decimal(20)


def _series_price(base: Decimal, amplitude: Decimal, ts: dt.datetime) -> Decimal:
    """A slow triangle wave. Integer maths on purpose — no float ever touches a
    price, not even a fake one, because the habit is the point."""
    period = 96                       # bars per full cycle
    phase = (int(ts.timestamp()) // 900) % period
    half = period // 2
    ramp = phase if phase < half else period - phase
    offset = amplitude * (Decimal(ramp * 2) / Decimal(half) - Decimal(1))
    return (base + offset).quantize(Decimal("0.01"))

# ==========================================================================
# one feed per asset class
# ==========================================================================

@dataclass(frozen=True)
class FeedReport:
    """One boot line per class, so "what data is this trading on" is as easy to
    answer as "what is armed"."""

    asset_class: AssetClass
    mode: Mode
    feed: str                       # the class actually serving it
    source: str                     # synthetic | okx | dhan
    detail: str
    fallback_from: Optional[str] = None   # the real source we could not use

    @property
    def real(self) -> bool:
        return self.source != FeedChoice.SYNTHETIC.value

    def line(self) -> str:
        return (f"{self.asset_class.value:<7} mode={self.mode.value:<5} "
                f"feed={self.feed:<17} source={self.source:<9} {self.detail}")


class CompositeFeed:
    """A `MarketFeed` over several feeds, routed by asset class.

    This exists because crypto and equity are priced by different vendors and
    have different prerequisites: OKX data is public, Dhan data needs the
    owner's credentials. Without routing, "I want real crypto prices" would
    also mean "I have Dhan credentials", and the honest answer to not having
    them would be to run everything on generated prices.

    THE SAFETY PROPERTY. `SyntheticBarFeed.bind()` refuses to serve a class the
    engine resolved to LIVE. A composite that bound its sub-feeds with the
    WHOLE mode map would keep that guard; a composite that bound them with an
    empty or filtered-down-to-paper map would route around it, and the
    synthetic feed would happily price a live book. So `bind()` here does
    exactly one thing with modes: it hands each sub-feed the modes of the
    classes THAT FEED IS ROUTED FOR — no more (which would fail a feed for a
    class it is not serving) and no less (which would hide a live class from
    the feed that is about to serve it). Every routed class must appear in the
    resolved map and every resolved class must be routed; neither is defaulted.
    """

    def __init__(self, routes: Mapping[AssetClass, Any],
                 instruments: Mapping[AssetClass, Sequence[Instrument]]) -> None:
        if not routes:
            raise ValueError("CompositeFeed needs at least one route")
        self._routes: dict[AssetClass, Any] = dict(routes)
        for asset_class, feed in self._routes.items():
            if feed is None:
                raise ValueError(f"no feed routed to {asset_class.value}")

        # symbol -> class, so bars() can route without asking a feed to
        # recognise a symbol that belongs to another one.
        self._owner: dict[str, AssetClass] = {}
        for asset_class, group in instruments.items():
            if asset_class not in self._routes:
                raise ValueError(
                    f"instruments are configured for {asset_class.value} but no "
                    "feed is routed to it; that class would tick with no data")
            for instrument in group:
                if instrument.asset_class is not asset_class:
                    raise ValueError(
                        f"{instrument.symbol} is {instrument.asset_class.value} but "
                        f"was listed under {asset_class.value}")
                previous = self._owner.get(instrument.symbol)
                if previous is not None and previous is not asset_class:
                    raise ValueError(
                        f"{instrument.symbol} appears in both {previous.value} and "
                        f"{asset_class.value}; one feed would silently shadow the other")
                self._owner[instrument.symbol] = asset_class
        self._bound = False

    # -- introspection -----------------------------------------------------

    @property
    def routes(self) -> Mapping[AssetClass, Any]:
        return dict(self._routes)

    def feed_for(self, asset_class: AssetClass) -> Any:
        """The feed serving a class. Raises for an unrouted one — answering
        with a default would price a class off another class's data."""
        try:
            return self._routes[asset_class]
        except KeyError:
            raise FeedError(
                f"no feed is routed to {asset_class.value}; routed: "
                + ", ".join(sorted(a.value for a in self._routes))) from None

    # -- bind --------------------------------------------------------------

    def bind(self, resolved_modes: Mapping[AssetClass, Mode]) -> "CompositeFeed":
        unrouted = sorted(a.value for a in resolved_modes if a not in self._routes)
        if unrouted:
            raise ValueError(
                f"the engine resolved {', '.join(unrouted)} but no feed is routed "
                "there; a class the engine will tick must not be left without data")
        missing = sorted(a.value for a in self._routes if a not in resolved_modes)
        if missing:
            # Not defaulted to paper. A class whose mode is unknown could be
            # live, and defaulting it to paper is precisely how a synthetic
            # feed would be allowed to bind for a live book.
            raise ValueError(
                f"no resolved mode for {', '.join(missing)}; refusing to bind a "
                "feed for a class whose mode is unknown")

        # One bind() per DISTINCT feed object, with the union of the classes it
        # serves — a feed shared by two classes must see both modes at once, or
        # a live class could be smuggled past it one call at a time.
        groups: list[tuple[Any, list[AssetClass]]] = []
        for asset_class in sorted(self._routes, key=lambda a: a.value):
            feed = self._routes[asset_class]
            for existing, classes in groups:
                if existing is feed:
                    classes.append(asset_class)
                    break
            else:
                groups.append((feed, [asset_class]))

        for feed, classes in groups:
            subset = {a: resolved_modes[a] for a in classes}
            bound = feed.bind(subset)
            if bound is not feed:
                # The routing table and the boot report both name the object
                # constructed above. A bind() that swapped it would leave both
                # describing a feed that is not the one serving the class.
                raise FeedError(
                    f"{type(feed).__name__}.bind() returned a different object "
                    f"({type(bound).__name__}); CompositeFeed cannot vouch for a "
                    "feed it did not route")
        self._bound = True
        return self

    def _require_bound(self, what: str) -> None:
        if not self._bound:
            raise RuntimeError(
                f"CompositeFeed.{what}() called before bind(); the live-mode "
                "check has not run")

    # -- delegation --------------------------------------------------------

    def bars(self, symbol: str, now: dt.datetime) -> list[Bar]:
        self._require_bound("bars")
        asset_class = self._owner.get(symbol)
        if asset_class is None:
            raise FeedError(
                f"{symbol!r} is not in any routed universe "
                f"({', '.join(sorted(self._owner)) or 'empty'}); refusing to "
                "return an empty series for a symbol nothing prices")
        return self.feed_for(asset_class).bars(symbol, now)

    def refresh_quotes(self, quotes: Any, asset_class: AssetClass,
                       now: dt.datetime) -> None:
        self._require_bound("refresh_quotes")
        self.feed_for(asset_class).refresh_quotes(quotes, asset_class, now)

    def extra_provider(self, asset_class: AssetClass, now: dt.datetime) -> dict:
        self._require_bound("extra_provider")
        feed = self.feed_for(asset_class)
        extra = feed.extra_provider(asset_class, now)
        if extra is None:
            raise FeedError(
                f"{type(feed).__name__}.extra_provider("
                f"{asset_class.value}) returned None; a strategy would read that "
                "as a context with no bar book")
        if BARS_KEY not in extra:
            # `{}` is not None and so used to pass — but downstream it is
            # indistinguishable from a quiet market, which is how a feed routed
            # to a class it cannot serve reads as "no signal" while the class
            # keeps ticking. A feed that has nothing to say about a class must
            # say so; it does not get to say nothing.
            raise FeedError(
                f"{type(feed).__name__}.extra_provider({asset_class.value}) "
                f"published no {BARS_KEY!r}; an absent bar book is a wiring "
                "failure, not an empty market. Route this class to a feed that "
                "can price it.")
        return extra


# ==========================================================================
# choosing the feeds
# ==========================================================================

def select_feeds(
    config: RunnerConfig,
    resolved: Mapping[AssetClass, Mode],
    instruments: Mapping[AssetClass, Sequence[Instrument]],
    *,
    dhan_client: Any = None,
    okx_exchange: Any = None,
) -> tuple[dict[AssetClass, Any], tuple[FeedReport, ...]]:
    """Pick one feed per class and say, per class, what it is.

    The fallback rule, and it is the only one:

        a real feed that was never CONFIGURED falls back to synthetic for a
        PAPER class and refuses to boot a LIVE one.

    "Never configured" means no Dhan client, no scrip ids, or a source that
    does not price this class at all. It does NOT mean broken: a feed that
    should work and does not raises in paper too, because a fault an operator
    cannot see is a fault that gets worse.
    """
    synthetic_universe = [i for group in instruments.values() for i in group]
    shared_synthetic: list[Any] = []          # built at most once, lazily

    def synthetic() -> "SyntheticBarFeed":
        if not shared_synthetic:
            shared_synthetic.append(SyntheticBarFeed(synthetic_universe))
        return shared_synthetic[0]

    routes: dict[AssetClass, Any] = {}
    reports: list[FeedReport] = []

    for asset_class in config.asset_classes:
        try:
            mode = resolved[asset_class]
        except KeyError:
            raise ValueError(
                f"no resolved mode for {asset_class.value}; a feed cannot be "
                "chosen for a class whose mode is unknown") from None

        source = _requested_source(config.feed, asset_class)
        if source == FeedChoice.SYNTHETIC.value:
            routes[asset_class] = synthetic()
            reports.append(FeedReport(
                asset_class=asset_class, mode=mode,
                feed=type(synthetic()).__name__,
                source=FeedChoice.SYNTHETIC.value,
                detail="GENERATED PRICES — not market data, paper only"))
            continue

        try:
            feed, detail = _build_real_feed(
                source, asset_class, instruments[asset_class], config,
                dhan_client=dhan_client, okx_exchange=okx_exchange)
        except FeedUnconfigured as exc:
            if mode is Mode.LIVE:
                # The whole point. A live class whose real feed is missing does
                # not quietly become a live class on a sine wave.
                raise RuntimeError(
                    f"{asset_class.value} resolves to LIVE and {ENV_FEED}="
                    f"{config.feed.value} cannot serve it: {exc}. Refusing to "
                    "boot a live book on generated prices — supply what the "
                    "feed needs, or do not arm this class."
                ) from exc
            LOG.warning(
                "boot: %s falls back to SyntheticBarFeed (GENERATED PRICES): "
                "%s. This class is PAPER, so it will run, but it is NOT "
                "trading on market data.", asset_class.value, exc)
            routes[asset_class] = synthetic()
            reports.append(FeedReport(
                asset_class=asset_class, mode=mode,
                feed=type(synthetic()).__name__,
                source=FeedChoice.SYNTHETIC.value,
                detail=f"FALLBACK from {source} ({exc}) — GENERATED PRICES, paper only",
                fallback_from=source))
            continue

        routes[asset_class] = feed
        reports.append(FeedReport(
            asset_class=asset_class, mode=mode, feed=type(feed).__name__,
            source=source, detail=detail))

    return routes, tuple(reports)


def _requested_source(choice: FeedChoice, asset_class: AssetClass) -> str:
    """Which source `TRADER_FEED` asks for, for ONE class."""
    if choice is FeedChoice.SYNTHETIC:
        return FeedChoice.SYNTHETIC.value
    if choice is FeedChoice.REAL:
        return (FeedChoice.OKX.value if asset_class is AssetClass.CRYPTO
                else FeedChoice.DHAN.value)
    return choice.value


def _build_real_feed(source: str, asset_class: AssetClass,
                     instruments: Sequence[Instrument], config: RunnerConfig,
                     *, dhan_client: Any, okx_exchange: Any) -> tuple[Any, str]:
    """Construct the real feed for one class, or raise FeedUnconfigured."""
    covers = SOURCE_COVERS[source]
    if asset_class is not covers:
        raise FeedUnconfigured(
            f"the {source} feed prices {covers.value} only")
    if not instruments:
        raise FeedUnconfigured(
            f"no {asset_class.value} instruments are configured")

    if source == FeedChoice.OKX.value:
        # No credentials anywhere on this path: OkxFeed builds a public
        # ccxt.okx with no key, and markets are loaded at bind().
        feed = OkxFeed(instruments, exchange=okx_exchange)
        return feed, (f"real OKX public spot, {len(instruments)} instruments "
                      f"@{feed.timeframe}, no credentials")

    # Dhan. The client is INJECTED and never built here: minting a second Dhan
    # access token would invalidate the owner's live one, so this file has no
    # code path that could authenticate.
    if dhan_client is None:
        raise FeedUnconfigured(
            "no Dhan market-data client was injected (Dhan needs the owner's "
            "credentials; see the README)")
    universe = _dhan_universe(instruments, config.dhan_security_ids)
    feed = DhanFeed(universe, dhan_client)
    return feed, (f"real Dhan NSE data via {type(dhan_client).__name__}, "
                  f"{len(universe)} instruments")


def _dhan_universe(instruments: Sequence[Instrument],
                   security_ids: Mapping[str, str]) -> tuple[Instrument, ...]:
    """Re-stamp the equity universe with real Dhan security ids.

    `default_instruments` deliberately carries fake broker ids so nothing in
    the paper universe can be mistaken for a live scrip, and `DhanFeed` rejects
    them for exactly that reason. Real Dhan data therefore needs the real ids,
    and their absence is a configuration gap — not a reason to price a symbol
    off a guess.
    """
    if not security_ids:
        raise FeedUnconfigured(
            f"no Dhan security ids are configured ({ENV_DHAN_IDS}); the paper "
            "universe carries placeholder broker ids that DhanFeed rejects")
    out: list[Instrument] = []
    missing: list[str] = []
    for instrument in instruments:
        security_id = security_ids.get(instrument.symbol)
        if security_id is None:
            missing.append(instrument.symbol)
            continue
        out.append(replace(instrument, broker_id=str(security_id)))
    if missing:
        # Not dropped. A universe quietly short of two names is a strategy that
        # never sees them and a book nobody notices is half-unmanaged.
        raise FeedUnconfigured(
            f"{ENV_DHAN_IDS} has no security_id for {', '.join(sorted(missing))}")
    return tuple(out)


# ==========================================================================
# the universe
# ==========================================================================

def default_instruments(asset_class: AssetClass) -> tuple[Instrument, ...]:
    """The paper universe. Broker ids are placeholders for crypto (ccxt uses the
    symbol) and obviously-fake numbers for equity, so nothing here can be
    mistaken for a live Dhan security id."""
    if asset_class is AssetClass.CRYPTO:
        return (
            Instrument("BTC/USDT", AssetClass.CRYPTO, "BTC/USDT",
                       lot_size=1, tick_size=Decimal("0.1"),
                       qty_precision=6, min_notional=Decimal("5")),
            Instrument("ETH/USDT", AssetClass.CRYPTO, "ETH/USDT",
                       lot_size=1, tick_size=Decimal("0.01"),
                       qty_precision=5, min_notional=Decimal("5")),
        )
    return (
        Instrument("RELIANCE", AssetClass.EQUITY, "PAPER-RELIANCE",
                   lot_size=1, tick_size=Decimal("0.05")),
        Instrument("INFY", AssetClass.EQUITY, "PAPER-INFY",
                   lot_size=1, tick_size=Decimal("0.05")),
    )


def default_strategies(asset_class: AssetClass,
                       instruments: Sequence[Instrument]) -> tuple[Any, ...]:
    if asset_class is AssetClass.CRYPTO:
        return (CryptoEmaTrendStrategy(instruments=list(instruments)),)
    return (EquityMeanReversionStrategy(instruments=list(instruments)),)


# ==========================================================================
# the stack
# ==========================================================================

@dataclass
class TraderStack:
    """Everything this process owns, kept together so it can be shut down in
    one place and inspected in tests."""

    config: RunnerConfig
    store: SQLitePositionStore
    gate_state: SQLiteGateStateStore
    control: api_mod.SqliteControlStore
    gate: OrderGate
    engine: Engine
    registry: api_mod.StrategyRegistry
    api_bus: api_mod.EventBus
    handle: api_mod.EngineHandle
    feed: Optional[Any] = None
    # What is serving each class, in the words the boot log used.
    feed_report: tuple[FeedReport, ...] = ()
    _bridge: Optional[asyncio.Task[None]] = None

    def feed_report_text(self) -> str:
        """The per-class data report, the counterpart to the engine's arm
        report. An operator answers "what is this trading on" from one and
        "what is armed" from the other, and both are printed at boot."""
        return "\n".join(r.line() for r in self.feed_report)

    async def start(self) -> None:
        # The bridge starts BEFORE the engine so the boot events — which carry
        # the arm report — reach the API stream rather than being published into
        # a bus nobody is draining yet.
        self._bridge = asyncio.create_task(
            _bridge_events(self.engine, self.api_bus, self.registry),
            name="trader-event-bridge")
        await self.engine.start()

    async def stop(self) -> None:
        await self.engine.stop()
        if self._bridge is not None:
            self._bridge.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._bridge
            self._bridge = None
        self.close()

    def close(self) -> None:
        """Close what the engine does not. The engine checkpoints the position
        store itself; these two are opened here and so are closed here."""
        for name in ("gate_state", "control"):
            handle = getattr(self, name)
            try:
                handle.close()
            except Exception as exc:            # pragma: no cover - shutdown path
                LOG.critical("closing %s failed: %r", name, exc)


def build_stack(config: Optional[RunnerConfig] = None,
                env: Optional[Mapping[str, str]] = None,
                clock: Callable[[], dt.datetime] = engine_mod._utc_now,
                *,
                dhan_client: Any = None,
                okx_exchange: Any = None,
                ) -> TraderStack:
    """Construct the whole system. Reads arming ONLY through gate/engine.

    Ordering matters in one place: the engine is built before the API handle so
    the handle can report the modes the engine ACTUALLY resolved, rather than
    the mode this process asked for. A dashboard that shows "paper" for a class
    the gate will execute live is worse than no dashboard.

    `dhan_client` is INJECTED and is never constructed here. Dhan permits one
    active access token at a time and the owner's live engine holds it, so a
    second one minted by this process would break real trading with real money.
    Wiring it is a deliberate edit at this call site, exactly like wiring a
    live adapter; see the README. `okx_exchange` is the same seam for OKX and
    is only needed by tests — OKX market data is public, so the default
    (a keyless ccxt client) is the right one in production.
    """
    cfg = config or RunnerConfig.from_env(env)
    source: Mapping[str, str] = os.environ if env is None else env

    store = SQLitePositionStore(cfg.db_path)
    gate_state = SQLiteGateStateStore(cfg.db_path, cfg.requested_mode)
    control = api_mod.SqliteControlStore(cfg.db_path)

    engine_config = EngineConfig(requested_mode=cfg.requested_mode)

    # Resolve each class's real mode with the ENGINE's own resolver, so this
    # file cannot invent a third opinion about what is armed.
    resolved: dict[AssetClass, Mode] = {}
    for asset_class in cfg.asset_classes:
        var = engine_mod.ARM_ENV[asset_class]
        armed = engine_mod._parse_arm_flag(var, source.get(var))
        resolved[asset_class] = (Mode.LIVE if cfg.requested_mode is Mode.LIVE
                                 and armed else Mode.PAPER)

    instruments = {a: default_instruments(a) for a in cfg.asset_classes}
    routes, feed_reports = select_feeds(
        cfg, resolved, instruments,
        dhan_client=dhan_client, okx_exchange=okx_exchange)
    # bind() is where a synthetic feed refuses a live class, and CompositeFeed
    # binds each sub-feed with the modes of ITS OWN classes so the refusal
    # cannot be routed around.
    feed = CompositeFeed(routes, instruments).bind(resolved)
    for report in feed_reports:
        LOG.info("boot: %s", report.line())

    circuits: dict[AssetClass, RiskCircuit] = {}
    wirings: list[ClassWiring] = []
    adapters: dict[AssetClass, Any] = {}
    quote_sources: dict[AssetClass, InMemoryQuoteSource] = {}

    for asset_class in cfg.asset_classes:
        mode = resolved[asset_class]
        circuits[asset_class] = RiskCircuit(store, mode, clock=clock)

        quotes = InMemoryQuoteSource()
        quote_sources[asset_class] = quotes
        paper = PaperAdapter(asset_class, quotes,
                             cfg.starting_paper_cash[asset_class])

        def extra_provider(ac: AssetClass, now: dt.datetime,
                           _quotes: InMemoryQuoteSource = quotes,
                           ) -> Mapping[str, Any]:
            # Marks and bars come from ONE source and are refreshed together.
            # A quote from a different clock than the bars is how a strategy's
            # stop is evaluated against a price the entry never saw.
            feed.refresh_quotes(_quotes, ac, now)
            return feed.extra_provider(ac, now)

        wirings.append(ClassWiring(
            asset_class=asset_class,
            paper_adapter=paper,
            risk=circuits[asset_class],
            strategies=default_strategies(asset_class, instruments[asset_class]),
            # No live adapter is constructed here. Wiring one is a separate,
            # deliberate edit; see the README.
            live_adapter=None,
            instruments=instruments[asset_class],
            tick_interval_s=cfg.tick_interval_s,
            extra_provider=extra_provider,
        ))
        adapters[asset_class] = paper

    gate = OrderGate(
        position_store=store,
        gate_state=gate_state,
        risk_circuit=_PerClassCircuit(circuits),
        # requested_mode MUST match EngineConfig's, or every approval is refused
        # with a mode mismatch. One value, read once, passed to both.
        config=GateConfig(requested_mode=cfg.requested_mode),
        env=env,
    )

    engine = Engine(wirings=wirings, store=store, gate=gate,
                    config=engine_config, clock=clock, env=env,
                    control=control)

    registry = api_mod.StrategyRegistry()
    for wiring in wirings:
        for strategy in wiring.strategies:
            registry.register(strategy.id, wiring.asset_class)

    api_bus = api_mod.EventBus()
    engine_modes = engine.runner_modes()
    handle = api_mod.EngineHandle(
        # The handle's mode is what this process REQUESTED. Per-class truth is
        # in `armed`, which is taken from the engine's own resolution.
        mode=cfg.requested_mode,
        asset_classes=tuple(cfg.asset_classes),
        ledger=store,
        circuit=_HandleCircuit(circuits, cfg.requested_mode),
        control=control,
        strategies=registry,
        bus=api_bus,
        adapters=adapters,
        armed={a: engine_modes[a] is Mode.LIVE for a in cfg.asset_classes},
        clock=clock,
    )

    return TraderStack(config=cfg, store=store, gate_state=gate_state,
                       control=control, gate=gate, engine=engine,
                       registry=registry, api_bus=api_bus, handle=handle,
                       feed=feed, feed_report=feed_reports)


class _HandleCircuit(_PerClassCircuit):
    """The API's view of the circuits.

    Identical routing to the gate's, plus a `mode` property, because
    `EngineHandle` refuses to be built against a circuit whose mode disagrees
    with the handle's — a live view reading a paper circuit shows a drawdown
    computed from the wrong book. `advance()` still raises: the API's own
    docstring promises it never calls the circuit's writer, and this makes that
    promise enforceable rather than merely stated.
    """

    def __init__(self, circuits: Mapping[AssetClass, RiskCircuit],
                 mode: Mode) -> None:
        super().__init__(circuits)
        self._mode = mode

    @property
    def mode(self) -> Mode:
        return self._mode


# ==========================================================================
# event bridge
# ==========================================================================

def _api_event(event: engine_mod.Event) -> api_mod.Event:
    """Translate an engine event into the API's shape.

    The two Event types are genuinely different: the engine's carries a rich
    `data` mapping with Decimals and datetimes in it, the API's requires JSON
    scalars so an unserialisable value cannot kill a live operator's stream
    mid-flight. `as_dict()` already flattens the first into the second's world.
    """
    payload = event.as_dict()
    data: dict[str, Any] = {
        "seq": payload["seq"],
        "message": payload["message"],
        "asset_class": payload["asset_class"],
        "mode": payload["mode"],
    }
    for key, value in payload["data"].items():
        if isinstance(value, (str, int, bool)) or value is None:
            data[f"d_{key}"] = value
        else:
            # Lists and nested mappings are already JSON-safe from _json_safe,
            # but the API's Event only accepts scalars. Render rather than drop:
            # a rejection whose `checks` vanished is a rejection nobody can
            # explain afterwards.
            data[f"d_{key}"] = repr(value)
    return api_mod.Event(kind=payload["kind"],
                         at=dt.datetime.fromisoformat(payload["at"]),
                         data=data)


async def _bridge_events(engine: Engine, api_bus: api_mod.EventBus,
                         registry: api_mod.StrategyRegistry) -> None:
    """Drain the engine's bus into the API's, and keep the registry honest.

    A display path. It must never be able to break the order path, so every
    failure here is logged and skipped rather than raised — with one deliberate
    exception: `asyncio.CancelledError` propagates, because a bridge that
    swallowed its own cancellation would keep a stopped engine's stream open.
    """
    subscription = engine.events.subscribe()
    try:
        async for event in subscription:
            try:
                api_bus.publish(_api_event(event))
            except Exception as exc:            # pragma: no cover - display path
                LOG.error("event bridge: could not publish %r: %r",
                          event.kind, exc)
            try:
                _update_registry(registry, event)
            except Exception as exc:            # pragma: no cover - display path
                LOG.error("event bridge: registry update failed: %r", exc)
    except asyncio.CancelledError:
        raise
    finally:
        engine.events.unsubscribe(subscription)


def _update_registry(registry: api_mod.StrategyRegistry,
                     event: engine_mod.Event) -> None:
    """Mirror the engine's strategy lifecycle into the API's registry.

    `/api/strategies` is the screen an operator checks to answer "is anything
    still running", and `StrategyStatus` is explicit that a null `last_tick_at`
    means "has never run in this process" — which is NOT "ran and did nothing".
    Leaving it null while the engine ticks would render every live strategy as a
    healthy-looking dead one, so ticks are mirrored as well as disables.
    """
    if event.kind is EventKind.STRATEGY_DISABLED:
        strategy_id = event.data.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            LOG.error("strategy_disabled event carried no strategy_id: %r",
                      event.data)
            return
        registry.disable(strategy_id, str(event.data.get("error") or event.message))
        return

    if event.kind is EventKind.TICK:
        # A class-level event: the engine evaluated every strategy still enabled
        # in that class. Disabled ones are skipped so a disabled strategy cannot
        # look like it is still being asked for proposals.
        for row in registry.snapshot():
            if row.asset_class is event.asset_class and row.enabled:
                registry.record_tick(row.strategy_id, event.at)
        return

    if event.kind is EventKind.INTENT:
        strategy_id = event.data.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            return
        if strategy_id not in {r.strategy_id for r in registry.snapshot()}:
            # Square-off and operator-flatten intents carry the position's
            # strategy_id, which may belong to a strategy that is no longer
            # registered. Recording it would invent a row.
            return
        record = api_mod.ProposalRecord(
            symbol=str(event.data.get("symbol", "?")),
            side=str(event.data.get("side", "?")),
            qty=str(event.data.get("qty", "?")),
            order_type=str(event.data.get("order_type", "?")),
            reason=str(event.data.get("reason", event.message)),
            structure_id=(str(event.data["structure_id"])
                          if event.data.get("structure_id") else None),
        )
        registry.record_tick(strategy_id, event.at, proposals=(record,),
                             reason=record.reason)


# ==========================================================================
# ASGI entry point
# ==========================================================================

def create_asgi_app(config: Optional[RunnerConfig] = None,
                    api_settings: Optional[api_mod.ApiSettings] = None,
                    env: Optional[Mapping[str, str]] = None) -> Any:
    """Build the API app with the engine running inside its lifespan.

    One process on purpose. `api/server.py` documents that its event bus and
    strategy registry only see what is published in the SAME process; running
    the engine here is what turns `/api/events` and `/api/strategies` from
    almost-empty into the truth.
    """
    stack = build_stack(config, env)
    app = api_mod.create_app(stack.handle, api_settings, env)

    @app.on_event("startup")
    async def _startup() -> None:                # pragma: no cover - ASGI hook
        await stack.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:               # pragma: no cover - ASGI hook
        await stack.stop()

    app.state.stack = stack
    return app


async def run_engine_only(config: Optional[RunnerConfig] = None,
                          env: Optional[Mapping[str, str]] = None) -> None:
    """Run the engine with no API. Used by `python3 -m runner --no-api`."""
    stack = build_stack(config, env=env)
    await stack.start()
    try:
        await asyncio.gather(*stack.engine._tasks)
    finally:
        await stack.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:   # pragma: no cover - CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)

    if "--no-api" in args:
        try:
            asyncio.run(run_engine_only())
        except KeyboardInterrupt:
            LOG.info("interrupted")
        return 0

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Either `pip install uvicorn` or run "
              "the engine alone with:  python3 -m runner --no-api",
              file=sys.stderr)
        return 2

    host = os.environ.get("TRADER_API_HOST", "127.0.0.1")
    port = int(os.environ.get("TRADER_API_PORT", "8080"))
    uvicorn.run(create_asgi_app(), host=host, port=port, log_level="info")
    return 0

if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
