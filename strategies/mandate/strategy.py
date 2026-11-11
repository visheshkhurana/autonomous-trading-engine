"""
Paper crypto mandate engine.

Replaces crypto_trend / crypto_breakout 15m EMA-spray as the registered
crypto decision path. Strategies stay pure: no I/O, no clock, no broker.
Paper vs live is an adapter choice this module cannot see.

Perps and options are paper/sim stubs. They execute only when
`MandateState.paper_derivatives` is True (runner sets False on a live
class). The live OkxAdapter is spot-only and is not taught perps here.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Optional, Sequence

from core.types import AssetClass, Instrument, Intent, MarketContext, Side

from strategies.base import (
    Bar, StrategyBase, StrategyInputError, atr, donchian,
)

from .constants import (
    ATR_PERIOD, ATR_STOP_MULT, HARD_STOP_FRACTION, MIN_RR,
    SPOT_WEIGHTS, STATE_KEY,
)
from .regime import RegimeRead, classify_regime, extras_from_ctx
from .risk import (
    cluster_allows, cluster_of, confidence_label, hard_stop,
    new_directional_blocked, size_notional,
)
from .score import annotate_from_extras, score_plan, vol_condition
from .types import (
    DecisionCard, InstrumentKind, MandateState, MandateTick, REGIME_CAPS,
    Regime, TradePlan, Verdict,
)

# cluster_of lives in risk; avoid a circular import of the string.
from .risk import CRYPTO_BETA


class CryptoMandateStrategy(StrategyBase):
    """1,000 USDT paper sleeve. Holding USDT is a valid position."""

    def __init__(self, *, instruments: Sequence[Instrument],
                 strategy_id: str = "crypto_mandate",
                 bar_interval: dt.timedelta = dt.timedelta(minutes=15),
                 max_loss_pct: Decimal = HARD_STOP_FRACTION) -> None:
        if not instruments:
            raise StrategyInputError("crypto_mandate needs at least one instrument")
        symbols = [i.symbol for i in instruments]
        if len(set(symbols)) != len(symbols):
            raise StrategyInputError(f"duplicate instruments in universe: {symbols}")
        for i in instruments:
            if i.asset_class is not AssetClass.CRYPTO:
                raise StrategyInputError(f"{i.symbol} is not a CRYPTO instrument")
        if not (Decimal(0) < max_loss_pct < Decimal(1)):
            raise StrategyInputError("max_loss_pct must be in (0, 1)")
        super().__init__(
            strategy_id=strategy_id, asset_class=AssetClass.CRYPTO,
            bar_interval=bar_interval,
        )
        self.instruments = tuple(instruments)
        self.max_loss_pct = max_loss_pct
        self.last_tick: Optional[MandateTick] = None
        self._paper_perps = tuple(
            _paper_perp(base) for base in ("BTC", "ETH", "SOL")
        )

    @property
    def min_bars(self) -> int:
        return max(ATR_PERIOD + 2, 50)

    def propose(self, ctx: MarketContext) -> Sequence[Intent]:
        tick = self.evaluate(ctx)
        self.last_tick = tick
        return self.finalize(tick.intents)

    def evaluate(self, ctx: MarketContext) -> MandateTick:
        state = _state_from_ctx(ctx)
        extras = extras_from_ctx(ctx.extra)
        books: dict[str, Optional[list[Bar]]] = {}
        quotes: dict[str, Optional[Decimal]] = {}
        for inst in self.instruments:
            pos = self.own_position(ctx, inst.symbol)
            books[inst.symbol] = self.bars_for(
                ctx, inst.symbol, min_bars=self.min_bars,
                must_have=pos is not None)
            quotes[inst.symbol] = self.known_quote(
                ctx, inst.symbol, must_have=pos is not None)

        btc_bars = books.get("BTC/USDT") or _first_book(books)
        if btc_bars is None:
            hold = _hold_card(
                Regime.RANGE_NEUTRAL, state,
                "no BTC book; holding USDT is valid",
            )
            return MandateTick(Regime.RANGE_NEUTRAL, (hold,), (), state)

        eth_bars = books.get("ETH/USDT")
        read = classify_regime(btc_bars, eth_bars, extras)
        cards: list[DecisionCard] = []
        intents: list[Intent] = []

        for inst in self.instruments:
            pos = self.own_position(ctx, inst.symbol)
            if pos is None:
                continue
            if pos.qty < 0:
                raise StrategyInputError(
                    f"{self.id}: holds a SHORT {inst.symbol} ({pos.qty}) on "
                    "spot, which this strategy never opens"
                )
            bars = books[inst.symbol]
            quote = quotes[inst.symbol]
            if bars is None or quote is None:
                continue
            exit_intent, card = self._exit(
                inst, pos.avg_price, pos.qty, bars, quote, read, state, extras)
            if exit_intent is not None and card is not None:
                intents.append(exit_intent)
                cards.append(card)

        halt = new_directional_blocked(state)
        if halt:
            cards.append(_hold_card(read.regime, state, halt))
            return MandateTick(read.regime, tuple(cards), tuple(intents), state)

        # Do not open a second directional while we already hold, or while
        # another strategy already sits on the symbol. Engine max-1 is the
        # ceiling; the mandate also treats BTC+ETH+SOL as one cluster.
        if self.own_positions(ctx) and not intents:
            cards.append(_hold_card(
                read.regime, state,
                "managing open book; USDT remainder stays unspent",
            ))
            return MandateTick(read.regime, tuple(cards), tuple(intents), state)

        if intents:
            # An exit this tick is enough. Do not immediately reverse.
            return MandateTick(read.regime, tuple(cards), tuple(intents), state)

        caps = REGIME_CAPS[read.regime]
        plan, reject = self._best_spot_plan(
            ctx, books, quotes, read, extras, state, caps.spot)
        if plan is None:
            if reject is not None:
                cards.append(reject)
            else:
                cards.append(_hold_card(
                    read.regime, state,
                    "no score-passing spot; holding USDT is a valid position "
                    "and idle capital is not a reason to trade",
                ))
            if state.paper_derivatives:
                cards.extend(self._gated_derivatives(
                    ctx, read, extras, state, caps))
            return MandateTick(read.regime, tuple(cards), (), state)

        card, intent = self._execute_or_reject(plan, read, state, extras)
        cards.append(card)
        if intent is not None:
            intents.append(intent)
        elif state.paper_derivatives:
            cards.extend(self._gated_derivatives(
                ctx, read, extras, state, caps))
        return MandateTick(read.regime, tuple(cards), tuple(intents), state)

    # -- plans -------------------------------------------------------------

    def _best_spot_plan(
        self, ctx: MarketContext,
        books: dict[str, Optional[list[Bar]]],
        quotes: dict[str, Optional[Decimal]],
        read: RegimeRead,
        extras: dict,
        state: MandateState,
        spot_cap: Decimal,
    ) -> tuple[Optional[TradePlan], Optional[DecisionCard]]:
        if spot_cap <= 0:
            return None, _hold_card(
                read.regime, state,
                f"{read.regime.value} caps new spot at 0; majority USDT",
            )
        if not read.regime.bullish:
            # Range may still take a high-score breakout against the 20% cap;
            # bear does not open spot longs.
            if read.regime is Regime.BEAR:
                return None, _hold_card(
                    read.regime, state,
                    "bear regime: no new spot longs; USDT is the position",
                )
        ranked: list[TradePlan] = []
        last_reject: Optional[DecisionCard] = None
        for inst in self.instruments:
            if self.held_by_anyone(ctx, inst.symbol):
                continue
            bars = books.get(inst.symbol)
            quote = quotes.get(inst.symbol)
            if bars is None or quote is None:
                continue
            plan = self._build_long_plan(inst, bars, quote, extras, eth_agrees=(
                read.regime.bullish if inst.symbol != "ETH/USDT" else True
            ))
            if plan is None:
                continue
            if plan.score.total < 65 or plan.reward_risk < MIN_RR:
                last_reject = self._card_from_plan(
                    plan, read, state, extras, Verdict.REJECT,
                    "score or R:R rejected",
                )
                continue
            ranked.append(plan)
        if not ranked:
            return None, last_reject
        # Bullish default mix: prefer BTC, then ETH, then SOL when scores tie.
        weight = {s: i for i, s in enumerate(SPOT_WEIGHTS)}
        ranked.sort(key=lambda p: (-p.score.total, weight.get(p.instrument.symbol, 9)))
        return ranked[0], None

    def _build_long_plan(
        self, instrument: Instrument, bars: Sequence[Bar], quote: Decimal,
        extras: dict, *, eth_agrees: Optional[bool],
        kind: InstrumentKind = InstrumentKind.SPOT,
        leverage: Decimal = Decimal("1"),
    ) -> Optional[TradePlan]:
        vol = atr(bars, ATR_PERIOD)
        if vol is None or vol <= 0:
            return None
        last = bars[-1]
        entry = quote
        prior = bars[:-1]
        if len(prior) < 10:
            return None
        swing_low = min(b.low for b in prior[-10:])
        atr_stop = entry - ATR_STOP_MULT * vol
        # Invalidation is structure (swing), not a fabricated tight stop that
        # would make every grind look like 2R.
        structure_stop = swing_low if swing_low < entry else atr_stop
        if structure_stop >= entry or structure_stop <= 0:
            return None
        stop = structure_stop
        risk = entry - stop
        lookback = prior[-20:] if len(prior) >= 20 else prior
        prior_high = max(b.high for b in lookback)
        prior_low = min(b.low for b in lookback)
        range_h = prior_high - prior_low
        if range_h <= 0:
            return None
        if prior_high > entry:
            structure_target = prior_high
        else:
            structure_target = entry + range_h
        if structure_target <= entry:
            return None
        rr = (structure_target - entry) / risk
        if rr < MIN_RR:
            return None
        t1 = entry + Decimal("2") * risk
        t2 = max(structure_target, entry + Decimal("2.5") * risk)
        channel = donchian(bars, 20, exclude_last=True)
        if channel is not None and channel.upper > t2:
            t2 = channel.upper
        score = score_plan(
            bars, rr, entry, extras, side_long=True, eth_agrees=eth_agrees,
        )
        funding, oi = annotate_from_extras(extras, True)
        thesis = (
            f"long {instrument.symbol} {kind.value} after {last.close} close; "
            f"stop {stop} ({ATR_STOP_MULT}x ATR {vol} / swing / "
            f"{HARD_STOP_FRACTION * 100}% floor), t1 {t1} (2R), t2 {t2}"
        )
        invalidation = (
            f"close <= {stop} or regime flips to {Regime.BEAR.value}"
        )
        return TradePlan(
            instrument=instrument, kind=kind, side=Side.BUY,
            entry=entry, stop=stop, target_1=t1, target_2=t2,
            reward_risk=rr, atr=vol, score=score, thesis=thesis,
            invalidation=invalidation, funding=funding, oi_signal=oi,
            vol_condition=vol_condition(bars), leverage=leverage,
            decision_ts=last.ts,
        )

    def _execute_or_reject(
        self, plan: TradePlan, read: RegimeRead, state: MandateState,
        extras: dict,
    ) -> tuple[DecisionCard, Optional[Intent]]:
        levered = plan.kind is InstrumentKind.PERP
        notional, max_loss, why = size_notional(state, plan, levered=levered)
        if notional is None:
            card = self._card_from_plan(
                plan, read, state, extras, Verdict.REJECT, why,
                max_loss=Decimal("0"), notional=Decimal("0"),
            )
            return card, None
        cluster_why = cluster_allows(state, plan, max_loss)
        if cluster_why:
            card = self._card_from_plan(
                plan, read, state, extras, Verdict.REJECT, cluster_why,
                max_loss=max_loss, notional=notional,
            )
            return card, None
        qty = (notional / plan.entry)
        qty = self.profile.quantize_qty(plan.instrument, qty)
        if qty <= 0:
            card = self._card_from_plan(
                plan, read, state, extras, Verdict.REJECT,
                "qty quantized to 0",
                max_loss=max_loss, notional=notional,
            )
            return card, None
        notional = qty * plan.entry
        margin = notional / plan.leverage if plan.leverage > 0 else notional
        card = self._card_from_plan(
            plan, read, state, extras, Verdict.EXECUTE, plan.thesis,
            max_loss=max_loss, notional=notional, margin=margin,
        )
        intent = self.make_intent(
            instrument=plan.instrument, side=plan.side, qty=qty,
            reason=card.reason_text(), decision_ts=self._decision_ts_from_plan(plan),
            tag="entry", ref_price=plan.entry, stop_price=plan.stop,
        )
        return card, intent

    def protective_stop(self, ctx, position):
        """ATR stop floored by the hard fraction, from the position's own entry."""
        if position.avg_price <= 0:
            return None
        try:
            bars = self.bars_for(
                ctx, position.instrument.symbol, min_bars=self.min_bars, must_have=False,
            )
        except Exception:
            return None
        floor = position.avg_price * (Decimal(1) - self.max_loss_pct)
        if not bars:
            return floor
        vol = atr(bars, ATR_PERIOD)
        if vol is None or vol <= 0:
            return floor
        return max(position.avg_price - ATR_STOP_MULT * vol, floor)

    def _decision_ts_from_plan(self, plan: TradePlan) -> dt.datetime:
        ts = plan.decision_ts
        if isinstance(ts, dt.datetime):
            return ts
        raise StrategyInputError("trade plan missing decision_ts")

    # -- exits -------------------------------------------------------------

    def _exit(
        self, instrument: Instrument, entry_price: Decimal, qty: Decimal,
        bars: Sequence[Bar], quote: Decimal, read: RegimeRead,
        state: MandateState, extras: dict,
    ) -> tuple[Optional[Intent], Optional[DecisionCard]]:
        if entry_price <= 0:
            raise StrategyInputError(
                f"{self.id}: {instrument.symbol} avg_price {entry_price}"
            )
        vol = atr(bars, ATR_PERIOD)
        if vol is None:
            raise StrategyInputError(
                f"{self.id}: cannot compute ATR for open {instrument.symbol}"
            )
        last = bars[-1]
        atr_stop = entry_price - ATR_STOP_MULT * vol
        floor = entry_price * (Decimal(1) - self.max_loss_pct)
        stop = max(atr_stop, floor)
        hit = last.close <= stop
        regime_kill = read.regime is Regime.BEAR
        if not hit and not regime_kill:
            return None, None
        pnl = (quote - entry_price) * qty
        why = (
            f"stop hit close {last.close} <= {stop}" if hit
            else f"regime {read.regime.value} invalidates the long"
        )
        card = DecisionCard(
            regime=read.regime, asset=instrument.symbol.split("/")[0],
            instrument=instrument.symbol, kind=InstrumentKind.SPOT,
            direction="sell", entry=entry_price, stop=stop,
            target_1=None, target_2=None,
            position_notional=abs(qty) * quote,
            margin_required=Decimal("0"), leverage=Decimal("1"),
            maximum_loss=max(Decimal("0"), -pnl),
            expected_profit=pnl, reward_risk=None,
            funding=str(extras.get("funding", "n/a")),
            oi_signal=str(extras.get("oi_change", "n/a")),
            vol_condition=vol_condition(bars),
            signal_score=0,
            portfolio_correlation=CRYPTO_BETA,
            current_nav=state.sleeve_nav,
            total_open_risk=state.open_risk,
            trade_thesis=why,
            invalidation="exit complete",
            confidence="exit",
            verdict=Verdict.EXECUTE,
            tag="exit",
            closed_pnl=pnl,
        )
        intent = self.make_intent(
            instrument=instrument, side=Side.SELL, qty=abs(qty),
            reason=card.reason_text(), decision_ts=last.ts, tag="exit",
            ref_price=quote,
        )
        return intent, card

    # -- paper derivatives -------------------------------------------------

    def _gated_derivatives(
        self, ctx: MarketContext, read: RegimeRead, extras: dict,
        state: MandateState, caps,
    ) -> list[DecisionCard]:
        """Paper/sim stubs. Never a live path — runner sets the flag False."""
        cards: list[DecisionCard] = []
        if not state.paper_derivatives:
            return cards
        if caps.perps <= 0:
            cards.append(_hold_card(
                read.regime, state,
                "perp sleeve capped at 0 this regime; no paper perp",
                asset="PERP",
            ))
        else:
            for stub in self._paper_perps:
                quote = ctx.quote(stub.symbol)
                if quote is None:
                    cards.append(_hold_card(
                        read.regime, state,
                        f"paper perp {stub.symbol} has no mark; not wired to "
                        "live OKX perps",
                        asset=stub.symbol,
                    ))
                    continue
                # A mark without a bar book is not enough to size a directional.
                cards.append(_hold_card(
                    read.regime, state,
                    f"paper perp {stub.symbol} quoted but not sized: "
                    "every directional perp needs entry/stop/TP/size/"
                    "invalidation and a spot-structure score; stub only",
                    asset=stub.symbol,
                ))
        premium_cap = state.sleeve_nav * caps.options
        if premium_cap <= 0:
            cards.append(_hold_card(
                read.regime, state,
                "options sleeve 0; keep USDT",
                asset="OPTIONS",
            ))
        else:
            # Never sell naked. If we cannot size a long premium inside the
            # cap, the 10% slice stays USDT.
            cards.append(_hold_card(
                read.regime, state,
                f"options paper stub: cannot size a defined-risk long premium "
                f"inside {premium_cap} USDT cap; keeping that slice in USDT. "
                "Never sell naked options. Live adapter is not enabled.",
                asset="OPTIONS",
            ))
        return cards

    # -- cards -------------------------------------------------------------

    def _card_from_plan(
        self, plan: TradePlan, read: RegimeRead, state: MandateState,
        extras: dict, verdict: Verdict, thesis: str,
        *, max_loss: Optional[Decimal] = None,
        notional: Optional[Decimal] = None,
        margin: Optional[Decimal] = None,
    ) -> DecisionCard:
        side_long = plan.entry > plan.stop
        return DecisionCard(
            regime=read.regime,
            asset=plan.asset,
            instrument=plan.instrument.symbol,
            kind=plan.kind,
            direction="buy" if side_long else "sell",
            entry=plan.entry,
            stop=plan.stop,
            target_1=plan.target_1,
            target_2=plan.target_2,
            position_notional=notional if notional is not None else Decimal("0"),
            margin_required=margin if margin is not None else Decimal("0"),
            leverage=plan.leverage,
            maximum_loss=max_loss if max_loss is not None else Decimal("0"),
            expected_profit=(plan.target_1 - plan.entry) / (plan.entry - plan.stop)
            * (max_loss or Decimal("0")),
            reward_risk=plan.reward_risk,
            funding=plan.funding,
            oi_signal=plan.oi_signal,
            vol_condition=plan.vol_condition,
            signal_score=plan.score.total,
            portfolio_correlation=cluster_of(plan.asset, plan.kind, side_long),
            current_nav=state.sleeve_nav,
            total_open_risk=state.open_risk + (max_loss or Decimal("0")),
            trade_thesis=thesis,
            invalidation=plan.invalidation,
            confidence=confidence_label(plan.score.total, read.regime),
            verdict=verdict,
            tag="entry",
        )


def _paper_perp(base: str) -> Instrument:
    return Instrument(
        f"{base}-PERP", AssetClass.CRYPTO, f"{base}/USDT:USDT",
        lot_size=1, tick_size=Decimal("0.1"), qty_precision=6,
        min_notional=Decimal("5"),
    )


def _state_from_ctx(ctx: MarketContext) -> MandateState:
    raw = ctx.extra.get(STATE_KEY)
    if raw is None:
        return MandateState()
    if isinstance(raw, MandateState):
        return raw
    if isinstance(raw, dict):
        return MandateState.from_mapping(raw)
    raise StrategyInputError(
        f"{STATE_KEY} must be MandateState or mapping, got {type(raw).__name__}"
    )


def _first_book(books: dict[str, Optional[list[Bar]]]) -> Optional[list[Bar]]:
    for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        bars = books.get(symbol)
        if bars:
            return bars
    for bars in books.values():
        if bars:
            return bars
    return None


def _hold_card(
    regime: Regime, state: MandateState, thesis: str, *, asset: str = "USDT",
) -> DecisionCard:
    return DecisionCard(
        regime=regime, asset=asset, instrument="USDT",
        kind=InstrumentKind.CASH, direction="hold",
        entry=None, stop=None, target_1=None, target_2=None,
        position_notional=Decimal("0"), margin_required=Decimal("0"),
        leverage=Decimal("1"), maximum_loss=Decimal("0"),
        expected_profit=Decimal("0"), reward_risk=None,
        funding="n/a", oi_signal="n/a", vol_condition="n/a",
        signal_score=0, portfolio_correlation="",
        current_nav=state.sleeve_nav, total_open_risk=state.open_risk,
        trade_thesis=thesis, invalidation="none — flat USDT",
        confidence="n/a", verdict=Verdict.WAIT, tag="hold",
    )


