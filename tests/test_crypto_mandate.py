"""Paper mandate engine — Vishesh's 1,000 USDT sleeve.

These tests pin the rules that replace crypto_trend / crypto_breakout as
the registered crypto path. No test here sets CRYPTO_LIVE, EQUITY_LIVE,
or TRADER_MODE=live.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

import runner
from adapters.okx import OkxAdapter
from adapters.paper import PaperAdapter
from core.types import AssetClass, Mode, OrderType, Side
from strategies.base import BARS_KEY, Bar
from strategies.mandate import (
    MARKET_KEY, STATE_KEY, CryptoMandateStrategy, MandateState, Regime,
    Verdict, classify_regime, score_plan,
)
from strategies.mandate.constants import SLEEVE_START
from strategies.mandate.risk import CRYPTO_BETA, new_directional_blocked
from tests import test_engine as eng_tests
from tests._strategy_support import coin, context, d, position, utc

INTERVAL = dt.timedelta(minutes=15)
BTC = coin("BTC/USDT")
ETH = coin("ETH/USDT")
SOL = coin("SOL/USDT")
NOW = utc(2026, 9, 2, 12, 0)


def _bar(i: int, close: object, *, start=None, volume: object = 100,
         spread: object = 1) -> Bar:
    start = start or utc(2026, 8, 1, 0, 0)
    c = d(close)
    sp = d(spread)
    return Bar.of(
        start + i * INTERVAL, c, c + sp, c - sp, c, volume,
    )


def grind_up(n: int = 220, start_px: object = 100, drift: object = "0.4",
             *, volume: object = 100, spike_last: bool = True) -> list[Bar]:
    """Monotone grind with an optional volume spike on the last bar."""
    px = d(start_px)
    step = d(drift)
    out: list[Bar] = []
    for i in range(n):
        vol = d(volume) * 3 if spike_last and i == n - 1 else d(volume)
        out.append(_bar(i, px, volume=vol, spread="1.2"))
        px += step
    return out


def pullback_setup(n: int = 220) -> list[Bar]:
    """Uptrend, shallow pullback, resume — structure offers ≥2R to the high."""
    out: list[Bar] = []
    px = d(100)
    for i in range(n - 20):
        out.append(_bar(i, px, volume=100, spread="1.2"))
        px += d("0.4")
    peak = px
    for j in range(12):
        px = peak - d("1.0") * (j + 1)
        out.append(_bar(n - 20 + j, px, volume=80, spread="1.0"))
    trough = px
    # Recover a little so the stop (trough low) is tight vs the prior high.
    # prior_high ≈ peak+1.2; entry = trough+3; risk ≈ 4; upside ≈ 13 → R:R > 2.
    resume = trough + d("3")
    for k in range(8):
        step_px = trough + (resume - trough) * d(k + 1) / d(8)
        vol = 300 if k == 7 else 110
        out.append(_bar(n - 8 + k, step_px, volume=vol, spread="1.0"))
    return out


def chop(n: int = 80, mid: object = 100) -> list[Bar]:
    """Range: no stacked EMAs, no 2R structure."""
    m = d(mid)
    out: list[Bar] = []
    for i in range(n):
        px = m + (d("2") if i % 2 == 0 else d("-2"))
        out.append(_bar(i, px, volume=50, spread="3"))
    return out


def strat(instruments=None, **kw) -> CryptoMandateStrategy:
    return CryptoMandateStrategy(instruments=instruments or [BTC, ETH, SOL], **kw)


def extras_bull() -> dict:
    return {
        MARKET_KEY: {
            "funding": Decimal("-0.0001"),
            "oi_change": "up",
            "liquidations": "short_squeeze",
            "cross_agree": True,
            "skew": "calls_bid",
        }
    }


def ctx_of(btc_bars, *, eth_bars=None, sol_bars=None, quotes=None,
           positions=(), state=None, extra=None, now=None):
    bars = {"BTC/USDT": btc_bars}
    if eth_bars is not None:
        bars["ETH/USDT"] = eth_bars
    if sol_bars is not None:
        bars["SOL/USDT"] = sol_bars
    last = btc_bars[-1]
    q = quotes if quotes is not None else {"BTC/USDT": last.close}
    payload = dict(extra or {})
    if state is not None:
        payload[STATE_KEY] = state
    payload.update(extras_bull())
    return context(now or last.ts, bars=bars, quotes=q,
                   positions=positions, extra=payload)


def execute_ctx():
    """A score-passing BTC long: pullback in a bull stack. Contract tests too."""
    bars = pullback_setup(220)
    return ctx_of(bars, quotes={"BTC/USDT": bars[-1].close})


# --------------------------------------------------------------------------
# regime
# --------------------------------------------------------------------------


def test_stacked_btc_and_eth_is_strong_bull() -> None:
    btc = grind_up(220, start_px=100)
    eth = grind_up(220, start_px=50, drift="0.3")
    read = classify_regime(btc, eth)
    assert read.regime is Regime.STRONG_BULL
    assert read.btc_ema200 is not None
    assert btc[-1].close > read.btc_ema20 > read.btc_ema50 > read.btc_ema200


def test_chop_is_range_neutral() -> None:
    read = classify_regime(chop())
    assert read.regime is Regime.RANGE_NEUTRAL


def test_breakdown_is_bear() -> None:
    bars = grind_up(220, start_px=220, drift="-0.5", spike_last=False)
    read = classify_regime(bars)
    assert read.regime is Regime.BEAR


def test_insufficient_history_is_range_not_a_fabricated_bull() -> None:
    bars = grind_up(20)
    read = classify_regime(bars)
    assert read.regime is Regime.RANGE_NEUTRAL
    assert "insufficient" in read.detail


def test_regime_caps_change_between_bull_range_and_bear() -> None:
    from strategies.mandate.types import REGIME_CAPS
    assert REGIME_CAPS[Regime.STRONG_BULL].spot > REGIME_CAPS[Regime.RANGE_NEUTRAL].spot
    assert REGIME_CAPS[Regime.RANGE_NEUTRAL].spot > REGIME_CAPS[Regime.BEAR].spot
    assert REGIME_CAPS[Regime.BEAR].spot == Decimal("0")
    assert REGIME_CAPS[Regime.RANGE_NEUTRAL].perps == Decimal("0")
    assert REGIME_CAPS[Regime.STRONG_BULL].reserve == Decimal("0.30")


# --------------------------------------------------------------------------
# score / no-trade
# --------------------------------------------------------------------------


def test_score_below_65_is_no_trade() -> None:
    bars = chop(80)
    breakdown = score_plan(bars, Decimal("1.2"), bars[-1].close)
    assert breakdown.total < 65
    tick = strat().evaluate(ctx_of(bars, extra={}))
    assert tick.intents == ()
    assert all(c.verdict is not Verdict.EXECUTE or c.tag == "exit"
               for c in tick.cards)


def test_usdt_hold_is_valid_when_there_is_no_setup() -> None:
    bars = chop(80)
    tick = strat().evaluate(ctx_of(bars, extra={}))
    assert tick.intents == ()
    holds = [c for c in tick.cards if c.asset == "USDT" or c.kind.value == "usdt"]
    assert holds, tick.cards
    assert any("USDT" in c.trade_thesis or "holding" in c.trade_thesis.lower()
               for c in tick.cards)


def test_rr_below_two_is_rejected() -> None:
    """Deep swing stop vs a nearby prior high — structure does not offer 2R."""
    bars = []
    px = d(100)
    for i in range(70):
        bars.append(_bar(i, px, volume=80, spread="1"))
        px += d("0.1")
    # Deep wick: structure stop far below, target only a couple of handles up.
    deep = list(bars)
    wick = _bar(70, px, volume=90, spread="1")
    # Replace low with a far swing without lifting the close.
    deep.append(Bar.of(wick.ts, wick.open, wick.high, d("40"), wick.close, 90))
    deep.append(_bar(71, px + d("0.2"), volume=90, spread="1"))
    tick = strat().evaluate(ctx_of(deep, extra={}))
    assert tick.intents == ()
    assert all(c.verdict is not Verdict.EXECUTE or c.tag != "entry"
               for c in tick.cards)


# --------------------------------------------------------------------------
# risk halt
# --------------------------------------------------------------------------


def test_three_consecutive_losses_stop_new_directional() -> None:
    state = MandateState(consecutive_losses=3)
    assert new_directional_blocked(state) is not None
    bars = pullback_setup(220)
    tick = strat().evaluate(ctx_of(bars, state=state))
    assert tick.intents == ()
    assert any("consecutive-loss" in c.trade_thesis for c in tick.cards)


def test_two_consecutive_losses_do_not_halt() -> None:
    assert new_directional_blocked(MandateState(consecutive_losses=2)) is None


def test_daily_two_percent_loss_halts_new_directional() -> None:
    state = MandateState(sleeve_nav=SLEEVE_START, daily_loss=Decimal("20"))
    assert new_directional_blocked(state) is not None
    bars = pullback_setup(220)
    tick = strat().evaluate(ctx_of(bars, state=state))
    assert tick.intents == ()
    assert any("daily loss" in c.trade_thesis for c in tick.cards)


def test_correlated_cluster_is_one_crypto_beta_thesis() -> None:
    from strategies.mandate.risk import cluster_of
    from strategies.mandate.types import InstrumentKind
    assert cluster_of("BTC", InstrumentKind.PERP, True) == CRYPTO_BETA
    assert cluster_of("ETH", InstrumentKind.PERP, True) == CRYPTO_BETA
    assert cluster_of("SOL", InstrumentKind.SPOT, True) == CRYPTO_BETA
    assert cluster_of("BTC", InstrumentKind.OPTION, True) == CRYPTO_BETA
    state = MandateState(
        open_cluster=CRYPTO_BETA,
        open_risk=Decimal("24"),
        open_symbols=("ETH/USDT",),
    )
    bars = pullback_setup(220)
    tick = strat().evaluate(ctx_of(bars, state=state))
    assert tick.intents == ()
    assert any("cluster" in c.trade_thesis.lower() or c.verdict is Verdict.REJECT
               for c in tick.cards)


# --------------------------------------------------------------------------
# execute path + paper fill
# --------------------------------------------------------------------------


def test_score_passing_spot_stamps_a_stop_and_is_not_unbounded_market() -> None:
    ctx = execute_ctx()
    strategy = strat()
    intents = list(strategy.propose(ctx))
    assert len(intents) == 1
    intent = intents[0]
    assert intent.side is Side.BUY
    assert intent.instrument.symbol == "BTC/USDT"
    assert intent.order_type is OrderType.MARKET
    assert intent.limit_price == ctx.quotes["BTC/USDT"]
    assert intent.limit_price > 0
    assert "STOP=" in intent.reason
    assert "STOP=n/a" not in intent.reason
    assert intent.qty > 0
    tick = strategy.last_tick
    assert tick is not None
    card = [c for c in tick.cards if c.verdict is Verdict.EXECUTE][0]
    assert card.stop is not None and card.stop < card.entry
    assert card.reward_risk is not None and card.reward_risk >= 2
    assert card.signal_score >= 65
    assert card.current_nav == SLEEVE_START


def test_paper_fill_of_a_score_passing_spot_with_stop_stamped() -> None:
    bars = pullback_setup(220)
    last = bars[-1]
    instrument = coin("BTC/USDT")
    strategy = CryptoMandateStrategy(instruments=[instrument])

    def extra(_ac, _now):
        return {
            BARS_KEY: {"BTC/USDT": bars},
            STATE_KEY: MandateState(),
            **extras_bull(),
        }

    h = eng_tests.build(
        strategies=[strategy],
        instruments=(instrument,),
        prices={"BTC/USDT": last.close},
        cash=Decimal("2000000"),
        clock_at=last.ts,
        extra_provider=extra,
        limits=runner.CRYPTO_CLASS_LIMITS,
    )
    assert isinstance(h.adapter, PaperAdapter)
    assert not isinstance(h.adapter, OkxAdapter)

    eng_tests.tick(h)

    held = h.inner.positions(Mode.PAPER, AssetClass.CRYPTO)
    assert len(held) == 1
    assert held[0].instrument.symbol == "BTC/USDT"
    assert held[0].qty > 0
    assert held[0].qty * last.close <= runner.CRYPTO_CLIP_USDT
    assert held[0].mode is Mode.PAPER
    filled = h.adapter.order("PAPER-crypto-000001")
    assert filled is not None
    assert filled.intent.limit_price == last.close
    assert "STOP=" in filled.intent.reason


def test_propose_is_pure() -> None:
    ctx = execute_ctx()
    s = strat()
    assert list(s.propose(ctx)) == list(s.propose(ctx))


def test_idle_synthetic_triangle_does_not_spray() -> None:
    """The default synthetic book is a triangle wave. Mandate must sit in USDT."""
    env = {
        "TRADER_DB": "/tmp/does-not-matter-mandate.db",
        "ENGINE_API_TOKEN": "t" * 32,
        "TRADER_CLASSES": "crypto",
    }
    # Use a temp path via the stack helper pattern without writing /tmp blindly
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        env["TRADER_DB"] = str(Path(td) / "t.db")
        s = runner.build_stack(env=env, clock=lambda: NOW)
        try:
            wired = s.engine._runners[AssetClass.CRYPTO].wiring.strategies
            ids = [x.id for x in wired]
            assert ids == ["crypto_mandate"]
            assert s.engine._runners[AssetClass.CRYPTO].mode is Mode.PAPER
        finally:
            s.close()
            s.store.close()


# --------------------------------------------------------------------------
# paper derivatives stay gated; no live arming
# --------------------------------------------------------------------------


def test_paper_derivatives_are_stubs_not_intents_without_a_book() -> None:
    bars = chop(80)
    state = MandateState(paper_derivatives=True)
    tick = strat().evaluate(ctx_of(bars, state=state, extra={}))
    assert all(i.instrument.symbol.endswith("/USDT") for i in tick.intents)
    assert any("perp" in c.trade_thesis.lower() or "options" in c.trade_thesis.lower()
               for c in tick.cards)


def test_live_flag_on_state_does_not_emit_perp_or_option_intents() -> None:
    bars = pullback_setup(220)
    state = MandateState(paper_derivatives=False)
    tick = strat().evaluate(ctx_of(bars, state=state))
    for intent in tick.intents:
        assert "-PERP" not in intent.instrument.symbol
        assert "C" not in intent.instrument.symbol[-2:] or "/USDT" in intent.instrument.symbol


def test_never_sells_naked_options() -> None:
    bars = pullback_setup(220)
    tick = strat().evaluate(ctx_of(bars))
    for intent in tick.intents:
        assert intent.side is Side.BUY or intent.instrument.symbol.endswith("/USDT")


# --------------------------------------------------------------------------
# runner wiring
# --------------------------------------------------------------------------


def test_default_crypto_universe_is_btc_eth_sol() -> None:
    symbols = [i.symbol for i in runner.default_instruments(AssetClass.CRYPTO)]
    assert symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_spray_stays_off_unless_explicitly_enabled() -> None:
    instruments = runner.default_instruments(AssetClass.CRYPTO)
    assert [s.id for s in runner.default_strategies(AssetClass.CRYPTO, instruments)] == [
        "crypto_mandate"
    ]
    sprayed = runner.default_strategies(
        AssetClass.CRYPTO, instruments, enable_crypto_spray=True)
    assert [s.id for s in sprayed] == [
        "crypto_mandate", "crypto_trend", "crypto_breakout",
    ]


def test_unrecognised_spray_flag_raises() -> None:
    with pytest.raises(ValueError, match="CRYPTO_EMA_SPRAY"):
        runner.parse_crypto_spray("maybe")


def test_class_limits_remain_fifty_two_fifty_max_one() -> None:
    assert runner.CRYPTO_CLIP_USDT == Decimal("50")
    assert runner.CRYPTO_DAILY_USDT == Decimal("250")
    assert runner.CRYPTO_CLASS_LIMITS.max_open_positions == 1
