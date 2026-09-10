"""
Crypto clip / daily / open-slot caps.

Strategies still propose one unit. The gate is what must make a live-ready
clip 50 USDT, not 1 BTC (~78k) or the leftover paper ETH/USDT 1 @ 2460.
These tests fail if CRYPTO_CLASS_LIMITS go back to equity-fraction-only
sizing, or if a reused ledger can inherit a 1-coin paper leftover.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

import runner
from core.gate import ClassLimits, GateConfig
from core.store import SQLitePositionStore
from core.types import (
    AssetClass, Fill, Instrument, Intent, MarketContext, Mode, OrderType, Side,
)

TOKEN = "t" * 32
NOW = dt.datetime(2026, 3, 2, 12, 0, tzinfo=dt.timezone.utc)
BTC_PX = Decimal("78000")
ETH_PX = Decimal("2460")
CLIP = Decimal("50")
DAILY = Decimal("250")


def base_env(tmp_path, **extra) -> dict[str, str]:
    env = {
        "TRADER_DB": str(tmp_path / "trader.db"),
        "ENGINE_API_TOKEN": TOKEN,
        "TRADER_CLASSES": "crypto",
    }
    env.update(extra)
    return env


def btc() -> Instrument:
    return runner.default_instruments(AssetClass.CRYPTO)[0]


def eth() -> Instrument:
    return runner.default_instruments(AssetClass.CRYPTO)[1]


def crypto_intent(*, instrument: Instrument | None = None, qty: str = "1",
                  key: str = "clip-probe", side: Side = Side.BUY,
                  strategy_id: str = "crypto_trend") -> Intent:
    inst = instrument or btc()
    return Intent(
        instrument=inst, side=side, qty=Decimal(qty),
        order_type=OrderType.MARKET, strategy_id=strategy_id,
        reason="clip-cap probe", idempotency_key=key,
    )


def crypto_ctx(*, equity: str = "2000000",
               quotes: dict[str, Decimal] | None = None) -> MarketContext:
    inst = btc()
    return MarketContext(
        now=NOW,
        quotes=quotes or {inst.symbol: BTC_PX, eth().symbol: ETH_PX},
        equity=Decimal(equity),
    )


@pytest.fixture()
def stack(tmp_path):
    made = []

    def build(**extra):
        s = runner.build_stack(env=base_env(tmp_path, **extra),
                               clock=lambda: NOW)
        made.append(s)
        return s

    yield build
    for s in made:
        s.close()
        s.store.close()


# --------------------------------------------------------------------------
# wired limits
# --------------------------------------------------------------------------


def test_crypto_clip_and_daily_caps_are_absolute_usdt(stack) -> None:
    """A 2M paper float must not license a 1-coin BTC clip. The operator
    caps are 50 USDT / clip and 250 USDT / day, not 10% of equity."""
    s = stack()
    limits = s.gate._OrderGate__config.limits_for(AssetClass.CRYPTO)
    assert limits is runner.CRYPTO_CLASS_LIMITS
    assert runner.CRYPTO_CLIP_USDT == CLIP
    assert runner.CRYPTO_DAILY_USDT == DAILY
    assert limits.max_order_notional == CLIP
    assert limits.max_daily_notional == DAILY
    assert limits.max_open_positions == 1


def test_one_btc_is_sized_to_fifty_usdt_not_one_coin(stack) -> None:
    """The defect: paper used qty 1, which is ~78k USDT on BTC if armed."""
    s = stack()
    approval = s.gate.check(crypto_intent(qty="1"), crypto_ctx())
    assert approval.approved, approval.reason
    assert approval.mode is Mode.PAPER
    sized = approval.intent.qty
    assert sized * BTC_PX <= CLIP
    assert sized * BTC_PX >= Decimal("5")   # above instrument min_notional
    assert sized < Decimal("1")
    # 50 / 78000 = 0.000641, snapped to qty_precision=6
    assert sized == Decimal("0.000641")


def test_one_eth_is_sized_to_fifty_usdt(stack) -> None:
    s = stack()
    approval = s.gate.check(
        crypto_intent(instrument=eth(), qty="1", key="eth-clip"),
        crypto_ctx())
    assert approval.approved, approval.reason
    assert approval.intent.qty * ETH_PX <= CLIP
    assert approval.intent.qty < Decimal("1")


def test_a_second_open_crypto_symbol_is_refused(stack) -> None:
    s = stack()
    book_eth = s.store
    book_eth.apply(
        Fill(order_id="seed-eth", instrument=eth(), side=Side.BUY,
             qty=Decimal("0.02"), price=ETH_PX, fees=Decimal("0"),
             at=NOW, mode=Mode.PAPER),
        strategy_id="crypto_trend",
    )
    approval = s.gate.check(crypto_intent(qty="1", key="btc-while-eth"),
                            crypto_ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "max_open_positions"


def test_daily_notional_refuses_a_sixth_fifty_usdt_clip(stack) -> None:
    """5 * 50 = 250. A sixth entry the same UTC day has no room."""
    s = stack()
    state = s.gate_state
    for i in range(5):
        state.record_submission(
            f"spent-{i}", AssetClass.CRYPTO, "2026-03-02", NOW,
            notional=CLIP)
    approval = s.gate.check(crypto_intent(qty="1", key="sixth"), crypto_ctx())
    assert not approval.approved
    assert approval.rejections[0].name == "daily_notional_cap"


def test_daily_notional_sizes_down_to_remaining_room(stack) -> None:
    s = stack()
    s.gate_state.record_submission(
        "spent-200", AssetClass.CRYPTO, "2026-03-02", NOW,
        notional=Decimal("200"))
    approval = s.gate.check(crypto_intent(qty="1", key="remain-50"),
                            crypto_ctx())
    assert approval.approved, approval.reason
    assert approval.intent.qty * BTC_PX <= Decimal("50")
    # 50 remaining / 78000 = 0.000641
    assert approval.intent.qty == Decimal("0.000641")


def test_equity_class_is_not_given_the_crypto_usdt_caps(stack) -> None:
    s = stack(TRADER_CLASSES="crypto,equity")
    equity_limits = s.gate._OrderGate__config.limits_for(AssetClass.EQUITY)
    crypto_limits = s.gate._OrderGate__config.limits_for(AssetClass.CRYPTO)
    assert crypto_limits.max_order_notional == CLIP
    assert equity_limits.max_order_notional is None
    assert equity_limits.max_daily_notional is None
    assert equity_limits.max_open_positions == ClassLimits().max_open_positions


def test_paper_is_still_the_default_and_nothing_is_armed(stack) -> None:
    s = stack()
    assert s.engine.config.requested_mode is Mode.PAPER
    assert s.engine.runner_modes()[AssetClass.CRYPTO] is Mode.PAPER
    assert s.handle.armed[AssetClass.CRYPTO] is False
    text = s.engine.arm_report_text()
    assert "CRYPTO_LIVE=disarmed" in text
    assert "mode=paper" in text


# --------------------------------------------------------------------------
# start flat — do not inherit the leftover 1-coin ETH
# --------------------------------------------------------------------------


def test_build_stack_flattens_inherited_one_coin_paper_eth(tmp_path) -> None:
    """Phase-0 leftover: PAPER ETH/USDT buy qty 1 @ 2460. A reused ledger
    must not hand that 1-coin row to the sized runner — it would consume
    the single open slot and be ~49x the clip."""
    db = str(tmp_path / "trader.db")
    leftover = eth()
    with SQLitePositionStore(db) as seed:
        seed.apply(
            Fill(order_id="PAPER-crypto-000011", instrument=leftover,
                 side=Side.BUY, qty=Decimal("1"), price=ETH_PX,
                 fees=Decimal("0"), at=NOW, mode=Mode.PAPER),
            strategy_id="crypto_trend",
        )
        held = seed.positions(Mode.PAPER, AssetClass.CRYPTO)
        assert len(held) == 1
        assert held[0].qty == Decimal("1")
        assert held[0].avg_price == ETH_PX

    s = runner.build_stack(env=base_env(tmp_path), clock=lambda: NOW)
    try:
        after = s.store.positions(Mode.PAPER, AssetClass.CRYPTO)
        assert after == [], (
            "inherited 1-coin ETH/USDT must not remain open on the sized book")
        approval = s.gate.check(crypto_intent(qty="1"), crypto_ctx())
        assert approval.approved, approval.reason
        assert approval.intent.qty * BTC_PX <= CLIP
    finally:
        s.close()
        s.store.close()


def test_a_clip_sized_paper_position_is_kept_across_rebuild(tmp_path) -> None:
    """Start-flat is for leftovers above the clip, not a wipe of every row."""
    db = str(tmp_path / "trader.db")
    qty = (CLIP / BTC_PX).quantize(Decimal("0.000001"))
    with SQLitePositionStore(db) as seed:
        seed.apply(
            Fill(order_id="paper-clip-ok", instrument=btc(),
                 side=Side.BUY, qty=qty, price=BTC_PX,
                 fees=Decimal("0"), at=NOW, mode=Mode.PAPER),
            strategy_id="crypto_trend",
        )
    s = runner.build_stack(env=base_env(tmp_path), clock=lambda: NOW)
    try:
        held = s.store.positions(Mode.PAPER, AssetClass.CRYPTO)
        assert len(held) == 1
        assert held[0].qty == qty
        assert held[0].instrument.symbol == "BTC/USDT"
    finally:
        s.close()
        s.store.close()


# --------------------------------------------------------------------------
# ClassLimits construction
# --------------------------------------------------------------------------


def test_absolute_caps_must_be_positive_decimals() -> None:
    with pytest.raises((ValueError, TypeError)):
        ClassLimits(max_order_notional=Decimal("0"))
    with pytest.raises((ValueError, TypeError)):
        ClassLimits(max_daily_notional=Decimal("-1"))
    with pytest.raises((ValueError, TypeError)):
        ClassLimits(max_order_notional=50)  # type: ignore[arg-type]


def test_a_clip_larger_than_the_daily_cap_does_not_construct() -> None:
    with pytest.raises(ValueError):
        ClassLimits(max_order_notional=Decimal("300"),
                    max_daily_notional=Decimal("250"))


def test_equity_default_limits_have_no_absolute_usdt_cap() -> None:
    defaults = ClassLimits()
    assert defaults.max_order_notional is None
    assert defaults.max_daily_notional is None
    assert GateConfig().limits_for(AssetClass.EQUITY) is GateConfig().default_limits
