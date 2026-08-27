"""Index-option debit spread driven by a regime read."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from core.types import OrderType, Side
from strategies.base import MissingDataError, StrategyInputError
from strategies.equity_trend import (
    IndexOptionTrendStrategy, OptionContract, RegimeRead, RegimeState, Right,
)
from tests._strategy_support import context, d, ist, option, position

NOW = ist(2026, 8, 24, 11, 0)          # Monday, inside the session
EXPIRY = dt.date(2026, 9, 3)           # 10 days out
NEAR_EXPIRY = dt.date(2026, 8, 25)     # 1 day out
STRIKES = [22300, 22400, 22500, 22600, 22700]
SPOT = "22500"

CALL_PREMIUM = {22300: "330", 22400: "250", 22500: "180", 22600: "120", 22700: "80"}
PUT_PREMIUM = {22300: "70", 22400: "110", 22500: "180", 22600: "250", 22700: "340"}


def sym(strike: int, right: Right, expiry: dt.date = EXPIRY) -> str:
    return f"NIFTY{expiry:%d%b%y}{strike}{right.value}".upper()


def contract(strike: int, right: Right, expiry: dt.date = EXPIRY) -> OptionContract:
    return OptionContract(
        instrument=option(sym(strike, right, expiry), expiry),
        underlying="NIFTY", strike=Decimal(strike), right=right,
    )


def chain(expiry: dt.date = EXPIRY) -> list[OptionContract]:
    return [contract(s, r, expiry) for s in STRIKES for r in (Right.CALL, Right.PUT)]


def quotes(expiry: dt.date = EXPIRY) -> dict[str, str]:
    q = {"NIFTY": SPOT}
    for s in STRIKES:
        q[sym(s, Right.CALL, expiry)] = CALL_PREMIUM[s]
        q[sym(s, Right.PUT, expiry)] = PUT_PREMIUM[s]
    return q


def regime(state=RegimeState.TREND_UP, confidence="0.75", as_of=None):
    return RegimeRead(state=state, confidence=d(confidence), as_of=as_of or NOW)


def make_ctx(*, now=NOW, state=RegimeState.TREND_UP, confidence="0.75",
             as_of=None, positions=(), expiry=EXPIRY, with_chain=True,
             with_regime=True, extra_quotes=None, drop_quotes=()):
    q = quotes(expiry)
    q.update(extra_quotes or {})
    for k in drop_quotes:
        q.pop(k, None)
    extra: dict = {}
    if with_regime:
        extra["regime"] = {"NIFTY": regime(state, confidence, as_of)}
    if with_chain:
        extra["option_chain"] = {"NIFTY": chain(expiry)}
    return context(now, bars={}, quotes=q, positions=positions, extra=extra)


def strat(**kw) -> IndexOptionTrendStrategy:
    params = dict(underlying="NIFTY", width_strikes=2,
                  min_confidence=Decimal("0.60"), min_days_to_expiry=3,
                  close_before_expiry_days=1)
    params.update(kw)
    return IndexOptionTrendStrategy(**params)


# ------------------------------------------------------------------ entry

def test_trend_up_opens_a_bull_call_debit_spread():
    intents = strat().propose(make_ctx())
    assert len(intents) == 2
    long_leg, short_leg = intents
    # The LONG leg is proposed first: if only one order fills we are long an
    # option (defined loss), never short one.
    assert long_leg.side is Side.BUY
    assert long_leg.instrument.symbol == sym(22500, Right.CALL)
    assert short_leg.side is Side.SELL
    assert short_leg.instrument.symbol == sym(22700, Right.CALL)
    assert long_leg.structure_id and long_leg.structure_id == short_leg.structure_id
    assert long_leg.qty == short_leg.qty == Decimal(75)
    assert long_leg.order_type is OrderType.LIMIT and long_leg.limit_price is not None
    assert "debit spread" in long_leg.reason and "Max loss is the 100 debit" in long_leg.reason


def test_trend_down_opens_a_bear_put_debit_spread_below_the_money():
    intents = strat().propose(make_ctx(state=RegimeState.TREND_DOWN))
    assert [i.instrument.symbol for i in intents] == [
        sym(22500, Right.PUT), sym(22300, Right.PUT)]
    assert [i.side for i in intents] == [Side.BUY, Side.SELL]


def test_a_range_regime_is_not_traded_by_a_directional_strategy():
    assert list(strat().propose(make_ctx(state=RegimeState.RANGE))) == []


def test_low_confidence_is_not_traded():
    assert list(strat().propose(make_ctx(confidence="0.40"))) == []


def test_a_stale_regime_read_is_treated_as_no_read():
    """Positioning on a regime computed an hour ago is worse than not trading;
    'stale' must not be silently accepted as 'current'."""
    old = NOW - dt.timedelta(minutes=45)
    assert list(strat().propose(make_ctx(as_of=old))) == []


def test_no_regime_published_means_no_entry():
    assert list(strat().propose(make_ctx(with_regime=False))) == []


def test_no_chain_means_no_entry():
    assert list(strat().propose(make_ctx(with_chain=False))) == []


def test_no_entry_without_a_spot_price():
    assert list(strat().propose(make_ctx(drop_quotes=["NIFTY"]))) == []


def test_no_entry_when_only_one_leg_can_be_priced():
    """Half a spread priced is not a spread."""
    ctx = make_ctx(drop_quotes=[sym(22700, Right.CALL)])
    assert list(strat().propose(ctx)) == []


def test_no_entry_when_the_quotes_imply_a_credit_rather_than_a_debit():
    """Taken for a credit this is a different structure with a different risk
    profile. Crossed or stale quotes are the usual cause."""
    ctx = make_ctx(extra_quotes={sym(22500, Right.CALL): "70"})   # below the 22700 at 80
    assert list(strat().propose(ctx)) == []


def test_expiry_too_near_is_skipped():
    """Entering a spread days from expiry buys gamma and sells nothing."""
    assert list(strat().propose(make_ctx(expiry=NEAR_EXPIRY))) == []


def test_a_chain_too_shallow_for_the_width_is_skipped():
    assert list(strat().propose(make_ctx())) != []
    assert list(strat(width_strikes=4).propose(make_ctx())) == []


def test_nothing_is_proposed_outside_the_session():
    assert list(strat().propose(make_ctx(now=ist(2026, 8, 24, 8, 0)))) == []
    assert list(strat().propose(make_ctx(now=ist(2026, 8, 23, 11, 0)))) == []


def test_the_same_regime_read_is_the_same_decision():
    a = strat().propose(make_ctx())
    b = strat().propose(make_ctx(now=NOW + dt.timedelta(minutes=4)))
    assert [i.idempotency_key for i in a] == [i.idempotency_key for i in b]
    assert a[0].structure_id == b[0].structure_id


def test_a_new_regime_read_is_a_new_decision():
    a = strat().propose(make_ctx())
    b = strat().propose(make_ctx(as_of=NOW - dt.timedelta(minutes=5)))
    assert a[0].idempotency_key != b[0].idempotency_key
    assert a[0].structure_id != b[0].structure_id


# --------------------------------------------------------------- positions

def spread_legs(structure_id="sid-1", expiry=EXPIRY, right=Right.CALL,
                long_strike=22500, short_strike=22700, opened_at=None):
    return [
        position(option(sym(long_strike, right, expiry), expiry), 75,
                 CALL_PREMIUM.get(long_strike, "180"), strategy_id="equity_trend_nifty",
                 structure_id=structure_id, opened_at=opened_at),
        position(option(sym(short_strike, right, expiry), expiry), -75,
                 CALL_PREMIUM.get(short_strike, "80"), strategy_id="equity_trend_nifty",
                 structure_id=structure_id, opened_at=opened_at),
    ]


def test_an_open_structure_blocks_a_second_one():
    ctx = make_ctx(positions=spread_legs())
    assert list(strat().propose(ctx)) == []


def test_a_regime_flip_closes_the_structure_short_leg_first():
    """Buying the short leg back first means a single fill leaves us long an
    option, not naked short one."""
    ctx = make_ctx(state=RegimeState.TREND_DOWN, positions=spread_legs())
    intents = strat().propose(ctx)
    assert len(intents) == 2
    assert intents[0].side is Side.BUY
    assert intents[0].instrument.symbol == sym(22700, Right.CALL)
    assert intents[1].side is Side.SELL
    assert intents[1].instrument.symbol == sym(22500, Right.CALL)
    assert all(i.structure_id == "sid-1" for i in intents)
    assert all(i.qty == Decimal(75) for i in intents)
    assert "regime flipped" in intents[0].reason


def test_a_range_regime_also_closes_a_directional_structure():
    ctx = make_ctx(state=RegimeState.RANGE, positions=spread_legs())
    assert len(strat().propose(ctx)) == 2


def test_the_structure_is_unwound_before_expiry():
    legs = spread_legs(expiry=NEAR_EXPIRY)
    ctx = make_ctx(expiry=NEAR_EXPIRY, positions=legs)
    intents = strat().propose(ctx)
    assert len(intents) == 2
    assert "to expiry" in intents[0].reason


def test_a_matching_regime_leaves_a_healthy_structure_alone():
    ctx = make_ctx(positions=spread_legs())
    assert list(strat().propose(ctx)) == []


def test_an_orphaned_short_leg_is_closed_as_naked_risk():
    legs = spread_legs(opened_at=NOW - dt.timedelta(minutes=30))
    ctx = make_ctx(positions=[legs[1]])
    intents = strat().propose(ctx)
    assert len(intents) == 1
    assert intents[0].side is Side.BUY
    assert "NAKED SHORT" in intents[0].reason


def test_a_leg_still_inside_the_grace_window_is_left_to_pair_up():
    """Both legs are placed in one pass; a lone leg seen seconds later is the
    other order still in flight, not an orphan."""
    legs = spread_legs(opened_at=NOW - dt.timedelta(minutes=1))
    ctx = make_ctx(positions=[legs[1]])
    assert list(strat().propose(ctx)) == []


def test_a_leg_of_unknown_age_is_not_assumed_to_be_in_flight():
    """opened_at is None, so the grace cannot be established. Closing a leg
    that was about to pair up costs a round trip; assuming a naked short is
    in flight has no bound."""
    legs = spread_legs(opened_at=None)
    ctx = make_ctx(positions=[legs[1]])
    assert len(strat().propose(ctx)) == 1


def test_legs_without_a_structure_id_are_closed():
    """Un-grouped legs cannot be unwound together by the executor, so they are
    not the defined-risk structure this strategy believes it holds."""
    legs = spread_legs(structure_id=None, opened_at=NOW - dt.timedelta(hours=1))
    ctx = make_ctx(positions=legs)
    intents = strat().propose(ctx)
    assert len(intents) == 2
    assert "malformed structure" in intents[0].reason


def test_legs_on_different_expiries_are_closed():
    legs = spread_legs()
    legs[1] = position(option(sym(22700, Right.CALL, NEAR_EXPIRY), NEAR_EXPIRY), -75,
                       "80", strategy_id="equity_trend_nifty", structure_id="sid-1")
    ctx = make_ctx(positions=legs,
                   extra_quotes={sym(22700, Right.CALL, NEAR_EXPIRY): "80"})
    intents = strat().propose(ctx)
    assert len(intents) == 2
    assert "malformed structure" in intents[0].reason


def test_no_quote_for_an_open_leg_raises_rather_than_sending_an_unpriced_exit():
    ctx = make_ctx(state=RegimeState.TREND_DOWN, positions=spread_legs(),
                   drop_quotes=[sym(22700, Right.CALL)])
    with pytest.raises(MissingDataError):
        strat().propose(ctx)


def foreign_legs():
    return [position(p.instrument, p.qty, p.avg_price, strategy_id="someone_else",
                     structure_id=p.structure_id) for p in spread_legs()]


def test_another_strategys_legs_are_never_unwound():
    """Unwinding a leg someone else opened turns their spread into a naked
    position — the exact failure this strategy orders its own exits to avoid."""
    legs = foreign_legs()
    ctx = make_ctx(state=RegimeState.TREND_DOWN, positions=legs)
    touched = {i.instrument.symbol for i in strat().propose(ctx)}
    assert touched.isdisjoint({p.instrument.symbol for p in legs})


def test_no_entry_on_an_instrument_another_strategy_already_holds():
    ctx = make_ctx(state=RegimeState.TREND_UP, positions=foreign_legs())
    assert list(strat().propose(ctx)) == []


# ------------------------------------------------------------ regime input

def test_an_unknown_regime_label_raises_instead_of_defaulting_to_range():
    """'I do not know what this means' and 'the market is ranging' are
    different facts and must not collapse into one."""
    ctx = context(NOW, bars={}, quotes=quotes(),
                  extra={"regime": {"NIFTY": {"state": "sideways-ish",
                                              "confidence": d("0.9"),
                                              "as_of": NOW}},
                         "option_chain": {"NIFTY": chain()}})
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_a_regime_read_as_a_plain_mapping_is_accepted():
    ctx = context(NOW, bars={}, quotes=quotes(),
                  extra={"regime": {"NIFTY": {"state": "trend_up",
                                              "confidence": "0.9",
                                              "as_of": NOW}},
                         "option_chain": {"NIFTY": chain()}})
    assert len(strat().propose(ctx)) == 2


def test_a_regime_read_stamped_in_the_future_raises():
    with pytest.raises(StrategyInputError):
        strat().propose(make_ctx(as_of=NOW + dt.timedelta(minutes=5)))


def test_a_regime_read_without_a_timestamp_raises():
    ctx = context(NOW, bars={}, quotes=quotes(),
                  extra={"regime": {"NIFTY": {"state": "trend_up",
                                              "confidence": "0.9"}},
                         "option_chain": {"NIFTY": chain()}})
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_a_naive_regime_timestamp_raises():
    with pytest.raises(StrategyInputError):
        RegimeRead(state=RegimeState.TREND_UP, confidence=d("0.8"),
                   as_of=dt.datetime(2026, 8, 24, 11, 0))


def test_confidence_outside_zero_to_one_raises():
    with pytest.raises(StrategyInputError):
        RegimeRead(state=RegimeState.TREND_UP, confidence=d("1.4"), as_of=NOW)


def test_a_duplicated_strike_in_the_chain_raises():
    """Two contracts claiming one strike means the chain merged two sources;
    picking either silently is a coin flip on which instrument gets sold."""
    dupe = chain() + [contract(22700, Right.CALL)]
    ctx = context(NOW, bars={}, quotes=quotes(),
                  extra={"regime": {"NIFTY": regime()},
                         "option_chain": {"NIFTY": dupe}})
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_a_contract_for_another_underlying_raises():
    foreign = OptionContract(
        instrument=option("BANKNIFTY03SEP2650000CE", EXPIRY),
        underlying="BANKNIFTY", strike=Decimal(50000), right=Right.CALL)
    ctx = context(NOW, bars={}, quotes=quotes(),
                  extra={"regime": {"NIFTY": regime()},
                         "option_chain": {"NIFTY": chain() + [foreign]}})
    with pytest.raises(StrategyInputError):
        strat().propose(ctx)


def test_an_option_contract_without_an_expiry_raises():
    from core.types import AssetClass, Instrument
    with pytest.raises(StrategyInputError):
        OptionContract(
            instrument=Instrument(symbol="X22500CE", asset_class=AssetClass.EQUITY,
                                  broker_id="x", lot_size=75),
            underlying="NIFTY", strike=Decimal(22500), right=Right.CALL)


def test_construction_rejects_bad_parameters():
    with pytest.raises(StrategyInputError):
        strat(underlying=" ")
    with pytest.raises(StrategyInputError):
        strat(width_strikes=0)
    with pytest.raises(StrategyInputError):
        strat(min_confidence=Decimal("0"))
    with pytest.raises(StrategyInputError):
        # Otherwise it opens a structure it is already obliged to close.
        strat(min_days_to_expiry=1, close_before_expiry_days=1)
