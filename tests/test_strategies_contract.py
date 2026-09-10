"""Properties every strategy in the book must hold, checked on all four.

These are the ones the rest of the engine relies on: the UI shows `reason`
verbatim, the gate dedups on `idempotency_key`, and the runner calls propose()
on every tick.
"""

from __future__ import annotations

from core.protocols import Strategy
from core.types import AssetClass, OrderType
from strategies import (
    CryptoBreakoutStrategy, CryptoEmaTrendStrategy, CryptoMandateStrategy,
    EquityMeanReversionStrategy, IndexOptionTrendStrategy,
)
from tests import test_crypto_breakout as bo
from tests import test_crypto_mandate as md
from tests import test_crypto_trend as ct
from tests import test_equity_meanrev as mr
from tests import test_equity_trend as et


def cases():
    """(strategy, context) pairs that each produce at least one intent."""
    return [
        (et.strat(), et.make_ctx()),
        (mr.strat(), mr.make_ctx()),
        (ct.strat(), ct.ctx_from(ct.CROSS_UP)[0]),
        (bo.strat(), bo.make_ctx(bo.RANGE_ROWS + [bo.BREAKOUT_ROW])[0]),
        (md.strat([md.BTC]), md.execute_ctx()),
    ]


def test_every_strategy_satisfies_the_frozen_protocol():
    for strategy, _ in cases():
        assert isinstance(strategy, Strategy)
        assert strategy.id and isinstance(strategy.asset_class, AssetClass)


def test_the_package_exports_the_registered_and_legacy_strategies():
    built = [type(s) for s, _ in cases()]
    assert set(built) == {IndexOptionTrendStrategy, EquityMeanReversionStrategy,
                          CryptoEmaTrendStrategy, CryptoBreakoutStrategy,
                          CryptoMandateStrategy}


def test_every_intent_names_its_instrument_and_says_something_specific():
    """`reason` is shown verbatim in the UI and is how the owner supervises an
    autonomous trader. A generic string is not supervision."""
    for strategy, ctx in cases():
        intents = strategy.propose(ctx)
        assert intents, f"{strategy.id} produced nothing to check"
        for i in intents:
            assert i.instrument.symbol in i.reason
            assert len(i.reason) > 40
            assert any(ch.isdigit() for ch in i.reason)


def test_every_intent_is_attributed_and_deduplicable():
    for strategy, ctx in cases():
        for i in strategy.propose(ctx):
            assert i.strategy_id == strategy.id
            assert i.idempotency_key
            assert i.qty > 0


def test_equity_never_proposes_a_market_order():
    """NFO rejects MARKET outright; a MARKET intent here is a rejected order
    at best and an unbounded fill price at worst."""
    for strategy, ctx in cases():
        if strategy.asset_class is not AssetClass.EQUITY:
            continue
        for i in strategy.propose(ctx):
            assert i.order_type is OrderType.LIMIT
            assert i.limit_price is not None and i.limit_price > 0


def test_propose_is_pure_and_repeatable():
    """Same context in, same intents out. Anything else means the strategy is
    reading something that is not in the context — a clock, a file, a socket."""
    for strategy, ctx in cases():
        first = list(strategy.propose(ctx))
        second = list(strategy.propose(ctx))
        assert first == second


def test_propose_leaves_the_context_untouched():
    for strategy, ctx in cases():
        before = (ctx.now, dict(ctx.quotes), tuple(ctx.positions), set(ctx.extra))
        strategy.propose(ctx)
        assert (ctx.now, dict(ctx.quotes), tuple(ctx.positions), set(ctx.extra)) == before


def test_an_empty_book_is_the_normal_case_and_is_cheap():
    """Called on every tick; the flat, no-signal path must not raise."""
    quiet = [
        (et.strat(), et.make_ctx(state=et.RegimeState.RANGE)),
        (mr.strat(), mr.make_ctx([100, 100, 100, 100, 100], quote="100")),
        (ct.strat(), ct.ctx_from(ct.NO_CROSS)[0]),
        (bo.strat(), bo.make_ctx(bo.RANGE_ROWS + [(101, 102, 100, 101)])[0]),
        (md.strat([md.BTC]), md.ctx_of(md.chop(80), extra={})),
    ]
    for strategy, ctx in quiet:
        assert list(strategy.propose(ctx)) == []


def test_the_strategies_are_distinct_ids():
    ids = [s.id for s, _ in cases()]
    assert len(set(ids)) == 5
