"""Option-chain construction from an injected Dhan scrip master.

No network, no Dhan credentials, no second token. The three attacks this
file exists to refuse:

1. a silently-short chain from dropped option rows,
2. a stale cache serving expired contracts as live,
3. a synthetic PAPER- identity reaching a LIVE class.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from core.profiles import IST
from core.types import AssetClass, Mode
from feeds.option_chain import (
    ChainIncomplete, LiveChainError, OptionChainIndex, StaleScripMaster,
)
from strategies.equity_trend import OptionContract, Right

LOADED = dt.datetime(2026, 8, 27, 10, 0, tzinfo=IST)
ON = dt.date(2026, 8, 27)

HEADER = (
    "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,"
    "SEM_TRADING_SYMBOL,SEM_LOT_UNITS,SEM_EXPIRY_DATE,SEM_TICK_SIZE,SEM_SERIES,"
    "SEM_STRIKE_PRICE,SEM_OPTION_TYPE\n"
)


def _csv(*rows: str) -> str:
    return HEADER + "".join(r + "\n" for r in rows)


NIFTY_CHAIN = _csv(
    "NSE,E,26000,INDEX,NIFTY,1,,0.05,EQ,,",
    "NSE,D,43001,OPTIDX,NIFTY-Sep2026-24000-CE,75,2026-09-24,0.05,,24000,CE",
    "NSE,D,43002,OPTIDX,NIFTY-Sep2026-24000-PE,75,2026-09-24,0.05,,24000,PE",
    "NSE,D,43003,OPTIDX,NIFTY-Sep2026-24100-CE,75,2026-09-24,0.05,,24100,CE",
    "NSE,D,43004,OPTIDX,NIFTY-Sep2026-24100-PE,75,2026-09-24,0.05,,24100,PE",
)


def test_a_complete_master_publishes_contracts_keyed_by_underlying() -> None:
    chain = OptionChainIndex.from_csv(NIFTY_CHAIN, loaded_at=LOADED)
    mapping = chain.as_mapping(ON, mode=Mode.PAPER)
    assert set(mapping) == {"NIFTY"}
    assert chain.index_underlyings() == ("NIFTY",)
    contracts = mapping["NIFTY"]
    assert all(isinstance(c, OptionContract) for c in contracts)
    assert {c.right for c in contracts} == {Right.CALL, Right.PUT}
    assert all(c.instrument.broker_id.isdigit() for c in contracts)
    assert all(c.instrument.asset_class is AssetClass.EQUITY for c in contracts)
    spots = chain.spot_instruments(mode=Mode.PAPER)
    assert len(spots) == 1 and spots[0].symbol == "NIFTY"
    assert spots[0].broker_id == "26000"
    assert spots[0].expiry is None
    watch = {i.symbol for i in chain.watchlist(ON, mode=Mode.PAPER)}
    assert "NIFTY" in watch
    assert any(s.endswith("-CE") or s.endswith("-PE") for s in watch)


def test_dropped_option_rows_refuse_a_silently_short_chain() -> None:
    """ATTACK: a row that looks like an option but is missing its security_id
    is dropped by the CSV parser. Publishing the remainder would leave a hole
    in the strike ladder the strategy cannot see."""
    short = _csv(
        "NSE,D,43001,OPTIDX,NIFTY-Sep2026-24000-CE,75,2026-09-24,0.05,,24000,CE",
        "NSE,D,,OPTIDX,NIFTY-Sep2026-24100-CE,75,2026-09-24,0.05,,24100,CE",
    )
    with pytest.raises(ChainIncomplete, match="silently-short"):
        OptionChainIndex.from_csv(short, loaded_at=LOADED)
    from adapters.dhan import DhanSecurityMap
    with pytest.raises(ChainIncomplete, match="silently-short"):
        OptionChainIndex.from_security_map(
            DhanSecurityMap.from_csv(short), loaded_at=LOADED)


def test_an_option_row_missing_strike_is_not_indexed() -> None:
    broken = _csv(
        "NSE,D,43001,OPTIDX,NOT-A-CONTRACT,75,2026-09-24,0.05,,,",
    )
    with pytest.raises(ChainIncomplete):
        OptionChainIndex.from_csv(broken, loaded_at=LOADED)


def test_stale_cache_does_not_serve_expired_contracts_as_live() -> None:
    """ATTACK: a cached master whose every contract has expired is served as
    a live chain. The strategy would then propose a spread on dead scrips."""
    expired = _csv(
        "NSE,D,43001,OPTIDX,NIFTY-Aug2026-24000-CE,75,2026-08-01,0.05,,24000,CE",
        "NSE,D,43002,OPTIDX,NIFTY-Aug2026-24000-PE,75,2026-08-01,0.05,,24000,PE",
    )
    chain = OptionChainIndex.from_csv(expired, loaded_at=LOADED)
    with pytest.raises(StaleScripMaster, match="expired"):
        chain.as_mapping(ON, mode=Mode.PAPER)


def test_expired_contracts_are_filtered_not_returned() -> None:
    mixed = _csv(
        "NSE,D,43001,OPTIDX,NIFTY-Aug2026-24000-CE,75,2026-08-01,0.05,,24000,CE",
        "NSE,D,43011,OPTIDX,NIFTY-Sep2026-24000-CE,75,2026-09-24,0.05,,24000,CE",
        "NSE,D,43012,OPTIDX,NIFTY-Sep2026-24000-PE,75,2026-09-24,0.05,,24000,PE",
    )
    chain = OptionChainIndex.from_csv(mixed, loaded_at=LOADED)
    mapping = chain.as_mapping(ON, mode=Mode.PAPER)
    expiries = {c.expiry for c in mapping["NIFTY"]}
    assert dt.date(2026, 8, 1) not in expiries
    assert expiries == {dt.date(2026, 9, 24)}


def test_a_paper_identity_cannot_reach_a_live_class() -> None:
    """ATTACK: a synthetic PAPER- broker_id is published into a LIVE chain,
    which is how a generated book gets routed at a real venue."""
    paper = _csv(
        "NSE,D,PAPER-1,OPTIDX,NIFTY-Sep2026-24000-CE,75,2026-09-24,"
        "0.05,,24000,CE",
    )
    chain = OptionChainIndex.from_csv(paper, loaded_at=LOADED)
    with pytest.raises(LiveChainError, match="LIVE"):
        chain.as_mapping(ON, mode=Mode.LIVE)
    mapping = chain.as_mapping(ON, mode=Mode.PAPER)
    assert mapping["NIFTY"][0].instrument.broker_id == "PAPER-1"


def test_loaded_at_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        OptionChainIndex.from_csv(NIFTY_CHAIN, loaded_at=dt.datetime(2026, 8, 27))


def test_this_module_never_fetches_or_authenticates_dhan() -> None:
    import inspect
    import feeds.option_chain as mod
    source = inspect.getsource(mod)
    assert "images.dhan.co" not in source
    assert "DHAN_ACCESS_TOKEN" not in source
    assert "dhanhq" not in source.lower()
    assert "requests." not in source
    assert "urllib" not in source
