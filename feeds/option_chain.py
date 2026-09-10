"""
Index-option chain from an injected Dhan scrip master.

Nothing here talks to a network, reads a Dhan token, or constructs a client.
The CSV is supplied as text (or via DhanSecurityMap, which itself never
fetches). The live engine holds the only permitted Dhan token; this
module exists so a paper bot can still *see* a chain, and so a later owner
arming of equity has somewhere to hang real contracts without minting a
second one.

Three attacks this file is written to refuse:

1. A silently-short chain, from option rows the CSV parser dropped.
2. A stale cache serving expired contracts as if they were live.
3. A synthetic PAPER- broker_id reaching a LIVE class.
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence

from adapters.dhan import (
    DhanSecurityMap,
    ScripMasterParse,
    ScripRecord,
    parse_scrip_master_report,
)
from core.types import AssetClass, Instrument, Mode
from strategies.equity_trend import OptionContract, Right

__all__ = [
    "ChainIncomplete",
    "LiveChainError",
    "OptionChainIndex",
    "StaleScripMaster",
]


class ChainIncomplete(RuntimeError):
    """Option rows were dropped while building the chain.

    Publishing the remainder would be a silently-short book: the strategy
    would pick ATM off a ladder with holes in it.
    """


class StaleScripMaster(RuntimeError):
    """The master cannot vouch that its contracts are still listed as of `on`."""


class LiveChainError(RuntimeError):
    """A paper/synthetic identity was about to be published for a live class."""


_OPTION_KINDS = frozenset({"OPTIDX", "OPTSTK"})
_SPOT_KINDS = frozenset({"INDEX", "EQUITY"})


class OptionChainIndex:
    """`ctx.extra["option_chain"]` keyed by underlying.

    Constructed from a parse of the scrip master. `loaded_at` is when THIS
    process indexed the CSV, not when Dhan published it — we cannot know the
    latter without a network. Combined with `on` (the trading date) it is
    enough to refuse expired contracts and to refuse a master whose every
    contract has already expired.
    """

    def __init__(
        self,
        records: Sequence[ScripRecord],
        *,
        loaded_at: dt.datetime,
        skipped_option_rows: Sequence[str] = (),
    ) -> None:
        if loaded_at.tzinfo is None or loaded_at.utcoffset() is None:
            raise ValueError(
                f"OptionChainIndex.loaded_at must be timezone-aware, got {loaded_at!r}"
            )
        self._loaded_at = loaded_at
        self._skipped = tuple(skipped_option_rows)
        self._by_underlying: dict[str, tuple[ScripRecord, ...]] = {}
        self._spot: dict[str, ScripRecord] = {}
        grouped: dict[str, list[ScripRecord]] = {}
        for rec in records:
            if rec.instrument_kind in _SPOT_KINDS:
                self._spot[rec.symbol] = rec
                continue
            if rec.instrument_kind not in _OPTION_KINDS:
                continue
            if rec.underlying is None or rec.strike is None or rec.option_type is None:
                raise ChainIncomplete(
                    f"{rec.symbol} is {rec.instrument_kind} but is missing "
                    f"underlying/strike/right; refusing to index a partial chain"
                )
            grouped.setdefault(rec.underlying, []).append(rec)
        self._by_underlying = {k: tuple(v) for k, v in grouped.items()}

    @property
    def loaded_at(self) -> dt.datetime:
        return self._loaded_at

    @property
    def skipped_option_rows(self) -> tuple[str, ...]:
        return self._skipped

    def underlyings(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_underlying))

    def index_underlyings(self) -> tuple[str, ...]:
        """Underlyings that have at least one OPTIDX row. These are the names
        IndexOptionTrendStrategy is allowed to be registered against."""
        names = [
            und for und, rows in self._by_underlying.items()
            if any(r.instrument_kind == "OPTIDX" for r in rows)
        ]
        return tuple(sorted(names))

    def instruments(self, on: dt.date, *, mode: Mode) -> tuple[Instrument, ...]:
        """Tradable option Instruments as of `on`, after the live/paper guard."""
        seen: dict[str, Instrument] = {}
        for contracts in self.as_mapping(on, mode=mode).values():
            for contract in contracts:
                seen[contract.instrument.symbol] = contract.instrument
        return tuple(seen.values())

    def spot_instruments(self, *, mode: Mode) -> tuple[Instrument, ...]:
        """Cash/index Instruments for OPTIDX underlyings that have a scrip row.

        IndexOptionTrendStrategy prices the ATM strike off `ctx.quote(underlying)`.
        Without this row in the engine watchlist the spot is simply absent and
        the strategy never enters — which looks like a quiet tape.
        """
        out: list[Instrument] = []
        for name in self.index_underlyings():
            rec = self._spot.get(name)
            if rec is None:
                continue
            out.append(self._instrument(rec, mode))
        return tuple(out)

    def watchlist(self, on: dt.date, *, mode: Mode) -> tuple[Instrument, ...]:
        """Everything the engine must snapshot quotes for: spots then legs."""
        return self.spot_instruments(mode=mode) + self.instruments(on, mode=mode)

    @classmethod
    def from_csv(cls, text: str, *, loaded_at: dt.datetime) -> "OptionChainIndex":
        return cls.from_parse(parse_scrip_master_report(text), loaded_at=loaded_at)

    @classmethod
    def from_parse(cls, parsed: ScripMasterParse, *,
                   loaded_at: dt.datetime) -> "OptionChainIndex":
        if parsed.skipped_option_rows:
            # Fail CLOSED. A chain built from the remainder is silently short
            # of every dropped strike, and the strategy would never know.
            raise ChainIncomplete(
                "refusing to publish a silently-short option chain; dropped "
                f"{len(parsed.skipped_option_rows)} option row(s): "
                + "; ".join(parsed.skipped_option_rows[:8])
            )
        return cls(parsed.records, loaded_at=loaded_at,
                   skipped_option_rows=parsed.skipped_option_rows)

    @classmethod
    def from_security_map(cls, mapping: DhanSecurityMap, *,
                          loaded_at: dt.datetime) -> "OptionChainIndex":
        skipped = mapping.skipped_option_rows()
        if skipped:
            raise ChainIncomplete(
                "refusing to publish a silently-short option chain; dropped "
                f"{len(skipped)} option row(s): "
                + "; ".join(skipped[:8])
            )
        return cls(mapping.records(), loaded_at=loaded_at,
                   skipped_option_rows=skipped)

    def as_mapping(self, on: dt.date, *, mode: Mode
                   ) -> dict[str, list[OptionContract]]:
        """The dict `IndexOptionTrendStrategy` reads from ctx.extra.

        Empty is not returned for an underlying whose every listed contract
        has expired — that is a stale master, not a quiet market. An
        underlying that was never in the CSV is simply absent.
        """
        if self._skipped:
            raise ChainIncomplete(
                "refusing to publish a silently-short option chain; dropped "
                f"{len(self._skipped)} option row(s)"
            )
        if not self._by_underlying:
            return {}

        out: dict[str, list[OptionContract]] = {}
        expired_only: list[str] = []
        for underlying, rows in self._by_underlying.items():
            live_rows = [r for r in rows if r.expiry is not None and r.expiry >= on]
            if not live_rows:
                expired_only.append(underlying)
                continue
            contracts: list[OptionContract] = []
            for rec in live_rows:
                instrument = self._instrument(rec, mode)
                right = Right(rec.option_type)  # type: ignore[arg-type]
                contracts.append(OptionContract(
                    instrument=instrument,
                    underlying=underlying,
                    strike=rec.strike,  # type: ignore[arg-type]
                    right=right,
                ))
            out[underlying] = contracts

        if expired_only and not out:
            raise StaleScripMaster(
                f"scrip master loaded at {self._loaded_at.isoformat()} has no "
                f"unexpired option contracts as of {on.isoformat()} "
                f"(underlyings: {', '.join(sorted(expired_only))}); refusing "
                "to serve expired contracts as a live chain"
            )
        if expired_only:
            # Some underlyings died, others live. The dead ones are ABSENT,
            # not present as []. Absent means "no chain for this name" and
            # the strategy will not register against them.
            pass
        return out

    @staticmethod
    def _instrument(rec: ScripRecord, mode: Mode) -> Instrument:
        broker_id = str(rec.security_id).strip()
        symbol = rec.symbol.strip()
        paperish = (
            not broker_id.isdigit()
            or broker_id.upper().startswith("PAPER")
            or symbol.upper().startswith("PAPER-")
        )
        if mode is Mode.LIVE and paperish:
            raise LiveChainError(
                f"synthetic/paper identity {symbol!r} broker_id={broker_id!r} "
                f"cannot be published into a LIVE option chain. A PAPER- "
                "contract reaching a live class is how a generated book gets "
                "routed at a real venue."
            )
        if not broker_id:
            raise ChainIncomplete(f"{symbol} has an empty security_id")
        return Instrument(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            broker_id=broker_id,
            lot_size=rec.lot_size,
            tick_size=rec.tick_size,
            expiry=rec.expiry,
        )
