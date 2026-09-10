"""SQLite persistence for mandate decision cards and sleeve bookkeeping.

I/O lives here, not inside CryptoMandateStrategy. The runner wraps propose()
and writes after each tick. Strategies stay pure.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("trader.mandate")

from core.types import Mode, Position

from .constants import SLEEVE_START, STATE_KEY
from .types import DecisionCard, MandateState, MandateTick, Verdict

UTC = dt.timezone.utc

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mandate_cards (
    id INTEGER PRIMARY KEY,
    at_utc TEXT NOT NULL,
    verdict TEXT NOT NULL,
    tag TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mandate_meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""


class MandateLedger:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._cx = sqlite3.connect(path)
        self._cx.row_factory = sqlite3.Row
        self._cx.executescript(_SCHEMA)
        self._cx.commit()

    def close(self) -> None:
        self._cx.close()

    def record(self, tick: MandateTick) -> None:
        now = None
        for card in tick.cards:
            payload = _card_json(card)
            # decision_ts is not on the card; use now-less ISO from thesis. The
            # runner stamps wall time at persist so 24h windows work.
            at = dt.datetime.now(tz=UTC).isoformat()
            now = at
            self._cx.execute(
                "INSERT INTO mandate_cards (at_utc, verdict, tag, payload) "
                "VALUES (?, ?, ?, ?)",
                (at, card.verdict.value, card.tag, payload),
            )
            LOG.info("mandate card %s %s %s score=%s — %s",
                     card.verdict.value, card.instrument, card.direction,
                     card.signal_score, card.trade_thesis)
            if card.tag == "exit" and card.closed_pnl is not None:
                if card.closed_pnl < 0:
                    self._bump("consecutive_losses", 1)
                else:
                    self._set("consecutive_losses", "0")
            if card.verdict is Verdict.EXECUTE and card.tag == "entry":
                self._bump("trades_24h_count", 1)
        if now is not None:
            self._set("last_tick", now)
        self._cx.commit()

    def snapshot(
        self,
        now: dt.datetime,
        *,
        mode: Mode,
        positions: Iterable[Position],
        quotes: dict[str, Decimal],
        realized_lifetime: Decimal,
        realized_today: Decimal,
        paper_derivatives: bool,
    ) -> MandateState:
        unreal = Decimal("0")
        symbols: list[str] = []
        for pos in positions:
            if pos.strategy_id != "crypto_mandate" or pos.qty == 0:
                continue
            mark = quotes.get(pos.instrument.symbol, pos.avg_price)
            if not isinstance(mark, Decimal) or mark <= 0:
                mark = pos.avg_price
            unreal += (mark - pos.avg_price) * pos.qty
            symbols.append(pos.instrument.symbol)
        nav = SLEEVE_START + unreal + realized_lifetime
        if nav <= 0:
            nav = SLEEVE_START
        stored_high = self._get_dec("equity_high", SLEEVE_START)
        high = max(stored_high, nav)
        if high != stored_high:
            self._set("equity_high", str(high))
            self._cx.commit()
        losses = int(self._get("consecutive_losses", "0"))
        trades = self._trades_since(now - dt.timedelta(hours=24))
        daily_loss = max(Decimal("0"), -realized_today)
        cluster = "crypto_long" if symbols else ""
        open_risk = self._open_risk()
        return MandateState(
            sleeve_nav=nav,
            equity_high=high,
            consecutive_losses=losses,
            daily_loss=daily_loss,
            trades_24h=trades,
            open_risk=open_risk,
            open_cluster=cluster,
            paper_derivatives=paper_derivatives,
            open_symbols=tuple(symbols),
        )

    def _open_risk(self) -> Decimal:
        rows = self._cx.execute(
            "SELECT payload FROM mandate_cards WHERE tag='entry' "
            "AND verdict='EXECUTE' ORDER BY id DESC LIMIT 8"
        ).fetchall()
        # Conservative: last execute card's maximum_loss if we still have
        # a position; runner snapshot also sees open_symbols.
        total = Decimal("0")
        for row in rows:
            payload = json.loads(row["payload"])
            total += Decimal(str(payload.get("maximum_loss", "0")))
            break
        return total

    def _trades_since(self, since: dt.datetime) -> int:
        stamp = since.astimezone(UTC).isoformat()
        row = self._cx.execute(
            "SELECT COUNT(*) AS n FROM mandate_cards "
            "WHERE tag='entry' AND verdict='EXECUTE' AND at_utc>=?",
            (stamp,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def _bump(self, key: str, n: int) -> None:
        cur = int(self._get(key, "0"))
        self._set(key, str(cur + n))

    def _get(self, key: str, default: str) -> str:
        row = self._cx.execute(
            "SELECT v FROM mandate_meta WHERE k=?", (key,)
        ).fetchone()
        return str(row["v"]) if row else default

    def _get_dec(self, key: str, default: Decimal) -> Decimal:
        return Decimal(self._get(key, str(default)))

    def _set(self, key: str, value: str) -> None:
        self._cx.execute(
            "INSERT INTO mandate_meta (k, v) VALUES (?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, value),
        )


def _card_json(card: DecisionCard) -> str:
    def enc(value: object) -> object:
        if isinstance(value, Decimal):
            return str(value)
        if hasattr(value, "value"):
            return getattr(value, "value")
        return value

    return json.dumps({
        "regime": card.regime.value,
        "asset": card.asset,
        "instrument": card.instrument,
        "verdict": card.verdict.value,
        "tag": card.tag,
        "score": card.signal_score,
        "maximum_loss": str(card.maximum_loss),
        "thesis": card.trade_thesis,
        "closed_pnl": enc(card.closed_pnl),
        "nav": str(card.current_nav),
    })


class PersistingMandate:
    """Strategy protocol wrapper. Forwards everything; persists last tick."""

    def __init__(self, inner, ledger: MandateLedger) -> None:
        self._inner = inner
        self._ledger = ledger

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def propose(self, ctx):
        intents = self._inner.propose(ctx)
        tick = getattr(self._inner, "last_tick", None)
        if tick is not None:
            self._ledger.record(tick)
        return intents


# Keep the extra key name in one place for the runner.
__all__ = ["MandateLedger", "PersistingMandate", "STATE_KEY"]
