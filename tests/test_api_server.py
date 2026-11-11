"""
Tests for api/server.py.

The interesting question for this module is never "does it render a healthy
book" but "what does it say when it cannot establish something, and can an
operator tell the difference". Every fake below therefore has a working mode and
a FAILING mode, and most of these tests are about the failing one.

Every safety rule in server.py has a test here that fails if the rule is
deleted:

  * app refuses to start without a token / with a short token / with CORS "*"
  * auth is enforced on paths the router does not even define
  * a failed ledger read is 503, never 200 with an empty book
  * unknown equity is null with a reason, never 0
  * an empty strategy registry is 503, never 200 with []
  * the API never calls RiskCircuit.advance()
  * the API never calls BrokerAdapter.place()
  * control state is durable across processes
  * resume is refused while a flatten is unacknowledged
  * control bodies cannot smuggle a mode or an arming flag
  * Decimals cross the wire as strings
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from decimal import Decimal
from typing import Any, Optional, Sequence

import pytest
from fastapi.testclient import TestClient

from api.server import (
    ApiSettings, ControlState, EngineHandle, Event, EventBus,
    GATE_REJECTION_PREFIX, ProposalRecord, SqliteControlStore, StrategyRegistry,
    create_app, day_start_instant, gate_rejection_ref,
)
from core.profiles import IST, CryptoProfile, EquityProfile, profile_for
from core.risk import CircuitAction, CircuitDecision, RiskCircuit
from core.store import SQLitePositionStore
from core.types import (
    Approval, AssetClass, CheckResult, Fill, Instrument, Intent, Mode,
    OrderRef, OrderStatus, OrderType, Position, Side,
)

UTC = dt.timezone.utc

# Thursday 2026-08-27, 10:00 IST — NSE open, and 04:30 UTC on the same date.
NOW = dt.datetime(2026, 8, 27, 10, 0, tzinfo=IST)
TOKEN = "a-token-long-enough-to-be-real"

NIFTY = Instrument(symbol="NIFTY26AUG24000CE", asset_class=AssetClass.EQUITY,
                   broker_id="43210", lot_size=75, tick_size=Decimal("0.05"),
                   expiry=dt.date(2026, 8, 27))
BTC = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO,
                 broker_id="BTC/USDT", qty_precision=6,
                 tick_size=Decimal("0.1"), min_notional=Decimal("5"))


# ==========================================================================
# fakes
# ==========================================================================

class FakeLedger:
    """LedgerReader with a failure switch on every method."""

    def __init__(self, positions: Sequence[Position] = (),
                 orders: Sequence[OrderRef] = (),
                 pnl: Decimal = Decimal("0")) -> None:
        self._positions = list(positions)
        self._orders = list(orders)
        self._pnl = pnl
        self.fail_positions: Optional[Exception] = None
        self.fail_orders: Optional[Exception] = None
        self.fail_pnl: Optional[Exception] = None
        self.pnl_calls: list[tuple[Mode, AssetClass, str]] = []

    def positions(self, mode: Mode,
                  asset_class: Optional[AssetClass] = None) -> Sequence[Position]:
        if self.fail_positions is not None:
            raise self.fail_positions
        return [p for p in self._positions
                if p.mode is mode
                and (asset_class is None or p.instrument.asset_class is asset_class)]

    def orders(self, mode: Mode, asset_class: Optional[AssetClass] = None,
               since_iso: Optional[str] = None,
               limit: Optional[int] = None) -> Sequence[OrderRef]:
        if self.fail_orders is not None:
            raise self.fail_orders
        rows = [o for o in self._orders
                if o.mode is mode
                and (asset_class is None or o.intent.instrument.asset_class is asset_class)]
        return rows if limit is None else rows[:limit]

    def realized_pnl(self, mode: Mode, asset_class: AssetClass,
                     since_iso: str) -> Decimal:
        self.pnl_calls.append((mode, asset_class, since_iso))
        if self.fail_pnl is not None:
            raise self.fail_pnl
        # Mirrors the real store: a naive instant is a bug, not a shifted window.
        parsed = dt.datetime.fromisoformat(since_iso)
        if parsed.tzinfo is None:
            raise ValueError(f"since_iso must be tz-aware, got {since_iso!r}")
        return self._pnl


class FakeAdapter:
    """BrokerAdapter that screams if the API ever tries to trade with it."""

    def __init__(self, asset_class: AssetClass, mode: Mode = Mode.PAPER,
                 equity_value: Optional[Decimal] = Decimal("500000"),
                 equity_error: Optional[Exception] = None) -> None:
        self.asset_class = asset_class
        self.mode = mode
        self._equity = equity_value
        self._equity_error = equity_error
        self.equity_calls = 0

    def equity(self) -> Optional[Decimal]:
        self.equity_calls += 1
        if self._equity_error is not None:
            raise self._equity_error
        return self._equity

    def place(self, intent: Intent, timeout_s: float = 30.0) -> OrderRef:
        raise AssertionError(
            "the API process placed an order. It holds no gate and must never "
            "reach a venue; flatten is a request the engine executes.")

    def cancel(self, order_id: str) -> bool:
        raise AssertionError("the API process cancelled an order")

    def positions(self) -> Sequence[Position]:
        raise AssertionError(
            "the API read positions from the broker; PositionStore is the "
            "authority and a venue read on a dashboard poll is a rate limit")

    def quote(self, instrument: Instrument) -> Optional[Decimal]:
        raise AssertionError("the API hit the venue for a quote on a poll")


class FakeCircuit:
    """RiskCircuitProto whose advance() is a tripwire."""

    def __init__(self, mode: Mode = Mode.PAPER,
                 decision: Optional[CircuitDecision] = None,
                 error: Optional[Exception] = None) -> None:
        self.mode = mode
        self._decision = decision
        self._error = error
        self.read_calls: list[tuple[AssetClass, Optional[Decimal]]] = []

    def read(self, asset_class: AssetClass,
             equity: Optional[Decimal]) -> CircuitDecision:
        self.read_calls.append((asset_class, equity))
        if self._error is not None:
            raise self._error
        if self._decision is not None:
            return self._decision
        if equity is None:
            return CircuitDecision(
                action=CircuitAction.PAUSE, drawdown_pct=None, daily_loss_pct=None,
                reason="equity unknown", equity_known=False, state_known=True)
        return CircuitDecision(
            action=CircuitAction.OK, drawdown_pct=Decimal("0.01"),
            daily_loss_pct=Decimal("0.002"), reason="clear",
            equity_known=True, state_known=True, size_multiplier=Decimal("1"))

    def advance(self, asset_class: AssetClass, equity: Decimal) -> CircuitDecision:
        raise AssertionError(
            "the API called RiskCircuit.advance(). That is the circuit's only "
            "writer and belongs to the order path; a dashboard refresh must not "
            "move the high-water mark.")


class MemoryControlStore:
    """Durable-shaped control store for unit tests.

    NOT a production implementation and never exported — an in-memory control
    store would tell an operator their pause took effect in a process the engine
    cannot see. Used here only where durability is not what is under test; the
    durability test uses the real SqliteControlStore on a real file.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        self.fail: Optional[Exception] = None
        self.fail_write: Optional[Exception] = None

    def read(self, scope: str) -> Optional[dict[str, str]]:
        if self.fail is not None:
            raise self.fail
        row = self.rows.get(scope)
        return dict(row) if row is not None else None

    def write(self, scope: str, row: Any) -> None:
        if self.fail_write is not None:
            raise self.fail_write
        self.rows[scope] = {str(k): str(v) for k, v in row.items()}

    def mutate(self, scope: str, fn: Any) -> dict[str, str]:
        if self.fail is not None:
            raise self.fail
        if self.fail_write is not None:
            raise self.fail_write
        updated = fn(self.read(scope))
        self.rows[scope] = {str(k): str(v) for k, v in updated.items()}
        return dict(self.rows[scope])


def build_handle(**kw: Any) -> EngineHandle:
    classes = kw.pop("asset_classes", (AssetClass.EQUITY, AssetClass.CRYPTO))
    mode = kw.pop("mode", Mode.PAPER)
    adapters = kw.pop("adapters", None)
    if adapters is None:
        adapters = {ac: FakeAdapter(ac, mode) for ac in classes}
    return EngineHandle(
        mode=mode,
        asset_classes=tuple(classes),
        ledger=kw.pop("ledger", None) or FakeLedger(),
        circuit=kw.pop("circuit", None) or FakeCircuit(mode),
        control=kw.pop("control", None) or MemoryControlStore(),
        strategies=kw.pop("strategies", None) or StrategyRegistry(),
        bus=kw.pop("bus", None) or EventBus(),
        adapters=adapters,
        armed=kw.pop("armed", {ac: False for ac in classes}),
        clock=kw.pop("clock", lambda: NOW),
        **kw,
    )


def settings(**kw: Any) -> ApiSettings:
    kw.setdefault("token", TOKEN)
    kw.setdefault("sse_poll_interval_s", 0.01)
    kw.setdefault("sse_heartbeat_s", 0.05)
    # TestClient's transport runs the app to completion and buffers the body,
    # so an unbounded stream would hang the test run forever. The bound is a
    # real production behaviour (see ApiSettings.sse_max_stream_s), not a test
    # hook — it is only made short here.
    kw.setdefault("sse_max_stream_s", 0.3)
    return ApiSettings(**kw)


def client(handle: Optional[EngineHandle] = None,
           api_settings: Optional[ApiSettings] = None) -> TestClient:
    app = create_app(handle or build_handle(), api_settings or settings())
    return TestClient(app)


AUTH = {"Authorization": f"Bearer {TOKEN}"}


# ==========================================================================
# startup refusals
# ==========================================================================

def test_app_refuses_to_start_without_a_token() -> None:
    with pytest.raises(RuntimeError) as exc:
        create_app(build_handle(), None, env={})
    assert "ENGINE_API_TOKEN" in str(exc.value)


def test_app_refuses_to_start_with_a_blank_token() -> None:
    with pytest.raises(RuntimeError):
        create_app(build_handle(), None, env={"ENGINE_API_TOKEN": "   "})


def test_app_refuses_a_token_short_enough_to_guess() -> None:
    with pytest.raises(ValueError) as exc:
        ApiSettings(token="short")
    assert "at least" in str(exc.value)


def test_app_refuses_wildcard_cors() -> None:
    with pytest.raises(ValueError) as exc:
        ApiSettings(token=TOKEN, cors_origins=("*",))
    assert "'*'" in str(exc.value)


def test_env_parses_a_cors_allowlist() -> None:
    resolved = ApiSettings.from_env({
        "ENGINE_API_TOKEN": TOKEN,
        "ENGINE_API_CORS_ORIGINS": "https://ui.example, https://ops.example ",
    })
    assert resolved.cors_origins == ("https://ui.example", "https://ops.example")


def test_env_with_wildcard_in_the_list_still_refuses() -> None:
    with pytest.raises(ValueError):
        ApiSettings.from_env({"ENGINE_API_TOKEN": TOKEN,
                              "ENGINE_API_CORS_ORIGINS": "https://ui.example,*"})


# ==========================================================================
# auth
# ==========================================================================

def test_health_and_live_are_the_unauthenticated_probes() -> None:
    c = client()
    assert c.get("/health").status_code == 200
    assert c.get("/health").json() == {"status": "ok"}
    live = c.get("/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"


def test_every_other_route_401s_without_a_token() -> None:
    c = client()
    for path in ("/ready", "/api/state", "/api/positions", "/api/orders",
                 "/api/strategies", "/api/risk", "/api/events"):
        resp = c.get(path)
        assert resp.status_code == 401, path
        assert resp.headers["www-authenticate"] == "Bearer"


def test_a_wrong_token_is_401() -> None:
    resp = client().get("/api/state", headers={"Authorization": "Bearer nope-nope-nope"})
    assert resp.status_code == 401


def test_a_non_bearer_scheme_is_401() -> None:
    resp = client().get("/api/state", headers={"Authorization": f"Basic {TOKEN}"})
    assert resp.status_code == 401


def test_a_path_the_router_never_defined_is_still_401() -> None:
    """Auth is middleware, not a decorator. A route added tomorrow without an
    auth dependency is still protected — and an unauthenticated caller cannot
    even enumerate which paths exist."""
    assert client().get("/api/some/future/endpoint").status_code == 401
    assert client().post("/api/control/arm").status_code == 401


def test_control_endpoints_401_without_a_token() -> None:
    c = client()
    for path in ("/api/control/pause", "/api/control/resume", "/api/control/flatten"):
        assert c.post(path, json={"reason": "x"}).status_code == 401


def test_a_valid_token_gets_through() -> None:
    assert client().get("/api/state", headers=AUTH).status_code == 200


# ==========================================================================
# CORS
# ==========================================================================

def test_preflight_is_answered_for_an_allowed_origin_without_a_token() -> None:
    """CORS must sit OUTSIDE auth: a preflight carries no Authorization header,
    so an auth-first stack would 401 it and the browser would never send the
    real request."""
    c = client(api_settings=settings(cors_origins=("https://ui.example",)))
    resp = c.options("/api/state", headers={
        "Origin": "https://ui.example",
        "Access-Control-Request-Method": "GET",
    })
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://ui.example"


def test_an_unlisted_origin_gets_no_allow_header() -> None:
    c = client(api_settings=settings(cors_origins=("https://ui.example",)))
    resp = c.get("/api/state", headers={**AUTH, "Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers


def test_no_cors_configured_means_no_cross_origin_access() -> None:
    resp = client().get("/api/state",
                        headers={**AUTH, "Origin": "https://ui.example"})
    assert "access-control-allow-origin" not in resp.headers


# ==========================================================================
# snapshot shape
# ==========================================================================

def test_state_snapshot_shape() -> None:
    positions = (
        Position(instrument=NIFTY, qty=Decimal("75"), avg_price=Decimal("120.50"),
                 mode=Mode.PAPER, strategy_id="equity_trend"),
    )
    control = MemoryControlStore()
    control.rows["__engine__"] = {"heartbeat_at": NOW.isoformat()}
    handle = build_handle(ledger=FakeLedger(positions=positions,
                                            pnl=Decimal("-1234.50")),
                          control=control)
    body = client(handle).get("/api/state", headers=AUTH).json()

    assert body["mode"] == "paper"
    assert body["engine"]["status"] == "alive"
    assert body["open_positions"] == 1
    assert set(body["classes"]) == {"equity", "crypto"}
    equity_block = body["classes"]["equity"]
    assert equity_block["base_currency"] == "INR"
    assert equity_block["market_open"] is True
    assert equity_block["armed_live"] is False
    assert equity_block["open_positions"] == 1
    assert equity_block["equity"] == {"value": "500000", "known": True,
                                      "detail": None}
    assert equity_block["realized_pnl_today"]["value"] == "-1234.50"
    assert equity_block["risk"]["action"] == "ok"
    assert equity_block["risk"]["entries_allowed"] is True
    assert equity_block["control"]["paused"] is False
    assert body["degraded"] is False
    assert body["problems"] == []


def test_state_reports_no_cross_currency_total() -> None:
    """INR and USDT do not add up. A summed 'total equity' would be the number
    a drawdown fraction gets divided by."""
    body = client().get("/api/state", headers=AUTH).json()
    assert "total_equity" not in body
    assert "equity" not in body
    assert body["classes"]["equity"]["base_currency"] == "INR"
    assert body["classes"]["crypto"]["base_currency"] == "USDT"


def test_every_money_value_is_a_string_never_a_json_number() -> None:
    """A JSON number is an IEEE double. A price that round-trips through one is
    no longer the price that filled."""
    positions = (
        Position(instrument=BTC, qty=Decimal("0.001234"),
                 avg_price=Decimal("64123.456789"), mode=Mode.PAPER,
                 strategy_id="crypto_trend"),
    )
    handle = build_handle(ledger=FakeLedger(positions=positions,
                                            pnl=Decimal("0.10")))
    raw = client(handle).get("/api/positions", headers=AUTH).text
    assert '"qty":"0.001234"' in raw.replace(" ", "")
    assert '"avg_price":"64123.456789"' in raw.replace(" ", "")

    state = client(handle).get("/api/state", headers=AUTH).text.replace(" ", "")
    assert '"value":"500000"' in state
    assert '"value":500000' not in state


# ==========================================================================
# unknown is not zero
# ==========================================================================

def test_unknown_equity_is_null_with_a_reason_never_zero() -> None:
    handle = build_handle(adapters={
        AssetClass.EQUITY: FakeAdapter(AssetClass.EQUITY, equity_value=None),
        AssetClass.CRYPTO: FakeAdapter(AssetClass.CRYPTO),
    })
    body = client(handle).get("/api/state", headers=AUTH).json()
    block = body["classes"]["equity"]["equity"]
    assert block["value"] is None
    assert block["known"] is False
    assert "could not establish" in block["detail"]
    assert body["degraded"] is True
    assert any(p["scope"] == "equity" for p in body["problems"])


def test_an_adapter_that_raises_on_equity_is_null_not_zero() -> None:
    handle = build_handle(adapters={
        AssetClass.EQUITY: FakeAdapter(AssetClass.EQUITY,
                                       equity_error=TimeoutError("broker down")),
        AssetClass.CRYPTO: FakeAdapter(AssetClass.CRYPTO),
    })
    body = client(handle).get("/api/state", headers=AUTH).json()
    block = body["classes"]["equity"]["equity"]
    assert block["value"] is None
    assert "broker down" in block["detail"]


def test_a_float_equity_is_refused_rather_than_shipped() -> None:
    """A float would become a JSON number and drift the denominator every
    drawdown fraction is computed against."""
    handle = build_handle(adapters={
        AssetClass.EQUITY: FakeAdapter(AssetClass.EQUITY, equity_value=500000.0),
        AssetClass.CRYPTO: FakeAdapter(AssetClass.CRYPTO),
    })
    block = client(handle).get("/api/state", headers=AUTH).json()["classes"]["equity"]["equity"]
    assert block["value"] is None
    assert "Decimal is required" in block["detail"]


def test_unknown_equity_blocks_entries_in_the_risk_block() -> None:
    handle = build_handle(adapters={
        AssetClass.EQUITY: FakeAdapter(AssetClass.EQUITY, equity_value=None),
        AssetClass.CRYPTO: FakeAdapter(AssetClass.CRYPTO),
    })
    block = client(handle).get("/api/state", headers=AUTH).json()["classes"]["equity"]["risk"]
    assert block["entries_allowed"] is False
    assert block["equity_known"] is False
    assert block["drawdown_fraction"] is None


def test_a_missing_adapter_is_reported_not_treated_as_zero_equity() -> None:
    handle = build_handle(adapters={AssetClass.CRYPTO: FakeAdapter(AssetClass.CRYPTO)})
    body = client(handle).get("/api/state", headers=AUTH).json()
    block = body["classes"]["equity"]["equity"]
    assert block["value"] is None
    assert "no adapter resolved" in block["detail"]


def test_a_failed_pnl_read_is_null_not_zero() -> None:
    ledger = FakeLedger()
    ledger.fail_pnl = sqlite_error = RuntimeError("db locked")
    body = client(build_handle(ledger=ledger)).get("/api/state", headers=AUTH).json()
    block = body["classes"]["equity"]["realized_pnl_today"]
    assert block["value"] is None
    assert "db locked" in block["detail"]
    assert body["degraded"] is True
    assert sqlite_error is ledger.fail_pnl


def test_a_genuine_zero_pnl_is_reported_as_zero() -> None:
    """The counterpart to the test above: a computed 0 must NOT look like a
    failure, or the operator learns to ignore both."""
    body = client(build_handle(ledger=FakeLedger(pnl=Decimal("0")))).get(
        "/api/state", headers=AUTH).json()
    block = body["classes"]["equity"]["realized_pnl_today"]
    assert block["value"] == "0"
    assert block["detail"] is None


# ==========================================================================
# a failed read is 503, never a plausible empty answer
# ==========================================================================

def test_a_failed_ledger_read_is_503_not_an_empty_book() -> None:
    ledger = FakeLedger()
    ledger.fail_positions = RuntimeError("database is locked")
    resp = client(build_handle(ledger=ledger)).get("/api/state", headers=AUTH)
    assert resp.status_code == 503
    assert "database is locked" in resp.json()["detail"]


def test_positions_503s_on_a_failed_read() -> None:
    ledger = FakeLedger()
    ledger.fail_positions = RuntimeError("disk I/O error")
    resp = client(build_handle(ledger=ledger)).get("/api/positions", headers=AUTH)
    assert resp.status_code == 503
    assert "positions" not in resp.json()


def test_a_genuinely_flat_book_is_200_with_an_empty_list() -> None:
    resp = client().get("/api/positions", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"mode": "paper", "asset_class": None,
                           "count": 0, "positions": []}


def test_orders_503s_on_a_failed_read() -> None:
    ledger = FakeLedger()
    ledger.fail_orders = RuntimeError("corrupt row")
    resp = client(build_handle(ledger=ledger)).get("/api/orders", headers=AUTH)
    assert resp.status_code == 503


def test_a_failed_control_read_is_503_not_an_unpaused_engine() -> None:
    control = MemoryControlStore()
    control.fail = RuntimeError("control table missing")
    resp = client(build_handle(control=control)).get("/api/state", headers=AUTH)
    assert resp.status_code == 503


def test_a_corrupt_control_row_is_503_not_an_unpaused_engine() -> None:
    control = MemoryControlStore()
    control.rows["equity"] = {"paused": "maybe"}
    resp = client(build_handle(control=control)).get("/api/state", headers=AUTH)
    assert resp.status_code == 503
    assert "corrupt" in resp.json()["detail"]


# ==========================================================================
# mode isolation
# ==========================================================================

def test_positions_are_mode_filtered_and_the_other_book_is_reachable() -> None:
    paper = Position(instrument=NIFTY, qty=Decimal("75"), avg_price=Decimal("100"),
                     mode=Mode.PAPER, strategy_id="s")
    live = Position(instrument=NIFTY, qty=Decimal("150"), avg_price=Decimal("101"),
                    mode=Mode.LIVE, strategy_id="s")
    c = client(build_handle(ledger=FakeLedger(positions=(paper, live))))

    default = c.get("/api/positions", headers=AUTH).json()
    assert [p["qty"] for p in default["positions"]] == ["75"]

    other = c.get("/api/positions?mode=live", headers=AUTH).json()
    assert other["mode"] == "live"
    assert [p["qty"] for p in other["positions"]] == ["150"]


def test_asset_class_filter_is_honoured() -> None:
    equity = Position(instrument=NIFTY, qty=Decimal("75"), avg_price=Decimal("100"),
                      mode=Mode.PAPER, strategy_id="s")
    crypto = Position(instrument=BTC, qty=Decimal("0.5"), avg_price=Decimal("60000"),
                      mode=Mode.PAPER, strategy_id="c")
    c = client(build_handle(ledger=FakeLedger(positions=(equity, crypto))))
    body = c.get("/api/positions?asset_class=crypto", headers=AUTH).json()
    assert [p["symbol"] for p in body["positions"]] == ["BTC/USDT"]


# ==========================================================================
# orders, including gate refusals
# ==========================================================================

def _rejected_approval() -> Approval:
    intent = Intent(instrument=NIFTY, side=Side.BUY, qty=Decimal("75"),
                    order_type=OrderType.LIMIT, strategy_id="equity_trend",
                    reason="regime turned bullish", limit_price=Decimal("120.50"),
                    idempotency_key="trend-2026-08-27-1")
    return Approval(False, Mode.PAPER, intent,
                    (CheckResult("allocation_cap", False,
                                 "exposure 210000 exceeds cap 150000"),))


def test_gate_rejection_ref_carries_the_gate_reason() -> None:
    ref = gate_rejection_ref(_rejected_approval(), NOW)
    assert ref.status is OrderStatus.REJECTED
    assert ref.order_id.startswith(GATE_REJECTION_PREFIX)
    assert "allocation_cap" in (ref.error or "")
    assert ref.filled_qty == Decimal("0")
    assert ref.fills == ()


def test_gate_rejection_ref_refuses_an_approved_approval() -> None:
    approved = Approval(True, Mode.PAPER, _rejected_approval().intent, ())
    with pytest.raises(ValueError):
        gate_rejection_ref(approved, NOW)


def test_gate_rejection_ref_requires_a_tz_aware_stamp() -> None:
    with pytest.raises(ValueError):
        gate_rejection_ref(_rejected_approval(), dt.datetime(2026, 8, 27, 10, 0))


def test_two_refusals_of_one_decision_do_not_collapse_into_one_row() -> None:
    """The ledger dedups on a row hash. Without the timestamp in the id, a
    strategy jamming against a cap all afternoon would show a single attempt."""
    approval = _rejected_approval()
    first = gate_rejection_ref(approval, NOW)
    second = gate_rejection_ref(approval, NOW + dt.timedelta(seconds=5))
    assert first.order_id != second.order_id


def test_orders_endpoint_separates_gate_refusals_from_broker_rejections() -> None:
    gate_row = gate_rejection_ref(_rejected_approval(), NOW)
    broker_row = OrderRef(
        order_id="DHAN-771", status=OrderStatus.REJECTED,
        intent=_rejected_approval().intent, mode=Mode.PAPER,
        error="RMS: insufficient margin")
    c = client(build_handle(ledger=FakeLedger(orders=(gate_row, broker_row))))
    rows = c.get("/api/orders", headers=AUTH).json()["orders"]
    by_origin = {r["origin"]: r for r in rows}
    assert by_origin["gate"]["error"].startswith("allocation_cap")
    assert by_origin["broker"]["error"] == "RMS: insufficient margin"
    assert by_origin["gate"]["status"] == "rejected"


def test_orders_rejects_a_malformed_since_window() -> None:
    """A bad client parameter must not masquerade as a broken ledger, and a
    naive instant must not silently shift the window by the local offset."""
    c = client()
    assert c.get("/api/orders?since=not-a-date", headers=AUTH).status_code == 422
    assert c.get("/api/orders?since=2026-08-27T00:00:00",
                 headers=AUTH).status_code == 422
    assert c.get("/api/orders?since=2026-08-27T00:00:00%2B05:30",
                 headers=AUTH).status_code == 200


def test_orders_limit_is_bounded() -> None:
    c = client()
    assert c.get("/api/orders?limit=0", headers=AUTH).status_code == 422
    assert c.get("/api/orders?limit=100000", headers=AUTH).status_code == 422


# ==========================================================================
# strategies
# ==========================================================================

def test_an_empty_registry_is_503_never_an_empty_bot_list() -> None:
    resp = client().get("/api/strategies", headers=AUTH)
    assert resp.status_code == 503
    body = resp.json()
    assert body["registered"] is False
    assert body["strategies"] is None
    assert "no strategies" in body["detail"]


def test_registered_strategies_are_listed() -> None:
    registry = StrategyRegistry()
    registry.register("equity_trend", AssetClass.EQUITY)
    registry.register("crypto_trend", AssetClass.CRYPTO)
    registry.record_tick("equity_trend", NOW,
                         proposals=[ProposalRecord("NIFTY26AUG24000CE", "buy",
                                                   "75", "limit", "regime bullish")],
                         reason="regime bullish")
    registry.record_error("crypto_trend", NOW, "quote feed timeout", disable=True)

    body = client(build_handle(strategies=registry)).get(
        "/api/strategies", headers=AUTH).json()
    assert body["registered"] is True
    assert body["count"] == 2
    assert body["enabled"] == 1
    rows = {r["strategy_id"]: r for r in body["strategies"]}
    assert rows["equity_trend"]["last_proposals"][0]["reason"] == "regime bullish"
    assert rows["crypto_trend"]["enabled"] is False
    assert rows["crypto_trend"]["disabled_reason"] == "quote feed timeout"
    assert rows["crypto_trend"]["error_count"] == 1


def test_a_strategy_that_never_ticked_is_visibly_distinct_from_a_quiet_one() -> None:
    registry = StrategyRegistry()
    registry.register("never_ran", AssetClass.EQUITY)
    registry.register("ran_quietly", AssetClass.EQUITY)
    registry.record_tick("ran_quietly", NOW, proposals=[], reason="no signal")

    body = client(build_handle(strategies=registry)).get(
        "/api/strategies", headers=AUTH).json()
    rows = {r["strategy_id"]: r for r in body["strategies"]}
    assert rows["never_ran"]["has_ticked"] is False
    assert rows["ran_quietly"]["has_ticked"] is True
    assert body["never_ticked"] == ["never_ran"]


def test_registry_refuses_a_status_update_for_an_unknown_strategy() -> None:
    registry = StrategyRegistry()
    with pytest.raises(KeyError):
        registry.record_tick("ghost", NOW)


def test_a_disabled_status_must_carry_a_reason() -> None:
    from api.server import StrategyStatus
    with pytest.raises(ValueError):
        StrategyStatus(strategy_id="x", asset_class=AssetClass.EQUITY,
                       enabled=False)


# ==========================================================================
# risk
# ==========================================================================

def test_risk_endpoint_reports_per_class_state_and_limits() -> None:
    store = SQLitePositionStore(":memory:")
    circuit = RiskCircuit(store, Mode.PAPER)
    body = client(build_handle(circuit=circuit)).get("/api/risk", headers=AUTH).json()
    equity = body["classes"]["equity"]
    assert equity["action"] == "ok"
    assert equity["entries_allowed"] is True
    assert equity["limits"]["flatten_drawdown"] == "0.15"
    # Fractions, and named as fractions: shipping 0.05 under a "_pct" key is how
    # a dashboard renders a 5% drawdown as 0.05%.
    assert "drawdown_fraction" in equity
    assert "drawdown_pct" not in equity
    store.close()


def test_the_api_never_advances_the_circuit() -> None:
    """FakeCircuit.advance() raises. Every read path must stay clear of it: the
    high-water mark is moved by the order path, never by a page refresh."""
    c = client()
    for path in ("/api/state", "/api/risk", "/api/positions", "/ready"):
        assert c.get(path, headers=AUTH).status_code in (200, 503), path
    c.post("/api/control/pause", json={"reason": "checking"}, headers=AUTH)
    c.post("/api/control/flatten", json={"reason": "checking"}, headers=AUTH)


def test_a_circuit_that_raises_is_reported_as_unreadable_not_clear() -> None:
    handle = build_handle(circuit=FakeCircuit(Mode.PAPER,
                                              error=RuntimeError("state row corrupt")))
    body = client(handle).get("/api/state", headers=AUTH).json()
    block = body["classes"]["equity"]["risk"]
    assert block["entries_allowed"] is False
    assert block["action"] is None
    assert "state row corrupt" in block["reason"]
    assert body["degraded"] is True


def test_a_flatten_decision_surfaces_as_must_flatten() -> None:
    decision = CircuitDecision(
        action=CircuitAction.FLATTEN, drawdown_pct=Decimal("0.18"),
        daily_loss_pct=Decimal("0.06"), reason="drawdown 18.00% at flatten rung",
        equity_known=True, state_known=True)
    handle = build_handle(circuit=FakeCircuit(Mode.PAPER, decision=decision))
    block = client(handle).get("/api/state", headers=AUTH).json()["classes"]["equity"]["risk"]
    assert block["must_flatten"] is True
    assert block["entries_allowed"] is False
    assert block["drawdown_fraction"] == "0.18"


# ==========================================================================
# handle validation
# ==========================================================================

def test_handle_refuses_an_unstated_arming_flag() -> None:
    with pytest.raises(ValueError) as exc:
        EngineHandle(
            mode=Mode.LIVE, asset_classes=(AssetClass.EQUITY,),
            ledger=FakeLedger(), circuit=FakeCircuit(Mode.LIVE),
            control=MemoryControlStore(), strategies=StrategyRegistry(),
            bus=EventBus(), armed={})
    assert "armed[equity] is not stated" in str(exc.value)


def test_handle_refuses_a_circuit_bound_to_the_other_mode() -> None:
    with pytest.raises(ValueError) as exc:
        EngineHandle(
            mode=Mode.LIVE, asset_classes=(AssetClass.EQUITY,),
            ledger=FakeLedger(), circuit=FakeCircuit(Mode.PAPER),
            control=MemoryControlStore(), strategies=StrategyRegistry(),
            bus=EventBus(), armed={AssetClass.EQUITY: True})
    assert "wrong book" in str(exc.value)


def test_handle_refuses_an_empty_asset_class_list() -> None:
    with pytest.raises(ValueError):
        EngineHandle(mode=Mode.PAPER, asset_classes=(), ledger=FakeLedger(),
                     circuit=FakeCircuit(), control=MemoryControlStore(),
                     strategies=StrategyRegistry(), bus=EventBus(), armed={})


def test_handle_refuses_a_naive_clock() -> None:
    handle = build_handle(clock=lambda: dt.datetime(2026, 8, 27, 10, 0))
    with pytest.raises(ValueError):
        handle.now()


def test_the_default_clock_is_the_wall_clock_and_is_tz_aware() -> None:
    handle = EngineHandle(
        mode=Mode.PAPER, asset_classes=(AssetClass.EQUITY,), ledger=FakeLedger(),
        circuit=FakeCircuit(), control=MemoryControlStore(),
        strategies=StrategyRegistry(), bus=EventBus(),
        armed={AssetClass.EQUITY: False})
    moment = handle.now()
    assert moment.tzinfo is not None
    assert abs((moment - dt.datetime.now(UTC)).total_seconds()) < 5


def test_armed_flags_are_reported_verbatim() -> None:
    handle = build_handle(mode=Mode.LIVE,
                          adapters={ac: FakeAdapter(ac, Mode.LIVE)
                                    for ac in (AssetClass.EQUITY, AssetClass.CRYPTO)},
                          circuit=FakeCircuit(Mode.LIVE),
                          armed={AssetClass.EQUITY: True, AssetClass.CRYPTO: False})
    body = client(handle).get("/api/state", headers=AUTH).json()
    assert body["classes"]["equity"]["armed_live"] is True
    assert body["classes"]["crypto"]["armed_live"] is False


# ==========================================================================
# readiness
# ==========================================================================

def test_ready_reports_the_resolved_adapter_per_class() -> None:
    body = client().get("/ready", headers=AUTH).json()
    assert body["ready"] is True
    assert body["classes"]["equity"] == {"resolved": True, "adapter": "FakeAdapter",
                                         "adapter_mode": "paper", "armed_live": False}


def test_ready_is_503_when_a_class_has_no_adapter() -> None:
    handle = build_handle(adapters={AssetClass.CRYPTO: FakeAdapter(AssetClass.CRYPTO)})
    resp = client(handle).get("/ready", headers=AUTH)
    assert resp.status_code == 503
    assert resp.json()["ready"] is False
    assert resp.json()["classes"]["equity"]["resolved"] is False


def test_ready_does_not_touch_the_venue() -> None:
    """A readiness probe runs on a timer. Wiring it to a broker turns a restart
    loop into a rate-limit ban."""
    adapters = {ac: FakeAdapter(ac) for ac in (AssetClass.EQUITY, AssetClass.CRYPTO)}
    client(build_handle(adapters=adapters)).get("/ready", headers=AUTH)
    assert all(a.equity_calls == 0 for a in adapters.values())


def test_ready_flags_a_live_adapter_on_a_disarmed_class() -> None:
    handle = build_handle(
        mode=Mode.LIVE, circuit=FakeCircuit(Mode.LIVE),
        adapters={ac: FakeAdapter(ac, Mode.LIVE)
                  for ac in (AssetClass.EQUITY, AssetClass.CRYPTO)},
        armed={AssetClass.EQUITY: False, AssetClass.CRYPTO: False})
    body = client(handle).get("/ready", headers=AUTH).json()
    assert any("coerce" in p["detail"] for p in body["problems"])


# ==========================================================================
# engine liveness
# ==========================================================================

def test_a_missing_heartbeat_is_unknown_not_dead_and_not_alive() -> None:
    body = client().get("/api/state", headers=AUTH).json()
    assert body["engine"]["status"] == "unknown"
    assert body["engine"]["heartbeat_at"] is None
    assert body["degraded"] is True


def test_a_fresh_heartbeat_reads_alive() -> None:
    control = MemoryControlStore()
    control.rows["__engine__"] = {"heartbeat_at": (NOW - dt.timedelta(seconds=3)).isoformat()}
    body = client(build_handle(control=control)).get("/api/state", headers=AUTH).json()
    assert body["engine"]["status"] == "alive"
    assert body["engine"]["age_s"] == pytest.approx(3.0)


def test_an_old_heartbeat_reads_stale() -> None:
    control = MemoryControlStore()
    control.rows["__engine__"] = {"heartbeat_at": (NOW - dt.timedelta(hours=2)).isoformat()}
    body = client(build_handle(control=control)).get("/api/state", headers=AUTH).json()
    assert body["engine"]["status"] == "stale"
    assert body["degraded"] is True


def test_a_naive_heartbeat_is_unknown_not_silently_shifted() -> None:
    control = MemoryControlStore()
    control.rows["__engine__"] = {"heartbeat_at": "2026-08-27T10:00:00"}
    body = client(build_handle(control=control)).get("/api/state", headers=AUTH).json()
    assert body["engine"]["status"] == "unknown"
    assert "timezone-aware" in body["engine"]["detail"]


# ==========================================================================
# control
# ==========================================================================

def test_pause_persists_and_is_reported() -> None:
    handle = build_handle()
    c = client(handle)
    resp = c.post("/api/control/pause",
                  json={"reason": "spread looks wrong", "actor": "vishesh"},
                  headers=AUTH)
    assert resp.status_code == 200
    applied = resp.json()["applied"]
    assert set(applied) == {"equity", "crypto"}
    assert applied["equity"]["paused"] is True
    assert applied["equity"]["paused_reason"] == "spread looks wrong"
    assert applied["equity"]["paused_actor_claimed"] == "vishesh"

    state = c.get("/api/state", headers=AUTH).json()
    assert state["classes"]["crypto"]["control"]["paused"] is True


def test_pause_can_target_one_class() -> None:
    c = client()
    c.post("/api/control/pause", json={"reason": "crypto only", "asset_class": "crypto"},
           headers=AUTH)
    state = c.get("/api/state", headers=AUTH).json()
    assert state["classes"]["crypto"]["control"]["paused"] is True
    assert state["classes"]["equity"]["control"]["paused"] is False


def test_pause_requires_a_reason() -> None:
    c = client()
    assert c.post("/api/control/pause", json={}, headers=AUTH).status_code == 422
    assert c.post("/api/control/pause", json={"reason": "   "},
                  headers=AUTH).status_code == 422


def test_control_bodies_cannot_smuggle_a_mode_or_an_arming_flag() -> None:
    """No endpoint may arm live trading or change a mode. `extra="forbid"` makes
    a field a future author might wire up a 422 today."""
    c = client()
    for payload in ({"reason": "x", "mode": "live"},
                    {"reason": "x", "armed": True},
                    {"reason": "x", "requested_mode": "live"},
                    {"reason": "x", "ENGINE_API_TOKEN": "new"}):
        resp = c.post("/api/control/pause", json=payload, headers=AUTH)
        assert resp.status_code == 422, payload


def test_control_does_not_touch_the_process_environment() -> None:
    before = dict(os.environ)
    c = client()
    c.post("/api/control/pause", json={"reason": "x"}, headers=AUTH)
    c.post("/api/control/flatten", json={"reason": "x"}, headers=AUTH)
    c.post("/api/control/resume", json={"reason": "x"}, headers=AUTH)
    assert dict(os.environ) == before


def test_control_rejects_a_class_this_engine_does_not_trade() -> None:
    handle = build_handle(asset_classes=(AssetClass.EQUITY,))
    resp = client(handle).post("/api/control/pause",
                               json={"reason": "x", "asset_class": "crypto"},
                               headers=AUTH)
    assert resp.status_code == 400


def test_resume_clears_an_operator_pause() -> None:
    c = client()
    c.post("/api/control/pause", json={"reason": "checking"}, headers=AUTH)
    resp = c.post("/api/control/resume", json={"reason": "all clear"}, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["applied"]["equity"]["paused"] is False


def test_resume_is_refused_while_a_flatten_is_unacknowledged() -> None:
    """Re-enabling entries into a book that is supposed to be closing is the
    exact wrong order of operations."""
    c = client()
    c.post("/api/control/flatten", json={"reason": "stepping away"}, headers=AUTH)
    resp = c.post("/api/control/resume", json={"reason": "changed my mind"},
                  headers=AUTH)
    assert resp.status_code == 409
    assert "unacknowledged" in resp.json()["detail"]


def test_resume_works_once_the_engine_acknowledges_the_flatten() -> None:
    control = MemoryControlStore()
    c = client(build_handle(control=control))
    flatten = c.post("/api/control/flatten", json={"reason": "stepping away"},
                     headers=AUTH).json()
    request_id = flatten["request_id"]
    for scope in ("equity", "crypto"):
        row = dict(control.rows[scope])
        row["flatten_acked_id"] = request_id
        row["flatten_acked_at"] = NOW.isoformat()
        control.rows[scope] = row
    resp = c.post("/api/control/resume", json={"reason": "book is flat"},
                  headers=AUTH)
    assert resp.status_code == 200


def test_a_guard_refusal_writes_nothing_at_all() -> None:
    """The guard runs across every target before the first write, so a
    multi-class action cannot half-apply and leave the classes disagreeing."""
    control = MemoryControlStore()
    c = client(build_handle(control=control))
    c.post("/api/control/flatten", json={"reason": "x", "asset_class": "crypto"},
           headers=AUTH)
    c.post("/api/control/pause", json={"reason": "x", "asset_class": "equity"},
           headers=AUTH)
    resp = c.post("/api/control/resume", json={"reason": "y"}, headers=AUTH)
    assert resp.status_code == 409
    # equity was resumable on its own, and must NOT have been resumed.
    assert control.rows["equity"]["paused"] == "1"


def test_flatten_records_a_request_and_does_not_execute_it() -> None:
    adapters = {ac: FakeAdapter(ac) for ac in (AssetClass.EQUITY, AssetClass.CRYPTO)}
    handle = build_handle(adapters=adapters)
    resp = client(handle).post("/api/control/flatten",
                               json={"reason": "closing the desk"}, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["executed"] is False
    assert body["request_id"]
    assert "next tick" in body["note"]
    assert body["applied"]["equity"]["flatten_pending"] is True
    # FakeAdapter.place/cancel raise AssertionError; reaching here proves the
    # API never touched them.


def test_flatten_also_pauses() -> None:
    c = client()
    body = c.post("/api/control/flatten", json={"reason": "closing"},
                  headers=AUTH).json()
    assert body["applied"]["equity"]["paused"] is True


def test_a_new_flatten_does_not_inherit_the_previous_acknowledgement() -> None:
    control = MemoryControlStore()
    c = client(build_handle(control=control))
    first = c.post("/api/control/flatten", json={"reason": "one"},
                   headers=AUTH).json()["request_id"]
    row = dict(control.rows["equity"])
    row["flatten_acked_id"] = first
    row["flatten_acked_at"] = NOW.isoformat()
    control.rows["equity"] = row

    second = c.post("/api/control/flatten", json={"reason": "two"},
                    headers=AUTH).json()
    equity = second["applied"]["equity"]
    assert equity["flatten_request_id"] != first
    assert equity["flatten_acked_id"] is None
    assert equity["flatten_pending"] is True


def test_a_control_write_failure_is_503_and_never_reports_success() -> None:
    control = MemoryControlStore()
    control.fail_write = RuntimeError("disk full")
    resp = client(build_handle(control=control)).post(
        "/api/control/pause", json={"reason": "x"}, headers=AUTH)
    assert resp.status_code == 503
    assert "disk full" in resp.json()["detail"]


def test_a_partial_control_application_names_the_classes_that_took_it() -> None:
    class HalfBrokenControl(MemoryControlStore):
        def mutate(self, scope: str, fn: Any) -> dict[str, str]:
            if scope == "crypto":
                raise RuntimeError("crypto row locked")
            return super().mutate(scope, fn)

    resp = client(build_handle(control=HalfBrokenControl())).post(
        "/api/control/pause", json={"reason": "x"}, headers=AUTH)
    assert resp.status_code == 503
    body = resp.json()
    assert body["applied"] == ["equity"]
    assert body["not_applied"] == ["crypto"]


def test_control_publishes_an_event() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    client(build_handle(bus=bus)).post("/api/control/pause",
                                       json={"reason": "spread looks wrong"},
                                       headers=AUTH)
    kinds = [d.event.kind for d in sub.drain() if d.event is not None]
    assert kinds == ["control.pause", "control.pause"]


# ==========================================================================
# control durability across processes
# ==========================================================================

def test_control_state_survives_a_separate_connection(tmp_path: Any) -> None:
    """The engine is a different process. A pause the engine cannot read is a
    pause that did not happen, however green the API response was."""
    path = str(tmp_path / "control.db")
    writer = SqliteControlStore(path)
    c = client(build_handle(control=writer))
    c.post("/api/control/pause", json={"reason": "stand down"}, headers=AUTH)

    reader = SqliteControlStore(path)
    state = ControlState.from_row(reader.read("equity") or {})
    assert state.paused is True
    assert state.paused_reason == "stand down"
    writer.close()
    reader.close()


def test_sqlite_control_mutate_is_a_read_modify_write_under_one_lock(
        tmp_path: Any) -> None:
    """Two operators pausing at once must not lose one of the pauses."""
    store = SqliteControlStore(str(tmp_path / "control.db"))
    store.write("equity", {"paused": "0"})
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def bump(n: int) -> None:
        def fn(row: Optional[Any]) -> dict[str, str]:
            current = dict(row or {})
            current[f"note{n}"] = str(n)
            return current
        try:
            barrier.wait(timeout=5)
            store.mutate("equity", fn)
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            errors.append(exc)

    threads = [threading.Thread(target=bump, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, errors
    final = store.read("equity") or {}
    assert sorted(k for k in final if k.startswith("note")) == [
        f"note{i}" for i in range(8)]
    store.close()


def test_sqlite_control_read_returns_none_only_for_a_missing_row(
        tmp_path: Any) -> None:
    store = SqliteControlStore(str(tmp_path / "control.db"))
    assert store.read("equity") is None
    store.write("equity", {"paused": "0"})
    assert store.read("equity") == {"paused": "0"}
    store.close()


def test_sqlite_control_refuses_non_string_values(tmp_path: Any) -> None:
    store = SqliteControlStore(str(tmp_path / "control.db"))
    with pytest.raises(TypeError):
        store.write("equity", {"paused": True})
    store.close()


def test_a_paused_row_without_a_timestamp_is_corrupt() -> None:
    with pytest.raises(ValueError):
        ControlState.from_row({"paused": "1"})


# ==========================================================================
# events / SSE
# ==========================================================================

def test_event_refuses_unserialisable_data() -> None:
    with pytest.raises(TypeError) as exc:
        Event(kind="fill", at=NOW, data={"price": Decimal("120.5")})
    assert "Stringify Decimals" in str(exc.value)


def test_event_requires_a_tz_aware_timestamp() -> None:
    with pytest.raises(ValueError):
        Event(kind="fill", at=dt.datetime(2026, 8, 27, 10, 0))


def test_a_slow_subscriber_never_blocks_the_publisher_and_is_told_it_missed() -> None:
    bus = EventBus(per_client_maxlen=2)
    sub = bus.subscribe()
    for i in range(5):
        bus.publish(Event(kind="tick", at=NOW, data={"n": i}))
    items = sub.drain()
    gaps = [d for d in items if d.event is None]
    kept = [d for d in items if d.event is not None]
    assert [d.gap for d in gaps] == [3]
    assert [d.event.data["n"] for d in kept] == [3, 4]


def test_a_reconnect_past_the_backlog_is_told_it_missed() -> None:
    bus = EventBus(backlog_maxlen=2)
    for i in range(5):
        bus.publish(Event(kind="tick", at=NOW, data={"n": i}))
    sub = bus.subscribe(after_seq=1)
    items = sub.drain()
    assert items[0].event is None
    assert items[0].gap == 2
    assert [d.event.data["n"] for d in items[1:]] == [3, 4]


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.unsubscribe(sub)
    bus.publish(Event(kind="tick", at=NOW))
    assert sub.drain() == []
    assert bus.subscriber_count == 0


def test_sse_replays_the_backlog_for_a_reconnecting_client() -> None:
    bus = EventBus()
    bus.publish(Event(kind="fill", at=NOW, data={"symbol": "BTC/USDT",
                                                 "price": "64000.1"}))
    bus.publish(Event(kind="gate.reject", at=NOW, data={"reason": "allocation_cap"}))
    resp = client(build_handle(bus=bus)).get(
        "/api/events", headers={**AUTH, "Last-Event-ID": "0"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert "event: fill" in text
    assert "event: gate.reject" in text
    assert '"price":"64000.1"' in text
    assert "id: 1" in text and "id: 2" in text


def test_sse_delivers_an_event_published_while_the_stream_is_open() -> None:
    bus = EventBus()
    c = client(build_handle(bus=bus))
    started = threading.Event()

    def publish_soon() -> None:
        started.wait(timeout=5)
        time.sleep(0.05)
        bus.publish(Event(kind="fill", at=NOW, data={"symbol": "NIFTY", "qty": "75"}))

    thread = threading.Thread(target=publish_soon, daemon=True)
    thread.start()
    started.set()
    # The transport buffers until the stream closes itself (sse_max_stream_s),
    # so the publish above lands mid-stream.
    text = c.get("/api/events", headers=AUTH).text
    thread.join(timeout=2)

    assert "event: fill" in text
    payload = next(l for l in text.splitlines()
                   if l.startswith("data: ") and "NIFTY" in l)
    assert json.loads(payload[len("data: "):])["symbol"] == "NIFTY"


def test_sse_streams_events_in_order_with_resumable_ids() -> None:
    bus = EventBus()
    for i in range(3):
        bus.publish(Event(kind="tick", at=NOW, data={"n": i}))
    text = client(build_handle(bus=bus)).get(
        "/api/events", headers={**AUTH, "Last-Event-ID": "0"}).text
    ids = [l for l in text.splitlines() if l.startswith("id: ")]
    assert ids == ["id: 1", "id: 2", "id: 3"]


def test_sse_tells_the_client_when_it_closes_the_stream() -> None:
    """A stream that just stops looks identical to a network failure. Saying
    goodbye is what makes the reconnect a resume rather than a guess."""
    text = client().get("/api/events", headers=AUTH).text
    assert "event: bye" in text
    assert "retry: " in text


def test_sse_releases_its_subscription_when_the_stream_ends() -> None:
    bus = EventBus()
    client(build_handle(bus=bus)).get("/api/events", headers=AUTH)
    assert bus.subscriber_count == 0


def test_sse_replays_a_gap_marker_when_the_backlog_is_short() -> None:
    bus = EventBus(backlog_maxlen=2)
    for i in range(5):
        bus.publish(Event(kind="tick", at=NOW, data={"n": i}))
    text = client(build_handle(bus=bus)).get(
        "/api/events", headers={**AUTH, "Last-Event-ID": "1"}).text
    assert "event: gap" in text
    assert '"missed":2' in text


# ==========================================================================
# day boundary — the input store.realized_pnl actually requires
# ==========================================================================

def test_equity_day_start_is_ist_midnight_as_a_tz_aware_instant() -> None:
    start = day_start_instant(EquityProfile(), NOW)
    assert start == dt.datetime(2026, 8, 26, 18, 30, tzinfo=UTC)
    assert start.astimezone(IST).hour == 0
    assert start.tzinfo is not None


def test_crypto_day_start_is_utc_midnight() -> None:
    start = day_start_instant(CryptoProfile(), NOW)
    assert start == dt.datetime(2026, 8, 27, 0, 0, tzinfo=UTC)


def test_day_start_is_derived_not_looked_up_per_asset_class() -> None:
    """Bisecting day_key means a profile whose timezone changes cannot leave a
    stale copy of that timezone behind in this module."""
    class ShiftedProfile(EquityProfile):
        def day_key(self, now: dt.datetime) -> str:
            return (now - dt.timedelta(hours=7)).astimezone(UTC).strftime("%Y-%m-%d")

    start = day_start_instant(ShiftedProfile(), NOW)
    assert start == dt.datetime(2026, 8, 27, 7, 0, tzinfo=UTC) - dt.timedelta(days=1)


def test_day_start_refuses_a_profile_whose_day_key_is_not_daily() -> None:
    class ConstantProfile(EquityProfile):
        def day_key(self, now: dt.datetime) -> str:
            return "always"

    with pytest.raises(ValueError) as exc:
        day_start_instant(ConstantProfile(), NOW)
    assert "not day-granular" in str(exc.value)


def test_state_passes_a_tz_aware_instant_to_realized_pnl() -> None:
    ledger = FakeLedger()
    client(build_handle(ledger=ledger)).get("/api/state", headers=AUTH)
    windows = {ac: since for _, ac, since in ledger.pnl_calls}
    assert windows[AssetClass.EQUITY] == "2026-08-26T18:30:00+00:00"
    assert windows[AssetClass.CRYPTO] == "2026-08-27T00:00:00+00:00"


# ==========================================================================
# end to end against the real store and the real circuit
# ==========================================================================

def test_snapshot_against_the_real_sqlite_store_and_risk_circuit() -> None:
    """The store raises on a naive `since_iso` and the circuit fails closed on
    an unreadable row. Wiring both for real is the tripwire for either contract
    drifting under this module."""
    store = SQLitePositionStore(":memory:")
    fill = Fill(order_id="OKX-1", instrument=BTC, side=Side.BUY,
                qty=Decimal("0.01"), price=Decimal("64000"), fees=Decimal("0.64"),
                at=NOW, mode=Mode.PAPER)
    store.apply(fill, strategy_id="crypto_trend")
    store.record_order(OrderRef(order_id="OKX-1", status=OrderStatus.FILLED,
                                intent=Intent(instrument=BTC, side=Side.BUY,
                                              qty=Decimal("0.01"),
                                              order_type=OrderType.MARKET,
                                              strategy_id="crypto_trend",
                                              reason="ema cross",
                                              idempotency_key="ct-1"),
                                mode=Mode.PAPER, filled_qty=Decimal("0.01"),
                                avg_price=Decimal("64000"), fills=(fill,)))
    store.record_order(gate_rejection_ref(_rejected_approval(), NOW))

    handle = build_handle(ledger=store, circuit=RiskCircuit(store, Mode.PAPER),
                          control=SqliteControlStore(":memory:"))
    c = client(handle)

    state = c.get("/api/state", headers=AUTH)
    assert state.status_code == 200
    body = state.json()
    assert body["classes"]["crypto"]["open_positions"] == 1
    assert body["classes"]["crypto"]["realized_pnl_today"]["value"] is not None
    assert body["classes"]["crypto"]["risk"]["entries_allowed"] is True

    positions = c.get("/api/positions?asset_class=crypto", headers=AUTH).json()
    assert positions["positions"][0]["symbol"] == "BTC/USDT"
    assert positions["positions"][0]["qty"] == "0.01"

    orders = c.get("/api/orders", headers=AUTH).json()["orders"]
    origins = {o["origin"] for o in orders}
    assert origins == {"broker", "gate"}
    gate_row = next(o for o in orders if o["origin"] == "gate")
    assert "allocation_cap" in gate_row["error"]
    store.close()


def test_the_real_store_would_reject_a_day_key_passed_as_a_window() -> None:
    """Pins the reason day_start_instant exists: a calendar date is not an
    instant, and passing one raises rather than shifting the daily-loss window
    by five and a half hours."""
    store = SQLitePositionStore(":memory:")
    with pytest.raises(ValueError):
        store.realized_pnl(Mode.PAPER, AssetClass.EQUITY,
                           EquityProfile().day_key(NOW))
    store.realized_pnl(Mode.PAPER, AssetClass.EQUITY,
                       day_start_instant(profile_for(AssetClass.EQUITY), NOW).isoformat())
    store.close()
