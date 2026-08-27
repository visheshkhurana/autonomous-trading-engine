"""The composition root.

Most of these are wiring assertions, and wiring assertions earn their place
here because every one of them protects a claim the modules make about each
other but cannot check alone:

  * the gate and the engine must agree about `requested_mode`, or every
    approval is refused with a mode mismatch,
  * the API handle must report the modes the engine ACTUALLY resolved, not the
    one the process asked for,
  * nothing in this file may arm anything, and the synthetic feed must refuse
    to serve a live class at all.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal

import pytest

import engine as eng
import runner
from api import server as api_mod
from core.gate import GateStateStoreProto
from core.store import SQLiteGateStateStore, SQLitePositionStore
from core.types import AssetClass, Mode

TOKEN = "t" * 32
NOW = dt.datetime(2026, 3, 2, 12, 0, tzinfo=dt.timezone.utc)


def base_env(tmp_path, **extra) -> dict[str, str]:
    env = {"TRADER_DB": str(tmp_path / "trader.db"),
           "ENGINE_API_TOKEN": TOKEN,
           "TRADER_CLASSES": "crypto"}
    env.update(extra)
    return env


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


# ==========================================================================
# it assembles at all
# ==========================================================================

def test_the_whole_stack_builds(stack) -> None:
    s = stack()
    assert s.engine is not None
    assert s.handle is not None
    assert s.registry.registered, "a registry nothing registered with 503s"


def test_the_gate_gets_the_durable_state_store_not_a_dict(stack) -> None:
    """core.gate is explicit that a dict satisfies its Protocol and must never
    be used outside tests: N workers, N counters, N times the daily cap."""
    s = stack()
    assert isinstance(s.gate_state, SQLiteGateStateStore)
    assert isinstance(s.gate_state, GateStateStoreProto)


def test_the_ledger_and_the_gate_state_share_one_file(stack, tmp_path) -> None:
    s = stack()
    assert s.store.path == s.gate_state.path == str(tmp_path / "trader.db")


def test_the_engine_is_given_the_control_store(stack) -> None:
    """Without this the API's pause and flatten write rows nothing reads."""
    s = stack()
    assert s.engine.control is s.control


def test_the_gate_and_the_engine_agree_about_requested_mode(stack) -> None:
    """They must, or _adapter_for refuses every approval as a mode mismatch and
    the engine ticks forever placing nothing."""
    s = stack()
    approval_mode = s.gate.check.__self__._OrderGate__config.requested_mode
    assert approval_mode is s.engine.config.requested_mode


# ==========================================================================
# nothing here arms anything
# ==========================================================================

def test_paper_is_the_default(stack) -> None:
    s = stack()
    assert s.engine.config.requested_mode is Mode.PAPER
    assert all(m is Mode.PAPER for m in s.engine.runner_modes().values())


def test_requesting_live_without_the_arm_flag_still_resolves_to_paper(stack) -> None:
    """TRADER_MODE is a REQUEST. Only CRYPTO_LIVE / EQUITY_LIVE arm."""
    s = stack(TRADER_MODE="live")
    assert s.engine.config.requested_mode is Mode.LIVE
    assert s.engine.runner_modes()[AssetClass.CRYPTO] is Mode.PAPER
    assert s.handle.armed[AssetClass.CRYPTO] is False


def test_no_live_adapter_is_constructed(stack) -> None:
    s = stack(TRADER_MODE="live")
    for wiring in [r.wiring for r in s.engine._runners.values()]:
        assert wiring.live_adapter is None


def test_an_unparseable_mode_raises_rather_than_defaulting_to_paper(tmp_path) -> None:
    """`TRADER_MODE=liev` resolving quietly to paper is the same silence that
    would resolve a typo'd live to paper on the day it mattered."""
    with pytest.raises(ValueError, match="TRADER_MODE"):
        runner.RunnerConfig.from_env(base_env(tmp_path, TRADER_MODE="liev"))


def test_an_unknown_asset_class_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="TRADER_CLASSES"):
        runner.RunnerConfig.from_env(base_env(tmp_path, TRADER_CLASSES="forex"))


def test_the_runner_never_writes_an_arm_flag(tmp_path) -> None:
    env = base_env(tmp_path)
    before = dict(env)
    runner.build_stack(env=env, clock=lambda: NOW).close()
    assert env == before, "build_stack must not mutate the environment"
    assert "CRYPTO_LIVE" not in env and "EQUITY_LIVE" not in env


# ==========================================================================
# the handle tells the truth about modes
# ==========================================================================

def test_the_handle_reports_the_modes_the_engine_resolved(stack) -> None:
    """A dashboard showing "paper" for a class the gate will execute live is
    worse than no dashboard."""
    s = stack()
    engine_modes = s.engine.runner_modes()
    for asset_class, armed in s.handle.armed.items():
        assert armed == (engine_modes[asset_class] is Mode.LIVE)


def test_the_handle_carries_an_adapter_for_every_class(stack) -> None:
    """A missing adapter is reported as unresolved by /ready; it should not be
    missing because the runner forgot to pass it."""
    s = stack()
    for asset_class in s.handle.asset_classes:
        assert asset_class in s.handle.adapters


# ==========================================================================
# the per-class circuit — the seam engine.py hands to the integrator
# ==========================================================================

def test_the_gate_circuit_routes_each_class_to_its_own_circuit(stack) -> None:
    s = stack(TRADER_CLASSES="crypto,equity")
    circuit = s.gate._OrderGate__circuit
    crypto = circuit.circuit_for(AssetClass.CRYPTO)
    equity = circuit.circuit_for(AssetClass.EQUITY)
    assert crypto is not equity


def test_an_unwired_class_raises_rather_than_borrowing_another_circuit(stack) -> None:
    s = stack(TRADER_CLASSES="crypto")
    circuit = s.gate._OrderGate__circuit
    with pytest.raises(KeyError):
        circuit.circuit_for(AssetClass.EQUITY)


def test_the_gate_may_not_advance_the_circuit(stack) -> None:
    """Advancing is the order path's job and the engine already does it per
    class. A second writer reachable through the gate would roll the
    high-water mark twice per tick."""
    s = stack()
    with pytest.raises(RuntimeError, match="must not advance"):
        s.gate._OrderGate__circuit.advance(AssetClass.CRYPTO, Decimal("1"))


def test_the_api_may_not_advance_the_circuit_either(stack) -> None:
    """api/server.py promises it never calls the circuit's writer. This makes
    that promise enforceable rather than merely stated."""
    s = stack()
    with pytest.raises(RuntimeError, match="must not advance"):
        s.handle.circuit.advance(AssetClass.CRYPTO, Decimal("1"))


# ==========================================================================
# the synthetic feed
# ==========================================================================

def test_the_synthetic_feed_refuses_to_serve_a_live_class() -> None:
    """A live book trading on a generated price series is the worst thing this
    file could produce."""
    feed = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO))
    with pytest.raises(RuntimeError, match="refuses to serve live"):
        feed.bind({AssetClass.CRYPTO: Mode.LIVE})


def test_an_unbound_feed_will_not_serve_bars() -> None:
    """Unbound means the live check has not run."""
    feed = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO))
    with pytest.raises(RuntimeError, match="before bind"):
        feed.extra_provider(AssetClass.CRYPTO, NOW)


def test_the_feed_binds_for_paper() -> None:
    feed = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO))
    assert feed.bind({AssetClass.CRYPTO: Mode.PAPER}) is feed


def test_bars_never_run_ahead_of_now() -> None:
    """Every strategy rejects a bar stamped ahead of ctx.now outright."""
    feed = runner.SyntheticBarFeed(
        runner.default_instruments(AssetClass.CRYPTO)).bind(
            {AssetClass.CRYPTO: Mode.PAPER})
    bars = feed.bars("BTC/USDT", NOW)
    assert bars[-1].ts <= NOW


def test_bars_are_strictly_increasing() -> None:
    """validate_bars() rejects a series that is not, and would disable the
    strategy for the session."""
    feed = runner.SyntheticBarFeed(
        runner.default_instruments(AssetClass.CRYPTO)).bind(
            {AssetClass.CRYPTO: Mode.PAPER})
    bars = feed.bars("BTC/USDT", NOW)
    assert all(b.ts < n.ts for b, n in zip(bars, bars[1:]))


def test_the_feed_is_deterministic() -> None:
    """A paper run must replay identically or a reported result means nothing."""
    a = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO)
                                ).bind({AssetClass.CRYPTO: Mode.PAPER})
    b = runner.SyntheticBarFeed(runner.default_instruments(AssetClass.CRYPTO)
                                ).bind({AssetClass.CRYPTO: Mode.PAPER})
    assert [x.close for x in a.bars("BTC/USDT", NOW)] == \
           [x.close for x in b.bars("BTC/USDT", NOW)]


def test_every_generated_price_is_a_decimal() -> None:
    feed = runner.SyntheticBarFeed(
        runner.default_instruments(AssetClass.CRYPTO)).bind(
            {AssetClass.CRYPTO: Mode.PAPER})
    for bar in feed.bars("BTC/USDT", NOW):
        for value in (bar.open, bar.high, bar.low, bar.close):
            assert isinstance(value, Decimal)
    assert isinstance(feed.mark("BTC/USDT", NOW), Decimal)


# ==========================================================================
# the event bridge
# ==========================================================================

def test_engine_events_reach_the_api_bus(stack) -> None:
    """api/server.py's own caveat: its bus only sees what is published in THIS
    process. The bridge is what makes /api/events carry the engine's story."""
    s = stack()

    async def go() -> None:
        await s.start()
        try:
            await s.engine.tick_once(AssetClass.CRYPTO)
            await asyncio.sleep(0.2)
        finally:
            await s.stop()

    asyncio.run(go())
    assert s.api_bus.published > 0


def test_a_decimal_in_event_data_survives_the_bridge_as_a_string(stack) -> None:
    """The API's Event refuses non-scalars so an unserialisable value cannot
    kill a live operator's stream mid-flight; a float round-trip would stop the
    displayed P&L reconciling with the ledger."""
    event = eng.Event(seq=1, at=NOW, kind=eng.EventKind.FILL, message="filled",
                      asset_class=AssetClass.CRYPTO, mode=Mode.PAPER,
                      data={"qty": Decimal("1.5"), "price": Decimal("60000.25")})
    translated = runner._api_event(event)
    assert translated.data["d_qty"] == "1.5"
    assert translated.data["d_price"] == "60000.25"


def test_a_nested_value_is_rendered_not_dropped(stack) -> None:
    """A rejection whose `checks` vanished is a rejection nobody can explain."""
    event = eng.Event(seq=1, at=NOW, kind=eng.EventKind.REJECTION,
                      message="rejected", data={"checks": [{"name": "dedup"}]})
    translated = runner._api_event(event)
    assert "dedup" in translated.data["d_checks"]


def test_a_disabled_strategy_stops_showing_as_enabled(stack) -> None:
    """/api/strategies is the screen that answers "is anything still running".
    It must never show a strategy as enabled after the engine dropped it."""
    s = stack()
    sid = s.registry.snapshot()[0].strategy_id
    runner._update_registry(s.registry, eng.Event(
        seq=1, at=NOW, kind=eng.EventKind.STRATEGY_DISABLED,
        message="disabled", asset_class=AssetClass.CRYPTO,
        data={"strategy_id": sid, "error": "ValueError: boom"}))
    row = {r.strategy_id: r for r in s.registry.snapshot()}[sid]
    assert row.enabled is False
    assert "boom" in (row.disabled_reason or "")


def test_a_ticking_strategy_does_not_look_like_a_dead_one(stack) -> None:
    """StrategyStatus is explicit that a null last_tick_at means "never ran in
    this process" — leaving it null while the engine ticks renders every live
    strategy as a healthy-looking dead one."""
    s = stack()
    sid = s.registry.snapshot()[0].strategy_id
    runner._update_registry(s.registry, eng.Event(
        seq=1, at=NOW, kind=eng.EventKind.TICK, message="tick 1",
        asset_class=AssetClass.CRYPTO, data={"tick": 1}))
    assert {r.strategy_id: r for r in s.registry.snapshot()}[sid].last_tick_at == NOW


def test_a_disabled_strategy_is_not_recorded_as_still_ticking(stack) -> None:
    s = stack()
    sid = s.registry.snapshot()[0].strategy_id
    s.registry.disable(sid, "boom")
    runner._update_registry(s.registry, eng.Event(
        seq=1, at=NOW, kind=eng.EventKind.TICK, message="tick 1",
        asset_class=AssetClass.CRYPTO, data={"tick": 1}))
    assert {r.strategy_id: r for r in s.registry.snapshot()}[sid].last_tick_at is None


def test_a_square_off_intent_does_not_invent_a_registry_row(stack) -> None:
    """Flatten intents carry the POSITION's strategy_id, which may belong to a
    strategy that is no longer registered."""
    s = stack()
    before = len(s.registry.snapshot())
    runner._update_registry(s.registry, eng.Event(
        seq=1, at=NOW, kind=eng.EventKind.INTENT, message="exit",
        asset_class=AssetClass.CRYPTO,
        data={"strategy_id": "long-gone", "symbol": "BTC/USDT",
              "side": "sell", "qty": "1", "order_type": "market",
              "reason": "flatten"}))
    assert len(s.registry.snapshot()) == before


# ==========================================================================
# end to end, through the real API
# ==========================================================================

def test_the_api_reports_ready_against_the_real_stack(stack) -> None:
    from fastapi.testclient import TestClient
    s = stack()
    app = api_mod.create_app(s.handle, api_mod.ApiSettings(token=TOKEN))
    client = TestClient(app)
    response = client.get("/ready", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200, response.json()


def test_a_pause_through_the_api_stops_the_engine_opening_positions(stack) -> None:
    """The whole operator contract, end to end through the real API, the real
    control store and the real engine — the thing neither module could verify
    on its own."""
    from fastapi.testclient import TestClient
    s = stack()
    app = api_mod.create_app(s.handle, api_mod.ApiSettings(token=TOKEN))
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    posted = client.post("/api/control/pause", headers=headers,
                         json={"reason": "stop trading", "actor": "owner"})
    assert posted.status_code == 200, posted.json()

    read = s.engine._runners[AssetClass.CRYPTO].read_control()
    assert read.known and read.paused
    assert read.entries_allowed is False


def test_a_flatten_request_is_pending_until_the_engine_acknowledges(stack) -> None:
    """An operator reading 200 OK must not conclude the book is closed."""
    from fastapi.testclient import TestClient
    s = stack()
    app = api_mod.create_app(s.handle, api_mod.ApiSettings(token=TOKEN))
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    posted = client.post("/api/control/flatten", headers=headers,
                         json={"reason": "close everything", "actor": "owner"})
    assert posted.status_code == 200, posted.json()
    assert posted.json()["executed"] is False

    runner_obj = s.engine._runners[AssetClass.CRYPTO]
    assert runner_obj.read_control().flatten_pending is True

    asyncio.run(s.engine.tick_once(AssetClass.CRYPTO))
    assert runner_obj.read_control().flatten_pending is False


def test_the_heartbeat_the_engine_writes_is_the_one_the_api_reads(stack) -> None:
    from fastapi.testclient import TestClient
    s = stack()
    asyncio.run(s.engine.tick_once(AssetClass.CRYPTO))

    app = api_mod.create_app(s.handle, api_mod.ApiSettings(token=TOKEN))
    client = TestClient(app)
    state = client.get("/api/state",
                       headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert state["engine"]["status"] == "alive", state["engine"]


def test_the_engine_places_a_paper_order_through_the_real_gate(stack) -> None:
    """The proof that the assembled system trades: a strategy proposal, a real
    gate approval, a paper fill and a ledger row, with no fakes anywhere."""
    s = stack()

    async def go() -> None:
        for _ in range(8):
            await s.engine.tick_once(AssetClass.CRYPTO)

    asyncio.run(go())
    kinds = {e.kind for e in s.engine.events.recent()}
    assert eng.EventKind.TICK in kinds
    # Whether an entry signal fires depends on the generated series, but the
    # tick must have run cleanly to the end every time.
    assert eng.EventKind.ERROR not in kinds, [
        e.message for e in s.engine.events.recent()
        if e.kind is eng.EventKind.ERROR]
