"""Kraken venue selection and arming. Default path must stay OKX.

No test here sets a Fly secret. KRAKEN_LIVE stays off unless a test names it.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

import runner
from adapters.kraken import KrakenAdapter
from adapters.okx import OkxAdapter
from adapters.paper import PaperAdapter
from core.types import AssetClass, Mode
from tests.test_runner import _FakePublicOkx, _SpyTradeExchange, base_env

NOW = dt.datetime(2026, 3, 2, 12, 0, tzinfo=dt.timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def test_kraken_live_defaults_off() -> None:
    assert runner.parse_kraken_live(None) is False
    assert runner.parse_kraken_live("") is False
    assert runner.parse_kraken_live("  ") is False
    assert runner.parse_kraken_live("0") is False
    assert runner.parse_kraken_live("off") is False
    assert runner.parse_kraken_live("1") is True
    assert runner.parse_kraken_live("true") is True


def test_unrecognised_kraken_live_raises() -> None:
    with pytest.raises(ValueError, match="KRAKEN_LIVE"):
        runner.parse_kraken_live("maybe")
    with pytest.raises(ValueError, match="KRAKEN_LIVE"):
        runner.parse_kraken_live("ture")


def test_crypto_venue_defaults_to_okx() -> None:
    assert runner.parse_crypto_venue(None) == "okx"
    assert runner.parse_crypto_venue("") == "okx"
    assert runner.parse_crypto_venue("OKX") == "okx"
    assert runner.parse_crypto_venue("kraken") == "kraken"


def test_unrecognised_crypto_venue_raises() -> None:
    with pytest.raises(ValueError, match="TRADER_CRYPTO_VENUE"):
        runner.parse_crypto_venue("binance")


def test_no_new_env_crypto_still_resolves_to_okx_adapter(tmp_path) -> None:
    """Regression guard for the live OKX book. Unset Kraken vars = OkxAdapter."""
    env = base_env(tmp_path)
    assert "TRADER_CRYPTO_VENUE" not in env
    assert "KRAKEN_LIVE" not in env
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        crypto = s.engine._runners[AssetClass.CRYPTO]
        assert isinstance(crypto.wiring.live_adapter, OkxAdapter)
        assert not isinstance(crypto.wiring.live_adapter, KrakenAdapter)
        assert isinstance(crypto.adapter, PaperAdapter)
        assert crypto.mode is Mode.PAPER
    finally:
        s.close()
        s.store.close()


def test_kraken_venue_with_flag_off_and_no_secrets_is_paper(tmp_path) -> None:
    """Absent credentials + KRAKEN_LIVE off → PaperAdapter. Say so at boot."""
    env = base_env(tmp_path, TRADER_CRYPTO_VENUE="kraken")
    assert "KRAKEN_API_KEY" not in env
    assert "KRAKEN_LIVE" not in env
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        crypto = s.engine._runners[AssetClass.CRYPTO]
        assert isinstance(crypto.adapter, PaperAdapter)
        assert isinstance(crypto.wiring.live_adapter, KrakenAdapter)
        assert crypto.mode is Mode.PAPER
        assert s.handle.armed[AssetClass.CRYPTO] is False
        report = s.engine.arm_report_text()
        assert "CRYPTO_LIVE=disarmed" in report
        assert "PaperAdapter" in report
    finally:
        s.close()
        s.store.close()


def test_crypto_live_on_okx_book_does_not_become_kraken(tmp_path) -> None:
    """CRYPTO_LIVE still arms OKX when no venue is named."""
    spy = _SpyTradeExchange()
    env = base_env(tmp_path, TRADER_MODE="live", CRYPTO_LIVE="1")
    s = runner.build_stack(
        env=env, clock=lambda: NOW,
        okx_exchange=_FakePublicOkx(),
        okx_trade_exchange=spy,
    )
    try:
        crypto = s.engine._runners[AssetClass.CRYPTO]
        assert crypto.mode is Mode.LIVE
        assert isinstance(crypto.adapter, OkxAdapter)
        assert not isinstance(crypto.adapter, KrakenAdapter)
    finally:
        s.close()
        s.store.close()


def test_kraken_does_not_inherit_crypto_live_without_kraken_live(tmp_path) -> None:
    """CRYPTO_LIVE arms OKX. Switching venue without KRAKEN_LIVE fails closed."""
    env = base_env(
        tmp_path,
        TRADER_MODE="live",
        CRYPTO_LIVE="1",
        TRADER_CRYPTO_VENUE="kraken",
    )
    with pytest.raises(RuntimeError, match="KRAKEN_LIVE"):
        runner.build_stack(env=env, clock=lambda: NOW)


def test_kraken_armed_without_secrets_refuses_to_boot(tmp_path) -> None:
    env = base_env(
        tmp_path,
        TRADER_MODE="live",
        CRYPTO_LIVE="1",
        TRADER_CRYPTO_VENUE="kraken",
        KRAKEN_LIVE="1",
    )
    with pytest.raises(RuntimeError, match="KRAKEN_API_KEY"):
        runner.build_stack(env=env, clock=lambda: NOW)


def test_kraken_goes_live_only_with_venue_and_both_arm_flags(tmp_path) -> None:
    """Owner path later: all gates set, fakes injected, no real order."""
    spy = _SpyTradeExchange()
    env = base_env(
        tmp_path,
        TRADER_MODE="live",
        CRYPTO_LIVE="1",
        TRADER_CRYPTO_VENUE="kraken",
        KRAKEN_LIVE="1",
    )
    s = runner.build_stack(
        env=env, clock=lambda: NOW,
        okx_exchange=_FakePublicOkx(),
        kraken_trade_exchange=spy,
    )
    try:
        crypto = s.engine._runners[AssetClass.CRYPTO]
        assert crypto.mode is Mode.LIVE
        assert isinstance(crypto.adapter, KrakenAdapter)
        assert crypto.adapter is crypto.wiring.live_adapter
        assert spy.calls == []
    finally:
        s.close()
        s.store.close()


def test_fly_toml_carries_no_kraken_arming_flag() -> None:
    if not (ROOT / "fly.toml").exists():
        pytest.skip("fly.toml is deployment config and is not part of this checkout")
    text = (ROOT / "fly.toml").read_text()
    env = __import__("tomllib").loads(text)["env"]
    assert "KRAKEN_LIVE" not in env
    assert "TRADER_CRYPTO_VENUE" not in env
    assert "KRAKEN_API_KEY" not in env
    assert "KRAKEN_API_SECRET" not in env
    env_block = text.split("[env]")[1].split("[mounts]")[0]
    for line in env_block.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        key = stripped.split("=")[0].strip()
        assert key != "KRAKEN_LIVE"
        assert key != "TRADER_CRYPTO_VENUE"


def test_default_feed_is_not_kraken() -> None:
    assert runner.RunnerConfig().feed is runner.FeedChoice.SYNTHETIC
    cfg = runner.RunnerConfig.from_env({"TRADER_FEED": "okx"})
    assert cfg.feed is runner.FeedChoice.OKX
    cfg_k = runner.RunnerConfig.from_env({"TRADER_FEED": "kraken"})
    assert cfg_k.feed is runner.FeedChoice.KRAKEN


def _kraken_public() -> object:
    from tests.test_okx_feed import DEFAULT_MARKETS, FakeExchange, spot_market
    markets = dict(DEFAULT_MARKETS)
    markets["SOL/USDT"] = spot_market("SOL/USDT")
    return FakeExchange(markets=markets)


def test_naming_kraken_venue_couples_the_public_feed(tmp_path) -> None:
    """One knob: TRADER_CRYPTO_VENUE=kraken selects KrakenFeed even if
    TRADER_FEED is still the paper default (synthetic) or fly's okx."""
    from feeds.kraken_feed import KrakenFeed
    from feeds.okx_feed import OkxFeed

    env = base_env(tmp_path, TRADER_CRYPTO_VENUE="kraken", TRADER_FEED="okx")
    assert "KRAKEN_LIVE" not in env
    s = runner.build_stack(
        env=env, clock=lambda: NOW, kraken_exchange=_kraken_public(),
    )
    try:
        crypto = s.engine._runners[AssetClass.CRYPTO]
        assert isinstance(crypto.adapter, PaperAdapter)
        assert isinstance(crypto.wiring.live_adapter, KrakenAdapter)
        assert crypto.mode is Mode.PAPER
        assert isinstance(s.feed.routes[AssetClass.CRYPTO], KrakenFeed)
        assert not isinstance(s.feed.routes[AssetClass.CRYPTO], OkxFeed)
        assert any(r.source == "kraken" for r in s.feed_report
                   if r.asset_class is AssetClass.CRYPTO)
    finally:
        s.close()
        s.store.close()


def test_unset_venue_does_not_retarget_the_okx_feed(tmp_path) -> None:
    """OKX live path: no TRADER_CRYPTO_VENUE keeps OkxFeed when asked."""
    from feeds.okx_feed import OkxFeed

    env = base_env(tmp_path, TRADER_FEED="okx")
    assert "TRADER_CRYPTO_VENUE" not in env
    s = runner.build_stack(env=env, clock=lambda: NOW)
    try:
        crypto = s.engine._runners[AssetClass.CRYPTO]
        assert isinstance(crypto.wiring.live_adapter, OkxAdapter)
        assert isinstance(s.feed.routes[AssetClass.CRYPTO], OkxFeed)
    finally:
        s.close()
        s.store.close()


def test_ready_surfaces_venue_live_adapter_flag_and_feed(tmp_path) -> None:
    from fastapi.testclient import TestClient
    from tests.test_runner import TOKEN

    env = base_env(tmp_path, TRADER_CRYPTO_VENUE="kraken")
    s = runner.build_stack(
        env=env, clock=lambda: NOW, kraken_exchange=_kraken_public(),
    )
    try:
        from api import server as api_mod
        app = api_mod.create_app(s.handle, api_mod.ApiSettings(token=TOKEN))
        body = TestClient(app).get(
            "/ready", headers={"Authorization": f"Bearer {TOKEN}"},
        ).json()
        assert body["ready"] is True
        assert body["crypto_venue"] == "kraken"
        assert body["kraken_live"] is False
        assert body["feed"] == "kraken"
        crypto = body["classes"]["crypto"]
        assert crypto["armed_live"] is False
        assert crypto["adapter"] == "PaperAdapter"
        assert crypto["live_adapter"] == "KrakenAdapter"
        assert crypto["feed"] == "kraken"
    finally:
        s.close()
        s.store.close()


def test_status_flag_prints_the_same_disarmed_shape(tmp_path, capsys) -> None:
    env = base_env(tmp_path, TRADER_CRYPTO_VENUE="kraken")
    payload = runner.status_report(
        env=env, clock=lambda: NOW, kraken_exchange=_kraken_public(),
    )
    assert payload["crypto_venue"] == "kraken"
    assert payload["kraken_live"] is False
    assert payload["feed"] == "kraken"
    assert payload["armed_live"] is False
    assert payload["adapter"] == "PaperAdapter"
    assert payload["live_adapter"] == "KrakenAdapter"
    assert payload["signal"] == "crypto_mandate"


def test_kraken_paper_reuses_crypto_mandate_not_a_new_scraper(tmp_path) -> None:
    env = base_env(tmp_path, TRADER_CRYPTO_VENUE="kraken")
    s = runner.build_stack(
        env=env, clock=lambda: NOW, kraken_exchange=_kraken_public(),
    )
    try:
        ids = [st.id for st in s.engine._runners[AssetClass.CRYPTO].wiring.strategies]
        assert "crypto_mandate" in ids
        assert not any("scraper" in i for i in ids)
        limits = runner.CRYPTO_CLASS_LIMITS
        assert limits.max_order_notional == runner.CRYPTO_CLIP_USDT == Decimal("100")
        assert limits.max_daily_notional == runner.CRYPTO_DAILY_USDT == Decimal("500")
        assert limits.max_open_positions == 1
        # Sleeve A numbers stay documentation for a later arm. The 1000
        # sleeve is not the clip. Max open stays 1.
        assert runner.KRAKEN_SLEEVE_A_USDT == Decimal("1000")
        assert runner.KRAKEN_SLEEVE_A_CLIP_USDT == Decimal("100")
        assert runner.KRAKEN_SLEEVE_A_DAILY_USDT == Decimal("500")
        assert runner.KRAKEN_SLEEVE_A_MAX_OPEN == 1
        assert limits.max_order_notional != runner.KRAKEN_SLEEVE_A_USDT
        assert limits.max_open_positions == runner.KRAKEN_SLEEVE_A_MAX_OPEN
        assert runner.parse_kraken_live(None) is False
        from strategies.mandate.constants import CONSECUTIVE_LOSS_STOP
        from strategies.mandate.risk import new_directional_blocked
        from strategies.mandate.types import MandateState
        assert CONSECUTIVE_LOSS_STOP == 3
        assert new_directional_blocked(
            MandateState(consecutive_losses=CONSECUTIVE_LOSS_STOP)
        )
        from strategies.mandate.research import research_extras
        assert research_extras() == {}
        assert research_extras({"funding": Decimal("-0.0001")}) == {}
    finally:
        s.close()
        s.store.close()


def test_research_stub_is_not_a_live_trigger() -> None:
    """Future scrape feeds scores via extras. It must not place or fetch."""
    from pathlib import Path
    from strategies.mandate.research import SCORE_EXTRA_KEYS, research_extras

    text = (
        Path(__file__).resolve().parents[1]
        / "strategies" / "mandate" / "research.py"
    ).read_text()
    for banned in ("urlopen", "requests", "http://", "https://", "create_order"):
        assert banned not in text, banned
    assert research_extras() == {}
    assert "funding" in SCORE_EXTRA_KEYS
    assert runner.parse_kraken_live(None) is False
