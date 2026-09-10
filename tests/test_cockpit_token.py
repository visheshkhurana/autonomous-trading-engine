"""The cockpit must never ship the engine token to the browser."""

from __future__ import annotations

import pytest

from pathlib import Path

VERCEL = Path(__file__).resolve().parents[2] / "vercel"

if not VERCEL.exists():
    pytest.skip("the Vercel cockpit is a sibling app and is not part of this checkout",
                allow_module_level=True)


def test_engine_token_is_not_a_next_public_var() -> None:
    forbidden = (
        "NEXT_PUBLIC_TRADER_ENGINE_TOKEN",
        "NEXT_PUBLIC_ENGINE_API_TOKEN",
        "NEXT_PUBLIC_TRADER_ENGINE_URL",
    )
    hits: list[str] = []
    for path in VERCEL.rglob("*"):
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx", ".mjs"}:
            continue
        if "node_modules" in path.parts:
            continue
        text = path.read_text(errors="replace")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.relative_to(VERCEL)}: {token}")
    assert hits == [], (
        "engine credentials must stay server-side; found " + ", ".join(hits)
    )


def test_proxy_reads_the_token_from_the_server_env() -> None:
    route = VERCEL / "app" / "api" / "trader" / "[...path]" / "route.ts"
    text = route.read_text()
    assert "process.env.TRADER_ENGINE_TOKEN" in text
    assert "process.env.NEXT_PUBLIC" not in text
    assert "NEXT_PUBLIC_TRADER_ENGINE_TOKEN" not in text
    assert "api/positions" in text
    assert "api/orders" in text


def test_cockpit_page_renders_unknown_not_zero() -> None:
    page = (VERCEL / "app" / "trader" / "page.tsx").read_text()
    assert "fmtPrice" in page
    assert "Positions unavailable" in page
    assert "Orders unavailable" in page
    lib = (VERCEL / "lib" / "trader.ts").read_text()
    assert 'return "unknown"' in lib
