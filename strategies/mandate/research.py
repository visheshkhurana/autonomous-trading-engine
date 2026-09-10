"""Paper/research-only extras hook for `crypto_mandate` scores.

Kraken v1 reuses the same regime+score path as OKX (`score_plan` /
`annotate_from_extras`). A later research or scrape module may publish
into these extra keys. This file does not fetch, scrape, or place. It
is not a live order trigger. `KRAKEN_LIVE` is not read here.
"""

from __future__ import annotations

from typing import Mapping

# Keys `score_plan` already understands. A future paper research source
# fills these; absent keys score as "unknown" and do not invent edge.
SCORE_EXTRA_KEYS: tuple[str, ...] = (
    "funding",
    "oi_change",
    "liquidations",
    "cross_agree",
    "skew",
    "iv",
)

RESEARCH_KEY = "mandate_research"


def research_extras(
    existing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return research extras to merge into ctx.extra.

    Today this is empty on purpose. Passing `existing` through would make
    a missing scrape look like a filled score. A future paper-only
    collector can replace this body; it must still not place orders.
    """
    del existing
    return {}
