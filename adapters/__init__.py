"""Broker adapters — the only code in this package that talks to a venue.

Only the PAPER adapter is re-exported. `DhanAdapter` and `OkxAdapter` must be
imported from their own modules:

    from adapters.dhan import DhanAdapter
    from adapters.okx import OkxAdapter

That is not tidiness. Both live adapters are constructed with credentials and,
once armed, place real orders; importing one should be a line a reviewer can
find with grep and a decision someone made on purpose, not something that
happens as a side effect of `import adapters`. Neither module reads a
credential or opens a connection at import time, and `adapters.okx` does not
even import ccxt — the exchange is injected — so this file loading them would
be harmless. Their absence from it is a signpost, not a guard.
"""

from .paper import (
    CryptoTakerFee, DefaultFeeModel, FeeModel, InMemoryQuoteSource,
    IndianChargeStack, PaperAdapter, Quote, QuoteSource, SlippageModel,
)

__all__ = [
    "CryptoTakerFee", "DefaultFeeModel", "FeeModel", "InMemoryQuoteSource",
    "IndianChargeStack", "PaperAdapter", "Quote", "QuoteSource", "SlippageModel",
]
