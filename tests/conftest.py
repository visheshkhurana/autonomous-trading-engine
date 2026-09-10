"""Pytest configuration for the trader test suite.

`network` marks the handful of tests that hit a LIVE PUBLIC endpoint (OKX
market data, no credentials). They are deselected by default with
`-m "not network"`; registering the marker here is what stops every one of them
emitting a PytestUnknownMarkWarning, which is how a real warning goes unread.
"""

from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "network: hits a live public endpoint")
