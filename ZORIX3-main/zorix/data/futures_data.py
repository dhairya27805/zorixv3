"""
zorix/data/futures_data.py

Futures data provider abstraction (Section 14/39) — same pattern as
options_data.py. yfinance has no reliable spot-vs-future basis or OI data
for NSE F&O, so this defaults to "unavailable" until a real provider is
wired into CustomFuturesProvider.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

FUTURES_POSITIONING_RULES = {
    # (price_up, oi_up) -> (label, interpretation)
    (True, True):   ("LONG BUILDUP", "Bullish futures positioning"),
    (False, True):  ("SHORT BUILDUP", "Bearish futures positioning"),
    (True, False):  ("SHORT COVERING", "Bullish reversal signal"),
    (False, False): ("LONG UNWINDING", "Bearish exit signal"),
}


@dataclass
class FuturesSnapshot:
    symbol: str
    spot: Optional[float]
    future: Optional[float]
    basis: Optional[float]
    basis_pct: Optional[float]
    open_interest: Optional[float]
    oi_change_pct: Optional[float]
    price_change_pct: Optional[float]
    days_to_expiry: Optional[int]
    positioning: Optional[str]
    interpretation: Optional[str]
    available: bool = True
    unavailable_reason: Optional[str] = None


def classify_positioning(price_change_pct: float, oi_change_pct: float) -> tuple[str, str]:
    """Section 14: Long Buildup / Short Buildup / Short Covering / Long Unwinding.

    This is a descriptive classification of observed price+OI behaviour,
    not a prediction — callers must present it as an additional signal,
    never as a guaranteed forecast.
    """
    price_up = price_change_pct >= 0
    oi_up = oi_change_pct >= 0
    return FUTURES_POSITIONING_RULES[(price_up, oi_up)]


class FuturesDataProvider(abc.ABC):
    @abc.abstractmethod
    def get_snapshot(self, symbol: str) -> FuturesSnapshot:
        raise NotImplementedError


class NullFuturesProvider(FuturesDataProvider):
    def get_snapshot(self, symbol: str) -> FuturesSnapshot:
        return FuturesSnapshot(
            symbol=symbol, spot=None, future=None, basis=None, basis_pct=None,
            open_interest=None, oi_change_pct=None, price_change_pct=None,
            days_to_expiry=None, positioning=None, interpretation=None,
            available=False,
            unavailable_reason=(
                "No futures data provider configured. Implement "
                "CustomFuturesProvider in zorix/data/futures_data.py "
                "against your F&O data API."
            ),
        )


class CustomFuturesProvider(FuturesDataProvider):
    """Template — wire in your F&O API's real calls."""

    def __init__(self, api_client=None):
        self.client = api_client

    def _fetch_raw(self, symbol: str) -> Optional[dict]:
        # TODO: replace with a real call, returning at minimum:
        #   {"spot": ..., "future": ..., "open_interest": ...,
        #    "prev_open_interest": ..., "prev_future": ..., "expiry": "YYYY-MM-DD"}
        return None

    def get_snapshot(self, symbol: str) -> FuturesSnapshot:
        raw = self._fetch_raw(symbol)
        if raw is None:
            return FuturesSnapshot(
                symbol=symbol, spot=None, future=None, basis=None, basis_pct=None,
                open_interest=None, oi_change_pct=None, price_change_pct=None,
                days_to_expiry=None, positioning=None, interpretation=None,
                available=False, unavailable_reason="Provider returned no data.",
            )

        spot, future = raw["spot"], raw["future"]
        basis = future - spot
        basis_pct = (basis / spot) * 100 if spot else None

        price_change_pct = (
            (future - raw["prev_future"]) / raw["prev_future"] * 100
            if raw.get("prev_future") else 0.0
        )
        oi_change_pct = (
            (raw["open_interest"] - raw["prev_open_interest"]) / raw["prev_open_interest"] * 100
            if raw.get("prev_open_interest") else 0.0
        )
        label, interp = classify_positioning(price_change_pct, oi_change_pct)

        days_to_expiry = None
        if raw.get("expiry"):
            import pandas as pd
            days_to_expiry = (pd.Timestamp(raw["expiry"]) - pd.Timestamp.now()).days

        return FuturesSnapshot(
            symbol=symbol, spot=spot, future=future, basis=basis, basis_pct=basis_pct,
            open_interest=raw.get("open_interest"), oi_change_pct=oi_change_pct,
            price_change_pct=price_change_pct, days_to_expiry=days_to_expiry,
            positioning=label, interpretation=interp, available=True,
        )
