"""
zorix/data/market_data.py

Data-provider abstraction for OHLCV market data (Section 39 of the spec).

    MarketDataProvider (abstract)
        └── YahooFinanceProvider

Adding a second provider (NSE, a broker API, etc.) later means writing one
more class that implements the same three methods — nothing else in the
codebase needs to change.
"""

from __future__ import annotations

import abc
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("zorix.data.market_data")


class DataUnavailableError(Exception):
    """Raised when a provider cannot return real data for a symbol.

    Callers must catch this and show 'data unavailable' in the UI —
    never fall back to synthetic/fabricated data (Section 3, Section 43).
    """


class MarketDataProvider(abc.ABC):
    """Abstract interface every market-data source must implement."""

    @abc.abstractmethod
    def fetch_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return a DataFrame indexed by date with columns
        Open, High, Low, Close, Volume. Raises DataUnavailableError on failure.
        """
        raise NotImplementedError

    def fetch_many(self, symbols: Dict[str, str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Fetch several symbols; failures are logged and skipped, not faked."""
        out: Dict[str, pd.DataFrame] = {}
        for name, ticker in symbols.items():
            try:
                out[name] = self.fetch_ohlcv(ticker, start, end)
            except DataUnavailableError as exc:
                logger.warning("Skipping %s (%s): %s", name, ticker, exc)
        return out


class YahooFinanceProvider(MarketDataProvider):
    """yfinance-backed provider. Free, no API key, no reliable historical
    options/F&O data — see data/options_data.py and data/futures_data.py
    for why those need a separate provider.
    """

    def fetch_ohlcv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
            from curl_cffi import requests as curl_requests
        except ImportError as exc:
            raise DataUnavailableError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from exc

        try:
            session = curl_requests.Session(impersonate="chrome")
            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, session=session)
        except Exception as exc:  # network errors, rate limits, etc.
            raise DataUnavailableError(f"yfinance download failed for {symbol}: {exc}") from exc

        if df is None or df.empty:
            raise DataUnavailableError(f"No data returned for {symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df


def apply_yield_scale(df: pd.DataFrame, yield_scale: float) -> pd.DataFrame:
    """Corrects Yahoo Finance treasury-yield tickers (^TNX, ^TYX, ^FVX) that
    report the value as yield*10 (e.g. 42.1 for a 4.21% yield). Applied once
    at load time so every downstream feature/model/UI consumer sees the
    actual yield, not a scaled proxy — never re-derived ad hoc in the UI.
    No-op when yield_scale == 1.0 (e.g. ^IRX, which already reports directly,
    or any ordinary price instrument).
    """
    if yield_scale == 1.0:
        return df
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col] / yield_scale
    return out


def align_to_trading_calendar(base: pd.DataFrame, other: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Reindex `other` onto `base`'s trading dates using forward-fill (never
    backward-fill — that would leak future values into past rows), then
    prefix its columns and merge into `base`.
    """
    aligned = other.reindex(base.index, method="ffill")
    aligned = aligned.add_prefix(f"{prefix}_")
    return base.join(aligned, how="left")


def fetch_cross_asset_frame(
    provider: MarketDataProvider,
    symbols: Dict[str, str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """Thin wrapper kept separate from Config so it's independently testable."""
    return provider.fetch_many(symbols, start, end)
