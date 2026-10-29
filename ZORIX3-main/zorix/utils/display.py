"""
zorix/utils/display.py

Asset-aware display helpers (Sections 7/18/20 of the multi-asset upgrade).
Kept separate from app.py so the "what label/format goes with which asset
class" logic is centralized and unit-testable, per Section 24's
config -> data provider -> feature engine -> model -> signal engine -> UI
layering — this is the last step in that chain, UI formatting only. It
never changes any underlying numbers used for features or models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from zorix.config.settings import InstrumentMeta

_INDIA_SUFFIXES = (".NS", ".BO")


@dataclass
class AssetDisplay:
    value_label: str          # "Current Price" | "Exchange Rate" | "Current Yield"
    change_label: str         # "Daily Return" | "Yield Change"
    volatility_label: str     # "Volatility" | "Yield Volatility"
    chart_type: str           # "candlestick" | "line"
    formatted_value: str


def equity_instrument_meta(ticker: str) -> InstrumentMeta:
    """Equities aren't in a fixed symbol map (free-text ticker entry), so
    their InstrumentMeta is derived from the ticker suffix rather than
    looked up — ₹ for NSE/BSE-listed tickers, no symbol otherwise, matching
    Section 7's "don't blindly display ₹ for every asset" instruction.
    """
    prefix = "₹" if ticker.upper().endswith(_INDIA_SUFFIXES) else ""
    return InstrumentMeta(yfinance_symbol=ticker, kind="price", currency_prefix=prefix)


def format_instrument_value(value: float, meta: InstrumentMeta) -> str:
    """Formats an already-scale-corrected value (yield_scale correction
    happens once, at load time, via data.market_data.apply_yield_scale —
    this function must NOT re-divide by yield_scale, or a yield would be
    corrected twice, e.g. 4.21% would incorrectly render as 0.42%).
    """
    if meta.kind == "yield":
        formatted = f"{value:,.2f}"
    elif meta.kind == "rate" and not meta.currency_prefix:
        # Conventionally-quoted USD pairs (EUR/USD, GBP/USD) use 4 decimals.
        formatted = f"{value:,.4f}"
    else:
        formatted = f"{value:,.2f}"
    return f"{meta.currency_prefix}{formatted}{meta.currency_suffix}"


def describe_asset(meta: InstrumentMeta) -> AssetDisplay:
    """Section 7/20: the label set and chart type change by instrument kind,
    not asset class name — a commodity and an equity both use 'Current
    Price' / candlesticks; a yield instrument does not.
    """
    if meta.kind == "yield":
        return AssetDisplay(
            value_label="Current Yield", change_label="Yield Change",
            volatility_label="Yield Volatility", chart_type="line", formatted_value="",
        )
    if meta.kind == "rate":
        return AssetDisplay(
            value_label="Exchange Rate", change_label="Daily Return",
            volatility_label="Volatility", chart_type="candlestick" if meta.ohlc_meaningful else "line",
            formatted_value="",
        )
    return AssetDisplay(
        value_label="Current Price", change_label="Daily Return",
        volatility_label="Volatility", chart_type="candlestick" if meta.ohlc_meaningful else "line",
        formatted_value="",
    )
