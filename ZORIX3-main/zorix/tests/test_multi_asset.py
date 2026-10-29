"""
zorix/tests/test_multi_asset.py

Tests for the multi-asset upgrade: centralized Forex/Bonds/Commodities
symbol maps, asset-aware display formatting, yield-scale correction, and
the four-state signal system (BUY/SELL/HOLD/NO TRADE) exercised with
controlled probability inputs per the spec's explicit testing request
(Section 25: "test all four signal states using controlled probability
inputs").

No network required — run with: pytest zorix/tests/test_multi_asset.py -v
"""

import numpy as np
import pandas as pd

from zorix.config.settings import DEFAULT_CONFIG
from zorix.data.market_data import apply_yield_scale
from zorix.signals.signal_engine import decide_signal
from zorix.utils.display import (describe_asset, equity_instrument_meta,
                                  format_instrument_value)


# ---- Centralized symbol maps: no duplicates, no empty entries ----

def test_forex_pairs_have_unique_tickers():
    cfg = DEFAULT_CONFIG
    tickers = [m.yfinance_symbol for m in cfg.FOREX.pairs.values()]
    assert len(tickers) == len(set(tickers))
    assert len(tickers) >= 4  # USD/INR, EUR/USD, GBP/USD, USD/JPY minimum


def test_bonds_cover_minimum_required_instruments():
    cfg = DEFAULT_CONFIG
    labels = " ".join(cfg.BONDS.instruments.keys())
    assert "10Y" in labels
    assert "30Y" in labels
    assert all(m.kind == "yield" for m in cfg.BONDS.instruments.values())


def test_commodities_meta_covers_minimum_required():
    cfg = DEFAULT_CONFIG
    names = set(cfg.COMMODITIES_META.instruments.keys())
    assert {"Gold", "Silver", "Crude Oil", "Copper"}.issubset(names)


# ---- Yield-scale correction ----

def test_yield_scale_correction_applied_once():
    """^TNX-style tickers report yield*10; a raw 42.1 should become 4.21."""
    df = pd.DataFrame({"Open": [42.0], "High": [42.5], "Low": [41.5], "Close": [42.1], "Volume": [0]})
    corrected = apply_yield_scale(df, yield_scale=10.0)
    assert corrected["Close"].iloc[0] == 4.21
    assert corrected["Open"].iloc[0] == 4.20


def test_yield_scale_noop_when_scale_is_one():
    """^IRX already reports the yield directly — no correction should be applied."""
    df = pd.DataFrame({"Open": [5.28], "High": [5.30], "Low": [5.27], "Close": [5.29], "Volume": [0]})
    unchanged = apply_yield_scale(df, yield_scale=1.0)
    pd.testing.assert_frame_equal(df, unchanged)


def test_display_formatting_does_not_double_correct_yield():
    """format_instrument_value must NOT re-divide by yield_scale — the
    correction happens exactly once, in apply_yield_scale at load time."""
    us10y = DEFAULT_CONFIG.BONDS.instruments["US 10Y Treasury Yield"]
    already_corrected_value = 4.21
    formatted = format_instrument_value(already_corrected_value, us10y)
    assert formatted == "4.21%"  # NOT "0.42%" (would be a double-correction bug)


# ---- Asset-aware display: currency formatting never blindly uses ₹ ----

def test_equity_india_ticker_gets_rupee_prefix():
    meta = equity_instrument_meta("RELIANCE.NS")
    assert meta.currency_prefix == "₹"
    assert format_instrument_value(3412.50, meta) == "₹3,412.50"


def test_equity_us_ticker_gets_no_currency_prefix():
    meta = equity_instrument_meta("AAPL")
    assert meta.currency_prefix == ""
    assert format_instrument_value(227.5, meta) == "227.50"


def test_forex_usd_quoted_pair_shown_as_plain_ratio():
    """EUR/USD should NOT get a ₹ symbol — Section 5's explicit example."""
    eurusd = DEFAULT_CONFIG.FOREX.pairs["EUR/USD"]
    formatted = format_instrument_value(1.1712, eurusd)
    assert "₹" not in formatted
    assert formatted == "1.1712"


def test_forex_usdinr_shown_with_rupee_prefix():
    usdinr = DEFAULT_CONFIG.FOREX.pairs["USD/INR"]
    formatted = format_instrument_value(83.42, usdinr)
    assert formatted == "₹83.42"


# ---- Asset-aware labels & chart type ----

def test_bond_instrument_uses_yield_label_and_line_chart():
    us10y = DEFAULT_CONFIG.BONDS.instruments["US 10Y Treasury Yield"]
    display = describe_asset(us10y)
    assert display.value_label == "Current Yield"
    assert display.chart_type == "line"
    assert "price" not in display.value_label.lower()  # must not mislabel a yield as a price


def test_equity_instrument_uses_price_label_and_candlestick():
    meta = equity_instrument_meta("TCS.NS")
    display = describe_asset(meta)
    assert display.value_label == "Current Price"
    assert display.chart_type == "candlestick"


def test_forex_instrument_uses_exchange_rate_label():
    eurusd = DEFAULT_CONFIG.FOREX.pairs["EUR/USD"]
    display = describe_asset(eurusd)
    assert display.value_label == "Exchange Rate"


# ---- Four-state signal system: controlled probability inputs (Section 25) ----
# proba layout is [SELL, HOLD, BUY]

def test_signal_state_buy_with_controlled_input():
    proba = np.array([0.05, 0.13, 0.82])
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "BUY"
    assert reason is None


def test_signal_state_sell_with_controlled_input():
    proba = np.array([0.80, 0.15, 0.05])
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "SELL"
    assert reason is None


def test_signal_state_hold_with_controlled_input():
    """HOLD is a genuine third predicted class here, not a fallback — the
    model's top class IS HOLD with a real edge over the runner-up."""
    proba = np.array([0.12, 0.68, 0.20])
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "HOLD"
    assert reason is None


def test_signal_state_no_trade_with_controlled_input():
    """NO TRADE fires on insufficient edge even when HOLD is not the top
    class — this is what distinguishes it from HOLD (Section 11)."""
    proba = np.array([0.36, 0.30, 0.34])  # SELL/BUY nearly tied, no real edge
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "NO TRADE"
    assert reason is not None


def test_hold_and_no_trade_are_genuinely_different_states():
    """Regression test for Section 11's core requirement: HOLD and NO TRADE
    must not collapse into the same thing."""
    hold_proba = np.array([0.15, 0.65, 0.20])
    no_trade_proba = np.array([0.34, 0.33, 0.33])
    hold_signal, _, hold_reason = decide_signal(hold_proba, min_edge=0.12, min_confidence=0.45)
    nt_signal, _, nt_reason = decide_signal(no_trade_proba, min_edge=0.12, min_confidence=0.45)

    assert hold_signal == "HOLD"
    assert nt_signal == "NO TRADE"
    assert hold_signal != nt_signal
    assert hold_reason is None          # HOLD needs no "insufficient edge" excuse — it's a real call
    assert nt_reason is not None        # NO TRADE must always explain why it declined to call a side


def test_signal_thresholds_are_configurable_not_hardcoded():
    """The same probabilities should flip from NO TRADE to BUY purely by
    changing config thresholds — proves thresholds aren't hardcoded into
    the decision logic (Section 10)."""
    proba = np.array([0.10, 0.30, 0.60])
    strict_signal, _, _ = decide_signal(proba, min_edge=0.35, min_confidence=0.65)
    loose_signal, _, _ = decide_signal(proba, min_edge=0.10, min_confidence=0.40)
    assert strict_signal == "NO TRADE"
    assert loose_signal == "BUY"
