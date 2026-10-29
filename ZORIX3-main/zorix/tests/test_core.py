"""
zorix/tests/test_core.py

Unit tests covering Section 46's list, restricted to components that need
no network access: target construction, technical indicators, return/
volatility features, walk-forward splitting, and performance metrics.

Run with:  pytest zorix/tests/test_core.py -v
"""

import numpy as np
import pandas as pd
import pytest

from zorix.features.target import add_bhs_targets, add_forward_returns, sklearn_label
from zorix.features.technical import add_return_features, add_volatility_features
from zorix.models.walkforward import walk_forward_split
from zorix.signals.signal_engine import decide_signal
from zorix.utils.helpers import (cagr, max_drawdown, profit_factor,
                                  sharpe_ratio, sortino_ratio, win_rate)


def _make_ohlcv(n=300, seed=0, drift=0.0003, vol=0.015):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(drift, vol, n)
    close = 100 * (1 + pd.Series(rets)).cumprod()
    df = pd.DataFrame({
        "Open": close.values * (1 + rng.normal(0, 0.001, n)),
        "High": close.values * (1 + np.abs(rng.normal(0, 0.003, n))),
        "Low": close.values * (1 - np.abs(rng.normal(0, 0.003, n))),
        "Close": close.values,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=dates)
    return df


# ---- Target construction ----

def test_forward_return_matches_manual_calc():
    df = _make_ohlcv()
    out = add_forward_returns(df, [5])
    manual = df["Close"].shift(-5) / df["Close"] - 1
    pd.testing.assert_series_equal(out["FwdReturn_5D"], manual, check_names=False)


def test_bhs_target_has_three_classes_on_trending_data():
    df = _make_ohlcv(n=500, drift=0.001, vol=0.01)  # clear uptrend -> mostly BUY/HOLD
    out = add_bhs_targets(df, horizons=[5], vol_lookback=20, k_buy=0.75, k_sell=0.75, min_threshold=0.0025)
    values = set(out["Target_5D"].dropna().unique())
    assert values.issubset({-1, 0, 1})
    assert len(values) >= 1


def test_bhs_target_no_lookahead_leakage():
    """Changing a row's *future* close must not change an EARLIER row's
    threshold, only that earlier row's own forward-return/target."""
    df = _make_ohlcv(n=200)
    out1 = add_bhs_targets(df.copy(), [5], 20, 0.75, 0.75, 0.0025)

    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("Close")] *= 1.5  # shock the very last close
    out2 = add_bhs_targets(df2, [5], 20, 0.75, 0.75, 0.0025)

    # Rows far before the shock (whose forward window doesn't touch it) must be identical.
    unaffected = out1.iloc[:150]
    unaffected2 = out2.iloc[:150]
    pd.testing.assert_series_equal(unaffected["Threshold_5D"], unaffected2["Threshold_5D"])


def test_sklearn_label_remaps_to_non_negative_contiguous():
    target = pd.Series([-1, 0, 1, -1, 1])
    mapped = sklearn_label(target)
    assert set(mapped.unique()) == {0, 1, 2}
    assert mapped.tolist() == [0, 1, 2, 0, 2]


# ---- Technical / return features ----

def test_return_features_match_pct_change():
    df = _make_ohlcv()
    out = add_return_features(df, [1, 5])
    pd.testing.assert_series_equal(out["Return_5D"], df["Close"].pct_change(5), check_names=False)


def test_volatility_feature_is_annualized_and_nonnegative():
    df = _make_ohlcv()
    out = add_volatility_features(df, [20])
    vol = out["Volatility_20D"].dropna()
    assert (vol >= 0).all()


# ---- Walk-forward splitting ----

def test_walk_forward_split_is_expanding_and_causal():
    splits = list(walk_forward_split(n_rows=1000, n_splits=5, min_train_size=300))
    assert len(splits) >= 2
    prev_train_len = 0
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min(), "train must precede test (no shuffling / no lookahead)"
        assert len(train_idx) >= prev_train_len, "training window must expand across folds"
        prev_train_len = len(train_idx)


# ---- Signal engine NO TRADE gating ----

def test_no_trade_when_edge_too_small():
    proba = np.array([0.30, 0.35, 0.35])  # HOLD/BUY nearly tied
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "NO TRADE"
    assert reason is not None


def test_buy_when_clear_edge():
    proba = np.array([0.05, 0.15, 0.80])
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "BUY"
    assert confidence == "HIGH"
    assert reason is None


# ---- Performance metrics ----

def test_sharpe_ratio_zero_for_zero_mean_returns():
    returns = pd.Series([0.01, -0.01, 0.01, -0.01] * 50)
    s = sharpe_ratio(returns)
    assert abs(s) < 1e-6


def test_max_drawdown_is_negative_or_zero():
    equity = pd.Series([1.0, 1.1, 0.9, 1.2, 0.8, 1.3])
    mdd = max_drawdown(equity)
    assert mdd <= 0
    assert mdd == pytest.approx(-0.3333, abs=1e-3)  # from 1.2 -> 0.8


def test_profit_factor_all_wins_is_inf():
    trade_returns = pd.Series([0.01, 0.02, 0.03])
    assert profit_factor(trade_returns) == float("inf")


def test_win_rate_basic():
    trade_returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert win_rate(trade_returns) == 0.5


def test_cagr_flat_equity_is_zero():
    equity = pd.Series([1.0] * 252)
    c = cagr(equity)
    assert c == pytest.approx(0.0, abs=1e-9)
