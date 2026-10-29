"""
zorix/features/target.py

Prediction target construction (Section 4): multi-horizon forward returns,
converted into a volatility-adjusted BUY / HOLD / SELL label.

    future_return_Nd = Close.shift(-N) / Close - 1

    BUY  if future_return >  k_buy  * recent_volatility
    SELL if future_return < -k_sell * recent_volatility
    HOLD otherwise

`recent_volatility` is computed using only data available up to and
including day T (a trailing rolling std), so the label for day T does not
use information from day T's own future window beyond the forward return
itself — the forward return is the target, not a feature.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

LABELS = {1: "SELL", 0: "HOLD", -1: "SELL"}  # placeholder, real map below
CLASS_NAMES = {1: "BUY", 0: "HOLD", -1: "SELL"}


def add_forward_returns(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"FwdReturn_{h}D"] = out["Close"].shift(-h) / out["Close"] - 1
    return out


def add_bhs_targets(
    df: pd.DataFrame,
    horizons: List[int],
    vol_lookback: int,
    k_buy: float,
    k_sell: float,
    min_threshold: float,
) -> pd.DataFrame:
    """Adds Target_{h}D (int in {-1,0,1}) and Threshold_{h}D (float, the
    volatility-adjusted cutoff actually used) for each horizon.
    """
    out = add_forward_returns(df, horizons)
    daily_vol = out["Close"].pct_change().rolling(vol_lookback).std()

    for h in horizons:
        threshold = (daily_vol * np.sqrt(h)).clip(lower=min_threshold)
        fwd = out[f"FwdReturn_{h}D"]

        target = pd.Series(0, index=out.index)  # HOLD
        target[fwd > k_buy * threshold] = 1      # BUY
        target[fwd < -k_sell * threshold] = -1   # SELL

        # Undefined (no forward window yet / no vol history) -> NaN, never a fake HOLD.
        target = target.where(fwd.notna() & threshold.notna())

        out[f"Target_{h}D"] = target
        out[f"Threshold_{h}D"] = threshold

    return out


def target_distribution(df: pd.DataFrame, horizon: int) -> Dict[str, float]:
    col = f"Target_{horizon}D"
    counts = df[col].value_counts(normalize=True).to_dict()
    return {CLASS_NAMES[k]: v for k, v in counts.items() if k in CLASS_NAMES}


def sklearn_label(target_col: pd.Series) -> pd.Series:
    """Remap {-1,0,1} -> {0,1,2} = {SELL,HOLD,BUY} for sklearn/xgboost
    classifiers, which expect non-negative contiguous class labels."""
    return target_col.map({-1: 0, 0: 1, 1: 2})


def sklearn_label_names() -> List[str]:
    return ["SELL", "HOLD", "BUY"]
