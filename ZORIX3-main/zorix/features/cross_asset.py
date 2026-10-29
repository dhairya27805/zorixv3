"""
zorix/features/cross_asset.py

Relative-strength (Section 11) and cross-asset (Section 12) features.

Design choice, per the spec: we compute *relationships* (relative returns,
rolling correlations, cross-asset momentum) and let the model learn what
they mean. We do NOT hardcode rules like "oil up = stock down".
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from zorix.data.market_data import align_to_trading_calendar


def add_relative_strength(df: pd.DataFrame, benchmark: pd.DataFrame,
                           windows=(5, 20)) -> pd.DataFrame:
    """Stock return vs benchmark return, and the spread ('relative strength'),
    for each window. `benchmark` must already be aligned to `df`'s index —
    use align_to_trading_calendar() first if it isn't.
    """
    out = df.copy()
    bench_close = benchmark["Close"] if "Close" in benchmark.columns else benchmark.iloc[:, 0]
    bench_close = bench_close.reindex(out.index, method="ffill")

    for w in windows:
        stock_ret = out["Close"].pct_change(w)
        bench_ret = bench_close.pct_change(w)
        out[f"RelStrength_{w}D"] = stock_ret - bench_ret
        out[f"BenchReturn_{w}D"] = bench_ret
    return out


def add_cross_asset_block(df: pd.DataFrame, other_assets: Dict[str, pd.DataFrame],
                           windows=(1, 5, 20), corr_window: int = 60) -> pd.DataFrame:
    """
    For each other asset (VIX, gold, USDINR, S&P500, ...): add its return,
    momentum and a rolling correlation of its returns with the primary
    asset's returns. `other_assets` values must be OHLCV-like frames with a
    Close column, already fetched via MarketDataProvider (never fabricated).
    """
    out = df.copy()
    primary_ret = out["Close"].pct_change()

    for name, frame in other_assets.items():
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close = frame["Close"].reindex(out.index, method="ffill")
        ret = close.pct_change()

        for w in windows:
            out[f"{name}_Return_{w}D"] = close.pct_change(w)

        out[f"{name}_Corr_{corr_window}D"] = primary_ret.rolling(corr_window).corr(ret)

    return out


def add_market_breadth(df: pd.DataFrame, breadth: pd.DataFrame | None) -> pd.DataFrame:
    """Section 19. `breadth` is an optional externally-supplied DataFrame with
    columns like advance_decline_ratio, pct_above_50dma, pct_above_200dma,
    new_highs, new_lows — indexed by date. If not supplied, breadth columns
    are simply omitted (never fabricated).
    """
    out = df.copy()
    if breadth is None or breadth.empty:
        return out
    aligned = breadth.reindex(out.index, method="ffill").add_prefix("Breadth_")
    return out.join(aligned, how="left")
