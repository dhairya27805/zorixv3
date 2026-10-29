"""
zorix/nifty_lab/intraday_model.py

Intraday forecasting: 5 / 15 / 30-minute ahead return + direction, trained
on a capped recent window of 1-min bars (see NiftyLabConfig.intraday_train_window
for why the window is capped).

Nothing here runs at import time — callers (ui.py) invoke train_intraday_models()
explicitly from a button handler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

from zorix.nifty_lab.config import NiftyLabConfig
from zorix.nifty_lab.features import build_intraday_features, build_intraday_targets

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_15",
    "sma_5", "sma_15", "sma_30",
    "std_5", "std_15", "std_30",
    "range_pct", "body_pct", "rsi_14",
    "minute_of_day", "day_of_week",
]


@dataclass
class IntradayHorizonResult:
    horizon_min: int
    reg_model: RandomForestRegressor
    clf_model: RandomForestClassifier
    train_mae: float
    train_rmse: float
    train_dir_acc: float
    n_train_rows: int
    last_forecast_ret: float = 0.0
    last_forecast_dir: int = 0
    last_forecast_price: float = 0.0
    # All four describe ONE observation, taken from the same row of the
    # target frame: the forecast origin bar, its close, the exact timestamp
    # the forecast is for, and that bar's actual close (None while that
    # bar has not been observed / is not a valid same-session target).
    forecast_origin: Optional[pd.Timestamp] = None
    origin_price: float = 0.0
    target_timestamp: Optional[pd.Timestamp] = None
    actual_price: Optional[float] = None


@dataclass
class IntradayResult:
    horizons: Dict[int, IntradayHorizonResult] = field(default_factory=dict)
    feature_cols: List[str] = field(default_factory=lambda: list(FEATURE_COLS))
    train_window_used: int = 0
    as_of: pd.Timestamp = None


def train_intraday_models(
    minute: pd.DataFrame,
    cfg: NiftyLabConfig,
    train_window: int = None,
) -> IntradayResult:
    """Trains one regressor + one classifier per horizon on the most recent
    `train_window` minute bars (default: cfg.intraday_train_window).
    `train_window` can be overridden to a longer window on explicit request.
    """
    window = train_window if train_window is not None else cfg.intraday_train_window

    recent = minute.tail(window + 50)  # small buffer so rolling features aren't all-NaN at the start
    feat = build_intraday_features(recent)
    feat_t = build_intraday_targets(feat, cfg.intraday_horizons_min, bars=recent)

    result = IntradayResult(train_window_used=min(window, len(feat)), as_of=feat.index[-1] if len(feat) else None)

    for h in cfg.intraday_horizons_min:
        ret_col = f"fwd_ret_{h}m"
        dir_col = f"fwd_dir_{h}m"

        # Chronological: drop only rows where the forward target is NaN
        # (the last h rows, since there's no future data for them yet).
        sub = feat_t.dropna(subset=[ret_col, dir_col])
        sub = sub.tail(window)
        if len(sub) < 50:
            continue

        X = sub[FEATURE_COLS].to_numpy()
        y_reg = sub[ret_col].to_numpy()
        y_clf = sub[dir_col].to_numpy()

        reg = RandomForestRegressor(
            n_estimators=cfg.intraday_n_estimators,
            max_depth=cfg.intraday_max_depth,
            n_jobs=cfg.intraday_n_jobs,
            random_state=cfg.random_state,
        )
        clf = RandomForestClassifier(
            n_estimators=cfg.intraday_n_estimators,
            max_depth=cfg.intraday_max_depth,
            n_jobs=cfg.intraday_n_jobs,
            random_state=cfg.random_state,
        )
        reg.fit(X, y_reg)
        clf.fit(X, y_clf)

        pred_reg = reg.predict(X)
        pred_clf = clf.predict(X)

        # Live forecast from the single most recent row available in the
        # full feature frame (not necessarily in `sub`, which was trimmed
        # by dropna on the forward target).
        latest_row = feat[FEATURE_COLS].iloc[[-1]].to_numpy()
        last_price = float(feat["Close"].iloc[-1])
        origin_ts = feat.index[-1]
        actual_close = feat_t[f"target_close_{h}m"].iloc[-1]  # same row as latest_row
        fcast_ret = float(reg.predict(latest_row)[0])
        fcast_dir = int(clf.predict(latest_row)[0])

        result.horizons[h] = IntradayHorizonResult(
            horizon_min=h,
            reg_model=reg,
            clf_model=clf,
            train_mae=float(mean_absolute_error(y_reg, pred_reg)),
            train_rmse=float(np.sqrt(mean_squared_error(y_reg, pred_reg))),
            train_dir_acc=float(accuracy_score(y_clf, pred_clf)),
            n_train_rows=len(sub),
            last_forecast_ret=fcast_ret,
            last_forecast_dir=fcast_dir,
            last_forecast_price=round(last_price * (1 + fcast_ret), 2),
            forecast_origin=origin_ts,
            origin_price=last_price,
            target_timestamp=origin_ts + pd.Timedelta(minutes=h),
            actual_price=None if pd.isna(actual_close) else float(actual_close),
        )

    return result
