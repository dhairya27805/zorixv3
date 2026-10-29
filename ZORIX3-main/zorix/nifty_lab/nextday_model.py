"""
zorix/nifty_lab/nextday_model.py

Next trading session's close (regression) and direction (classification),
trained on the full daily history (742 sessions — small enough that there
is no need for the intraday model's training-window cap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

from zorix.nifty_lab.config import NiftyLabConfig
from zorix.nifty_lab.features import build_daily_features, build_nextday_targets

FEATURE_COLS = [
    "ret_1d", "ret_5d",
    "sma_5d", "sma_10d", "sma_20d",
    "std_5d", "std_10d", "std_20d",
    "range_pct", "gap_pct", "rsi_14", "day_of_week",
]


@dataclass
class NextDayResult:
    reg_model: RandomForestRegressor
    clf_model: RandomForestClassifier
    train_mae: float
    train_rmse: float
    train_dir_acc: float
    n_train_rows: int
    feature_cols: List[str]
    last_session: pd.Timestamp
    last_close: float
    forecast_close: float
    forecast_ret: float
    forecast_dir: int


def train_nextday_model(daily: pd.DataFrame, cfg: NiftyLabConfig) -> NextDayResult:
    feat = build_daily_features(daily)
    feat_t = build_nextday_targets(feat)

    sub = feat_t.dropna(subset=["next_close", "next_dir"])

    X = sub[FEATURE_COLS].to_numpy()
    y_reg = sub["next_ret"].to_numpy()
    y_clf = sub["next_dir"].to_numpy()

    reg = RandomForestRegressor(
        n_estimators=cfg.nextday_n_estimators,
        max_depth=cfg.nextday_max_depth,
        n_jobs=cfg.nextday_n_jobs,
        random_state=cfg.random_state,
    )
    clf = RandomForestClassifier(
        n_estimators=cfg.nextday_n_estimators,
        max_depth=cfg.nextday_max_depth,
        n_jobs=cfg.nextday_n_jobs,
        random_state=cfg.random_state,
    )
    reg.fit(X, y_reg)
    clf.fit(X, y_clf)

    pred_reg = reg.predict(X)
    pred_clf = clf.predict(X)

    latest_row = feat[FEATURE_COLS].iloc[[-1]].to_numpy()
    last_close = float(feat["Close"].iloc[-1])
    fcast_ret = float(reg.predict(latest_row)[0])
    fcast_dir = int(clf.predict(latest_row)[0])

    return NextDayResult(
        reg_model=reg,
        clf_model=clf,
        train_mae=float(mean_absolute_error(y_reg, pred_reg)),
        train_rmse=float(np.sqrt(mean_squared_error(y_reg, pred_reg))),
        train_dir_acc=float(accuracy_score(y_clf, pred_clf)),
        n_train_rows=len(sub),
        feature_cols=FEATURE_COLS,
        last_session=feat.index[-1],
        last_close=last_close,
        forecast_close=round(last_close * (1 + fcast_ret), 2),
        forecast_ret=fcast_ret,
        forecast_dir=fcast_dir,
    )
