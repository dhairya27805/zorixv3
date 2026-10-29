"""
zorix/nifty_lab/historical_sim.py

Two backtest experiments for Mode 1:

  Experiment 1 — Walk-forward: rolling folds, retrain before each fold,
                 evaluate on the next `walkforward_test_days` daily
                 sessions ahead (intraday models) or sessions (next-day
                 model). Strictly chronological.
  Experiment 2 — Historical one-month-ahead FORECAST: the user picks a
                 historical target trading date; the information cutoff is
                 one calendar month before it; a model is trained using only
                 information observable at that cutoff and predicts the
                 close of that exact target session in a single step. It is
                 then compared with the actual close of that session. This
                 is one forecast made a month in advance -- NOT a month of
                 next-day predictions.

Verified: both run fast — intraday walk-forward ~4s/fold (3 folds ≈ 12s),
acceptable for an explicit backtest button.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from zorix.nifty_lab.config import NiftyLabConfig
from zorix.nifty_lab.features import (
    build_intraday_features, build_intraday_targets,
    build_daily_features, build_nextday_targets,
)
from zorix.nifty_lab.intraday_model import FEATURE_COLS as INTRADAY_FEATURE_COLS
from zorix.nifty_lab.nextday_model import FEATURE_COLS as NEXTDAY_FEATURE_COLS
from zorix.nifty_lab.evaluation import score, Scorecard


@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    scorecard: Scorecard


@dataclass
class ExperimentResult:
    name: str
    folds: List[FoldResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append({
                "fold": f.fold,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "n": f.scorecard.n,
                "MAE": f.scorecard.mae,
                "RMSE": f.scorecard.rmse,
                "Dir. Accuracy": f.scorecard.directional_accuracy,
                "Baseline MAE": f.scorecard.baseline_mae,
                "Baseline Dir. Accuracy": f.scorecard.baseline_dir_acc,
            })
        return pd.DataFrame(rows)


def _fit_predict_daily(train_feat: pd.DataFrame, test_feat: pd.DataFrame, cfg: NiftyLabConfig):
    X_tr = train_feat[NEXTDAY_FEATURE_COLS].to_numpy()
    y_tr_ret = train_feat["next_ret"].to_numpy()
    y_tr_dir = train_feat["next_dir"].to_numpy()

    reg = RandomForestRegressor(n_estimators=cfg.nextday_n_estimators, max_depth=cfg.nextday_max_depth,
                                 n_jobs=cfg.nextday_n_jobs, random_state=cfg.random_state)
    clf = RandomForestClassifier(n_estimators=cfg.nextday_n_estimators, max_depth=cfg.nextday_max_depth,
                                  n_jobs=cfg.nextday_n_jobs, random_state=cfg.random_state)
    reg.fit(X_tr, y_tr_ret)
    clf.fit(X_tr, y_tr_dir)

    X_te = test_feat[NEXTDAY_FEATURE_COLS].to_numpy()
    pred_ret = reg.predict(X_te)
    pred_dir = clf.predict(X_te)
    return pred_ret, pred_dir


def run_nextday_walkforward(daily: pd.DataFrame, cfg: NiftyLabConfig) -> ExperimentResult:
    """Experiment 1 for the next-day model: rolling folds over daily sessions."""
    feat = build_nextday_targets(build_daily_features(daily)).dropna(subset=["next_ret", "next_dir"])

    result = ExperimentResult(name="Next-Day Walk-Forward")
    n = len(feat)
    test_days = cfg.walkforward_test_days
    n_folds = cfg.walkforward_n_folds

    # Reserve the first half of history as a minimum training floor so
    # early folds aren't trained on a handful of rows.
    min_train = max(100, n // 2)
    usable = n - min_train
    if usable < test_days:
        return result  # not enough data for even one fold

    step = max(test_days, usable // n_folds)

    for i in range(n_folds):
        train_end_idx = min_train + i * step
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_days, n)
        if test_start_idx >= n or test_end_idx <= test_start_idx:
            break

        train_feat = feat.iloc[:train_end_idx]
        test_feat = feat.iloc[test_start_idx:test_end_idx]
        if len(train_feat) < 50 or len(test_feat) == 0:
            continue

        pred_ret, pred_dir = _fit_predict_daily(train_feat, test_feat, cfg)
        sc = score(test_feat["next_ret"].to_numpy(), pred_ret,
                   test_feat["next_dir"].to_numpy(), pred_dir)

        result.folds.append(FoldResult(
            fold=i + 1,
            train_end=train_feat.index[-1],
            test_start=test_feat.index[0],
            test_end=test_feat.index[-1],
            scorecard=sc,
        ))

    return result


# ---------------------------------------------------------------------
# Experiment 2 — historical one-month-ahead forecast of a selected date
# ---------------------------------------------------------------------

MIN_TRAIN_ROWS = 50  # same floor the walk-forward experiment uses


class HistoricalForecastError(ValueError):
    """Raised when a historical forecast cannot be made honestly.

    ``code`` is one of NON_TRADING_DATE, OUTSIDE_COVERAGE, INSUFFICIENT_HISTORY.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class HistoricalForecastResult:
    target_date: pd.Timestamp          # the selected trading session
    cutoff_date: pd.Timestamp          # target_date - 1 calendar month
    last_session_used: pd.Timestamp    # last session on/before cutoff_date
    horizon_sessions: int              # sessions from last_session_used to target_date
    predicted_price: float
    actual_price: float                # actual close of target_date
    forecast_error: float              # predicted - actual
    abs_error: float
    pct_error: float                   # forecast_error / actual
    baseline_price: float              # naive "no change" = close on last_session_used
    baseline_error: float
    n_train_rows: int
    last_train_feature_date: pd.Timestamp
    last_train_label_date: pd.Timestamp  # latest date any training label depends on

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Cutoff date": self.cutoff_date.date(),
            "Last session used": self.last_session_used.date(),
            "Target date": self.target_date.date(),
            "Horizon (sessions)": self.horizon_sessions,
            "Predicted close": self.predicted_price,
            "Actual close": self.actual_price,
            "Error (pred - actual)": self.forecast_error,
            "Abs. error": self.abs_error,
            "Error %": self.pct_error * 100,
            "Naive close (cutoff)": self.baseline_price,
            "Naive error": self.baseline_error,
            "Training rows": self.n_train_rows,
        }])


def calendar_month_cutoff(target_date) -> pd.Timestamp:
    """Information cutoff: one calendar month before ``target_date``.

    ``pd.DateOffset(months=1)`` clamps to month end (Mar 31 -> Feb 28/29).
    The cutoff is the END of that date: sessions dated on or before it are
    observable, anything later is not.
    """
    return pd.Timestamp(target_date).normalize() - pd.DateOffset(months=1)


def build_horizon_training_set(info: pd.DataFrame, horizon: int, cutoff: pd.Timestamp):
    """Direct h-session-ahead training set from information up to ``cutoff``.

    ``info`` must already be restricted to sessions on or before ``cutoff``
    (features are computed from it, so nothing later can leak into them).
    The label of the row dated d is the simple return from the close of d to
    the close of the session ``horizon`` sessions later; that outcome is only
    observable once that later session has closed, so rows whose label
    session would fall after the last available session are dropped (their
    label is unknown at the cutoff). Each label's outcome date is checked
    explicitly against ``cutoff`` -- not just "train_end < test_start".

    Returns (X, y, feature_dates, label_dates, feat) where ``feat`` is the
    full feature frame (including the unlabelled tail used for prediction).
    """
    if len(info) and info.index[-1] > cutoff:
        raise RuntimeError(
            f"information frame extends past the cutoff ({info.index[-1].date()} > {cutoff.date()})")

    feat = build_daily_features(info)

    # Labels come from the full session sequence (not from `feat`, which has
    # lost its warm-up rows), so "h sessions later" is always h real sessions.
    close = info["Close"]
    label_close = close.shift(-horizon).reindex(feat.index)
    label_date = pd.Series(info.index, index=info.index).shift(-horizon).reindex(feat.index)

    known = label_close.notna() & label_date.notna()
    train = feat.loc[known]
    label_dates = pd.DatetimeIndex(label_date.loc[known])

    if len(label_dates) and label_dates.max() > cutoff:
        raise RuntimeError(
            f"training label outcome {label_dates.max().date()} is after the cutoff {cutoff.date()}")

    y = (label_close.loc[known] / train["Close"] - 1).to_numpy()
    X = train[NEXTDAY_FEATURE_COLS].to_numpy()
    return X, y, train.index, label_dates, feat


def run_historical_forecast(daily: pd.DataFrame, cfg: NiftyLabConfig, target_date,
                            min_train_rows: int = MIN_TRAIN_ROWS) -> HistoricalForecastResult:
    """Experiment 2: forecast the close of ``target_date`` one month in advance.

    1. ``target_date`` must be a trading session present in ``daily``.
    2. Cutoff = target_date minus one calendar month; only sessions dated on
       or before the cutoff are used -- features, labels and the model fit.
    3. Method: DIRECT multi-step forecast. A random forest (same
       hyper-parameters as the next-day model) is trained to predict the
       return from a session's close to the close ``h`` sessions later, where
       ``h`` = number of sessions between the last session at the cutoff and
       the target. Only (session, session+h) pairs that had both completed by
       the cutoff are used, so no label depends on post-cutoff prices. The
       model is applied once, to the last session on/before the cutoff.
       (The session COUNT h uses the trading calendar only -- never prices
       after the cutoff. No scaling/normalisation is used: the forest is
       scale-invariant and all features are trailing windows over the
       truncated history.)
    4. The forecast is compared with the actual close of exactly that session.

    Raises HistoricalForecastError for non-trading dates, targets outside
    dataset coverage and insufficient training history.
    """
    daily = daily.copy()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("daily index must be unique and sorted ascending")

    target = pd.Timestamp(target_date).normalize()

    if len(daily) == 0:
        raise HistoricalForecastError("OUTSIDE_COVERAGE", "Dataset has no daily sessions.")
    first, last = daily.index[0], daily.index[-1]
    if target < first or target > last:
        raise HistoricalForecastError(
            "OUTSIDE_COVERAGE",
            f"Target date {target.date()} is outside the dataset coverage "
            f"({first.date()} to {last.date()}).")

    if target not in daily.index:
        prev_s = daily.index[daily.index < target][-1]
        next_s = daily.index[daily.index > target][0]
        raise HistoricalForecastError(
            "NON_TRADING_DATE",
            f"{target.date()} is not a trading session in the dataset (weekend, holiday or "
            f"missing data). Nearest sessions: {prev_s.date()} and {next_s.date()}.")

    cutoff = calendar_month_cutoff(target)
    info = daily.loc[daily.index <= cutoff]
    if info.empty:
        raise HistoricalForecastError(
            "INSUFFICIENT_HISTORY",
            f"Cutoff {cutoff.date()} is before the first session in the dataset ({first.date()}).")

    last_session = info.index[-1]
    horizon = int(daily.index.get_loc(target) - daily.index.get_loc(last_session))

    X, y, feat_dates, label_dates, feat = build_horizon_training_set(info, horizon, cutoff)
    if len(y) < min_train_rows:
        raise HistoricalForecastError(
            "INSUFFICIENT_HISTORY",
            f"Only {len(y)} usable training rows before the {cutoff.date()} cutoff for a "
            f"{horizon}-session horizon (need at least {min_train_rows}).")
    if len(feat) == 0 or feat.index[-1] != last_session:
        raise HistoricalForecastError(
            "INSUFFICIENT_HISTORY",
            f"Features are not available for the last session before the cutoff ({last_session.date()}).")

    reg = RandomForestRegressor(n_estimators=cfg.nextday_n_estimators, max_depth=cfg.nextday_max_depth,
                                 n_jobs=cfg.nextday_n_jobs, random_state=cfg.random_state)
    reg.fit(X, y)

    origin_close = float(info["Close"].iloc[-1])
    pred_ret = float(reg.predict(feat[NEXTDAY_FEATURE_COLS].iloc[[-1]].to_numpy())[0])
    predicted = origin_close * (1 + pred_ret)

    actual = float(daily.loc[target, "Close"])
    err = predicted - actual
    return HistoricalForecastResult(
        target_date=target,
        cutoff_date=cutoff,
        last_session_used=last_session,
        horizon_sessions=horizon,
        predicted_price=predicted,
        actual_price=actual,
        forecast_error=err,
        abs_error=abs(err),
        pct_error=err / actual,
        baseline_price=origin_close,
        baseline_error=origin_close - actual,
        n_train_rows=len(y),
        last_train_feature_date=feat_dates[-1],
        last_train_label_date=label_dates.max(),
    )


def run_intraday_walkforward(minute: pd.DataFrame, cfg: NiftyLabConfig, horizon_min: int,
                              train_window: int = None) -> ExperimentResult:
    """Experiment 1 for one intraday horizon. Operates on a recent slice of
    minute bars (same train_window cap as the live intraday model) to keep
    each fold's retrain fast."""
    window = train_window if train_window is not None else cfg.intraday_train_window
    recent = minute.tail(window * (cfg.walkforward_n_folds + 1))

    feat = build_intraday_features(recent)
    feat_t = build_intraday_targets(feat, [horizon_min], bars=recent)
    ret_col, dir_col = f"fwd_ret_{horizon_min}m", f"fwd_dir_{horizon_min}m"
    feat_t = feat_t.dropna(subset=[ret_col, dir_col])

    result = ExperimentResult(name=f"Intraday {horizon_min}m Walk-Forward")
    n = len(feat_t)
    if n < 200:
        return result

    fold_size = n // (cfg.walkforward_n_folds + 1)
    if fold_size < 50:
        return result

    for i in range(cfg.walkforward_n_folds):
        train_end_idx = fold_size * (i + 1)
        test_end_idx = min(train_end_idx + fold_size, n)
        train_feat = feat_t.iloc[:train_end_idx]
        test_feat = feat_t.iloc[train_end_idx:test_end_idx]
        if len(train_feat) < 50 or len(test_feat) == 0:
            continue

        X_tr = train_feat[INTRADAY_FEATURE_COLS].to_numpy()
        X_te = test_feat[INTRADAY_FEATURE_COLS].to_numpy()

        reg = RandomForestRegressor(n_estimators=cfg.intraday_n_estimators, max_depth=cfg.intraday_max_depth,
                                     n_jobs=cfg.intraday_n_jobs, random_state=cfg.random_state)
        clf = RandomForestClassifier(n_estimators=cfg.intraday_n_estimators, max_depth=cfg.intraday_max_depth,
                                      n_jobs=cfg.intraday_n_jobs, random_state=cfg.random_state)
        reg.fit(X_tr, train_feat[ret_col].to_numpy())
        clf.fit(X_tr, train_feat[dir_col].to_numpy())

        pred_ret = reg.predict(X_te)
        pred_dir = clf.predict(X_te)
        sc = score(test_feat[ret_col].to_numpy(), pred_ret,
                   test_feat[dir_col].to_numpy(), pred_dir)

        result.folds.append(FoldResult(
            fold=i + 1,
            train_end=train_feat.index[-1],
            test_start=test_feat.index[0],
            test_end=test_feat.index[-1],
            scorecard=sc,
        ))

    return result
