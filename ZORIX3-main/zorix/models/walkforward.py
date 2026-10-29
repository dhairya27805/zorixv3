"""
zorix/models/walkforward.py

Walk-forward validation (Section 8) — the highest-priority fix called out
in the spec. Replaces a single random 80/20 split with an expanding-window
evaluation: train on everything up to fold boundary i, test on the next
chunk, then grow the training window and repeat. Reports the distribution
of metrics across folds, not just the best (or only) one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from zorix.models.splits import infer_horizon
from zorix.utils.helpers import (cagr, max_drawdown, profit_factor,
                                  sharpe_ratio, sortino_ratio)


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: Dict[str, float]


@dataclass
class WalkForwardReport:
    folds: List[FoldResult] = field(default_factory=list)
    oos_proba: pd.DataFrame = field(default_factory=pd.DataFrame)  # index=date, cols=SELL,HOLD,BUY

    def summary(self) -> pd.DataFrame:
        rows = [f.metrics for f in self.folds]
        df = pd.DataFrame(rows)
        summary = pd.DataFrame({
            "mean": df.mean(numeric_only=True),
            "median": df.median(numeric_only=True),
            "std": df.std(numeric_only=True),
        })
        return summary


def oos_predictions_to_signals(oos_proba: pd.DataFrame, min_edge: float, min_confidence: float) -> pd.Series:
    """Converts the walk-forward report's out-of-sample probabilities into
    BUY/SELL/HOLD/NO TRADE signal strings using the same NO TRADE gating
    logic the live app uses (signals/signal_engine.py), so backtested
    signals match what the deployed signal engine would actually have said
    — not a naive argmax that ignores the confidence gate.
    """
    from zorix.signals.signal_engine import decide_signal  # local import: avoids a cycle at module load

    if oos_proba.empty:
        return pd.Series(dtype=object)

    signals = {}
    for date, row in oos_proba.iterrows():
        proba = row[["SELL", "HOLD", "BUY"]].values.astype(float)
        if np.isnan(proba).any():
            continue
        signal, _, _ = decide_signal(proba, min_edge, min_confidence)
        signals[date] = signal
    return pd.Series(signals)


def walk_forward_split(n_rows: int, n_splits: int, min_train_size: int, gap: int = 0):
    """Expanding-window indices: fold i's test set is the next contiguous
    chunk after fold i's (growing) train set. Falls back to sklearn's
    TimeSeriesSplit which already implements exactly this. `gap` rows (set to
    the label horizon) are purged between train and test so overlapping label
    windows can't leak.
    """
    usable_rows = n_rows - min_train_size
    if usable_rows <= n_splits:
        n_splits = max(2, usable_rows // 50)  # degrade gracefully on short histories
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    all_idx = np.arange(n_rows)
    # Restrict candidate split points to rows after min_train_size so the
    # first fold always has a real minimum training window.
    eligible = all_idx[min_train_size:]
    for train_idx_rel, test_idx_rel in tscv.split(eligible):
        train_idx = np.concatenate([all_idx[:min_train_size], eligible[train_idx_rel]])
        test_idx = eligible[test_idx_rel]
        yield train_idx, test_idx


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    fwd_return_col: str,
    train_fn: Callable[[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], object],
    predict_fn: Callable[[object, pd.DataFrame], np.ndarray],
    n_splits: int,
    min_train_size: int,
    collect_predictions: bool = False,
    horizon: Optional[int] = None,
) -> WalkForwardReport:
    """
    train_fn(X_train, X_test, y_train, y_test) -> fitted model
    predict_fn(model, X_test) -> proba array (n_test, 3) [SELL, HOLD, BUY]

    A simple long-only strategy return is derived per fold: on days the
    model's argmax class is BUY, earn fwd_return_col's per-period return;
    on SELL, earn its negative (short); on HOLD, earn 0. This is a
    simplified proxy for Sharpe/Sortino/drawdown reporting during model
    selection — the full transaction-cost-aware backtester is a separate
    module.

    If collect_predictions=True, the report's `oos_proba` DataFrame is
    populated with every fold's out-of-sample probabilities, indexed by
    date — each row's prediction comes only from a model trained on
    strictly earlier data (expanding window), so this is safe to feed
    into the backtester as a genuine out-of-sample signal series.
    """
    report = WalkForwardReport()
    clean = df.dropna(subset=feature_cols + [target_col, fwd_return_col]).reset_index()
    date_col = clean.columns[0]
    oos_records = []
    # Purge gap = label horizon (inferred from e.g. "Target_5D" unless given).
    gap = infer_horizon(target_col) if horizon is None else int(horizon)
    # Targets arrive as {-1,0,1}; models/argmax use {0,1,2} = {SELL,HOLD,BUY}.
    if (clean[target_col] == -1).any():
        clean[target_col] = clean[target_col].map({-1: 0, 0: 1, 1: 2}).astype(int)

    for fold_i, (train_idx, test_idx) in enumerate(
        walk_forward_split(len(clean), n_splits, min_train_size, gap=gap), start=1
    ):
        train, test = clean.iloc[train_idx], clean.iloc[test_idx]
        if train.empty or test.empty:
            continue

        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]

        model = train_fn(X_train, X_test, y_train, y_test)
        proba = predict_fn(model, X_test)
        preds = proba.argmax(axis=1)  # 0=SELL,1=HOLD,2=BUY

        from sklearn.metrics import accuracy_score, f1_score
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        }

        direction = np.where(preds == 2, 1, np.where(preds == 0, -1, 0))
        strat_returns = pd.Series(direction * test[fwd_return_col].values, index=test.index)
        equity = (1 + strat_returns.fillna(0)).cumprod()

        metrics["sharpe"] = sharpe_ratio(strat_returns)
        metrics["sortino"] = sortino_ratio(strat_returns)
        metrics["max_drawdown"] = max_drawdown(equity)
        metrics["profit_factor"] = profit_factor(strat_returns[direction != 0])
        metrics["cagr"] = cagr(equity)

        report.folds.append(FoldResult(
            fold=fold_i,
            train_start=str(train[date_col].iloc[0]), train_end=str(train[date_col].iloc[-1]),
            test_start=str(test[date_col].iloc[0]), test_end=str(test[date_col].iloc[-1]),
            metrics=metrics,
        ))

        if collect_predictions:
            fold_oos = pd.DataFrame(proba, columns=["SELL", "HOLD", "BUY"], index=test[date_col].values)
            oos_records.append(fold_oos)

    if collect_predictions and oos_records:
        report.oos_proba = pd.concat(oos_records).sort_index()
        report.oos_proba = report.oos_proba[~report.oos_proba.index.duplicated(keep="last")]

    return report
