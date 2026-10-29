"""
zorix/nifty_lab/evaluation.py

Shared metrics + naive baselines used by both the live models and the
historical simulations, so "is the model actually any good" has a
consistent, chronologically-fair yardstick everywhere in Mode 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


@dataclass
class Scorecard:
    mae: float
    rmse: float
    directional_accuracy: float
    baseline_mae: float
    baseline_rmse: float
    baseline_dir_acc: float
    n: int

    @property
    def beats_baseline_mae(self) -> bool:
        return self.mae < self.baseline_mae

    @property
    def beats_baseline_dir(self) -> bool:
        return self.directional_accuracy > self.baseline_dir_acc


def naive_persistence_forecast(y_true_ret: np.ndarray) -> np.ndarray:
    """Baseline: 'no change' — predicted forward return is always 0."""
    return np.zeros_like(y_true_ret)


def naive_majority_direction(y_true_dir: np.ndarray) -> np.ndarray:
    """Baseline: always predict the majority class observed in y_true_dir itself,
    used only as a fixed reference point for in-sample scorecards, never fit
    into the model's own training loop."""
    majority = int(round(np.mean(y_true_dir))) if len(y_true_dir) else 0
    return np.full_like(y_true_dir, majority)


def score(y_true_ret: np.ndarray, y_pred_ret: np.ndarray,
          y_true_dir: np.ndarray, y_pred_dir: np.ndarray) -> Scorecard:
    baseline_ret = naive_persistence_forecast(y_true_ret)
    baseline_dir = naive_majority_direction(y_true_dir)

    return Scorecard(
        mae=float(mean_absolute_error(y_true_ret, y_pred_ret)),
        rmse=float(np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))),
        directional_accuracy=float(accuracy_score(y_true_dir, y_pred_dir)),
        baseline_mae=float(mean_absolute_error(y_true_ret, baseline_ret)),
        baseline_rmse=float(np.sqrt(mean_squared_error(y_true_ret, baseline_ret))),
        baseline_dir_acc=float(accuracy_score(y_true_dir, baseline_dir)),
        n=len(y_true_ret),
    )


def scorecard_to_frame(sc: Scorecard, label: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "segment": label,
        "n": sc.n,
        "MAE": sc.mae,
        "RMSE": sc.rmse,
        "Dir. Accuracy": sc.directional_accuracy,
        "Baseline MAE": sc.baseline_mae,
        "Baseline RMSE": sc.baseline_rmse,
        "Baseline Dir. Accuracy": sc.baseline_dir_acc,
        "Beats baseline (MAE)": sc.beats_baseline_mae,
        "Beats baseline (Dir.)": sc.beats_baseline_dir,
    }])
