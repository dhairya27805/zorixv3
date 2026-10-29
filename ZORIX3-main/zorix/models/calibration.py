"""
zorix/models/calibration.py

Probability calibration (Section 9). Distinguishes:
  - "prediction confidence"    = max class probability the model outputs
  - "probability calibration"  = whether that probability is actually
                                   trustworthy (e.g. among all days the model
                                   said "70% BUY", did BUY actually happen
                                   ~70% of the time?)

Uses sklearn's CalibratedClassifierCV (Platt = sigmoid, or isotonic).
Calibration must be fit on a held-out slice never used for the base model's
training, to avoid leaking test information into the calibration itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize


@dataclass
class CalibrationReport:
    brier_before: float
    brier_after: float
    calibration_curve_before: pd.DataFrame  # columns: bin_mid, predicted, actual
    calibration_curve_after: pd.DataFrame
    method: str


def calibrate_model(base_estimator, X_train, y_train, method: Literal["sigmoid", "isotonic"] = "isotonic"):
    """Wraps an already-configured (unfit) sklearn-compatible estimator with
    cross-validated calibration. Returns a fitted CalibratedClassifierCV.
    """
    calibrated = CalibratedClassifierCV(base_estimator, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def _binned_calibration_curve(y_true_bin: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_mid, predicted, actual = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (proba >= lo) & (proba < hi if i < n_bins - 1 else proba <= hi)
        if mask.sum() == 0:
            continue
        bin_mid.append((lo + hi) / 2)
        predicted.append(float(proba[mask].mean()))
        actual.append(float(y_true_bin[mask].mean()))
    return pd.DataFrame({"bin_mid": bin_mid, "predicted": predicted, "actual": actual})


def evaluate_calibration(
    y_test: np.ndarray, proba_before: np.ndarray, proba_after: np.ndarray, target_class: int = 2
) -> CalibrationReport:
    """Compares calibration for one class (default class=2 -> BUY) before/after."""
    y_bin = label_binarize(y_test, classes=[0, 1, 2])[:, target_class]

    brier_before = brier_score_loss(y_bin, proba_before[:, target_class])
    brier_after = brier_score_loss(y_bin, proba_after[:, target_class])

    curve_before = _binned_calibration_curve(y_bin, proba_before[:, target_class])
    curve_after = _binned_calibration_curve(y_bin, proba_after[:, target_class])

    return CalibrationReport(
        brier_before=float(brier_before),
        brier_after=float(brier_after),
        calibration_curve_before=curve_before,
        calibration_curve_after=curve_after,
        method="isotonic/sigmoid (see CalibratedClassifierCV config)",
    )
