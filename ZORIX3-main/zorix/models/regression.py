"""
zorix/models/regression.py

Separate return-regression models (Priority 1 of this phase): predict the
continuous forward return over the chosen horizon, rather than only a
discrete BUY/HOLD/SELL class. This is what feeds "Expected Return" in the
signal/risk output — previously app.py used a historical average as a
rough placeholder; this replaces that with an actual out-of-sample model
prediction.

Kept as its own module (not bolted onto ModelTrainer) because regression
and classification use different metrics, different sklearn base classes,
and are conceptually separate questions ("how much" vs "which bucket").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from zorix.models.trainer import LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE

if XGBOOST_AVAILABLE:
    import xgboost as xgb
if LIGHTGBM_AVAILABLE:
    import lightgbm as lgb


@dataclass
class RegressionResult:
    name: str
    model: object
    predictions_test: Optional[np.ndarray]
    metrics: Dict[str, float]
    ran: bool = True
    skip_reason: Optional[str] = None


@dataclass
class RegressionReport:
    results: List[RegressionResult] = field(default_factory=list)
    selected_name: Optional[str] = None                 # chosen on a VALIDATION slice of train
    validation_scores: Dict[str, float] = field(default_factory=dict)
    selection_metric: str = "mae"

    def available_models(self) -> List[str]:
        return [r.name for r in self.results if r.ran]

    def best(self) -> Optional[RegressionResult]:
        """The model chosen on the validation slice carved from the training
        data. Test metrics in `results` are reporting only and never select."""
        for r in self.results:
            if r.ran and r.name == self.selected_name:
                return r
        return None


def _evaluate_regression(name: str, y_test: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }
    try:
        metrics["r2"] = float(r2_score(y_test, preds))
    except Exception:
        metrics["r2"] = float("nan")

    # Directional accuracy: how often the sign of the predicted return
    # matches the sign of the realized return — the metric that actually
    # matters for a BUY/SELL decision, independent of magnitude error.
    same_sign = (np.sign(preds) == np.sign(y_test))
    metrics["directional_accuracy"] = float(same_sign.mean())
    return metrics


class ReturnRegressionTrainer:
    """Trains Linear Regression, Random Forest, XGBoost and LightGBM
    regressors on the same train/test split used for the classifiers, all
    predicting the continuous forward return for one horizon.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()

    def _scale(self, X_train, X_test):
        X_tr = self.scaler.fit_transform(X_train)
        X_te = self.scaler.transform(X_test)
        return X_tr, X_te

    def train_linear(self, X_train, X_test, y_train, y_test) -> RegressionResult:
        X_tr, X_te = self._scale(X_train, X_test)
        model = LinearRegression()
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        return RegressionResult("Linear Regression", model, preds,
                                 _evaluate_regression("Linear", y_test.values, preds))

    def train_random_forest(self, X_train, X_test, y_train, y_test) -> RegressionResult:
        model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                       n_jobs=-1, random_state=self.random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return RegressionResult("Random Forest Regressor", model, preds,
                                 _evaluate_regression("RF", y_test.values, preds))

    def train_xgboost(self, X_train, X_test, y_train, y_test) -> RegressionResult:
        if not XGBOOST_AVAILABLE:
            return RegressionResult("XGBoost Regressor", None, None, {}, ran=False,
                                     skip_reason="xgboost not installed")
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective="reg:squarederror", random_state=self.random_state, n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)  # never touch test data during fit
        preds = model.predict(X_test)
        return RegressionResult("XGBoost Regressor", model, preds,
                                 _evaluate_regression("XGB", y_test.values, preds))

    def train_lightgbm(self, X_train, X_test, y_train, y_test) -> RegressionResult:
        if not LIGHTGBM_AVAILABLE:
            return RegressionResult("LightGBM Regressor", None, None, {}, ran=False,
                                     skip_reason="lightgbm not installed")
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=-1, verbosity=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return RegressionResult("LightGBM Regressor", model, preds,
                                 _evaluate_regression("LGBM", y_test.values, preds))

    def _select_on_validation(self, X_train, y_train, gap: int, val_frac: float, metric: str):
        """Fit every candidate on the early part of TRAIN, score on the later
        part (purged by `gap` rows). Returns (best_name, {name: score})."""
        n = len(X_train)
        cut = int(n * (1 - val_frac))
        tr_end = cut - gap
        if tr_end < 50 or n - cut < 20:
            return None, {}
        Xa, ya = X_train.iloc[:tr_end], y_train.iloc[:tr_end]
        Xv, yv = X_train.iloc[cut:], y_train.iloc[cut:]
        scores: Dict[str, float] = {}
        for fn in (self.train_linear, self.train_random_forest, self.train_xgboost, self.train_lightgbm):
            r = fn(Xa, Xv, ya, yv)
            if r.ran and metric in r.metrics and not np.isnan(r.metrics[metric]):
                scores[r.name] = r.metrics[metric]
        if not scores:
            return None, {}
        higher_better = metric in ("r2", "directional_accuracy")
        pick = max if higher_better else min
        return pick(scores, key=scores.get), scores

    def train_all(self, X_train, X_test, y_train, y_test, gap: int = 0,
                  val_frac: float = 0.2, select_metric: str = "mae") -> RegressionReport:
        # Selection first (it re-fits self.scaler); the final fits below leave the
        # scaler fit on the full training set.
        name, scores = self._select_on_validation(X_train, y_train, gap, val_frac, select_metric)
        report = RegressionReport(selected_name=name, validation_scores=scores,
                                  selection_metric=select_metric)
        report.results.append(self.train_linear(X_train, X_test, y_train, y_test))
        report.results.append(self.train_random_forest(X_train, X_test, y_train, y_test))
        report.results.append(self.train_xgboost(X_train, X_test, y_train, y_test))
        report.results.append(self.train_lightgbm(X_train, X_test, y_train, y_test))
        return report
