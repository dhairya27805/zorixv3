"""
zorix/models/trainer.py

Multi-model layer (Section 6): Logistic Regression, Random Forest, XGBoost,
LightGBM (if installed), LSTM/GRU (if TensorFlow is installed) — all
producing 3-class (SELL/HOLD/BUY) probabilities.

Every optional model is wrapped so that a missing dependency disables just
that model and is reported, rather than crashing the app (Section 6,
Section 43). Nothing here ever claims a model ran if it didn't
(Section 32, Section 50 #10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                              roc_auc_score)
from sklearn.preprocessing import RobustScaler, label_binarize

logger = logging.getLogger("zorix.models.trainer")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf  # noqa: F401
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@dataclass
class ModelResult:
    name: str
    model: object
    proba_test: np.ndarray       # shape (n_test, 3) columns = [SELL, HOLD, BUY]
    metrics: Dict[str, float]
    ran: bool = True
    skip_reason: Optional[str] = None


@dataclass
class TrainingReport:
    results: List[ModelResult] = field(default_factory=list)

    def available_models(self) -> List[str]:
        return [r.name for r in self.results if r.ran]

    def skipped_models(self) -> Dict[str, str]:
        return {r.name: r.skip_reason for r in self.results if not r.ran}


def _balanced_sample_weight(y: pd.Series) -> np.ndarray:
    """Per-sample weights equivalent to sklearn's class_weight="balanced"
    (weight_c = n_samples / (n_classes * n_samples_c)), for estimators
    (XGBoost) that take sample_weight instead of a class_weight kwarg."""
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    weight_by_class = {c: len(y_arr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_by_class[v] for v in y_arr])


def _evaluate(name: str, y_test: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """3-class metrics: accuracy, macro-F1, one-vs-rest ROC-AUC, mean Brier."""
    preds = proba.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }
    try:
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        metrics["roc_auc_ovr"] = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        metrics["roc_auc_ovr"] = float("nan")

    try:
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        briers = [brier_score_loss(y_bin[:, c], proba[:, c]) for c in range(3)]
        metrics["brier_mean"] = float(np.mean(briers))
    except Exception:
        metrics["brier_mean"] = float("nan")

    return metrics


class ModelTrainer:
    """Trains every available model on the same train/test split and returns
    a TrainingReport. X must already be numeric/imputed; y must be in
    {0,1,2} = {SELL,HOLD,BUY} (see features/target.py:sklearn_label).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()

    def _scale(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        X_tr = self.scaler.fit_transform(X_train)   # fit ONLY on train
        X_te = self.scaler.transform(X_test)
        return X_tr, X_te

    def train_logistic(self, X_train, X_test, y_train, y_test) -> ModelResult:
        X_tr, X_te = self._scale(X_train, X_test)
        # Note: sklearn >=1.5 removed the multi_class kwarg — LogisticRegression
        # now automatically uses a multinomial loss for problems with >2 classes.
        # class_weight="balanced": the BHS label is imbalanced by construction
        # (HOLD is the plurality class). An unweighted fit on noisy/weak-signal
        # features collapses to argmax-always-HOLD even when the per-class mean
        # probabilities are reasonably spread, because HOLD only has to win a
        # plurality, not a majority, of every row. Balancing the loss removes
        # that free ride for the majority class so the decision boundary reflects
        # actual separability rather than the label prior.
        clf = LogisticRegression(max_iter=2000, C=0.5, random_state=self.random_state,
                                  class_weight="balanced")
        clf.fit(X_tr, y_train)
        proba = clf.predict_proba(X_te)
        return ModelResult("Logistic Regression", clf, proba, _evaluate("Logistic", y_test, proba))

    def train_random_forest(self, X_train, X_test, y_train, y_test) -> ModelResult:
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                      n_jobs=-1, random_state=self.random_state,
                                      class_weight="balanced")
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        return ModelResult("Random Forest", clf, proba, _evaluate("RandomForest", y_test, proba))

    def train_xgboost(self, X_train, X_test, y_train, y_test) -> ModelResult:
        if not XGBOOST_AVAILABLE:
            return ModelResult("XGBoost", None, None, {}, ran=False,
                                skip_reason="xgboost not installed")
        clf = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=self.random_state, n_jobs=-1,
        )
        # XGBoost has no multiclass class_weight kwarg, so balancing is done via
        # per-sample weights (same effect as class_weight="balanced" above).
        sample_weight = _balanced_sample_weight(y_train)
        clf.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)  # never touch test data during fit
        proba = clf.predict_proba(X_test)
        return ModelResult("XGBoost", clf, proba, _evaluate("XGBoost", y_test, proba))

    def train_lightgbm(self, X_train, X_test, y_train, y_test) -> ModelResult:
        if not LIGHTGBM_AVAILABLE:
            return ModelResult("LightGBM", None, None, {}, ran=False,
                                skip_reason="lightgbm not installed")
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, objective="multiclass", num_class=3,
            random_state=self.random_state, n_jobs=-1, verbosity=-1,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        return ModelResult("LightGBM", clf, proba, _evaluate("LightGBM", y_test, proba))

    def train_recurrent(self, X_train, X_test, y_train, y_test, cell_type: str = "LSTM",
                         seq_len: int = 60, epochs: int = 40, batch_size: int = 32,
                         units: int = 64) -> ModelResult:
        """Shared implementation for both LSTM and GRU (Section 6, Model 5) —
        same architecture, only the recurrent layer type differs.
        """
        name = cell_type.upper()
        if not TENSORFLOW_AVAILABLE:
            return ModelResult(name, None, None, {}, ran=False,
                                skip_reason="tensorflow not installed")
        import tensorflow as tf
        from tensorflow.keras import layers, models

        X_tr, X_te = self._scale(X_train, X_test)

        def make_sequences(X, y, n):
            Xs, ys, idx = [], [], []
            for i in range(n, len(X)):
                Xs.append(X[i - n:i])
                ys.append(y[i])
                idx.append(i)
            return np.array(Xs), np.array(ys), idx

        y_train_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

        Xtr_seq, ytr_seq, _ = make_sequences(X_tr, y_train_arr, seq_len)
        Xte_seq, yte_seq, keep_idx = make_sequences(X_te, y_test_arr, seq_len)

        if len(Xtr_seq) < 50 or len(Xte_seq) < 5:
            return ModelResult(name, None, None, {}, ran=False,
                                skip_reason="Not enough rows for the sequence length configured")

        recurrent_layer = layers.GRU(units, return_sequences=False) if name == "GRU" \
            else layers.LSTM(units, return_sequences=False)

        model = models.Sequential([
            layers.Input(shape=(seq_len, X_tr.shape[1])),
            recurrent_layer,
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(Xtr_seq, ytr_seq, validation_split=0.1, epochs=epochs,
                  batch_size=batch_size, callbacks=[es], verbose=0)

        proba_partial = model.predict(Xte_seq, verbose=0)
        # Pad back to full test length so downstream code can align by index;
        # the first `seq_len` test rows have no prediction (NaN).
        proba_full = np.full((len(X_te), 3), np.nan)
        proba_full[keep_idx] = proba_partial

        metrics = _evaluate(name, yte_seq, proba_partial)
        return ModelResult(name, model, proba_full, metrics)

    def train_lstm(self, X_train, X_test, y_train, y_test, **kwargs) -> ModelResult:
        return self.train_recurrent(X_train, X_test, y_train, y_test, cell_type="LSTM", **kwargs)

    def train_gru(self, X_train, X_test, y_train, y_test, **kwargs) -> ModelResult:
        return self.train_recurrent(X_train, X_test, y_train, y_test, cell_type="GRU", **kwargs)

    def train_all(self, X_train, X_test, y_train, y_test,
                   include_lstm: bool = False, include_gru: bool = False,
                   **recurrent_kwargs) -> TrainingReport:
        report = TrainingReport()
        report.results.append(self.train_logistic(X_train, X_test, y_train, y_test))
        report.results.append(self.train_random_forest(X_train, X_test, y_train, y_test))
        report.results.append(self.train_xgboost(X_train, X_test, y_train, y_test))
        report.results.append(self.train_lightgbm(X_train, X_test, y_train, y_test))
        if include_lstm:
            report.results.append(self.train_lstm(X_train, X_test, y_train, y_test, **recurrent_kwargs))
        if include_gru:
            report.results.append(self.train_gru(X_train, X_test, y_train, y_test, **recurrent_kwargs))
        return report
