"""
zorix/tests/test_signal_balance.py

Regression tests for the "standard model almost always outputs HOLD" bug.

Root cause (see FIXES.md): the BUY/HOLD/SELL label is imbalanced by
construction (HOLD is the plurality class, typically ~50-55% of rows).
On noisy/weak-signal features, an unweighted classifier doesn't need to
actually separate the classes to minimize training loss — predicting
HOLD everywhere is already a low-loss strategy, so argmax(predict_proba)
collapses to HOLD on nearly every row even when the per-class mean
probabilities are reasonably spread out. Combined with the NO TRADE
edge/confidence gate, real deployments saw almost no BUY/SELL signals.

Fix: class_weight="balanced" (LogReg, RandomForest, LightGBM) and
equivalent balanced sample_weight (XGBoost, which has no multiclass
class_weight kwarg) in zorix/models/trainer.py and
zorix/models/ensemble.py's base-model factories.

These tests cover:
  1. decide_signal still produces every one of BUY / SELL / HOLD / NO
     TRADE for the probability inputs that should produce them (sanity
     check the gate itself was never the problem).
  2. On synthetic imbalanced data, a model trained with the fixed
     ModelTrainer does NOT collapse to predicting HOLD on (almost) every
     test row -- the specific failure mode this fix addresses.
  3. _balanced_sample_weight produces the expected per-class weights.
"""

import numpy as np
import pandas as pd
import pytest

from zorix.config.settings import DEFAULT_CONFIG as cfg
from zorix.features.target import add_bhs_targets, sklearn_label
from zorix.features.technical import build_technical_features
from zorix.models.splits import purged_split_points
from zorix.models.trainer import ModelTrainer, _balanced_sample_weight
from zorix.signals.signal_engine import decide_signal


def _make_ohlcv(n=1200, seed=0, drift=0.0, vol=0.012):
    """Near-random-walk synthetic price series (no real edge expected) --
    deliberately hard for a classifier, which is exactly the regime where
    the always-HOLD collapse showed up."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2013-01-01", periods=n)
    rets = rng.normal(drift, vol, n)
    wiggle = np.sin(np.linspace(0, 20, n)) * 0.001  # mild structure so features aren't pure noise
    close = 100 * (1 + pd.Series(rets) + wiggle).cumprod()
    df = pd.DataFrame({
        "Open": close.values * (1 - 0.001),
        "High": close.values * (1 + 0.004),
        "Low": close.values * (1 - 0.004),
        "Close": close.values,
        "Volume": rng.integers(100_000, 1_000_000, n),
    }, index=dates)
    return df


def _train_test_split_for_horizon(df, horizon=5):
    df = build_technical_features(df, cfg)
    df = add_bhs_targets(df, cfg.HORIZONS, cfg.VOL_LOOKBACK, cfg.K_BUY, cfg.K_SELL, cfg.MIN_THRESHOLD)
    target_col, fwd_col = f"Target_{horizon}D", f"FwdReturn_{horizon}D"
    feature_cols = [c for c in df.columns
                    if c not in cfg.DROP_COLS
                    and not c.startswith("Threshold_")
                    and not c.startswith("Target_")
                    and not c.startswith("FwdReturn_")]
    clean = df.dropna(subset=feature_cols + [target_col, fwd_col]).copy()
    X = clean[feature_cols]
    y = sklearn_label(clean[target_col])
    train_end, split = purged_split_points(len(X), cfg.TEST_SIZE, horizon)
    return X.iloc[:train_end], X.iloc[split:], y.iloc[:train_end], y.iloc[split:]


# ---- decide_signal: all four outcomes are actually reachable ----

def test_decide_signal_produces_buy():
    proba = np.array([0.05, 0.15, 0.80])
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "BUY"
    assert reason is None


def test_decide_signal_produces_sell():
    proba = np.array([0.78, 0.14, 0.08])
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "SELL"
    assert reason is None


def test_decide_signal_produces_hold_when_hold_genuinely_wins():
    # HOLD clearly ahead of both runners-up -> a real HOLD call, not a gate artifact.
    proba = np.array([0.12, 0.76, 0.12])
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "HOLD"
    assert reason is None


def test_decide_signal_produces_no_trade_when_edge_too_thin():
    proba = np.array([0.30, 0.35, 0.35])
    signal, confidence, reason = decide_signal(proba, min_edge=0.12, min_confidence=0.45)
    assert signal == "NO TRADE"
    assert reason is not None


# ---- The actual bug: trained models must not collapse to all-HOLD ----

def test_balanced_sample_weight_matches_sklearn_formula():
    y = pd.Series([0, 0, 0, 1, 1, 2])  # 3x class0, 2x class1, 1x class2
    w = _balanced_sample_weight(y)
    n, k = len(y), 3
    expected = {0: n / (k * 3), 1: n / (k * 2), 2: n / (k * 1)}
    for yi, wi in zip(y, w):
        assert wi == pytest.approx(expected[yi])


def test_trained_models_do_not_collapse_to_always_hold():
    """This is the core regression test for the bug report: on imbalanced,
    weak-signal data, no trained model's predicted-class distribution on
    the held-out set should be dominated by HOLD to the degree the
    unweighted models were (>=95% HOLD, reproduced pre-fix). With
    class_weight="balanced" applied, HOLD's share of predictions should
    stay within a reasonable band of its share of the *labels*, not
    swallow almost everything regardless of the input features.
    """
    df = _make_ohlcv(n=1500, seed=1)
    X_train, X_test, y_train, y_test = _train_test_split_for_horizon(df, horizon=5)

    # Sanity: HOLD is the plurality label class (confirms the regime the
    # bug occurs in), but not an overwhelming majority.
    hold_frac_labels = (y_test == 1).mean()
    assert 0.35 < hold_frac_labels < 0.75

    trainer = ModelTrainer(random_state=42)
    report = trainer.train_all(X_train, X_test, y_train, y_test, include_lstm=False, include_gru=False)

    ran = [r for r in report.results if r.ran]
    assert len(ran) >= 2, "expected at least Logistic Regression + Random Forest to run"

    for result in ran:
        preds = result.proba_test.argmax(axis=1)
        hold_frac_preds = (preds == 1).mean()
        # The bug produced >=0.95 (observed 0.97-0.99) regardless of the
        # true label mix. After the fix this must not happen.
        assert hold_frac_preds < 0.90, (
            f"{result.name} predicted HOLD on {hold_frac_preds:.0%} of test rows "
            f"(label HOLD share was {hold_frac_labels:.0%}) -- looks like the "
            f"always-HOLD collapse is back."
        )
        # Also confirm it isn't collapsing to a *different* single class instead.
        pred_class_counts = pd.Series(preds).value_counts(normalize=True)
        assert pred_class_counts.max() < 0.92


def test_end_to_end_signal_distribution_is_not_hold_only():
    """Averaging the (fixed) base models' probabilities and running every
    test-set row through decide_signal must yield more than one signal
    type -- i.e. the pipeline is actually capable of calling BUY/SELL,
    not just HOLD/NO TRADE forever."""
    df = _make_ohlcv(n=1500, seed=2)
    X_train, X_test, y_train, y_test = _train_test_split_for_horizon(df, horizon=5)

    trainer = ModelTrainer(random_state=42)
    report = trainer.train_all(X_train, X_test, y_train, y_test, include_lstm=False, include_gru=False)
    probas = np.mean([r.proba_test for r in report.results if r.ran], axis=0)

    signals = [decide_signal(row, cfg.MIN_EDGE, cfg.MIN_CONFIDENCE)[0] for row in probas]
    distinct_signals = set(signals)
    assert "BUY" in distinct_signals or "SELL" in distinct_signals, (
        f"Pipeline never produced BUY or SELL across {len(signals)} test rows: {pd.Series(signals).value_counts()}"
    )
