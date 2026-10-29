"""
Tests for the Part-1 leakage/evaluation fixes (standard yfinance model only).
No network needed. Run:  python -m pytest zorix/tests/test_part1_leakage.py -v
or (no pytest):          python zorix/tests/test_part1_leakage.py
"""
import numpy as np
import pandas as pd

from zorix.features.regime import (_hmm_filtered_posteriors, attach_regime_features,
                                   detect_regime)
from zorix.features.target import add_bhs_targets, sklearn_label
from zorix.models.ensemble import (build_oof_base_predictions, fit_stacking,
                                   predict_stack_new)
from zorix.models.regression import ReturnRegressionTrainer
from zorix.models.splits import infer_horizon, purged_split_points
from zorix.models.walkforward import run_walk_forward, walk_forward_split


def _prices(n=700, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    return pd.DataFrame({"Close": close}, index=idx)


def _xy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx, columns=list("abcd"))
    y = pd.Series(rng.integers(0, 3, n), index=idx)
    return X, y


# ---- 1. purged chronological split ----
def test_infer_horizon():
    assert infer_horizon("Target_5D") == 5 and infer_horizon("FwdReturn_20D") == 20
    assert infer_horizon("whatever") == 0


def test_purged_split_has_no_label_overlap():
    for h in (1, 5, 20):
        n = 500
        train_end, test_start = purged_split_points(n, 0.2, h)
        assert test_start == 400 and train_end == 400 - h
        # last train label uses Close[train_end-1+h]; must be before test start
        assert (train_end - 1) + h < test_start
        # old behaviour (no purge) overlapped:
        assert (test_start - 1) + h >= test_start


# ---- 2. targets ----
def test_targets_nan_when_forward_window_or_vol_missing():
    out = add_bhs_targets(_prices(120), [5], vol_lookback=20, k_buy=0.5, k_sell=0.5, min_threshold=0.002)
    assert out["Target_5D"].iloc[-5:].isna().all(), "last h rows have no future -> NaN, not HOLD"
    assert out["Target_5D"].iloc[:19].isna().all(), "no vol history yet -> NaN"
    assert set(out["Target_5D"].dropna().unique()) <= {-1, 0, 1}
    assert set(sklearn_label(out["Target_5D"].dropna()).unique()) <= {0, 1, 2}


# ---- 3. regime causality ----
def _assert_causal(engine_df, cut):
    full = detect_regime(engine_df)
    trunc = detect_regime(engine_df.iloc[:cut])
    common = trunc.labels.index
    assert len(common) > 0
    assert (full.labels.reindex(common) == trunc.labels).all(), "labels changed when future data removed"
    assert np.allclose(full.confidence.reindex(common).values, trunc.confidence.values)
    return full


def test_regime_gmm_is_causal():
    df = _prices(700)
    full = _assert_causal(df, 500)
    assert full.engine in ("hmm", "gmm")
    # perturbing ONLY the future must not change past labels
    df2 = df.copy()
    df2.iloc[550:, 0] *= np.linspace(1, 3, len(df2) - 550)
    r2 = detect_regime(df2)
    past = full.labels.index[full.labels.index < df.index[550]]
    assert (full.labels.loc[past] == r2.labels.reindex(past)).all()


def test_regime_quantile_fallback_is_causal():
    df = _prices(200)  # too short for model engine (min_fit=252) -> quantile
    full = _assert_causal(df, 150)
    assert full.engine == "quantile"


def test_regime_features_nan_before_warmup():
    df = _prices(700)
    reg = detect_regime(df)
    out = attach_regime_features(df, reg)
    cols = [c for c in out.columns if c.startswith("Regime_") and c != "Regime_Engine"]
    assert out[cols].iloc[:100].isna().all().all(), "warm-up rows must be NaN, not 0 = 'not in regime'"
    assert out[cols].iloc[-1].notna().all()


def test_hmm_filter_is_causal_and_matches_bruteforce():
    class Fake:  # 2-state HMM, 1-D emissions; same interface the filter uses
        startprob_ = np.array([0.6, 0.4])
        transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])

        def _compute_log_likelihood(self, X):
            mu = np.array([-1.0, 1.0])
            return -0.5 * (X - mu[None, :]) ** 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 1))
    m = Fake()
    post = _hmm_filtered_posteriors(m, X)
    # brute-force P(s_t | x_0..x_t) by enumerating all state paths up to t
    import itertools
    logb = m._compute_log_likelihood(X)
    for t in range(len(X)):
        w = np.zeros(2)
        for path in itertools.product([0, 1], repeat=t + 1):
            p = m.startprob_[path[0]] * np.exp(logb[0, path[0]])
            for k in range(1, t + 1):
                p *= m.transmat_[path[k - 1], path[k]] * np.exp(logb[k, path[k]])
            w[path[-1]] += p
        assert np.allclose(post[t], w / w.sum()), f"mismatch at t={t}"
    # causality: first 5 posteriors identical when computed on the 5-row prefix
    assert np.allclose(post[:5], _hmm_filtered_posteriors(m, X[:5]))


# ---- 4. stacking ----
class _Recorder:
    """Majority-class dummy that records the index set it was fit on."""
    fits = []

    def fit(self, X, y):
        _Recorder.fits.append(np.array(X.index))
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, X):
        return np.tile([0.3, 0.3, 0.4], (len(X), 1))


def test_oof_is_purged_by_gap():
    X, y = _xy(300)
    _Recorder.fits = []
    gap = 5
    oof = build_oof_base_predictions(X, y, {"R": _Recorder}, n_splits=4, gap=gap)
    # first predicted row must be >= gap rows after the end of the first fold's train set
    pos = np.where(oof.notna().all(axis=1))[0]
    assert len(pos) > 0 and pos.min() >= len(_Recorder.fits[0]) + gap


def test_stack_is_fit_on_train_only_and_scores_holdout():
    X, y = _xy(400)
    train_end, split = purged_split_points(len(X), 0.2, 5)
    Xtr, ytr, Xte = X.iloc[:train_end], y.iloc[:train_end], X.iloc[split:]
    _Recorder.fits = []
    res = fit_stacking(Xtr, ytr, None, {"R": _Recorder}, n_splits=4, gap=5)
    seen = set(np.concatenate(_Recorder.fits))
    assert seen <= set(Xtr.index), "stack fit touched rows outside the training set"
    assert not (seen & set(Xte.index))
    p = predict_stack_new(res, Xte)
    assert p.shape == (len(Xte), 3) and np.allclose(p.sum(axis=1), 1)


# ---- 5. model selection ----
def test_regression_selection_ignores_test_data():
    rng = np.random.default_rng(3)
    n = 500
    idx = pd.bdate_range("2019-01-01", periods=n)
    X = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=list("abcde"))
    y = pd.Series(0.01 * X["a"] + rng.normal(0, 0.01, n), index=idx)
    Xtr, ytr, Xte, yte = X.iloc[:380], y.iloc[:380], X.iloc[400:], y.iloc[400:]
    t = lambda: ReturnRegressionTrainer(random_state=0)
    r1 = t().train_all(Xtr, Xte, ytr, yte, gap=5)
    r2 = t().train_all(Xtr, Xte, ytr, -yte * 50 + 7, gap=5)  # sabotage the test labels
    assert r1.selected_name is not None and r1.selected_name == r2.selected_name
    assert r1.validation_scores == r2.validation_scores
    assert r1.best().name == r1.selected_name
    assert r1.results[0].metrics != r2.results[0].metrics  # test metrics DO change (reporting only)


# ---- 6. walk-forward ----
def test_walk_forward_gap_purges_horizon():
    for train_idx, test_idx in walk_forward_split(1000, 5, 300, gap=5):
        assert test_idx.min() - train_idx.max() > 5


def test_run_walk_forward_uses_horizon_gap_and_handles_raw_labels():
    n = 900
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2017-01-01", periods=n)
    df = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=list("abc"))
    df["Target_5D"] = rng.integers(-1, 2, n)          # raw {-1,0,1} as built by add_bhs_targets
    df["FwdReturn_5D"] = rng.normal(0, 0.02, n)
    seen = []

    def train_fn(Xtr, Xte, ytr, yte):
        assert set(ytr.unique()) <= {0, 1, 2}, "labels must be remapped to {0,1,2}"
        seen.append((Xtr.index.max(), Xte.index.min()))
        return None

    predict_fn = lambda m, X: np.tile([0.2, 0.3, 0.5], (len(X), 1))
    rep = run_walk_forward(df, list("abc"), "Target_5D", "FwdReturn_5D", train_fn, predict_fn,
                           n_splits=4, min_train_size=300)
    assert len(rep.folds) >= 2
    for tr_end, te_start in seen:
        assert te_start - tr_end > 5  # run_walk_forward re-indexes rows 0..n-1


if __name__ == "__main__":
    import sys
    import traceback
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
