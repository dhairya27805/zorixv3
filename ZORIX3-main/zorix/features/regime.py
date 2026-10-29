"""
zorix/features/regime.py

Market regime detection (Section 18): BULL/BEAR x LOW/HIGH VOL.

Preference order, all graceful (Section 6 / Section 43):
    1. hmmlearn.GaussianHMM   (preferred — proper Hidden Markov Model)
    2. sklearn GaussianMixture (unsupervised fallback if hmmlearn missing)
    3. Quantile-based regime rule (always available, no extra dependency)

Whichever engine actually ran is recorded on the result so the UI never
claims "HMM regime detection" when it silently fell back to something else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeResult:
    labels: pd.Series          # per-row regime label, e.g. "BULL / HIGH VOL"
    confidence: pd.Series      # per-row confidence in [0, 1]
    engine: str                # "hmm" | "gmm" | "quantile"
    current_regime: Optional[str]
    current_confidence: Optional[float]


_REGIME_NAMES = ["BEAR / LOW VOL", "BEAR / HIGH VOL", "BULL / LOW VOL", "BULL / HIGH VOL"]


def _build_regime_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    ret = df["Close"].pct_change()
    trend = df["Close"].pct_change(lookback)
    vol = ret.rolling(lookback).std() * np.sqrt(252)
    feats = pd.DataFrame({"trend": trend, "vol": vol}, index=df.index)
    return feats.dropna()


def _label_from_trend_vol(trend: float, vol: float, vol_median: float) -> str:
    bull = trend >= 0
    high_vol = vol >= vol_median
    if bull and not high_vol:
        return "BULL / LOW VOL"
    if bull and high_vol:
        return "BULL / HIGH VOL"
    if not bull and not high_vol:
        return "BEAR / LOW VOL"
    return "BEAR / HIGH VOL"


def _hmm_filtered_posteriors(model, X: np.ndarray) -> np.ndarray:
    """Causal (forward-algorithm) state posteriors P(state_t | x_0..x_t).
    hmmlearn's predict/predict_proba are smoothed over the WHOLE sequence
    (they use future observations), so they must not be used for features."""
    from scipy.special import logsumexp
    logb = model._compute_log_likelihood(X)
    loga = np.log(np.clip(model.transmat_, 1e-300, None))
    la = np.log(np.clip(model.startprob_, 1e-300, None)) + logb[0]
    la = la - logsumexp(la)
    out = np.empty_like(logb)
    out[0] = la
    for t in range(1, len(X)):
        la = logb[t] + logsumexp(la[:, None] + loga, axis=0)
        la = la - logsumexp(la)
        out[t] = la
    return np.exp(out)


def _fit(engine: str, X: np.ndarray, n_regimes: int):
    if engine == "hmm":
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM(n_components=n_regimes, covariance_type="diag",
                           n_iter=200, random_state=42).fit(X)
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=n_regimes, random_state=42, n_init=3).fit(X)


def _causal_model_regime(feats: pd.DataFrame, engine: str, n_regimes: int,
                         min_fit: int, refit_every: int):
    """Expanding-window regime model. The model for block [s, e) is fit ONLY on
    rows [0, s); state->label mapping and the vol median also use only [0, s).
    Row t therefore never depends on rows after t (HMM uses forward filtering).
    Rows before `min_fit` get no regime."""
    X = feats.values
    n = len(X)
    labels = np.full(n, None, dtype=object)
    conf = np.full(n, np.nan)
    for s in range(min_fit, n, refit_every):
        e = min(s + refit_every, n)
        model = _fit(engine, X[:s], n_regimes)
        vol_median = float(np.median(X[:s, 1]))
        state_label = [_label_from_trend_vol(m[0], m[1], vol_median) for m in model.means_]
        if engine == "hmm":
            post = _hmm_filtered_posteriors(model, X[:e])[s:e]
        else:
            post = model.predict_proba(X[s:e])
        for k, i in enumerate(range(s, e)):
            labels[i] = state_label[int(post[k].argmax())]
            conf[i] = float(post[k].max())
    keep = np.array([l is not None for l in labels])
    return (pd.Series(labels[keep], index=feats.index[keep]),
            pd.Series(conf[keep], index=feats.index[keep]))


def detect_regime(df: pd.DataFrame, lookback: int = 20, n_regimes: int = 4,
                  min_fit: int = 252, refit_every: int = 63) -> RegimeResult:
    """Causal regime detection: the label/confidence at date t uses only data
    up to t (see _causal_model_regime). Rows without enough history are
    dropped from the result (NaN after alignment)."""
    feats = _build_regime_features(df, lookback)
    if feats.empty:
        empty = pd.Series(dtype=object)
        return RegimeResult(empty, empty, "unavailable", None, None)

    labels = conf_series = None
    engine = "quantile"
    if len(feats) > min_fit:
        for eng in ("hmm", "gmm"):
            try:
                labels, conf_series = _causal_model_regime(feats, eng, n_regimes, min_fit, refit_every)
                engine = eng
                break
            except Exception:  # ImportError (no hmmlearn) or fit failure -> next engine
                labels = conf_series = None

    if labels is None:
        # Quantile rule with an EXPANDING (causal) vol median.
        vol_med = feats["vol"].expanding(min_periods=60).median()
        valid = vol_med.notna()
        f = feats[valid]
        labels = pd.Series([_label_from_trend_vol(t, v, m) for t, v, m
                            in zip(f["trend"], f["vol"], vol_med[valid])], index=f.index)
        conf_series = pd.Series(0.5, index=f.index)  # no probabilistic confidence available
        engine = "quantile"

    if labels.empty:
        empty = pd.Series(dtype=object)
        return RegimeResult(empty, empty, "unavailable", None, None)

    current_regime = labels.iloc[-1]
    current_confidence = float(conf_series.iloc[-1])
    return RegimeResult(labels, conf_series, engine, current_regime, current_confidence)


def attach_regime_features(df: pd.DataFrame, regime: RegimeResult) -> pd.DataFrame:
    """Adds regime as one-hot columns + confidence, aligned to df's index.
    Rows before the regime model had enough lookback data get NaN (dropped
    later during training, same as every other rolling feature)."""
    out = df.copy()
    if regime.engine == "unavailable":
        return out

    aligned_labels = regime.labels.reindex(out.index)
    aligned_conf = regime.confidence.reindex(out.index)

    for name in _REGIME_NAMES:
        col = (aligned_labels == name).astype(float)
        col[aligned_labels.isna()] = np.nan  # no regime yet != "not in this regime"
        out[f"Regime_{name.replace(' / ', '_').replace(' ', '')}"] = col
    out["Regime_Confidence"] = aligned_conf
    out["Regime_Engine"] = regime.engine
    return out
