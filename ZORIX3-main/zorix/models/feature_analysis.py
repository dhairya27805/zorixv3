"""
zorix/models/feature_analysis.py

Feature ablation and feature stability (Priority 1 of this phase).

Feature ablation here is implemented via permutation importance
(sklearn.inspection.permutation_importance): each feature's values are
independently shuffled and the drop in held-out performance is measured.
This is a standard, efficient stand-in for "remove the feature and
retrain" ablation — it isolates each feature's marginal contribution
without the cost of N full retrains, and unlike raw
`.feature_importances_` it is measured on held-out data, not training
data, so it isn't inflated by overfitting.

Feature stability measures whether importance rankings are consistent
across walk-forward folds (i.e., whether a feature that looks important
is a persistent, real relationship, or noise from a single lucky split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from zorix.models.splits import infer_horizon
from zorix.models.walkforward import walk_forward_split


@dataclass
class AblationResult:
    importances: pd.DataFrame  # columns: feature, importance_mean, importance_std
    scoring: str
    n_repeats: int


def run_feature_ablation(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = "accuracy",
    n_repeats: int = 10,
    random_state: int = 42,
) -> AblationResult:
    """`model` must already be fitted. Returns features sorted by importance
    (most important first)."""
    result = permutation_importance(
        model, X_test, y_test, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return AblationResult(df, scoring, n_repeats)


@dataclass
class StabilityResult:
    per_fold_importance: pd.DataFrame   # rows=features, cols=fold_1..fold_n
    mean_importance: pd.Series
    std_importance: pd.Series
    coefficient_of_variation: pd.Series  # std/|mean| — lower = more stable
    mean_rank_correlation: float         # average pairwise Spearman rank corr across folds
    n_folds_used: int


def run_feature_stability(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_fn: Callable[[pd.DataFrame, pd.Series], object],
    n_splits: int = 5,
    min_train_size: int = 400,
    scoring: str = "accuracy",
    n_repeats: int = 5,
    random_state: int = 42,
) -> StabilityResult:
    """
    train_fn(X_train, y_train) -> fitted model (sklearn-compatible, with
    .predict_proba or .predict as required by `scoring`).

    For each walk-forward fold: fit on that fold's train window, compute
    permutation importance on that fold's test window, collect into a
    features x folds matrix, then report mean/std/coefficient-of-variation
    per feature and the average pairwise Spearman rank correlation between
    folds' importance rankings (higher = more stable across time).
    """
    clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    fold_importances = []

    for fold_i, (train_idx, test_idx) in enumerate(
        walk_forward_split(len(clean), n_splits, min_train_size, gap=infer_horizon(target_col)), start=1
    ):
        train, test = clean.iloc[train_idx], clean.iloc[test_idx]
        if len(test) < 20:
            continue
        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]

        model = train_fn(X_train, y_train)
        result = permutation_importance(
            model, X_test, y_test, scoring=scoring, n_repeats=n_repeats,
            random_state=random_state, n_jobs=-1,
        )
        fold_importances.append(pd.Series(result.importances_mean, index=feature_cols, name=f"fold_{fold_i}"))

    if not fold_importances:
        empty = pd.Series(dtype=float)
        return StabilityResult(pd.DataFrame(), empty, empty, empty, float("nan"), 0)

    matrix = pd.concat(fold_importances, axis=1)
    mean_imp = matrix.mean(axis=1)
    std_imp = matrix.std(axis=1)
    cov = (std_imp / mean_imp.abs().replace(0, np.nan)).fillna(np.inf)

    # Average pairwise Spearman correlation of feature rankings between folds.
    corrs = []
    cols = matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho, _ = spearmanr(matrix[cols[i]], matrix[cols[j]])
            if not np.isnan(rho):
                corrs.append(rho)
    mean_rank_corr = float(np.mean(corrs)) if corrs else float("nan")

    return StabilityResult(
        per_fold_importance=matrix,
        mean_importance=mean_imp.sort_values(ascending=False),
        std_importance=std_imp,
        coefficient_of_variation=cov,
        mean_rank_correlation=mean_rank_corr,
        n_folds_used=len(fold_importances),
    )
