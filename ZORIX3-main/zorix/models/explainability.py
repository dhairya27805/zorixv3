"""
zorix/models/explainability.py

SHAP explainability (Section 23), Priority 4 of this phase.

Only wired up for tree-based models (XGBoost, LightGBM, RandomForest) via
shap.TreeExplainer, which is exact and fast for trees — no KernelExplainer
(slow, approximate) is used. If `shap` isn't installed, or the model isn't
tree-based, falls back to the model's own `.feature_importances_` and says
so explicitly, so the UI never claims "SHAP explanation" when it isn't one
(Section 43 graceful-degradation pattern, same as everywhere else).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_TREE_MODEL_MODULES = ("xgboost", "lightgbm", "sklearn.ensemble")


@dataclass
class ExplanationResult:
    feature_contributions: pd.DataFrame  # columns: feature, contribution (signed, class-specific)
    method: str                          # "shap" | "feature_importance" | "unavailable"
    target_class: Optional[int]
    note: str


def _is_tree_model(model) -> bool:
    module = type(model).__module__
    return any(module.startswith(m) for m in _TREE_MODEL_MODULES)


def explain_prediction(
    model,
    X_row: pd.DataFrame,
    background: Optional[pd.DataFrame] = None,
    target_class: int = 2,  # default BUY
    top_n: int = 10,
) -> ExplanationResult:
    """
    X_row: a single-row DataFrame (the instance being explained).
    background: optional sample of training data for the explainer; for
    tree models this is used only for interventional feature perturbation
    and is optional.
    """
    if SHAP_AVAILABLE and _is_tree_model(model):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_row)

            # Multiclass tree models: shap_values is a list of arrays (one per
            # class) or a 3D array depending on the SHAP/model version.
            if isinstance(shap_values, list):
                values = shap_values[target_class][0]
            elif np.asarray(shap_values).ndim == 3:
                values = np.asarray(shap_values)[0, :, target_class]
            else:
                values = np.asarray(shap_values)[0]

            df = pd.DataFrame({
                "feature": X_row.columns,
                "contribution": values,
            })
            df["abs_contribution"] = df["contribution"].abs()
            df = df.sort_values("abs_contribution", ascending=False).head(top_n)
            df = df.drop(columns="abs_contribution")

            return ExplanationResult(
                feature_contributions=df, method="shap", target_class=target_class,
                note="SHAP TreeExplainer values for the predicted class; positive = pushes toward this class.",
            )
        except Exception as exc:
            # Fall through to feature_importances_ fallback below rather than crash.
            fallback_note = f"SHAP failed ({exc}); showing global feature importance instead."
    else:
        fallback_note = (
            "SHAP not available for this model (either 'shap' isn't installed, "
            "or this model type isn't a supported tree model)."
        )

    if hasattr(model, "feature_importances_"):
        df = pd.DataFrame({
            "feature": X_row.columns,
            "contribution": model.feature_importances_,
        }).sort_values("contribution", ascending=False).head(top_n)
        return ExplanationResult(
            feature_contributions=df, method="feature_importance", target_class=None,
            note=fallback_note + " Note: this is GLOBAL importance, not specific to this prediction.",
        )

    return ExplanationResult(
        feature_contributions=pd.DataFrame(columns=["feature", "contribution"]),
        method="unavailable", target_class=None,
        note="No explainability available for this model type.",
    )
