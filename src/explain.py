"""SHAP-based explanation layer.

For one patient's prediction, decompose how much each clinical feature pushed the risk up (+) or down (-)
and report it back in "clinician language".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import humanize


class Explainer:
    def __init__(self, model, columns):
        self.model = model
        self.columns = columns
        self._explainer = shap.TreeExplainer(model)

    def explain_one(self, x_row: pd.DataFrame, top_k: int = 5) -> dict:
        """Explain a single patient. Returns the top features that raised/lowered the risk."""
        proba = float(self.model.predict_proba(x_row)[0, 1])

        sv = self._explainer.shap_values(x_row)
        # binary classification: contributions toward class 1 (disease)
        contrib = _positive_class_shap(sv)[0]

        rows = []
        for col, val, c in zip(self.columns, x_row.iloc[0].values, contrib):
            rows.append(
                {
                    "feature": col,
                    "label": humanize(col, val),
                    "impact": float(c),
                    "direction": "risk ↑" if c > 0 else "risk ↓",
                }
            )
        rows.sort(key=lambda r: abs(r["impact"]), reverse=True)

        return {
            "risk_probability": proba,
            "top_factors": rows[:top_k],
            "all_factors": rows,
        }


def _positive_class_shap(shap_values):
    """Normalize the version-dependent SHAP output shape to class 1 (n_samples, n_features)."""
    if isinstance(shap_values, list):  # [class0, class1]
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return arr[:, :, 1]
    return arr
