"""SHAP 기반 설명 레이어.

한 환자의 예측에 대해 각 임상 지표가 위험을 얼마나 밀었는지(+)/낮췄는지(-)
분해하여 '의사 언어'로 돌려준다.
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
        """단일 환자 설명. 위험을 높인/낮춘 상위 지표를 반환."""
        proba = float(self.model.predict_proba(x_row)[0, 1])

        sv = self._explainer.shap_values(x_row)
        # 이진 분류: 클래스 1(질환 있음)에 대한 기여도
        contrib = _positive_class_shap(sv)[0]

        rows = []
        for col, val, c in zip(self.columns, x_row.iloc[0].values, contrib):
            rows.append(
                {
                    "feature": col,
                    "label": humanize(col, val),
                    "impact": float(c),
                    "direction": "위험 ↑" if c > 0 else "위험 ↓",
                }
            )
        rows.sort(key=lambda r: abs(r["impact"]), reverse=True)

        return {
            "risk_probability": proba,
            "top_factors": rows[:top_k],
            "all_factors": rows,
        }


def _positive_class_shap(shap_values):
    """SHAP 버전별 출력 형태를 클래스1 (n_samples, n_features)로 정규화."""
    if isinstance(shap_values, list):  # [class0, class1]
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return arr[:, :, 1]
    return arr
