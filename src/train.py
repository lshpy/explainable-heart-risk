"""Train the heart disease risk model.

The goal is an "explainable" baseline, not an accuracy race.
Why a tree model: SHAP TreeExplainer gives fast, exact attribution.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split

from .data import load_xy

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"

RANDOM_STATE = 42


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_and_save() -> dict:
    X, y = load_xy()
    model = build_model()

    # honest performance estimate via cross-validation
    cv_proba = cross_val_predict(
        model, X, y, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    cv_auc = roc_auc_score(y, cv_proba)
    cv_acc = accuracy_score(y, (cv_proba >= 0.5).astype(int))

    # final fit on all data, then save
    model.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_PATH)

    metrics = {"cv_auc": round(cv_auc, 3), "cv_accuracy": round(cv_acc, 3)}
    return metrics


def load_model():
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["columns"]


if __name__ == "__main__":
    m = train_and_save()
    print(f"5-fold CV AUC={m['cv_auc']}  accuracy={m['cv_accuracy']}")
    print(f"Model saved: {MODEL_PATH}")
