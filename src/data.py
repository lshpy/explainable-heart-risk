"""Load and cache the UCI Heart Disease (Cleveland) data.

Public dataset, so no privacy/IRB issues.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, TARGET_COLUMN

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "processed.cleveland.data"

COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def _download() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if not RAW_PATH.exists():
        import urllib.request

        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    return RAW_PATH


def load_dataframe() -> pd.DataFrame:
    """Return the cleaned dataframe. Missing ('?') values are median-imputed."""
    path = _download()
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    # impute missing values with the column median (a few in ca, thal)
    df = df.fillna(df.median(numeric_only=True))
    # binarize target: 0 = no disease, 1 = disease
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
    return df


def load_xy():
    df = load_dataframe()
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


if __name__ == "__main__":
    X, y = load_xy()
    print(f"{len(X)} samples, {X.shape[1]} features")
    print(f"disease {int(y.sum())} / no disease {int((1 - y).sum())}")
