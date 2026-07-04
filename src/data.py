"""UCI Heart Disease (Cleveland) 데이터 로드 및 캐싱.

공개 데이터셋이라 개인정보·IRB 이슈 없음.
출처: https://archive.ics.uci.edu/dataset/45/heart+disease
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
    """정제된 데이터프레임 반환. 결측('?') 행은 중앙값 대치."""
    path = _download()
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    # 결측치는 열 중앙값으로 대치 (ca, thal에 소수 존재)
    df = df.fillna(df.median(numeric_only=True))
    # 타깃 이진화: 0 = 질환 없음, 1 = 질환 있음
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
    return df


def load_xy():
    df = load_dataframe()
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


if __name__ == "__main__":
    X, y = load_xy()
    print(f"샘플 {len(X)}건, 지표 {X.shape[1]}개")
    print(f"질환 있음 {int(y.sum())} / 없음 {int((1 - y).sum())}")
