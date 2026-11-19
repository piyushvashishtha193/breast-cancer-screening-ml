"""Data loading and splitting for the breast cancer screening project."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return the 30 WDBC features and a target where 1 = malignant.

    scikit-learn ships the labels as 0 = malignant, 1 = benign. I flip them so
    that the positive class is the one we care about catching (malignant),
    which makes recall/sensitivity read naturally.
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    y = (raw.target == 0).astype(int).rename("malignant")
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified 75/25 split, fixed seed so every run is comparable."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
