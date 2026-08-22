"""Train and evaluate an explainable breast cancer screening classifier.

This educational model supports data-science learning only. It must never be
used as a medical diagnosis or a substitute for a qualified clinician.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the curated Wisconsin Diagnostic Breast Cancer dataset."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    # sklearn labels malignant as 0 and benign as 1. A positive screen is
    # therefore defined as malignant (1) to make screening metrics intuitive.
    y = (dataset.target == 0).astype(int)
    y.name = "positive_screen"
    return X, y


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def train_and_save() -> dict:
    """Fit the model, save reusable artifacts, and return evaluation metrics."""
    ARTIFACTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    X, y = load_data()
    X.to_csv(DATA / "breast_cancer_features.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    model = build_model()
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.50).astype(int)
    cm = confusion_matrix(y_test, prediction)
    importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    top_features = (
        pd.DataFrame({"feature": X.columns, "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "average_precision": round(float(average_precision_score(y_test, probability)), 4),
        "sensitivity": round(float(cm[1, 1] / cm[1].sum()), 4),
        "specificity": round(float(cm[0, 0] / cm[0].sum()), 4),
        "test_samples": int(len(y_test)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, prediction, output_dict=True),
    }
    joblib.dump({"model": model, "feature_names": list(X.columns)}, ARTIFACTS / "screening_model.joblib")
    top_features.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    results = train_and_save()
    print(json.dumps({k: v for k, v in results.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))
