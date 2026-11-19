"""Compare, tune and evaluate the screening classifiers.

Run from the repo root:
    python -m src.train

Outputs
    artifacts/screening_model.joblib   selected model + threshold (for app.py)
    reports/*.json / *.csv             data audit, statistical tests and model metrics
    figures/*.png                      every chart in the report

Educational project only - not a diagnostic tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate

from src.data import RANDOM_STATE, load_data, split
from src.models import candidates

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
TARGET_SENSITIVITY = 0.98  # used to pick the screening threshold


def evaluate_at_threshold(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred),
        "specificity": float(tn / (tn + fp)),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compare_models(X_train, y_train, X_test, y_test):
    """Grid-search each model, then score it with 5-fold CV and on the test set."""
    rows, fitted, fold_scores = [], {}, []
    for name, (pipe, grid) in candidates().items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=CV, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        cv = cross_validate(
            clone(best),
            X_train,
            y_train,
            cv=CV,
            scoring=["roc_auc", "accuracy", "recall", "precision", "f1"],
        )
        for i, s in enumerate(cv["test_roc_auc"]):
            fold_scores.append({"model": name, "fold": i + 1, "roc_auc": s})
        proba = best.predict_proba(X_test)[:, 1]
        tm = evaluate_at_threshold(y_test, proba)
        rows.append(
            {
                "model": name,
                "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
                "cv_roc_auc_mean": cv["test_roc_auc"].mean(),
                "cv_roc_auc_std": cv["test_roc_auc"].std(),
                "cv_accuracy_mean": cv["test_accuracy"].mean(),
                "cv_recall_mean": cv["test_recall"].mean(),
                "cv_recall_std": cv["test_recall"].std(),
                "cv_precision_mean": cv["test_precision"].mean(),
                "cv_f1_mean": cv["test_f1"].mean(),
                **{f"test_{k}": v for k, v in tm.items()},
            }
        )
        fitted[name] = best
    table = pd.DataFrame(rows)
    return table, fitted, pd.DataFrame(fold_scores)


def pick_threshold(model, X_train, y_train, target=TARGET_SENSITIVITY):
    """Highest threshold whose out-of-fold sensitivity on TRAIN is >= target.

    The test set is never used here, so the test numbers stay honest.
    """
    oof = cross_val_predict(clone(model), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    sweep = []
    for t in grid:
        pred = (oof >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()
        sweep.append({"threshold": t, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp)})
    sweep = pd.DataFrame(sweep)
    ok = sweep[sweep["sensitivity"] >= target]
    chosen = float(ok["threshold"].max()) if len(ok) else 0.5
    return chosen, sweep


def run() -> dict:
    for d in (ARTIFACTS, REPORTS, FIGURES):
        d.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    table, fitted, folds = compare_models(X_train, y_train, X_test, y_test)
    # choose on cross-validation, not on the test set
    table = table.sort_values(["cv_roc_auc_mean", "cv_recall_mean"], ascending=False).reset_index(drop=True)
    best_name = table.loc[0, "model"]
    best = fitted[best_name]

    threshold, sweep = pick_threshold(best, X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    default_eval = evaluate_at_threshold(y_test, proba, 0.5)
    tuned_eval = evaluate_at_threshold(y_test, proba, threshold)

    imp = permutation_importance(
        best, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    joblib.dump(
        {
            "model": best,
            "model_name": best_name,
            "threshold": threshold,
            "feature_names": list(X.columns),
            "feature_medians": X.median().to_dict(),
            "feature_min": X.min().to_dict(),
            "feature_max": X.max().to_dict(),
        },
        ARTIFACTS / "screening_model.joblib",
    )

    table.round(4).to_csv(REPORTS / "model_comparison.csv", index=False)
    folds.round(4).to_csv(REPORTS / "cv_fold_scores.csv", index=False)
    sweep.round(4).to_csv(REPORTS / "threshold_sweep.csv", index=False)
    importance.round(5).to_csv(REPORTS / "feature_importance.csv", index=False)

    summary = {
        "dataset": {
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "malignant": int(y.sum()),
            "benign": int((y == 0).sum()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "selected_model": best_name,
        "selection_rule": "highest mean 5-fold CV ROC-AUC on the training set",
        "target_sensitivity": TARGET_SENSITIVITY,
        "chosen_threshold": threshold,
        "test_at_0.5": default_eval,
        "test_at_tuned_threshold": tuned_eval,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "table": table,
        "fitted": fitted,
        "folds": folds,
        "best_name": best_name,
        "best": best,
        "threshold": threshold,
        "sweep": sweep,
        "importance": importance,
        "summary": summary,
    }


def train_and_save() -> dict:
    """Kept for app.py: train everything and return the summary."""
    return run()["summary"]


def run_eda(X, y) -> dict:
    """Data audit + statistical tests, saved to reports/."""
    from src import eda

    REPORTS.mkdir(exist_ok=True)
    summary = eda.audit(X, y)
    tests = eda.univariate_tests(X, y)
    var = eda.pca_variance(X)
    summary["significant_features_bonferroni"] = int(tests["significant"].sum())
    summary["components_for_95pct_variance"] = int((var["cumulative"] < 0.95).sum() + 1)
    (REPORTS / "data_audit.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    # p-values are tiny, so keep them in scientific notation instead of rounding them to zero
    tests.round({c: 4 for c in tests.columns if not c.startswith("p_")}).to_csv(
        REPORTS / "statistical_tests.csv", index=False, float_format="%.6g"
    )
    eda.describe_by_class(X, y).round(4).to_csv(REPORTS / "descriptive_stats.csv", index=False)
    var.round(4).to_csv(REPORTS / "pca_variance.csv", index=False)
    return {"audit": summary, "tests": tests, "pca": var}


if __name__ == "__main__":
    from src import clustering, plots

    res = run()
    stats_res = run_eda(res["X"], res["y"])
    km = clustering.run(res["X"], res["y"])
    (REPORTS / "clustering.json").write_text(
        json.dumps(km["summary"], indent=2, default=float), encoding="utf-8"
    )
    plots.make_all(res, km, stats_res, FIGURES)

    cols = ["model", "cv_roc_auc_mean", "cv_recall_mean", "test_accuracy"]
    cols += ["test_sensitivity", "test_specificity", "test_roc_auc"]
    print(res["table"][cols].round(3).to_string(index=False))
    print(f"\nSelected: {res['best_name']}  |  screening threshold: {res['threshold']}")
    print("Test @0.50 :", {k: round(v, 3) for k, v in res["summary"]["test_at_0.5"].items()})
    print("Test @tuned:", {k: round(v, 3) for k, v in res["summary"]["test_at_tuned_threshold"].items()})
    print("K-means:", km["summary"])
