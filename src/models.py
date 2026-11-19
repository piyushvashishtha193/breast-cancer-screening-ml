"""Candidate classifiers and their hyper-parameter grids.

These are the four supervised models covered during the YBI internship
(logistic regression, KNN, decision tree, random forest). Every model sits in a
Pipeline so the scaler is only ever fitted on training folds.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import RANDOM_STATE


def candidates() -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_grid)}."""
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "KNN": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            # trees don't need scaling, the step is kept only so all pipelines look alike
            Pipeline(
                [
                    ("scale", "passthrough"),
                    ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__max_depth": [3, 4, 5, 6, None], "clf__min_samples_leaf": [1, 3, 5]},
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("scale", "passthrough"),
                    (
                        "clf",
                        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 6, 10],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
    }
