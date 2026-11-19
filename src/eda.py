"""Data audit, descriptive statistics and univariate tests.

This is the data-analysis half of the project: before any model is trained,
check the data quality and measure how strongly each feature separates the
two classes on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

KEY_FEATURES = [
    "mean radius",
    "mean texture",
    "mean concave points",
    "worst area",
    "worst concavity",
    "mean smoothness",
]


def feature_group(name: str) -> str:
    if name.startswith("mean "):
        return "mean"
    if name.startswith("worst "):
        return "worst"
    return "standard error"


def audit(X: pd.DataFrame, y: pd.Series) -> dict:
    """Missing values, duplicates, IQR outliers and skewness."""
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    is_out = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
    skew = X.skew()
    return {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "all_numeric": bool(all(np.issubdtype(t, np.number) for t in X.dtypes)),
        "malignant_share": round(float(y.mean()), 4),
        "rows_with_any_iqr_outlier": int(is_out.any(axis=1).sum()),
        "outliers_per_feature": is_out.sum().sort_values(ascending=False).astype(int).to_dict(),
        "features_skew_above_1": int((skew > 1).sum()),
        "skewness": skew.round(3).sort_values(ascending=False).to_dict(),
    }


def describe_by_class(X: pd.DataFrame, y: pd.Series, features=KEY_FEATURES) -> pd.DataFrame:
    """Median and IQR per class, the robust summary used in the report."""
    rows = []
    for f in features:
        for label, name in [(0, "benign"), (1, "malignant")]:
            s = X.loc[y == label, f]
            rows.append(
                {
                    "feature": f,
                    "class": name,
                    "mean": s.mean(),
                    "sd": s.std(),
                    "median": s.median(),
                    "q1": s.quantile(0.25),
                    "q3": s.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def univariate_tests(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Mann-Whitney U test for every feature (malignant vs benign).

    Mann-Whitney is used instead of a t-test because 22 of the 30 features have
    a skewness above 1, so a normality assumption would not hold. Effect size
    is the rank-biserial correlation (-1 to 1).
    p-values are Bonferroni-corrected for the 30 tests.
    """
    ben, mal = X[y == 0], X[y == 1]
    rows = []
    for f in X.columns:
        u, p = stats.mannwhitneyu(mal[f], ben[f], alternative="two-sided")
        rows.append(
            {
                "feature": f,
                "group": feature_group(f),
                "median_benign": ben[f].median(),
                "median_malignant": mal[f].median(),
                "u_statistic": u,
                "p_value": p,
                "p_bonferroni": min(p * X.shape[1], 1.0),
                "rank_biserial": 2 * u / (len(mal) * len(ben)) - 1,
                "point_biserial_r": stats.pointbiserialr(y, X[f])[0],
            }
        )
    out = pd.DataFrame(rows)
    out["significant"] = out["p_bonferroni"] < 0.05
    return out.sort_values("rank_biserial", key=np.abs, ascending=False).reset_index(drop=True)


def pca_variance(X: pd.DataFrame) -> pd.DataFrame:
    pca = PCA().fit(StandardScaler().fit_transform(X))
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame(
        {"component": np.arange(1, len(ratio) + 1), "explained": ratio, "cumulative": np.cumsum(ratio)}
    )
