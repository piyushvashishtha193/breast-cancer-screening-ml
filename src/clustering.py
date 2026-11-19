"""Unsupervised check: does K-means find the benign/malignant split on its own?

The labels are NOT used to fit anything here. They are only used afterwards to
see how well the clusters line up with the diagnosis.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data import RANDOM_STATE


def run(X: pd.DataFrame, y: pd.Series, k_range=range(2, 9)) -> dict:
    Z = StandardScaler().fit_transform(X)

    scan = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        scan.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_)})
    scan = pd.DataFrame(scan)

    km2 = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(Z)
    labels = km2.labels_
    crosstab = pd.crosstab(pd.Series(labels, name="cluster"), y.rename("malignant"))
    # map each cluster to its majority diagnosis, then count agreement
    majority = crosstab.idxmax(axis=1)
    agreement = (pd.Series(labels).map(majority).values == y.values).mean()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Z)

    summary = {
        "best_silhouette_k": int(scan.loc[scan["silhouette"].idxmax(), "k"]),
        "silhouette_k2": round(float(scan.loc[scan["k"] == 2, "silhouette"].iloc[0]), 4),
        "adjusted_rand_index_k2": round(float(adjusted_rand_score(y, labels)), 4),
        "cluster_label_agreement_k2": round(float(agreement), 4),
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "crosstab": {int(c): {int(k): int(v) for k, v in row.items()} for c, row in crosstab.iterrows()},
    }
    return {"scan": scan, "labels": labels, "coords": coords, "pca": pca, "summary": summary}
