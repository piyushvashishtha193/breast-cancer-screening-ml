"""All figures used in the notebook and the internship report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.eda import KEY_FEATURES

BENIGN, MALIGNANT = "#2a78d6", "#eb6834"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _save(fig, out: Path, name: str):
    fig.savefig(out / name)
    plt.close(fig)


def class_balance(y, out):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(["Benign (0)", "Malignant (1)"], counts.values, color=[BENIGN, MALIGNANT], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 6, f"{v}  ({v / counts.sum():.1%})", ha="center", color=INK
        )
    ax.set_ylabel("Number of samples")
    ax.set_title("Class distribution of the WDBC dataset")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)
    _save(fig, out, "fig01_class_balance.png")


def feature_distributions(X, y, out):
    feats = KEY_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, f in zip(axes.ravel(), feats):
        sns.kdeplot(x=X[f][y == 0], ax=ax, fill=True, color=BENIGN, alpha=0.35, linewidth=2, label="Benign")
        sns.kdeplot(
            x=X[f][y == 1], ax=ax, fill=True, color=MALIGNANT, alpha=0.35, linewidth=2, label="Malignant"
        )
        ax.set_title(f)
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Distribution of selected features by diagnosis", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out, "fig02_feature_distributions.png")


def boxplots(X, y, out):
    feats = KEY_FEATURES
    d = X[feats].assign(diagnosis=y.map({0: "Benign", 1: "Malignant"}))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, f in zip(axes.ravel(), feats):
        sns.boxplot(
            data=d,
            x="diagnosis",
            y=f,
            hue="diagnosis",
            order=["Benign", "Malignant"],
            palette={"Benign": BENIGN, "Malignant": MALIGNANT},
            width=0.5,
            linewidth=1.1,
            flierprops={"markersize": 3, "markerfacecolor": INK2, "markeredgecolor": INK2},
            ax=ax,
            legend=False,
        )
        ax.set_title(f)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(
        "Box plots of selected features by diagnosis (points beyond whiskers = IQR outliers)",
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, out, "fig03_boxplots.png")


def effect_sizes(tests, out):
    d = tests.sort_values("rank_biserial")
    colors = [
        {"mean": BENIGN, "worst": MALIGNANT, "standard error": "#1baf7a"}[g] if s else "#b5b4af"
        for g, s in zip(d["group"], d["significant"])
    ]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.barh(d["feature"], d["rank_biserial"], color=colors, height=0.7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Rank-biserial effect size (malignant vs benign)")
    ax.set_title("Univariate effect size of each feature (Mann-Whitney U)")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Patch(color=BENIGN, label="mean"),
            Patch(color="#1baf7a", label="standard error"),
            Patch(color=MALIGNANT, label="worst"),
            Patch(color="#b5b4af", label="not significant (Bonferroni)"),
        ],
        loc="lower right",
        fontsize=8.5,
    )
    _save(fig, out, "fig05_effect_sizes.png")


def pca_variance(var, out):
    d = var.head(15)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(d["component"], d["explained"], color=BENIGN, width=0.6, label="Individual")
    ax.plot(d["component"], d["cumulative"], color=MALIGNANT, marker="o", ms=6, lw=2, label="Cumulative")
    ax.axhline(0.95, color=INK2, ls=":", lw=1)
    ax.text(15.3, 0.95, "95%", va="center", color=INK2, fontsize=9)
    ax.set(
        xlabel="Principal component",
        ylabel="Share of total variance",
        ylim=(0, 1.05),
        title="Variance explained by principal components (standardised features)",
    )
    ax.set_xticks(d["component"])
    ax.legend(loc="center right")
    ax.grid(axis="x", visible=False)
    _save(fig, out, "fig06_pca_variance.png")


def correlation_heatmap(X, out):
    mean_cols = [c for c in X.columns if c.startswith("mean ")]
    corr = X[mean_cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
        xticklabels=[c.replace("mean ", "") for c in mean_cols],
        yticklabels=[c.replace("mean ", "") for c in mean_cols],
    )
    ax.set_title("Correlation between the ten 'mean' features")
    _save(fig, out, "fig04_correlation_heatmap.png")


def cv_boxplot(folds, order, out):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sns.boxplot(
        data=folds,
        x="roc_auc",
        y="model",
        order=order,
        palette=MODEL_COLORS,
        width=0.5,
        ax=ax,
        linewidth=1.2,
        fliersize=0,
        hue="model",
        legend=False,
        hue_order=order,
    )
    sns.stripplot(data=folds, x="roc_auc", y="model", order=order, color=INK, size=4, ax=ax, jitter=0.12)
    ax.set_xlabel("ROC-AUC across 5 stratified folds (training set)")
    ax.set_ylabel("")
    ax.set_title("Cross-validated ROC-AUC by model")
    _save(fig, out, "fig07_cv_roc_auc.png")


def roc_pr_curves(fitted, X_test, y_test, order, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for c, name in zip(MODEL_COLORS, order):
        p = fitted[name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        a1.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        a2.plot(rec, prec, color=c, lw=2, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
    a1.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1)
    a1.set(
        xlabel="False positive rate (1 - specificity)",
        ylabel="True positive rate (sensitivity)",
        title="ROC curves - test set",
    )
    a2.set(
        xlabel="Recall (sensitivity)",
        ylabel="Precision",
        title="Precision-recall curves - test set",
        ylim=(0.5, 1.02),
    )
    a1.legend(loc="lower right", fontsize=8.5)
    a2.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, out, "fig08_roc_pr_curves.png")


def test_metric_bars(table, order, out):
    metrics = ["test_accuracy", "test_sensitivity", "test_specificity", "test_f1"]
    labels = ["Accuracy", "Sensitivity", "Specificity", "F1-score"]
    t = table.set_index("model").loc[order, metrics]
    x = np.arange(len(metrics))
    w = 0.19
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (name, c) in enumerate(zip(order, MODEL_COLORS)):
        ax.bar(x + (i - 1.5) * w, t.loc[name].values, w - 0.02, color=c, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("Score (threshold 0.50)")
    ax.set_title("Test-set performance of the four tuned models")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    ax.grid(axis="x", visible=False)
    _save(fig, out, "fig09_test_metrics.png")


def confusion_pair(best, X_test, y_test, threshold, name, out):
    p = best.predict_proba(X_test)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, t in zip(axes, [0.5, threshold]):
        cm = confusion_matrix(y_test, (p >= t).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.grid(False)
        ax.set_title(f"{name}, threshold = {t:.2f}")
    fig.tight_layout()
    _save(fig, out, "fig11_confusion_matrices.png")


def threshold_sweep(sweep, threshold, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(sweep["threshold"], sweep["sensitivity"], color=MALIGNANT, lw=2, label="Sensitivity")
    ax.plot(sweep["threshold"], sweep["specificity"], color=BENIGN, lw=2, label="Specificity")
    ax.axvline(0.5, color=INK2, ls=":", lw=1)
    ax.axvline(threshold, color=INK, ls="--", lw=1.2)
    ax.text(threshold - 0.01, 0.63, f"chosen = {threshold:.2f}", color=INK, fontsize=9, ha="right")
    ax.text(0.51, 0.63, "default 0.50", color=INK2, fontsize=9)
    ax.set(
        xlabel="Decision threshold",
        ylabel="Out-of-fold rate (training set)",
        ylim=(0.6, 1.02),
        title="Sensitivity / specificity trade-off",
    )
    ax.legend(loc="lower right")
    _save(fig, out, "fig10_threshold_sweep.png")


def importance_bar(importance, name, out, top=10):
    d = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(
        d["feature"],
        d["importance"],
        xerr=d["std"],
        color=BENIGN,
        height=0.6,
        error_kw={"ecolor": INK2, "elinewidth": 1},
    )
    ax.set_xlabel("Mean drop in test ROC-AUC when the feature is shuffled")
    ax.set_title(f"Top {top} permutation importances - {name}")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig12_feature_importance.png")


def calibration(best, X_test, y_test, name, out):
    p = best.predict_proba(X_test)[:, 1]
    frac, mean_pred = calibration_curve(y_test, p, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac, marker="o", ms=7, color=MALIGNANT, lw=2, label=name)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed fraction malignant",
        title="Calibration curve - test set",
    )
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, out, "fig13_calibration.png")


def kmeans_scan(scan, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.5))
    a1.plot(scan["k"], scan["inertia"], marker="o", color=BENIGN, lw=2, ms=7)
    a1.set(xlabel="Number of clusters k", ylabel="Inertia (within-cluster SS)", title="Elbow method")
    a2.plot(scan["k"], scan["silhouette"], marker="o", color=MALIGNANT, lw=2, ms=7)
    a2.set(xlabel="Number of clusters k", ylabel="Mean silhouette score", title="Silhouette analysis")
    fig.tight_layout()
    _save(fig, out, "fig14_kmeans_k_selection.png")


def pca_clusters(coords, y, labels, pca, out):
    ev = pca.explained_variance_ratio_
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for v, c, lab in [(0, BENIGN, "Benign"), (1, MALIGNANT, "Malignant")]:
        m = y.values == v
        a1.scatter(
            coords[m, 0],
            coords[m, 1],
            s=14,
            color=c,
            alpha=0.7,
            label=lab,
            edgecolors="white",
            linewidths=0.4,
        )
    for v, c in [(0, "#1baf7a"), (1, "#6250d6")]:
        m = labels == v
        a2.scatter(
            coords[m, 0],
            coords[m, 1],
            s=14,
            color=c,
            alpha=0.7,
            label=f"Cluster {v}",
            edgecolors="white",
            linewidths=0.4,
        )
    a1.set_title("True diagnosis")
    a2.set_title("K-means clusters (k = 2, labels not used)")
    for a in (a1, a2):
        a.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
        a.legend(loc="upper right", fontsize=8.5)
    a1.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    fig.tight_layout()
    _save(fig, out, "fig15_pca_kmeans.png")


def make_all(res, km, stats_res, out: Path):
    out.mkdir(exist_ok=True)
    order = list(res["table"]["model"])
    class_balance(res["y"], out)
    feature_distributions(res["X"], res["y"], out)
    boxplots(res["X"], res["y"], out)
    correlation_heatmap(res["X"], out)
    effect_sizes(stats_res["tests"], out)
    pca_variance(stats_res["pca"], out)
    cv_boxplot(res["folds"], order, out)
    roc_pr_curves(res["fitted"], res["X_test"], res["y_test"], order, out)
    test_metric_bars(res["table"], order, out)
    confusion_pair(res["best"], res["X_test"], res["y_test"], res["threshold"], res["best_name"], out)
    threshold_sweep(res["sweep"], res["threshold"], out)
    importance_bar(res["importance"], res["best_name"], out)
    calibration(res["best"], res["X_test"], res["y_test"], res["best_name"], out)
    kmeans_scan(km["scan"], out)
    pca_clusters(km["coords"], res["y"], km["labels"], km["pca"], out)
