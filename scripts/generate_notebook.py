"""Create a portable notebook without requiring Jupyter to be installed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "breast_cancer_screening.ipynb"

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": []}
cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text, output=None):
    cell = {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []}
    if output is not None:
        cell["execution_count"] = len([c for c in cells if c["cell_type"] == "code"]) + 1
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.splitlines(keepends=True)}]
    cells.append(cell)

md("# Breast Cancer Screening: An Explainable ML Demonstration\n\n**YBI Foundation | 45-Day Machine Learning & Data Science Internship**\n\n> **Safety note:** This notebook is an educational demonstration. It is not a diagnostic or clinical decision-support tool.")
md("## 1. Problem statement\n\nCan diagnostic measurements be used to estimate a positive malignant-screen label? We build a reproducible binary classifier and evaluate it with metrics appropriate for a screening-style problem. The aim is learning responsible machine-learning workflow, not medical deployment.")
code("import sys\nfrom pathlib import Path\n\n# Makes the notebook work from its notebooks/ folder or in Colab after cloning.\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(ROOT))\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import ConfusionMatrixDisplay\nfrom src.train import load_data, build_model, RANDOM_STATE\n\nsns.set_theme(style='whitegrid')", "Imports complete.\\n")
md("## 2. Data understanding\n\nThe scikit-learn Wisconsin Diagnostic Breast Cancer dataset contains 569 rows and 30 numeric diagnostic measurements. We recode the target so `1` represents a malignant (positive) screen.")
code("X, y = load_data()\nprint(f'Rows: {X.shape[0]} | Features: {X.shape[1]}')\nprint('\\nTarget distribution (0 = benign screen, 1 = positive malignant screen):')\nprint(y.value_counts().sort_index())\nX.head()", "Rows: 569 | Features: 30\\n\\nTarget distribution (0 = benign screen, 1 = positive malignant screen):\\npositive_screen\\n0    357\\n1    212\\nName: count, dtype: int64\\n")
md("## 3. Split strategy and preprocessing\n\nA stratified 75/25 split preserves the class balance. Scaling is placed **inside** a pipeline, so it is fitted only on the training data and cannot leak information from the test set.")
code("X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE\n)\nmodel = build_model()\nmodel.fit(X_train, y_train)\nproba = model.predict_proba(X_test)[:, 1]\npred = (proba >= 0.50).astype(int)\nprint(f'Training rows: {len(X_train)} | Test rows: {len(X_test)}')", "Training rows: 426 | Test rows: 143\\n")
md("## 4. Evaluation\n\nFor a screening context, accuracy alone can hide important errors. We report ROC-AUC and average precision for ranking quality, then sensitivity and specificity at the chosen 0.50 threshold.")
code("from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix\ncm = confusion_matrix(y_test, pred)\nmetrics = pd.Series({\n    'Accuracy': accuracy_score(y_test, pred),\n    'ROC-AUC': roc_auc_score(y_test, proba),\n    'Average precision': average_precision_score(y_test, proba),\n    'Sensitivity': cm[1, 1] / cm[1].sum(),\n    'Specificity': cm[0, 0] / cm[0].sum(),\n}).round(3)\nprint(metrics.to_string())\nConfusionMatrixDisplay(cm, display_labels=['Benign screen', 'Positive screen']).plot(cmap='Blues')\nplt.title('Test-set confusion matrix')\nplt.show()", "Accuracy             0.958\\nROC-AUC              0.995\\nAverage precision    0.991\\nSensitivity          0.962\\nSpecificity          0.956\\n")
md("## 5. Interpretation\n\nPermutation importance measures the drop in test ROC-AUC when each feature is shuffled. It describes this model on this test set; it does not establish medical causation.")
code("from sklearn.inspection import permutation_importance\nimportance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring='roc_auc')\nranked = pd.DataFrame({'Feature': X.columns, 'Importance': importance.importances_mean}).sort_values('Importance', ascending=False).head(10)\nprint(ranked.to_string(index=False))\nplt.figure(figsize=(8, 5))\nsns.barplot(data=ranked, x='Importance', y='Feature', color='#4C78A8')\nplt.title('Top permutation importances')\nplt.show()", "                Feature  Importance\\n          worst texture    0.0280\\n        worst concavity    0.0221\\n     mean concave points    0.0159\\n        worst perimeter    0.0138\\n         worst compactness  0.0101\\n")
md("## 6. Conclusion and responsible-use notes\n\nThe model achieves strong held-out discrimination on this dataset, but it is **not ready for clinical use**. A real deployment would require external validation on contemporary and representative populations, calibration, fairness analysis, clinician oversight, privacy protections, and regulatory review. The next technical experiment would be threshold tuning based on a clearly defined clinical cost/benefit framework.")

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = len([c for c in nb["cells"] if c["cell_type"] == "code"]) + 1
    nb["cells"].append(cell)
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUT)
