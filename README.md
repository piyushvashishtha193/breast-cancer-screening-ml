# Breast Cancer Screening ML Demonstration

An end-to-end machine-learning project created for the **YBI Foundation 45-Day Machine Learning & Data Science Internship**.

> **Important:** This is an educational data-science demonstration, not a medical device. It cannot diagnose cancer, replace laboratory testing, or guide treatment. Any health concern should be reviewed by a qualified clinician.

## Project goal

Build an interpretable binary-classification pipeline that estimates whether an observation receives a **positive malignant-screen label** from diagnostic features. The work emphasizes a reproducible workflow: data understanding, preprocessing, stratified evaluation, sensitivity-aware metrics, explainability, and a simple interactive demo.

## Highlights

- Uses the built-in Wisconsin Diagnostic Breast Cancer dataset from scikit-learn (569 observations, 30 numeric features)
- Clean `Pipeline` with `StandardScaler` and class-balanced `LogisticRegression`
- Stratified train/test split for a fair class distribution
- Evaluation with ROC-AUC, average precision, sensitivity, specificity, and a confusion matrix
- Permutation feature importance for model interpretation
- Interactive Streamlit demonstration with clear safety messaging
- Executed Jupyter notebook ready for Colab/GitHub review

## Repository structure

```text
├── app.py                         # Interactive Streamlit demonstration
├── notebooks/
│   └── breast_cancer_screening.ipynb
├── src/
│   └── train.py                   # Reproducible training and artifact creation
├── artifacts/                     # Generated model, metrics, feature importance
├── data/                          # Generated local feature export (ignored by Git)
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

The app automatically trains the small demonstration model if the model file is not present.

## Notebook

Open `notebooks/breast_cancer_screening.ipynb` in Google Colab or Jupyter. It is organized for assessment: objective, data audit, split strategy, model training, evaluation, explainability, limitations, and conclusion. Execute all cells before sharing so results remain visible to reviewers.

## Methodology

1. Load the scikit-learn dataset and recode the target so `1` means a malignant (positive) screen.
2. Reserve a stratified 25% test set before fitting any preprocessing.
3. Standardize features inside a pipeline to avoid train/test leakage.
4. Fit a class-balanced logistic-regression classifier.
5. Evaluate discrimination and screening-relevant error patterns rather than accuracy alone.
6. Rank the most influential measurements using permutation importance.

## Limitations and ethical notes

- The source dataset is small and may not represent current clinical populations, devices, or local demographics.
- Labels and features are historical research data, not a substitute for real clinical workflows.
- A probability is not a diagnosis, and a threshold encodes a trade-off between false positives and false negatives.
- Clinical use would require external validation, calibration, bias evaluation, governance, privacy safeguards, and regulatory review.

## Dataset attribution

The project uses `sklearn.datasets.load_breast_cancer`, which provides the Wisconsin Diagnostic Breast Cancer dataset. Dataset details and citations are available in the [scikit-learn documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset).

## Internship submission checklist

- [x] Original project topic selected
- [x] Documented and reproducible notebook included
- [x] Model training and evaluation included
- [x] Results saved as visible notebook output
- [x] GitHub-ready README and requirements included
- [x] Shareable Colab workflow supported

## Author

Piyush Vashishtha
