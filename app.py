"""Streamlit demo for the educational screening model.

streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data import load_data
from src.train import train_and_save

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "screening_model.joblib"

st.set_page_config(page_title="Breast Cancer Screening | ML Internship", page_icon="🩺", layout="wide")
st.title("Breast Cancer Screening")
st.caption("Data analysis and machine learning project - YBI Foundation Data Science & ML Internship")
st.warning("Educational project only. This is not a diagnosis and must not be used for medical decisions.")

if not MODEL_PATH.exists():
    with st.spinner("Training the model for the first time (about 30 s)..."):
        train_and_save()

bundle = joblib.load(MODEL_PATH)
model, features, threshold = bundle["model"], bundle["feature_names"], bundle["threshold"]
medians, lo, hi = bundle["feature_medians"], bundle["feature_min"], bundle["feature_max"]

X, y = load_data()

with st.sidebar:
    st.header("Input measurements")
    for f in features:  # first run: start every input at the dataset median
        st.session_state.setdefault(f"in_{f}", float(medians[f]))
    if st.button("Load a random sample from the dataset"):
        idx = X.sample(1).index[0]
        for f in features:
            st.session_state[f"in_{f}"] = float(X.loc[idx, f])
        st.session_state["true_label"] = int(y.loc[idx])
    if st.button("Reset to median values"):
        for f in features:
            st.session_state[f"in_{f}"] = float(medians[f])
        st.session_state.pop("true_label", None)
    values = {}
    groups = {
        "Mean features": [f for f in features if f.startswith("mean ")],
        "Standard-error features": [f for f in features if f.endswith(" error")],
        "Worst features": [f for f in features if f.startswith("worst ")],
    }
    for title, cols in groups.items():
        with st.expander(title, expanded=(title == "Mean features")):
            for f in cols:
                values[f] = st.number_input(
                    f,
                    min_value=float(lo[f]) * 0.5,
                    max_value=float(hi[f]) * 1.5,
                    format="%.4f",
                    key=f"in_{f}",
                )

row = pd.DataFrame([values], columns=features)
prob = float(model.predict_proba(row)[0, 1])
flag = prob >= threshold

tab_screen, tab_data = st.tabs(["Screening", "Data explorer"])

with tab_screen:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", bundle["model_name"])
    c2.metric("Malignant probability", f"{prob:.1%}")
    c3.metric("Screening threshold", f"{threshold:.2f}")

    if flag:
        st.error("Screen result: FLAG FOR REVIEW - the probability is at or above the screening threshold.")
    else:
        st.success("Screen result: below the screening threshold.")
    if "true_label" in st.session_state:
        actual = "Malignant" if st.session_state["true_label"] else "Benign"
        st.caption(f"Actual label of the loaded sample: {actual}")

    st.divider()
    st.subheader("Why the threshold is not 0.50")
    st.write(
        "In screening, missing a malignant case (false negative) is worse than an extra follow-up test "
        "(false positive). The threshold was chosen on the training data so that out-of-fold sensitivity "
        "is at least 98%. See the notebook and README for the full evaluation."
    )
    cm_png = ROOT / "figures" / "fig11_confusion_matrices.png"
    if cm_png.exists():
        st.image(str(cm_png), caption="Test-set confusion matrices at 0.50 and at the screening threshold")

with tab_data:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Samples", len(X))
    d2.metric("Features", X.shape[1])
    d3.metric("Missing values", int(X.isna().sum().sum()))
    d4.metric("Malignant share", f"{y.mean():.1%}")

    feature = st.selectbox("Feature", features, index=features.index("mean concave points"))
    labelled = pd.DataFrame({feature: X[feature], "diagnosis": y.map({0: "Benign", 1: "Malignant"})})
    left, right = st.columns([3, 2])
    with left:
        hist = (
            labelled.assign(bin=pd.cut(labelled[feature], bins=30))
            .groupby(["bin", "diagnosis"], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        hist.index = [round(b.mid, 4) for b in hist.index]
        st.bar_chart(hist, color=["#2a78d6", "#eb6834"], x_label=feature, y_label="Samples")
    with right:
        summary = labelled.groupby("diagnosis")[feature].describe()[["mean", "std", "50%", "min", "max"]]
        st.dataframe(summary.rename(columns={"50%": "median"}).round(4))
        tests_csv = ROOT / "reports" / "statistical_tests.csv"
        if tests_csv.exists():
            t = pd.read_csv(tests_csv).set_index("feature").loc[feature]
            st.write(f"Mann-Whitney U, Bonferroni-corrected p = {t['p_bonferroni']:.2e}")
            st.write(f"Rank-biserial effect size = {t['rank_biserial']:.2f}")
