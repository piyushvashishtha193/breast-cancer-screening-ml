"""Streamlit demo for the educational screening model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.train import train_and_save

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "screening_model.joblib"

st.set_page_config(page_title="Breast Cancer Screening | ML Internship", page_icon="🩺")
st.title("Breast Cancer Screening: ML Demonstration")
st.warning("Educational demonstration only — this tool is not a diagnosis and must not guide medical decisions.")

if not MODEL_PATH.exists():
    with st.spinner("Preparing the demonstration model..."):
        train_and_save()

bundle = joblib.load(MODEL_PATH)
model, feature_names = bundle["model"], bundle["feature_names"]
st.caption("Enter diagnostic measurements to see the model's estimated positive-screen probability.")

with st.sidebar:
    st.header("Measurements")
    values = {}
    for feature in feature_names:
        values[feature] = st.number_input(feature.replace("_", " ").title(), value=0.0, format="%.4f")

if st.button("Estimate screening probability", type="primary"):
    row = pd.DataFrame([values], columns=feature_names)
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Positive-screen probability", f"{probability:.1%}")
    st.info("A positive model screen should be reviewed by a qualified healthcare professional; it is not a diagnosis.")

st.divider()
st.subheader("Project notes")
st.write("The model is logistic regression with standardized input features. See the notebook and README for evaluation, limitations, and ethical considerations.")
