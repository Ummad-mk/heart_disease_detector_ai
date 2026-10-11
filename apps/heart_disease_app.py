"""
apps/heart_disease_app.py
=========================
Streamlit web application – Heart Disease Predictor.

Run locally
-----------
    streamlit run apps/heart_disease_app.py

Must be run from the project root so that `src/` is on the Python path.
"""
import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from src.heart_disease import model as hd_model

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #e63946;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #6c757d;
        text-align: center; margin-bottom: 2rem;
    }
    .result-box-positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 1.5rem; border-radius: 12px; color: white;
        text-align: center; font-size: 1.4rem; font-weight: 700; margin-top: 1rem;
    }
    .result-box-negative {
        background: linear-gradient(135deg, #6bcb77, #2ecc71);
        padding: 1.5rem; border-radius: 12px; color: white;
        text-align: center; font-size: 1.4rem; font-weight: 700; margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e63946, #c1121f);
        color: white; border: none; border-radius: 8px;
        font-size: 1.1rem; font-weight: 600; padding: 0.6rem 2rem;
        width: 100%; transition: transform 0.2s;
    }
    .stButton > button:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">❤️ Heart Disease Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">ML Assignment 2 – Task 2 | UCI Heart Disease Dataset</p>',
    unsafe_allow_html=True,
)

# ── Load model & encoders ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return hd_model.load()

try:
    clf, encoders = load_artifacts()
except FileNotFoundError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()

# ── About ─────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this App & Selected Features"):
    st.markdown("""
    **Dataset**: UCI Heart Disease (Cleveland)  
    **Algorithm**: Support Vector Classifier (SVC, RBF kernel)

    **4 Selected Input Attributes** (Assignment constraint):
    | Feature | Description |
    |---------|-------------|
    | `age` | Age of the patient (years) |
    | `thalach` | Maximum heart rate during stress test |
    | `chol` | Serum cholesterol level (mg/dl) |
    | `cp` | Chest pain type (0=typical angina … 3=asymptomatic) |
    """)

st.divider()

# ── Input Form ───────────────────────────────────────────────────────────
st.subheader("📋 Enter Patient Information")
col1, col2 = st.columns(2)

with col1:
    age  = st.number_input("🧑 Age (years)", 20, 100, 55, 1, help="Patient age in years")
    chol = st.number_input("🩸 Serum Cholesterol (mg/dl)", 100, 600, 230, 1)

with col2:
    thalach = st.number_input("💓 Max Heart Rate Achieved", 60, 220, 150, 1)
    cp = st.selectbox(
        "💢 Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 – Typical Angina",
            1: "1 – Atypical Angina",
            2: "2 – Non-Anginal Pain",
            3: "3 – Asymptomatic",
        }[x],
    )

# ── Predict ───────────────────────────────────────────────────────────────
st.markdown("")
if st.button("🔍 Predict Heart Disease Risk"):
    # cp is numeric 0-3; LabelEncoder maps values in sorted order (same mapping)
    features = np.array([[age, thalach, chol, cp]])
    pred = clf.predict(features)[0]

    if pred == 1:
        st.markdown(
            '<div class="result-box-positive">🚨 Prediction: HEART DISEASE DETECTED<br>'
            '<small>Please consult a cardiologist immediately.</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-box-negative">✅ Prediction: NO HEART DISEASE<br>'
            '<small>Keep maintaining a healthy lifestyle!</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("📊 Input Summary")
    import pandas as pd
    st.dataframe(pd.DataFrame({
        "Feature": ["Age", "Max Heart Rate", "Cholesterol", "Chest Pain Type"],
        "Value":   [age, thalach, chol,
                    f"{cp} ({['Typical','Atypical','Non-Anginal','Asymptomatic'][cp]})"],
    }), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>ML Assignment 2 – Heart Disease Prediction | "
    "UCI Heart Disease Dataset | SVC (RBF Kernel)</small></center>",
    unsafe_allow_html=True,
)
