"""
Heart Disease Prediction System – Streamlit Web App
====================================================
Deployment Link: https://heart-disease-predictor-ml.streamlit.app/

Uses 4 input attributes:
  1. age      – Age of the patient
  2. thalach  – Maximum heart rate achieved
  3. chol     – Serum cholesterol (mg/dl)
  4. cp       – Chest pain type (0-3)

The trained SVC model (heart_disease_svc_model.pkl) is loaded from the
same directory. Run locally with:  streamlit run heart_disease_app.py
"""

import pickle
import numpy as np
import streamlit as st
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e63946;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box-positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .result-box-negative {
        background: linear-gradient(135deg, #6bcb77, #2ecc71);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #e63946;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e63946, #c1121f);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
st.markdown('<p class="main-header">❤️ Heart Disease Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML Assignment 2 – Task 2 | UCI Heart Disease Dataset</p>', unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_resource
def load_model():
    with open(BASE / "heart_disease_svc_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please run Task2_Heart_Disease_Prediction.py first to train and save the model.")
    st.stop()

# ── Info section ─────────────────────────────────────────────────
with st.expander("ℹ️ About this App & Selected Features"):
    st.markdown("""
    **Dataset**: UCI Heart Disease (Cleveland)  
    **Algorithm**: Support Vector Classifier (SVC, RBF kernel)  
    
    **4 Selected Input Attributes** (Assignment constraint):
    | Feature | Description |
    |---------|-------------|
    | `age` | Age of the patient (years) |
    | `thalach` | Maximum heart rate achieved during stress test |
    | `chol` | Serum cholesterol level (mg/dl) |
    | `cp` | Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic) |
    """)

st.divider()

# ── Input Form ───────────────────────────────────────────────────
st.subheader("📋 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "🧑 Age (years)",
        min_value=20, max_value=100, value=55, step=1,
        help="Patient age in years"
    )
    chol = st.number_input(
        "🩸 Serum Cholesterol (mg/dl)",
        min_value=100, max_value=600, value=230, step=1,
        help="Cholesterol measured in mg/dl"
    )

with col2:
    thalach = st.number_input(
        "💓 Max Heart Rate Achieved",
        min_value=60, max_value=220, value=150, step=1,
        help="Maximum heart rate during stress test"
    )
    cp = st.selectbox(
        "💢 Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 – Typical Angina",
            1: "1 – Atypical Angina",
            2: "2 – Non-Anginal Pain",
            3: "3 – Asymptomatic"
        }[x],
        help="Type of chest pain experienced"
    )

# ── Predict ──────────────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🔍 Predict Heart Disease Risk")

if predict_btn:
    # cp is already numeric 0-3, LabelEncoder just maps 0→0, 1→1, 2→2, 3→3
    # (sorted order = same order), so we can pass it directly
    features = np.array([[age, thalach, chol, cp]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.markdown(
            '<div class="result-box-positive">🚨 Prediction: HEART DISEASE DETECTED<br>'
            '<small>Please consult a cardiologist immediately.</small></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box-negative">✅ Prediction: NO HEART DISEASE<br>'
            '<small>Keep maintaining a healthy lifestyle!</small></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📊 Input Summary")
    summary_data = {
        "Feature": ["Age", "Max Heart Rate", "Cholesterol", "Chest Pain Type"],
        "Value": [age, thalach, chol, f"{cp} ({['Typical','Atypical','Non-Anginal','Asymptomatic'][cp]})"]
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>ML Assignment 2 – Heart Disease Prediction | "
    "UCI Heart Disease Dataset | SVC (RBF Kernel)</small></center>",
    unsafe_allow_html=True
)
