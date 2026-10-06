"""
Titanic Passenger Survival Prediction – Streamlit Web App
==========================================================
Assignment 2 – Task 1 | Deployment

Run locally:  streamlit run titanic_app.py
"""

import os
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    font-size: 2.4rem;
    font-weight: 700;
    color: #1d3557;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.sub-header {
    font-size: 1rem;
    color: #6c757d;
    text-align: center;
    margin-bottom: 1.5rem;
}
.result-survived {
    background: linear-gradient(135deg, #2ecc71, #27ae60);
    padding: 1.6rem 1rem;
    border-radius: 14px;
    color: white;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(46,204,113,0.35);
    animation: fadeIn 0.4s ease;
}
.result-not-survived {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    padding: 1.6rem 1rem;
    border-radius: 14px;
    color: white;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(231,76,60,0.35);
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.stButton > button {
    background: linear-gradient(135deg, #1d3557, #457b9d) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🚢 Titanic Survival Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML Assignment 2 – Task 1 &nbsp;|&nbsp; SVC Model &nbsp;|&nbsp; Titanic Dataset</p>', unsafe_allow_html=True)

# ── Load model & encoders ─────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_resource
def load_model():
    model_path = BASE / "titanic_svc_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_encoders():
    le_sex = LabelEncoder()
    le_sex.fit(["female", "male"])          # 0=female, 1=male
    le_embarked = LabelEncoder()
    le_embarked.fit(["C", "Q", "S"])        # 0=C, 1=Q, 2=S
    return le_sex, le_embarked

try:
    model = load_model()
    le_sex, le_embarked = build_encoders()
except FileNotFoundError:
    st.error("⚠️ Model file not found (`titanic_svc_model.pkl`). "
             "Run `Task1_Titanic_Survival_Prediction.py` first to train and save the model.")
    st.stop()

# ── About expander ────────────────────────────────────────────────
with st.expander("ℹ️ About This App"):
    st.markdown("""
    **Dataset:** Titanic (seaborn built-in, 891 passengers)  
    **Algorithm:** Support Vector Classifier (SVC, RBF kernel)  
    **Features:** Passenger Class, Sex, Age, SibSp, Parch, Fare, Port of Embarkation  
    **Training accuracy:** ~63.6%
    """)

st.divider()
st.subheader("📋 Enter Passenger Information")

# ── Input form ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

CLASS_LABELS = {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}
PORT_LABELS  = {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}

with col1:
    pclass   = st.selectbox("🎫 Passenger Class", options=[1, 2, 3],
                             format_func=lambda x: CLASS_LABELS[x])
    sex      = st.selectbox("👤 Sex", options=["female", "male"],
                             format_func=lambda x: x.capitalize())
    age      = st.slider("🎂 Age (years)", min_value=1, max_value=80, value=28)
    fare     = st.number_input("💰 Fare Paid ($)", min_value=0.0,
                                max_value=600.0, value=30.0, step=0.5)

with col2:
    sibsp    = st.number_input("👫 Siblings / Spouses Aboard", min_value=0, max_value=8,  value=0)
    parch    = st.number_input("👨‍👩‍👧 Parents / Children Aboard",  min_value=0, max_value=6,  value=0)
    embarked = st.selectbox("⚓ Port of Embarkation", options=["S", "C", "Q"],
                             format_func=lambda x: PORT_LABELS[x])

# ── Predict ───────────────────────────────────────────────────────
st.markdown("")
if st.button("🔍 Predict Survival Chances"):
    sex_enc      = le_sex.transform([sex])[0]
    embarked_enc = le_embarked.transform([embarked])[0]
    X            = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])
    prediction   = model.predict(X)[0]

    if prediction == 1:
        st.markdown(
            '<div class="result-survived">'
            '✅ &nbsp;SURVIVED<br>'
            '<small style="font-weight:400">This passenger would likely have survived!</small>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-not-survived">'
            '❌ &nbsp;DID NOT SURVIVE<br>'
            '<small style="font-weight:400">This passenger would likely not have survived.</small>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("**Input summary:**")
    import pandas as pd
    summary = pd.DataFrame({
        "Feature": ["Class", "Sex", "Age", "Siblings/Spouses", "Parents/Children", "Fare", "Embarked"],
        "Value":   [CLASS_LABELS[pclass], sex.capitalize(), age, sibsp, parch, f"${fare:.2f}", PORT_LABELS[embarked]],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>ML Assignment 2 – Task 1 &nbsp;|&nbsp; "
    "Titanic Survival Prediction &nbsp;|&nbsp; SVC (RBF Kernel)</small></center>",
    unsafe_allow_html=True,
)
