"""
apps/titanic_app.py
===================
Streamlit web application – Titanic Passenger Survival Predictor.

Run locally
-----------
    streamlit run apps/titanic_app.py

Must be run from the project root so that `src/` is on the Python path.
"""
import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from src.titanic import model as titanic_model

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    font-size: 2.4rem; font-weight: 700; color: #1d3557;
    text-align: center; margin-bottom: 0.2rem; letter-spacing: -0.5px;
}
.sub-header {
    font-size: 1rem; color: #6c757d;
    text-align: center; margin-bottom: 1.5rem;
}
.result-survived {
    background: linear-gradient(135deg, #2ecc71, #27ae60);
    padding: 1.6rem 1rem; border-radius: 14px; color: white;
    text-align: center; font-size: 1.5rem; font-weight: 700; margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(46,204,113,0.35); animation: fadeIn 0.4s ease;
}
.result-not-survived {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    padding: 1.6rem 1rem; border-radius: 14px; color: white;
    text-align: center; font-size: 1.5rem; font-weight: 700; margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(231,76,60,0.35); animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.stButton > button {
    background: linear-gradient(135deg, #1d3557, #457b9d) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-size: 1.05rem !important; font-weight: 600 !important;
    padding: 0.65rem 2rem !important; width: 100% !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🚢 Titanic Survival Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">ML Assignment 2 – Task 1 &nbsp;|&nbsp; SVC Model &nbsp;|&nbsp; Titanic Dataset</p>',
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return titanic_model.load()

@st.cache_resource
def build_encoders():
    le_sex = LabelEncoder().fit(["female", "male"])      # 0=female, 1=male
    le_emb = LabelEncoder().fit(["C", "Q", "S"])         # 0=C, 1=Q, 2=S
    return le_sex, le_emb

try:
    clf = load_model()
    le_sex, le_embarked = build_encoders()
except FileNotFoundError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()

# ── About ─────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About This App"):
    st.markdown("""
    **Dataset:** Titanic (seaborn built-in, 891 passengers)  
    **Algorithm:** Support Vector Classifier (SVC, RBF kernel)  
    **Features:** Passenger Class · Sex · Age · SibSp · Parch · Fare · Port of Embarkation
    """)

st.divider()
st.subheader("📋 Enter Passenger Information")

# ── Input form ────────────────────────────────────────────────────────────
CLASS_LABELS = {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}
PORT_LABELS  = {"S": "Southampton (S)",  "C": "Cherbourg (C)",    "Q": "Queenstown (Q)"}

col1, col2 = st.columns(2)

with col1:
    pclass   = st.selectbox("🎫 Passenger Class", [1, 2, 3], format_func=lambda x: CLASS_LABELS[x])
    sex      = st.selectbox("👤 Sex", ["female", "male"], format_func=str.capitalize)
    age      = st.slider("🎂 Age (years)", 1, 80, 28)
    fare     = st.number_input("💰 Fare Paid ($)", 0.0, 600.0, 30.0, 0.5)

with col2:
    sibsp    = st.number_input("👫 Siblings / Spouses Aboard", 0, 8, 0)
    parch    = st.number_input("👨‍👩‍👧 Parents / Children Aboard", 0, 6, 0)
    embarked = st.selectbox("⚓ Port of Embarkation", ["S", "C", "Q"], format_func=lambda x: PORT_LABELS[x])

# ── Predict ───────────────────────────────────────────────────────────────
st.markdown("")
if st.button("🔍 Predict Survival Chances"):
    X = np.array([[
        pclass,
        le_sex.transform([sex])[0],
        age, sibsp, parch, fare,
        le_embarked.transform([embarked])[0],
    ]])
    pred = clf.predict(X)[0]

    if pred == 1:
        st.markdown(
            '<div class="result-survived">✅ &nbsp;SURVIVED<br>'
            '<small style="font-weight:400">This passenger would likely have survived!</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-not-survived">❌ &nbsp;DID NOT SURVIVE<br>'
            '<small style="font-weight:400">This passenger would likely not have survived.</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("**Input summary:**")
    import pandas as pd
    st.dataframe(pd.DataFrame({
        "Feature": ["Class", "Sex", "Age", "Siblings/Spouses", "Parents/Children", "Fare", "Embarked"],
        "Value":   [CLASS_LABELS[pclass], sex.capitalize(), age, sibsp, parch, f"${fare:.2f}", PORT_LABELS[embarked]],
    }), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>ML Assignment 2 – Task 1 &nbsp;|&nbsp; "
    "Titanic Survival Prediction &nbsp;|&nbsp; SVC (RBF Kernel)</small></center>",
    unsafe_allow_html=True,
)
