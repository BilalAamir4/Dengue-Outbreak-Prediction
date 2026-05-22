# Dashboard.py — Main entry point for Dengue Outbreak Prediction Dashboard

import streamlit as st
import os

# ── Page configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Dengue Outbreak Prediction",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Dengue Outbreak Prediction | Big Data Analytics | Spring 2026"
    }
)

# ── Load custom CSS ────────────────────────────────────────────────────
def load_css():
    css_path = 'assets/style.css'
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Sidebar navigation header ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/mosquito-emoji.png", width=80)
    st.title("🦟 Dengue Predictor")
    st.markdown("---")
    st.markdown("**Big Data Analytics**")
    st.markdown("Spring 2026")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - 🏠 **Home** — Project overview
    - 📊 **EDA Dashboard** — Data exploration
    - 🔴 **Classification** — Outbreak zone prediction
    - 📈 **Regression** — Case count prediction
    - 📋 **Model Evaluation** — Performance metrics
    - 📁 **Batch Prediction** — Upload CSV
    """)
    st.markdown("---")

    # Model status indicator
    import os
    clf_ready = os.path.exists('models/clf_xgb_model.pkl')
    reg_ready = os.path.exists('models/reg_xgb_model.pkl')
    st.markdown("### Model Status")
    st.markdown(f"{'✅' if clf_ready else '❌'} Classification Model")
    st.markdown(f"{'✅' if reg_ready else '❌'} Regression Model")

    if not (clf_ready and reg_ready):
        st.warning("Run training scripts first:\n`python src/train_clf.py`\n`python src/train_reg.py`")

# ── Main landing content ───────────────────────────────────────────────
st.title("🦟 Dengue Outbreak Prediction System")
st.markdown("### Big Data Analytics — Spring 2026")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Dataset Size", "60,000 rows", "31 features")
with col2:
    st.metric("CLF AUC-ROC", "0.9945", "XGBoost")
with col3:
    st.metric("REG R²", "0.8971", "XGBoost")
with col4:
    st.metric("Cities Covered", "14", "9 regions")

st.markdown("---")
st.markdown("""
**Use the sidebar to navigate between pages:**
- **EDA Dashboard** — Explore the dataset with interactive charts
- **Classification Predictor** — Predict if a dengue outbreak will occur
- **Regression Predictor** — Predict how many cases will occur
- **Model Evaluation** — View full performance metrics, ROC curves, confusion matrices
- **Batch Prediction** — Upload a CSV file to get predictions for multiple records
""")
