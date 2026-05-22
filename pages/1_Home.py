# pages/1_Home.py

import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Home | Dengue Prediction", page_icon="🏠", layout="wide")

st.title("🏠 Project Overview")
st.markdown("## Dengue Outbreak Prediction Using Machine Learning")
st.markdown("**Subject:** Big Data Analytics | **Semester:** Spring 2026")
st.markdown("---")

# Problem statement
st.markdown("### Problem Statement")
st.info("""
Dengue fever affects 390 million people annually worldwide. This project builds a complete 
machine learning pipeline to predict dengue outbreaks using climate, vegetation, and 
epidemiological data from 14 cities across Pakistan, Sri Lanka, Puerto Rico, and Peru.

**Two prediction tasks:**
1. **Classification** — Will there be a dengue outbreak this week? (Yes/No)
2. **Regression** — How many dengue cases will occur this week? (Count prediction)
""")

st.markdown("---")
st.markdown("### Dataset Summary")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Dataset: dengue_training_60k.csv**")
    dataset_info = {
        "Property": ["Total Rows", "Total Columns", "Null Values", "Duplicate Rows",
                     "Year Range", "Cities", "Regions", "Outbreak Rate"],
        "Value":    ["60,000", "31", "0", "0",
                     "1990 – 2024", "14", "9", "5.01%"]
    }
    st.dataframe(pd.DataFrame(dataset_info), use_container_width=True, hide_index=True)

with col2:
    st.markdown("**Feature Categories**")
    feat_info = {
        "Category": ["Identifiers", "Time", "Vegetation (NDVI)", "Reanalysis Weather",
                     "Station Weather", "Socioeconomic", "Lag Features", "Targets"],
        "Count": [2, 3, 4, 10, 5, 2, 3, 2]
    }
    st.dataframe(pd.DataFrame(feat_info), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### Model Performance Summary")

# Load metrics if available
clf_metrics_path = 'models/clf_metrics.json'
reg_metrics_path = 'models/reg_metrics.json'

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🔴 Classification (outbreak_zone)")
    if os.path.exists(clf_metrics_path):
        with open(clf_metrics_path) as f:
            cm = json.load(f)
        data = {
            "Metric":    ["Accuracy", "F1 Score", "AUC-ROC", "Avg Precision", "CV AUC-ROC"],
            "Random Forest": [f"{cm['rf']['accuracy']:.4f}", f"{cm['rf']['f1']:.4f}",
                              f"{cm['rf']['auc_roc']:.4f}", f"{cm['rf']['avg_prec']:.4f}",
                              f"{cm['rf']['cv_auc_mean']:.4f} ± {cm['rf']['cv_auc_std']:.4f}"],
            "XGBoost ✓":     [f"{cm['xgb']['accuracy']:.4f}", f"{cm['xgb']['f1']:.4f}",
                              f"{cm['xgb']['auc_roc']:.4f}", f"{cm['xgb']['avg_prec']:.4f}",
                              f"{cm['xgb']['cv_auc_mean']:.4f} ± {cm['xgb']['cv_auc_std']:.4f}"],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    else:
        st.warning("Train the classification model first: `python src/train_clf.py`")

with col2:
    st.markdown("#### 📈 Regression (total_cases)")
    if os.path.exists(reg_metrics_path):
        with open(reg_metrics_path) as f:
            rm = json.load(f)
        data = {
            "Metric":    ["R²", "RMSE (log)", "MAE (log)", "RMSE (cases)", "CV R²"],
            "Random Forest": [f"{rm['rf']['r2']:.4f}", f"{rm['rf']['rmse_log']:.4f}",
                              f"{rm['rf']['mae_log']:.4f}", f"{rm['rf']['rmse_raw']:.2f}",
                              f"{rm['rf']['cv_r2_mean']:.4f} ± {rm['rf']['cv_r2_std']:.4f}"],
            "XGBoost ✓":     [f"{rm['xgb']['r2']:.4f}", f"{rm['xgb']['rmse_log']:.4f}",
                              f"{rm['xgb']['mae_log']:.4f}", f"{rm['xgb']['rmse_raw']:.2f}",
                              f"{rm['xgb']['cv_r2_mean']:.4f} ± {rm['xgb']['cv_r2_std']:.4f}"],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    else:
        st.warning("Train the regression model first: `python src/train_reg.py`")

st.markdown("---")
st.markdown("### Pipeline Summary")
steps = [
    ("1. Data Loading",        "60,000 rows × 31 columns, 0 nulls"),
    ("2. Column Removal",      "11 dropped — leakage + multicollinearity (r > 0.95)"),
    ("3. Label Encoding",      "city & region → numeric (LabelEncoder)"),
    ("4. Feature Engineering", "4–5 new features: ndvi_mean, temp_range, precip_humidity, lag_trend, lag_acceleration"),
    ("5. Target Transform",    "CLF: outbreak_zone binary | REG: log1p(total_cases) → skew 4.93 → 0.64"),
    ("6. Outlier Capping",     "IQR Winsorization — all rows preserved"),
    ("7. Feature Selection",   "CLF: MI + RFE union → 19 features | REG: MI regression → 16 features"),
    ("8. Scaling",             "RobustScaler — median-centered, IQR-scaled"),
    ("9. Imbalance Handling",  "CLF: SMOTE (48k → 91k balanced) | REG: log1p transform"),
    ("10. Models",             "Random Forest + XGBoost for both tasks"),
    ("11. Evaluation",         "CLF: F1, AUC-ROC, CV | REG: R², RMSE, MAE, CV"),
]
for step, detail in steps:
    st.markdown(f"**{step}** — {detail}")
