# pages/3_Classification_Predictor.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Classification Predictor", page_icon="🔴", layout="wide")

@st.cache_resource
def load_clf_models():
    rf        = joblib.load('models/clf_rf_model.pkl')
    xgb       = joblib.load('models/clf_xgb_model.pkl')
    scaler    = joblib.load('models/clf_scaler.pkl')
    le_city   = joblib.load('models/clf_le_city.pkl')
    le_region = joblib.load('models/clf_le_region.pkl')
    return rf, xgb, scaler, le_city, le_region

st.title("🔴 Classification Predictor")
st.markdown("### Predict: Will there be a dengue outbreak this week?")
st.markdown("---")

if not os.path.exists('models/clf_xgb_model.pkl'):
    st.error("Classification model not found. Run: `python src/train_clf.py`")
    st.stop()

rf, xgb_model, scaler, le_city, le_region = load_clf_models()

with st.sidebar:
    st.markdown("### Model Selection")
    model_choice = st.radio("Choose Model", ["XGBoost (Recommended)", "Random Forest"])
    st.markdown("---")
    st.markdown("### About This Model")
    st.markdown("""
    **Target:** outbreak_zone (0/1)  
    **Best Model:** XGBoost  
    **AUC-ROC:** 0.9945  
    **F1 Score:** 0.8322  
    **CV AUC:** 0.9952 ± 0.0004
    """)

st.markdown("### Enter Weekly Features")

# ── Input form ────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

CITIES  = ['Colombo','Gampaha','Kandy','Galle','Jaffna','Ampara',
           'Lahore','Karachi','Peshawar','Islamabad','Quetta','Muzaffarabad',
           'San_Juan','Iquitos']
REGIONS = ['Sri_Lanka','Punjab_PK','Sindh_PK','KPK_PK','ICT_PK',
           'Balochistan_PK','AJK_PK','Puerto_Rico','Peru']

with col1:
    st.markdown("**Location & Time**")
    city      = st.selectbox("City", CITIES)
    region    = st.selectbox("Region", REGIONS)
    weekofyear = st.slider("Week of Year", 1, 52, 30)

with col2:
    st.markdown("**Climate Features**")
    humidity    = st.slider("Relative Humidity (%)", 30.0, 99.0, 75.0, 0.5)
    precip      = st.slider("Precipitation (mm)", 0.0, 420.0, 35.0, 1.0)
    temp_min_k  = st.slider("Min Air Temp (K)", 282.0, 304.0, 295.0, 0.1)
    temp_max_k  = st.slider("Max Air Temp (K)", 292.0, 318.0, 303.0, 0.1)
    tdtr_k      = st.slider("Diurnal Temp Range (K)", 1.0, 18.0, 6.0, 0.1)
    dew_point_k = st.slider("Dew Point Temp (K)", 283.0, 302.0, 294.0, 0.1)
    avg_temp_k  = st.slider("Avg Air Temp (K)", 288.0, 308.0, 299.0, 0.1)
    precip_r    = st.slider("Reanalysis Precip (kg/m²)", 0.0, 600.0, 34.0, 1.0)

with col3:
    st.markdown("**Lag & Socioeconomic**")
    lag1 = st.number_input("Lag 1 Cases (last week)",  min_value=0, max_value=300, value=5)
    lag4 = st.number_input("Lag 4 Cases (4 weeks ago)",min_value=0, max_value=300, value=4)
    pop_density  = st.slider("Population Density (0–1)", 0.1, 1.0, 0.7, 0.01)
    hospital_cap = st.slider("Hospital Capacity (0–1)",  0.1, 1.0, 0.5, 0.01)

if st.button("🔴 Predict Outbreak Zone", use_container_width=True, type="primary"):
    # Build input dict matching CLF_FEATURES exactly
    input_data = {
        'city':   city,
        'region': region,
        'weekofyear': weekofyear,
        'precipitation_amt_mm': precip,
        'reanalysis_avg_temp_k': avg_temp_k,
        'reanalysis_dew_point_temp_k': dew_point_k,
        'reanalysis_max_air_temp_k': temp_max_k,
        'reanalysis_min_air_temp_k': temp_min_k,
        'reanalysis_relative_humidity_percent': humidity,
        'reanalysis_tdtr_k': tdtr_k,
        'reanalysis_precip_amt_kg_per_m2': precip_r,
        'population_density': pop_density,
        'hospital_capacity_index': hospital_cap,
        'lag1_cases': lag1,
        'lag4_cases': lag4,
        # Dummy NDVI columns (will be aggregated to ndvi_mean)
        'ndvi_ne': 0.15, 'ndvi_nw': 0.15, 'ndvi_se': 0.15, 'ndvi_sw': 0.15,
    }
    df_input = pd.DataFrame([input_data])

    # Preprocess
    import sys; sys.path.insert(0, '.')
    from src.preprocess_clf import preprocess_for_classification
    try:
        X_proc, _, _, _, _ = preprocess_for_classification(
            df_input, fit=False, scaler=scaler,
            le_city=le_city, le_region=le_region)

        model = xgb_model if "XGBoost" in model_choice else rf
        pred  = model.predict(X_proc)[0]
        prob  = model.predict_proba(X_proc)[0][1]

        st.markdown("---")
        st.markdown("### Prediction Result")
        col1, col2, col3 = st.columns(3)

        if pred == 1:
            col1.error(f"🚨 **OUTBREAK PREDICTED**")
        else:
            col1.success(f"✅ **NO OUTBREAK PREDICTED**")

        col2.metric("Outbreak Probability", f"{prob*100:.1f}%")
        col3.metric("Confidence", f"{'High' if abs(prob-0.5)>0.3 else 'Moderate' if abs(prob-0.5)>0.15 else 'Low'}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Outbreak Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': "#DC2626" if prob > 0.5 else "#059669"},
                'steps': [
                    {'range': [0, 30],  'color': '#DCFCE7'},
                    {'range': [30, 60], 'color': '#FEF9C3'},
                    {'range': [60, 100],'color': '#FEE2E2'},
                ],
                'threshold': {'line': {'color': "red", 'width': 4},
                              'thickness': 0.75, 'value': 50}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Engineered features
        st.markdown("#### Computed Input Features")
        st.json({
            "ndvi_mean":       round(0.15, 4),
            "temp_range_k":    round(temp_max_k - temp_min_k, 4),
            "precip_humidity": round(precip * humidity / 100, 4),
            "lag_trend":       int(lag1 - lag4),
        })
    except Exception as e:
        st.error(f"Prediction error: {e}")
