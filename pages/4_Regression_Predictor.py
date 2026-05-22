# pages/4_Regression_Predictor.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Regression Predictor", page_icon="📈", layout="wide")

@st.cache_resource
def load_reg_models():
    rf        = joblib.load('models/reg_rf_model.pkl')
    xgb       = joblib.load('models/reg_xgb_model.pkl')
    scaler    = joblib.load('models/reg_scaler.pkl')
    le_city   = joblib.load('models/reg_le_city.pkl')
    le_region = joblib.load('models/reg_le_region.pkl')
    return rf, xgb, scaler, le_city, le_region

st.title("📈 Regression Predictor")
st.markdown("### Predict: How many dengue cases will occur this week?")
st.markdown("---")

if not os.path.exists('models/reg_xgb_model.pkl'):
    st.error("Regression model not found. Run: `python src/train_reg.py`")
    st.stop()

rf, xgb_model, scaler, le_city, le_region = load_reg_models()

with st.sidebar:
    st.markdown("### Model Selection")
    model_choice = st.radio("Choose Model", ["XGBoost (Recommended)", "Random Forest"])
    st.markdown("---")
    st.markdown("### About This Model")
    st.markdown("""
    **Target:** total_cases (count)  
    **Transform:** log1p → expm1  
    **Best Model:** XGBoost  
    **R²:** 0.8971  
    **RMSE:** 6.42 cases  
    **CV R²:** 0.8968 ± 0.0006
    """)

st.markdown("### Enter Weekly Features")

CITIES  = ['Colombo','Gampaha','Kandy','Galle','Jaffna','Ampara',
           'Lahore','Karachi','Peshawar','Islamabad','Quetta','Muzaffarabad',
           'San_Juan','Iquitos']
REGIONS = ['Sri_Lanka','Punjab_PK','Sindh_PK','KPK_PK','ICT_PK',
           'Balochistan_PK','AJK_PK','Puerto_Rico','Peru']

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Location & Time**")
    city       = st.selectbox("City", CITIES)
    region     = st.selectbox("Region", REGIONS)
    weekofyear = st.slider("Week of Year", 1, 52, 35)

with col2:
    st.markdown("**Climate Features**")
    humidity   = st.slider("Relative Humidity (%)", 30.0, 99.0, 80.0, 0.5)
    precip     = st.slider("Precipitation (mm)", 0.0, 420.0, 50.0, 1.0)
    temp_min_k = st.slider("Min Air Temp (K)", 282.0, 304.0, 296.0, 0.1)
    temp_max_k = st.slider("Max Air Temp (K)", 292.0, 318.0, 304.0, 0.1)
    tdtr_k     = st.slider("Diurnal Temp Range (K)", 1.0, 18.0, 5.0, 0.1)
    precip_r   = st.slider("Reanalysis Precip (kg/m²)", 0.0, 600.0, 50.0, 1.0)

with col3:
    st.markdown("**Lag & Socioeconomic**")
    lag1 = st.number_input("Lag 1 Cases (last week)",  min_value=0, max_value=300, value=10)
    lag4 = st.number_input("Lag 4 Cases (4 weeks ago)",min_value=0, max_value=300, value=7)
    pop_density  = st.slider("Population Density (0–1)", 0.1, 1.0, 0.75, 0.01)
    hospital_cap = st.slider("Hospital Capacity (0–1)",  0.1, 1.0, 0.55, 0.01)

if st.button("📈 Predict Case Count", use_container_width=True, type="primary"):
    input_data = {
        'city': city, 'region': region, 'weekofyear': weekofyear,
        'precipitation_amt_mm': precip,
        'reanalysis_max_air_temp_k': temp_max_k,
        'reanalysis_min_air_temp_k': temp_min_k,
        'reanalysis_relative_humidity_percent': humidity,
        'reanalysis_tdtr_k': tdtr_k,
        'reanalysis_precip_amt_kg_per_m2': precip_r,
        'population_density': pop_density,
        'hospital_capacity_index': hospital_cap,
        'lag1_cases': lag1,
        'lag4_cases': lag4,
        'ndvi_ne': 0.15, 'ndvi_nw': 0.15, 'ndvi_se': 0.15, 'ndvi_sw': 0.15,
        'reanalysis_avg_temp_k': (temp_min_k + temp_max_k) / 2,
    }
    df_input = pd.DataFrame([input_data])

    import sys; sys.path.insert(0, '.')
    from src.preprocess_reg import preprocess_for_regression
    try:
        X_proc, _, _, _, _, _ = preprocess_for_regression(
            df_input, fit=False, scaler=scaler,
            le_city=le_city, le_region=le_region)

        model    = xgb_model if "XGBoost" in model_choice else rf
        pred_log = model.predict(X_proc)[0]
        pred_raw = int(np.expm1(pred_log))

        lag_acceleration = lag1 - 2 * lag4
        trend_label = "📈 Rising" if lag_acceleration > 0 else "📉 Falling" if lag_acceleration < 0 else "➡ Stable"

        st.markdown("---")
        st.markdown("### Prediction Result")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Cases", f"{max(0, pred_raw)}", "this week")
        col2.metric("log1p Prediction", f"{pred_log:.4f}", "internal model output")
        col3.metric("Outbreak Trend", trend_label)
        col4.metric("Severity",
                    "🔴 High" if pred_raw > 30 else "🟡 Medium" if pred_raw > 10 else "🟢 Low")

        # Bar chart showing predicted vs typical
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Predicted Cases', 'Dataset Median (3)', 'Dataset Mean (8.7)', 'Dataset 75th pct (8)'],
            y=[max(0, pred_raw), 3, 8.7, 8],
            marker_color=['#DC2626' if pred_raw > 30 else '#D97706' if pred_raw > 10 else '#059669',
                          '#6B7280', '#6B7280', '#6B7280']
        ))
        fig.update_layout(title='Predicted Cases vs Dataset Benchmarks',
                          yaxis_title='Cases', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Computed Features")
        st.json({
            "lag_trend":        int(lag1 - lag4),
            "lag_acceleration": int(lag_acceleration),
            "temp_range_k":     round(temp_max_k - temp_min_k, 2),
            "precip_humidity":  round(precip * humidity / 100, 2),
            "ndvi_mean":        0.15,
        })
    except Exception as e:
        st.error(f"Prediction error: {e}")
