# pages/6_Batch_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import sys
import os

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")
sys.path.insert(0, '.')

st.title("📁 Batch Prediction")
st.markdown("### Upload a CSV file to get predictions for multiple records at once")
st.markdown("---")

@st.cache_resource
def load_all_models():
    models = {}
    try:
        models['clf_xgb']    = joblib.load('models/clf_xgb_model.pkl')
        models['clf_scaler'] = joblib.load('models/clf_scaler.pkl')
        models['clf_le_city']   = joblib.load('models/clf_le_city.pkl')
        models['clf_le_region'] = joblib.load('models/clf_le_region.pkl')
        models['reg_xgb']    = joblib.load('models/reg_xgb_model.pkl')
        models['reg_scaler'] = joblib.load('models/reg_scaler.pkl')
        models['reg_le_city']   = joblib.load('models/reg_le_city.pkl')
        models['reg_le_region'] = joblib.load('models/reg_le_region.pkl')
    except Exception as e:
        st.error(f"Model loading error: {e}")
    return models

models = load_all_models()

st.markdown("### Required CSV Format")
st.markdown("Your CSV must contain the following columns (same as the training dataset):")

required_cols = [
    'city','region','weekofyear','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw',
    'precipitation_amt_mm','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k',
    'reanalysis_max_air_temp_k','reanalysis_min_air_temp_k',
    'reanalysis_relative_humidity_percent','reanalysis_tdtr_k',
    'reanalysis_precip_amt_kg_per_m2','population_density',
    'hospital_capacity_index','lag1_cases','lag4_cases'
]
st.code(', '.join(required_cols))

# Sample CSV download
sample_data = {col: ['San_Juan' if col=='city' else 'Puerto_Rico' if col=='region'
                      else 30 if col=='weekofyear' else 0.15 if 'ndvi' in col
                      else 35.0 if 'precip' in col else 299.0 if 'temp' in col
                      else 75.0 if 'humidity' in col else 0.7 if 'density' in col
                      else 0.5 if 'capacity' in col else 5 if 'lag' in col else 0.0]
               for col in required_cols}
sample_df = pd.DataFrame(sample_data)
csv_sample = sample_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇ Download Sample CSV Template",
                   csv_sample, "sample_input.csv", "text/csv")

st.markdown("---")
uploaded = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded is not None:
    df_upload = pd.read_csv(uploaded)
    st.markdown(f"**Uploaded:** {len(df_upload)} rows × {len(df_upload.columns)} columns")
    st.dataframe(df_upload.head(5), use_container_width=True)

    if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
        from src.preprocess_clf import preprocess_for_classification
        from src.preprocess_reg  import preprocess_for_regression

        results = df_upload.copy()

        # Classification
        try:
            X_clf, _, _, _, _ = preprocess_for_classification(
                df_upload.copy(), fit=False,
                scaler=models['clf_scaler'],
                le_city=models['clf_le_city'],
                le_region=models['clf_le_region'])
            results['predicted_outbreak_zone'] = models['clf_xgb'].predict(X_clf)
            results['outbreak_probability_%']  = (
                models['clf_xgb'].predict_proba(X_clf)[:,1] * 100).round(1)
        except Exception as e:
            st.error(f"Classification error: {e}")

        # Regression
        try:
            X_reg, _, _, _, _, _ = preprocess_for_regression(
                df_upload.copy(), fit=False,
                scaler=models['reg_scaler'],
                le_city=models['reg_le_city'],
                le_region=models['reg_le_region'])
            pred_log = models['reg_xgb'].predict(X_reg)
            results['predicted_cases'] = np.maximum(0, np.expm1(pred_log)).astype(int)
        except Exception as e:
            st.error(f"Regression error: {e}")

        st.markdown("### Prediction Results")
        st.dataframe(results, use_container_width=True)

        # Summary stats
        if 'predicted_outbreak_zone' in results.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records",    len(results))
            c2.metric("Outbreak Alerts",  int(results['predicted_outbreak_zone'].sum()))
            c3.metric("Mean Predicted Cases",
                      f"{results['predicted_cases'].mean():.1f}" if 'predicted_cases' in results.columns else "N/A")

        # Download results
        out_csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Predictions CSV",
                           out_csv, "predictions_output.csv", "text/csv",
                           use_container_width=True)
