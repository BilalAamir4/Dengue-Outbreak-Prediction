# pages/5_Model_Evaluation.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve
import os

st.set_page_config(page_title="Model Evaluation", page_icon="📋", layout="wide")

st.title("📋 Model Evaluation")
st.markdown("Complete performance metrics for both Classification and Regression models")
st.markdown("---")

tab1, tab2 = st.tabs(["🔴 Classification Evaluation", "📈 Regression Evaluation"])

# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Classification — Predicting outbreak_zone")

    if not os.path.exists('models/clf_metrics.json'):
        st.error("Run `python src/train_clf.py` first.")
        st.stop()

    with open('models/clf_metrics.json') as f:
        cm = json.load(f)

    y_test     = np.array(cm['y_test'])
    y_pred_rf  = np.array(cm['y_pred_rf'])
    y_pred_xgb = np.array(cm['y_pred_xgb'])
    y_prob_rf  = np.array(cm['y_prob_rf'])
    y_prob_xgb = np.array(cm['y_prob_xgb'])

    # Metrics table
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGB Accuracy",      f"{cm['xgb']['accuracy']:.4f}")
    c2.metric("XGB F1 Score",       f"{cm['xgb']['f1']:.4f}")
    c3.metric("XGB AUC-ROC",        f"{cm['xgb']['auc_roc']:.4f}")
    c4.metric("XGB 5-Fold CV AUC",  f"{cm['xgb']['cv_auc_mean']:.4f} ± {cm['xgb']['cv_auc_std']:.4f}")

    col1, col2 = st.columns(2)

    # Confusion matrices
    with col1:
        model_sel = st.radio("Select model for confusion matrix",
                             ["XGBoost", "Random Forest"], horizontal=True)
        cm_data  = cm['xgb']['confusion_matrix'] if model_sel == "XGBoost" \
                   else cm['rf']['confusion_matrix']
        fig = px.imshow(cm_data,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['No Outbreak', 'Outbreak'],
                        y=['No Outbreak', 'Outbreak'],
                        text_auto=True, color_continuous_scale='Blues',
                        title=f'{model_sel} — Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC curves
        fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name=f"RF (AUC={cm['rf']['auc_roc']:.4f})",
                                 line=dict(color='#059669', width=2)))
        fig.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, name=f"XGB (AUC={cm['xgb']['auc_roc']:.4f})",
                                 line=dict(color='#2563EB', width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', mode='lines',
                                 line=dict(color='gray', dash='dash')))
        fig.update_layout(title='ROC Curves', xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Precision-Recall
        prec_rf, rec_rf, _   = precision_recall_curve(y_test, y_prob_rf)
        prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec_rf, y=prec_rf, name=f"RF (AP={cm['rf']['avg_prec']:.4f})",
                                 line=dict(color='#059669', width=2)))
        fig.add_trace(go.Scatter(x=rec_xgb, y=prec_xgb, name=f"XGB (AP={cm['xgb']['avg_prec']:.4f})",
                                 line=dict(color='#2563EB', width=2)))
        fig.update_layout(title='Precision-Recall Curves',
                          xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importance
        fi = pd.DataFrame(list(cm['xgb']['feature_importances'].items()),
                          columns=['Feature', 'Importance']).sort_values('Importance')
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     title='XGBoost Feature Importances (Classification)',
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # CV folds
    st.markdown("### 5-Fold Cross-Validation AUC-ROC Scores")
    cv_data = pd.DataFrame({
        'Fold':          ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'],
        'Random Forest': cm['rf']['cv_folds'],
        'XGBoost':       cm['xgb']['cv_folds']
    })
    st.dataframe(cv_data, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Regression — Predicting total_cases")

    if not os.path.exists('models/reg_metrics.json'):
        st.error("Run `python src/train_reg.py` first.")
        st.stop()

    with open('models/reg_metrics.json') as f:
        rm = json.load(f)

    y_test_log    = np.array(rm['y_test_log'])
    y_test_raw    = np.array(rm['y_test_raw'])
    y_pred_rf_log = np.array(rm['y_pred_rf_log'])
    y_pred_rf_raw = np.array(rm['y_pred_rf_raw'])
    y_pred_xgb_log= np.array(rm['y_pred_xgb_log'])
    y_pred_xgb_raw= np.array(rm['y_pred_xgb_raw'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGB R²",          f"{rm['xgb']['r2']:.4f}")
    c2.metric("XGB RMSE (cases)",f"{rm['xgb']['rmse_raw']:.2f}")
    c3.metric("XGB MAE (cases)", f"{rm['xgb']['mae_raw']:.2f}")
    c4.metric("XGB 5-Fold CV R²",f"{rm['xgb']['cv_r2_mean']:.4f} ± {rm['xgb']['cv_r2_std']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        model_r = st.radio("Model for scatter plot", ["XGBoost","Random Forest"], horizontal=True)
        y_pred_use = y_pred_xgb_log if model_r=="XGBoost" else y_pred_rf_log
        r2_use = rm['xgb']['r2'] if model_r=="XGBoost" else rm['rf']['r2']
        sample = np.random.choice(len(y_test_log), 3000, replace=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test_log[sample], y=y_pred_use[sample],
                                 mode='markers', marker=dict(color='#2563EB', opacity=0.3, size=4),
                                 name='Predictions'))
        mn, mx = y_test_log.min(), y_test_log.max()
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                 line=dict(color='red', dash='dash'), name='Perfect Fit'))
        fig.update_layout(title=f'{model_r} — Predicted vs Actual (log space)\nR²={r2_use:.4f}',
                          xaxis_title='Actual log1p(cases)',
                          yaxis_title='Predicted log1p(cases)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Residuals
        resid = y_test_log - y_pred_use
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred_use[sample], y=resid[sample],
                                 mode='markers', marker=dict(color='#059669', opacity=0.3, size=4)))
        fig.add_hline(y=0, line_color='red', line_dash='dash')
        fig.update_layout(title=f'{model_r} — Residual Plot',
                          xaxis_title='Predicted', yaxis_title='Residual')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        metrics_comp = pd.DataFrame({
            'Metric':          ['R²','RMSE (log)','MAE (log)','RMSE (cases)','MAE (cases)'],
            'Random Forest':   [rm['rf']['r2'], rm['rf']['rmse_log'], rm['rf']['mae_log'],
                                rm['rf']['rmse_raw'], rm['rf']['mae_raw']],
            'XGBoost ✓':       [rm['xgb']['r2'],rm['xgb']['rmse_log'],rm['xgb']['mae_log'],
                                rm['xgb']['rmse_raw'], rm['xgb']['mae_raw']],
        })
        st.dataframe(metrics_comp.round(4), use_container_width=True, hide_index=True)

    with col2:
        fi = pd.DataFrame(list(rm['xgb']['feature_importances'].items()),
                          columns=['Feature','Importance']).sort_values('Importance')
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     title='XGBoost Feature Importances (Regression)',
                     color='Importance', color_continuous_scale='Greens')
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 5-Fold Cross-Validation R² Scores")
    cv_data = pd.DataFrame({
        'Fold':          ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'],
        'Random Forest': rm['rf']['cv_folds'],
        'XGBoost':       rm['xgb']['cv_folds']
    })
    st.dataframe(cv_data.round(4), use_container_width=True, hide_index=True)
