# src/train_reg.py
# Train Random Forest + XGBoost regressors for dengue case count prediction

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from src.preprocess_reg import prepare_train_test_reg


def train_regression_models(csv_path='data/dengue_training_60k.csv',
                              model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    # ── Preprocessing ─────────────────────────────────────────────────
    print("Step 1: Preprocessing...")
    X_train, X_test, y_train, y_test, y_test_raw, scaler, le_city, le_region = \
        prepare_train_test_reg(csv_path)

    # ── Random Forest Regressor ───────────────────────────────────────
    print("\nStep 2: Training Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf_log = rf.predict(X_test)
    y_pred_rf_raw = np.expm1(y_pred_rf_log)

    rf_metrics = {
        'r2':       float(r2_score(y_test, y_pred_rf_log)),
        'rmse_log': float(np.sqrt(mean_squared_error(y_test, y_pred_rf_log))),
        'mae_log':  float(mean_absolute_error(y_test, y_pred_rf_log)),
        'rmse_raw': float(np.sqrt(mean_squared_error(y_test_raw, y_pred_rf_raw))),
        'mae_raw':  float(mean_absolute_error(y_test_raw, y_pred_rf_raw)),
    }
    print(f"  RF  → R²={rf_metrics['r2']:.4f}  "
          f"RMSE={rf_metrics['rmse_raw']:.2f} cases  "
          f"MAE={rf_metrics['mae_raw']:.2f} cases")

    # ── XGBoost Regressor ─────────────────────────────────────────────
    print("\nStep 3: Training XGBoost Regressor...")
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='rmse',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb_log = xgb.predict(X_test)
    y_pred_xgb_raw = np.expm1(y_pred_xgb_log)

    xgb_metrics = {
        'r2':       float(r2_score(y_test, y_pred_xgb_log)),
        'rmse_log': float(np.sqrt(mean_squared_error(y_test, y_pred_xgb_log))),
        'mae_log':  float(mean_absolute_error(y_test, y_pred_xgb_log)),
        'rmse_raw': float(np.sqrt(mean_squared_error(y_test_raw, y_pred_xgb_raw))),
        'mae_raw':  float(mean_absolute_error(y_test_raw, y_pred_xgb_raw)),
    }
    print(f"  XGB → R²={xgb_metrics['r2']:.4f}  "
          f"RMSE={xgb_metrics['rmse_raw']:.2f} cases  "
          f"MAE={xgb_metrics['mae_raw']:.2f} cases")

    # ── Cross-Validation ──────────────────────────────────────────────
    print("\nStep 4: 5-Fold Cross-Validation...")
    X_full  = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_full  = np.concatenate([y_train, y_test])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rf  = cross_val_score(rf,  X_full, y_full, cv=kf, scoring='r2', n_jobs=-1)
    cv_xgb = cross_val_score(xgb, X_full, y_full, cv=kf, scoring='r2', n_jobs=-1)
    print(f"  RF  CV R²: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
    print(f"  XGB CV R²: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

    rf_metrics['cv_r2_mean']  = float(cv_rf.mean())
    rf_metrics['cv_r2_std']   = float(cv_rf.std())
    rf_metrics['cv_folds']    = cv_rf.tolist()
    xgb_metrics['cv_r2_mean'] = float(cv_xgb.mean())
    xgb_metrics['cv_r2_std']  = float(cv_xgb.std())
    xgb_metrics['cv_folds']   = cv_xgb.tolist()

    rf_metrics['feature_importances']  = dict(zip(
        X_train.columns, rf.feature_importances_.tolist()))
    xgb_metrics['feature_importances'] = dict(zip(
        X_train.columns, xgb.feature_importances_.tolist()))

    # ── Save Everything ───────────────────────────────────────────────
    print("\nStep 5: Saving models...")
    joblib.dump(rf,        f'{model_dir}/reg_rf_model.pkl')
    joblib.dump(xgb,       f'{model_dir}/reg_xgb_model.pkl')
    joblib.dump(scaler,    f'{model_dir}/reg_scaler.pkl')
    joblib.dump(le_city,   f'{model_dir}/reg_le_city.pkl')
    joblib.dump(le_region, f'{model_dir}/reg_le_region.pkl')

    all_metrics = {
        'rf':  rf_metrics,
        'xgb': xgb_metrics,
        'y_test_log':     y_test.tolist(),
        'y_test_raw':     y_test_raw.tolist(),
        'y_pred_rf_log':  y_pred_rf_log.tolist(),
        'y_pred_xgb_log': y_pred_xgb_log.tolist(),
        'y_pred_rf_raw':  y_pred_rf_raw.tolist(),
        'y_pred_xgb_raw': y_pred_xgb_raw.tolist(),
        'features':       list(X_train.columns)
    }
    with open(f'{model_dir}/reg_metrics.json', 'w') as f:
        json.dump(all_metrics, f)

    print(f"\n✅ Regression training complete. All files saved to '{model_dir}/'")
    return rf, xgb, all_metrics


if __name__ == '__main__':
    train_regression_models()
