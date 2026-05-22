# src/train_clf.py
# Train Random Forest + XGBoost classifiers on dengue outbreak data

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score,
                             average_precision_score)
from xgboost import XGBClassifier
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from src.preprocess_clf import prepare_train_test_clf, CLF_FEATURES


def train_classification_models(csv_path='data/dengue_training_60k.csv',
                                  model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    # ── Preprocessing ─────────────────────────────────────────────────
    print("Step 1: Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, le_city, le_region = \
        prepare_train_test_clf(csv_path)

    # ── Random Forest ─────────────────────────────────────────────────
    print("\nStep 2: Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf  = rf.predict(X_test)
    y_prob_rf  = rf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'f1':       float(f1_score(y_test, y_pred_rf)),
        'auc_roc':  float(roc_auc_score(y_test, y_prob_rf)),
        'avg_prec': float(average_precision_score(y_test, y_prob_rf)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist(),
        'report':   classification_report(y_test, y_pred_rf,
                        target_names=['No Outbreak', 'Outbreak'])
    }
    print(f"  RF  → Accuracy={rf_metrics['accuracy']:.4f}  "
          f"F1={rf_metrics['f1']:.4f}  AUC={rf_metrics['auc_roc']:.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────
    print("\nStep 3: Training XGBoost...")
    scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    xgb_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_xgb)),
        'f1':       float(f1_score(y_test, y_pred_xgb)),
        'auc_roc':  float(roc_auc_score(y_test, y_prob_xgb)),
        'avg_prec': float(average_precision_score(y_test, y_prob_xgb)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist(),
        'report':   classification_report(y_test, y_pred_xgb,
                        target_names=['No Outbreak', 'Outbreak'])
    }
    print(f"  XGB → Accuracy={xgb_metrics['accuracy']:.4f}  "
          f"F1={xgb_metrics['f1']:.4f}  AUC={xgb_metrics['auc_roc']:.4f}")

    # ── Cross-Validation ──────────────────────────────────────────────
    print("\nStep 4: 5-Fold Cross-Validation...")
    # Reload full scaled dataset for CV (without SMOTE inflation)
    df_full = pd.read_csv(csv_path)
    from src.preprocess_clf import preprocess_for_classification
    X_full, y_full, _, _, _ = preprocess_for_classification(
        df_full, fit=False, scaler=scaler, le_city=le_city, le_region=le_region)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rf  = cross_val_score(rf,  X_full, y_full, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_xgb = cross_val_score(xgb, X_full, y_full, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  RF  CV AUC: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
    print(f"  XGB CV AUC: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

    rf_metrics['cv_auc_mean']  = float(cv_rf.mean())
    rf_metrics['cv_auc_std']   = float(cv_rf.std())
    rf_metrics['cv_folds']     = cv_rf.tolist()
    xgb_metrics['cv_auc_mean'] = float(cv_xgb.mean())
    xgb_metrics['cv_auc_std']  = float(cv_xgb.std())
    xgb_metrics['cv_folds']    = cv_xgb.tolist()

    # Feature importances
    rf_metrics['feature_importances']  = dict(zip(
        X_train.columns, rf.feature_importances_.tolist()))
    xgb_metrics['feature_importances'] = dict(zip(
        X_train.columns, xgb.feature_importances_.tolist()))

    # ── Save Everything ───────────────────────────────────────────────
    print("\nStep 5: Saving models...")
    joblib.dump(rf,        f'{model_dir}/clf_rf_model.pkl')
    joblib.dump(xgb,       f'{model_dir}/clf_xgb_model.pkl')
    joblib.dump(scaler,    f'{model_dir}/clf_scaler.pkl')
    joblib.dump(le_city,   f'{model_dir}/clf_le_city.pkl')
    joblib.dump(le_region, f'{model_dir}/clf_le_region.pkl')

    # Save metrics and test data for Streamlit evaluation page
    all_metrics = {
        'rf':  rf_metrics,
        'xgb': xgb_metrics,
        'y_test':     y_test.tolist(),
        'y_pred_rf':  y_pred_rf.tolist(),
        'y_pred_xgb': y_pred_xgb.tolist(),
        'y_prob_rf':  y_prob_rf.tolist(),
        'y_prob_xgb': y_prob_xgb.tolist(),
        'features':   list(X_train.columns)
    }
    with open(f'{model_dir}/clf_metrics.json', 'w') as f:
        json.dump(all_metrics, f)

    print(f"\n✅ Classification training complete. All files saved to '{model_dir}/'")
    return rf, xgb, all_metrics


if __name__ == '__main__':
    train_classification_models()
