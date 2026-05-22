import os
import json

expected_files = [
    'models/clf_rf_model.pkl',
    'models/clf_xgb_model.pkl',
    'models/clf_scaler.pkl',
    'models/clf_le_city.pkl',
    'models/clf_le_region.pkl',
    'models/clf_metrics.json',
    'models/reg_rf_model.pkl',
    'models/reg_xgb_model.pkl',
    'models/reg_scaler.pkl',
    'models/reg_le_city.pkl',
    'models/reg_le_region.pkl',
    'models/reg_metrics.json',
]

print("Checking model files...")
all_ok = True
for f in expected_files:
    exists = os.path.exists(f)
    size   = os.path.getsize(f) if exists else 0
    status = "✅" if exists else "❌ MISSING"
    print(f"  {status}  {f}  ({size/1024:.1f} KB)")
    if not exists:
        all_ok = False

if all_ok:
    # Quick load test
    import joblib
    clf_xgb = joblib.load('models/clf_xgb_model.pkl')
    reg_xgb = joblib.load('models/reg_xgb_model.pkl')
    with open('models/clf_metrics.json') as f:
        clf_m = json.load(f)
    with open('models/reg_metrics.json') as f:
        reg_m = json.load(f)

    print("\n🎯 TRAINING RESULTS SUMMARY")
    print(f"  CLF XGBoost → AUC-ROC: {clf_m['xgb']['auc_roc']:.4f}  F1: {clf_m['xgb']['f1']:.4f}")
    print(f"  REG XGBoost → R²:      {reg_m['xgb']['r2']:.4f}  RMSE: {reg_m['xgb']['rmse_raw']:.2f} cases")
    print("\n✅ All models verified. Ready to build Streamlit app.")
else:
    print("\n❌ Some files are missing. Re-run the training scripts.")
