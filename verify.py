import os, json

print("=" * 60)
print("DENGUE PROJECT — FINAL STRUCTURE VERIFICATION")
print("=" * 60)

checks = {
    "Dataset":     ["data/dengue_training_60k.csv"],
    "Source Code": ["src/__init__.py","src/preprocess_clf.py",
                    "src/preprocess_reg.py","src/train_clf.py","src/train_reg.py"],
    "Models":      ["models/clf_rf_model.pkl","models/clf_xgb_model.pkl",
                    "models/clf_scaler.pkl","models/clf_le_city.pkl",
                    "models/clf_le_region.pkl","models/clf_metrics.json",
                    "models/reg_rf_model.pkl","models/reg_xgb_model.pkl",
                    "models/reg_scaler.pkl","models/reg_le_city.pkl",
                    "models/reg_le_region.pkl","models/reg_metrics.json"],
    "Pages":       ["pages/1_Home.py","pages/2_EDA_Dashboard.py",
                    "pages/3_Classification_Predictor.py",
                    "pages/4_Regression_Predictor.py",
                    "pages/5_Model_Evaluation.py",
                    "pages/6_Batch_Prediction.py"],
    "Config":      ["Dashboard.py","requirements.txt","README.md",
                    ".streamlit/config.toml","assets/style.css"],
}

all_ok = True
for category, files in checks.items():
    print(f"\n{category}:")
    for f in files:
        ok = os.path.exists(f)
        sz = f"({os.path.getsize(f)/1024:.1f} KB)" if ok else ""
        print(f"  {'✅' if ok else '❌ MISSING'} {f} {sz}")
        if not ok: all_ok = False

print("\n" + "=" * 60)
if all_ok:
    with open('models/clf_metrics.json') as f: cm = json.load(f)
    with open('models/reg_metrics.json') as f: rm = json.load(f)
    print("🎯 ALL FILES PRESENT — FINAL RESULTS:")
    print(f"  CLF XGBoost AUC-ROC : {cm['xgb']['auc_roc']:.4f}")
    print(f"  CLF XGBoost F1      : {cm['xgb']['f1']:.4f}")
    print(f"  CLF 5-Fold CV AUC   : {cm['xgb']['cv_auc_mean']:.4f} ± {cm['xgb']['cv_auc_std']:.4f}")
    print(f"  REG XGBoost R²      : {rm['xgb']['r2']:.4f}")
    print(f"  REG XGBoost RMSE    : {rm['xgb']['rmse_raw']:.2f} cases")
    print(f"  REG 5-Fold CV R²    : {rm['xgb']['cv_r2_mean']:.4f} ± {rm['xgb']['cv_r2_std']:.4f}")
    print("\n✅ PROJECT COMPLETE. Run: streamlit run Dashboard.py")
else:
    print("❌ Some files missing. Check the errors above.")
print("=" * 60)
