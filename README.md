# 🦟 Dengue Outbreak Prediction System

**Big Data Analytics — Spring 2026**

A machine learning web application that predicts dengue fever outbreaks and case counts using environmental, climatic, and epidemiological features. Built with Streamlit, scikit-learn, and XGBoost.

---

## 📊 Results

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Classification | XGBoost | AUC-ROC | **0.9945** |
| Classification | XGBoost | F1 Score | **0.8322** |
| Regression | XGBoost | R² | **0.8971** |
| Regression | XGBoost | RMSE | **6.42 cases** |

---

## 🗂️ Project Structure

```
dengue_project/
│
├── data/
│   └── dengue_training_60k.csv       # 60,000 rows, 31 features (not tracked by git)
│
├── models/                           # Trained model artifacts (not tracked by git)
│   ├── clf_rf_model.pkl
│   ├── clf_xgb_model.pkl
│   ├── clf_scaler.pkl
│   ├── clf_le_city.pkl
│   ├── clf_le_region.pkl
│   ├── clf_features.json
│   ├── clf_metrics.json
│   ├── reg_rf_model.pkl
│   ├── reg_xgb_model.pkl
│   ├── reg_scaler.pkl
│   ├── reg_le_city.pkl
│   ├── reg_le_region.pkl
│   ├── reg_features.json
│   └── reg_metrics.json
│
├── src/
│   ├── __init__.py
│   ├── preprocess_clf.py             # Classification preprocessing pipeline
│   ├── preprocess_reg.py             # Regression preprocessing pipeline
│   ├── train_clf.py                  # Train RF + XGBoost classifiers
│   └── train_reg.py                  # Train RF + XGBoost regressors
│
├── pages/
│   ├── 1_Home.py                     # Project overview & live metrics
│   ├── 2_EDA_Dashboard.py            # Interactive charts (filterable)
│   ├── 3_Classification_Predictor.py # Outbreak Yes/No + probability gauge
│   ├── 4_Regression_Predictor.py     # Predicted case count
│   ├── 5_Model_Evaluation.py         # ROC curves, confusion matrix, feature importance
│   └── 6_Batch_Prediction.py         # Upload CSV → download predictions
│
├── assets/
│   └── style.css                     # Custom Streamlit theming
│
├── .streamlit/
│   └── config.toml                   # Streamlit theme config
│
├── Dashboard.py                      # Main Streamlit entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/dengue_project.git
cd dengue_project
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `dengue_training_60k.csv` inside the `data/` folder.

```
data/dengue_training_60k.csv
```

---

## 🏋️ Train the Models

Run once before launching the app. Each script saves `.pkl` and `.json` artifacts to `models/`.

```bash
python src/train_clf.py   # Classification — ~5 min
python src/train_reg.py   # Regression    — ~6 min
```

Verify everything trained correctly:

```bash
python verify_models.py
```

---

## 🚀 Run the App

```bash
streamlit run Dashboard.py
```

Opens at **http://localhost:8501**

---

## 📱 App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview with live model metrics |
| 📊 EDA Dashboard | Interactive charts filterable by city and year |
| 🔴 Classification Predictor | Enter features → Outbreak Yes/No + probability gauge |
| 📈 Regression Predictor | Enter features → Predicted dengue case count |
| 📋 Model Evaluation | ROC curves, confusion matrices, feature importances, CV tables |
| 📁 Batch Prediction | Upload a CSV → download a predictions file |

---

## 🧠 ML Pipeline

### Features
- **19 features** used for classification, **16** for regression
- 11 high-correlation / leaky columns dropped (r > 0.95)
- 4–5 engineered features: `ndvi_mean`, `temp_range_k`, `precip_humidity`, `lag_trend`, `lag_acceleration`

### Preprocessing
- `RobustScaler` for numerical features
- `LabelEncoder` for `city` and `region`
- IQR Winsorization for outlier capping
- **SMOTE** applied on training set only (classification) to handle class imbalance (56,992 : 3,008)
- `log1p` target transform for regression

### Models
- Random Forest and XGBoost trained for both tasks
- 5-fold cross-validation
- Hyperparameter tuning via `RandomizedSearchCV`

---

## 📦 Dependencies

```
streamlit==1.35.0
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.2
xgboost==2.0.3
imbalanced-learn==0.12.2
matplotlib==3.8.4
seaborn==0.13.2
plotly==5.22.0
joblib==1.4.2
scipy==1.13.0
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: imbalanced-learn` | `pip install imbalanced-learn` |
| `ModuleNotFoundError: xgboost` | `pip install xgboost` |
| `FileNotFoundError: dengue_training_60k.csv` | Copy dataset to `data/` folder |
| Model `.pkl` file not found | Run `python src/train_clf.py` and `python src/train_reg.py` |
| Port 8501 already in use | `streamlit run Dashboard.py --server.port 8502` |
| `LabelEncoder` unknown city error | City must be one of the 14 cities present in the training dataset |
| Streamlit page not showing | Ensure file is in `pages/` with the `N_Name.py` naming convention |
