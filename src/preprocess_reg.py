# src/preprocess_reg.py
# Regression Preprocessing Pipeline
# Target: total_cases (continuous count, 0–227)
# Key difference from classification: log1p transform on target, no SMOTE

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ── Columns to drop (leakage + multicollinearity r > 0.95) ──────────────
DROP_COLS_REG = [
    'outbreak_zone',                        # classification target — leakage
    'year',                                 # zero mutual information
    'month',                                # r=0.995 with weekofyear
    'reanalysis_air_temp_k',                # r=0.995 with avg_temp_k
    'station_precip_mm',                    # r=0.994 with precipitation_amt_mm
    'station_min_temp_c',                   # r=0.989 with reanalysis_min_air_temp_k
    'station_max_temp_c',                   # r=0.985 with reanalysis_max_air_temp_k
    'station_avg_temp_c',                   # r=0.985 with reanalysis_avg_temp_k
    'station_diur_temp_rng_c',              # r=0.972 with reanalysis_tdtr_k
    'reanalysis_specific_humidity_g_per_kg',# r=0.969 with humidity_percent
    'lag2_cases',                           # r=0.992 with lag1_cases
]

# ── Final 16 features selected by MI regression ──────────────────────────
REG_FEATURES = [
    'lag1_cases', 'lag4_cases', 'lag_acceleration', 'city', 'region',
    'hospital_capacity_index', 'reanalysis_relative_humidity_percent',
    'weekofyear', 'population_density', 'lag_trend',
    'precip_humidity', 'temp_range_k', 'reanalysis_tdtr_k',
    'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
    'precipitation_amt_mm'
]


def preprocess_for_regression(df_input, fit=True, scaler=None,
                               le_city=None, le_region=None):
    """
    Full preprocessing pipeline for regression.

    Parameters:
        df_input : pd.DataFrame — raw dataset
        fit      : bool — True for training, False for inference
        scaler   : fitted RobustScaler (required if fit=False)
        le_city  : fitted LabelEncoder for city (required if fit=False)
        le_region: fitted LabelEncoder for region (required if fit=False)

    Returns:
        X_scaled    : pd.DataFrame of scaled features
        y_log       : np.array of log1p(total_cases) — None if target absent
        y_raw       : np.array of raw total_cases — None if target absent
        scaler, le_city, le_region : fitted transformers
    """
    df = df_input.copy()

    # ── Step 1: Drop redundant / leaky columns ──────────────────────────
    cols_to_drop = [c for c in DROP_COLS_REG if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # ── Step 2: Label Encoding ─────────────────────────────────────────
    if fit:
        le_city   = LabelEncoder()
        le_region = LabelEncoder()
        df['city']   = le_city.fit_transform(df['city'])
        df['region'] = le_region.fit_transform(df['region'])
    else:
        df['city']   = le_city.transform(df['city'])
        df['region'] = le_region.transform(df['region'])

    # ── Step 3: Feature Engineering (5 new features) ──────────────────
    df['ndvi_mean']        = df[['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw']].mean(axis=1)
    df['temp_range_k']     = df['reanalysis_max_air_temp_k'] - df['reanalysis_min_air_temp_k']
    df['precip_humidity']  = (df['precipitation_amt_mm'] *
                              df['reanalysis_relative_humidity_percent'] / 100)
    df['lag_trend']        = df['lag1_cases'] - df['lag4_cases']
    df['lag_acceleration'] = df['lag1_cases'] - 2 * df['lag4_cases']   # extra vs classification
    df.drop(columns=['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'],
            errors='ignore', inplace=True)

    # ── Step 4: Extract target + log1p transform ──────────────────────
    y_raw = df['total_cases'].values.copy() if 'total_cases' in df.columns else None
    y_log = np.log1p(y_raw) if y_raw is not None else None
    X = df.drop(columns=['total_cases'], errors='ignore')

    # ── Step 5: Outlier Capping (IQR Winsorization) ───────────────────
    for col in X.select_dtypes(include=np.number).columns:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = X[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # ── Step 6: Select final features ────────────────────────────────
    available = [f for f in REG_FEATURES if f in X.columns]
    X = X[available]

    # ── Step 7: RobustScaler ──────────────────────────────────────────
    if fit:
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=available)

    return X_scaled, y_log, y_raw, scaler, le_city, le_region


def prepare_train_test_reg(csv_path='data/dengue_training_60k.csv'):
    """Load data, preprocess, return train/test splits."""
    df = pd.read_csv(csv_path)
    X_scaled, y_log, y_raw, scaler, le_city, le_region = preprocess_for_regression(
        df, fit=True)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42)
    y_test_raw = np.expm1(y_test)

    print(f"✅ REG preprocessing done")
    print(f"   Train: {X_train.shape} | log1p target mean: {y_train.mean():.4f}")
    print(f"   Test:  {X_test.shape}  | raw cases mean:    {y_test_raw.mean():.2f}")

    return X_train, X_test, y_train, y_test, y_test_raw, scaler, le_city, le_region


if __name__ == '__main__':
    prepare_train_test_reg()
