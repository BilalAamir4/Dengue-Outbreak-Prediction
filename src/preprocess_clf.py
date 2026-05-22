# src/preprocess_clf.py
# Classification Preprocessing Pipeline
# Target: outbreak_zone (0 = No Outbreak, 1 = Outbreak)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ── Columns to drop (leakage + multicollinearity r > 0.95) ──────────────
DROP_COLS_CLF = [
    'total_cases',                          # regression target — leakage
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

# ── Final 19 features selected by MI + RFE union ─────────────────────────
CLF_FEATURES = [
    'lag1_cases', 'lag4_cases', 'lag_trend', 'weekofyear',
    'city', 'region', 'hospital_capacity_index', 'population_density',
    'reanalysis_relative_humidity_percent', 'temp_range_k',
    'precipitation_amt_mm', 'reanalysis_dew_point_temp_k',
    'precip_humidity', 'reanalysis_min_air_temp_k',
    'reanalysis_precip_amt_kg_per_m2', 'reanalysis_avg_temp_k',
    'reanalysis_tdtr_k', 'reanalysis_max_air_temp_k', 'ndvi_mean'
]


def preprocess_for_classification(df_input, fit=True, scaler=None,
                                   le_city=None, le_region=None):
    """
    Full preprocessing pipeline for classification.

    Parameters:
        df_input : pd.DataFrame — raw dataset
        fit      : bool — True for training, False for inference
        scaler   : fitted RobustScaler (required if fit=False)
        le_city  : fitted LabelEncoder for city (required if fit=False)
        le_region: fitted LabelEncoder for region (required if fit=False)

    Returns:
        X_scaled : pd.DataFrame of scaled features
        y        : pd.Series of outbreak_zone labels (None if not present)
        scaler, le_city, le_region : fitted transformers
    """
    df = df_input.copy()

    # ── Step 1: Drop redundant / leaky columns ──────────────────────────
    cols_to_drop = [c for c in DROP_COLS_CLF if c in df.columns]
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

    # ── Step 3: Feature Engineering (4 new features) ──────────────────
    df['ndvi_mean']       = df[['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw']].mean(axis=1)
    df['temp_range_k']    = df['reanalysis_max_air_temp_k'] - df['reanalysis_min_air_temp_k']
    df['precip_humidity'] = (df['precipitation_amt_mm'] *
                             df['reanalysis_relative_humidity_percent'] / 100)
    df['lag_trend']       = df['lag1_cases'] - df['lag4_cases']
    df.drop(columns=['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'],
            errors='ignore', inplace=True)

    # ── Step 4: Extract target ─────────────────────────────────────────
    y = df['outbreak_zone'].copy() if 'outbreak_zone' in df.columns else None
    X = df.drop(columns=['outbreak_zone'], errors='ignore')

    # ── Step 5: Outlier Capping (IQR Winsorization) ───────────────────
    for col in X.select_dtypes(include=np.number).columns:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = X[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # ── Step 6: Select final features ────────────────────────────────
    available = [f for f in CLF_FEATURES if f in X.columns]
    X = X[available]

    # ── Step 7: RobustScaler ──────────────────────────────────────────
    if fit:
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=available)

    return X_scaled, y, scaler, le_city, le_region


def prepare_train_test_clf(csv_path='data/dengue_training_60k.csv'):
    """Load data, preprocess, apply SMOTE, return train/test splits."""
    df = pd.read_csv(csv_path)
    X_scaled, y, scaler, le_city, le_region = preprocess_for_classification(df, fit=True)

    # Train/test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE on training set only
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"✅ CLF preprocessing done")
    print(f"   Train (after SMOTE): {X_train_sm.shape} | "
          f"Class dist: {dict(pd.Series(y_train_sm).value_counts())}")
    print(f"   Test: {X_test.shape} | Outbreak rate: {y_test.mean():.4f}")

    return X_train_sm, X_test, y_train_sm, y_test, scaler, le_city, le_region


if __name__ == '__main__':
    prepare_train_test_clf()
