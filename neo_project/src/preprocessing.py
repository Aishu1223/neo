"""
preprocessing.py
Handles data loading and preprocessing for the NEO Hazard Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Feature columns used for training
FEATURE_COLS = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]
TARGET_COL = "hazardous"
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")


def load_data(filepath: str) -> pd.DataFrame:
    """Load the NEO dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns and handle missing values."""
    # Drop non-numeric / leakage columns if present
    drop_cols = [c for c in ["sentry_object", "orbiting_body_Earth"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    # Fill remaining NaN with column median
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    print(f"[INFO] After cleaning: {df.shape[0]} rows")
    return df


def get_features_target(df: pd.DataFrame):
    """Split dataframe into feature matrix X and target vector y."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].astype(int).values
    return X, y


def scale_features(X_train, X_test, save_scaler: bool = True):
    """Fit a StandardScaler on training data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved to {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, scaler


def load_scaler():
    """Load the fitted scaler from disk."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run train.py first.")
    return joblib.load(SCALER_PATH)


def preprocess_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
    load → clean → split → scale
    Returns X_train, X_test, y_train, y_test, scaler
    """
    df = load_data(filepath)
    df = clean_data(df)
    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "neo_cleaned.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(DATA_PATH)
    print("[INFO] Preprocessing complete.")
    print(f"       X_train shape : {X_train.shape}")
    print(f"       X_test shape  : {X_test.shape}")
