"""
train.py
Trains a RandomForest classifier on the NEO dataset,
evaluates it, and saves the model + metrics using MLflow for experiment tracking.
"""

import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── MLflow (experiment tracking) ──────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARN] mlflow not installed. Metrics will be saved to JSON only.")

from preprocessing import preprocess_pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH  = os.path.join(BASE_DIR, "data", "neo_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# ── Hyperparameters ────────────────────────────────────────────────────────────
PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def train():
    # 1. Preprocess
    print("[INFO] Starting preprocessing ...")
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline(DATA_PATH)

    # 2. Train
    print("[INFO] Training RandomForestClassifier ...")
    clf = RandomForestClassifier(**PARAMS)
    clf.fit(X_train, y_train)

    # 3. Evaluate
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n[RESULT] Accuracy : {metrics['accuracy']}")
    print(f"[RESULT] ROC-AUC  : {metrics['roc_auc']}")
    print("\n" + classification_report(y_test, y_pred))

    # 4. Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")

    # 5. Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved → {METRICS_PATH}")

    # 6. MLflow experiment tracking
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("NEO_Hazard_Prediction")
        with mlflow.start_run(run_name="RandomForest_v1"):
            mlflow.log_params(PARAMS)
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("roc_auc",  metrics["roc_auc"])
            mlflow.sklearn.log_model(clf, "model")
            mlflow.log_artifact(METRICS_PATH)
        print("[INFO] MLflow run logged.")

    return clf, metrics


if __name__ == "__main__":
    train()
