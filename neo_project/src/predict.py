"""
predict.py
Prediction utilities – loads the saved model + scaler and returns predictions.
"""

import os
import joblib
import numpy as np
import logging

logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "..", "models", "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

FEATURE_COLS = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run train.py first.")
    return joblib.load(SCALER_PATH)


def predict_single(
    est_diameter_min: float,
    est_diameter_max: float,
    relative_velocity: float,
    miss_distance: float,
    absolute_magnitude: float,
) -> dict:
    """
    Predict whether a single NEO is hazardous.

    Returns a dict with:
        - prediction  : 0 (Not Hazardous) or 1 (Hazardous)
        - label       : human-readable string
        - probability : probability of being hazardous (0–1)
    """
    model  = load_model()
    scaler = load_scaler()

    features = np.array([[
        est_diameter_min,
        est_diameter_max,
        relative_velocity,
        miss_distance,
        absolute_magnitude,
    ]])

    features_scaled = scaler.transform(features)
    prediction      = int(model.predict(features_scaled)[0])
    probability     = float(model.predict_proba(features_scaled)[0][1])

    label = "🚨 Hazardous" if prediction == 1 else "✅ Not Hazardous"

    result = {
        "prediction":  prediction,
        "label":       label,
        "probability": round(probability, 4),
    }

    # Log prediction
    logging.info(
        "INPUT=[%s] | PREDICTION=%s | PROB=%.4f",
        ", ".join(map(str, [est_diameter_min, est_diameter_max,
                             relative_velocity, miss_distance, absolute_magnitude])),
        label,
        probability,
    )

    return result


if __name__ == "__main__":
    # Quick test
    result = predict_single(
        est_diameter_min=0.3,
        est_diameter_max=0.7,
        relative_velocity=60000,
        miss_distance=30000000,
        absolute_magnitude=18.5,
    )
    print(result)
