"""
app.py  –  Streamlit Frontend for NEO Hazard Prediction
Run with:  streamlit run app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌌 NEO Hazard Predictor",
    page_icon="☄️",
    layout="centered",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("☄️ Near-Earth Object (NEO) Hazard Predictor")
st.markdown(
    """
    Enter the physical and orbital characteristics of a Near-Earth Object below.  
    The model will predict whether it is **potentially hazardous**.
    """
)
st.divider()

# ── Sidebar – model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "**Model:** Random Forest Classifier\n\n"
        "**Dataset:** NASA NEO dataset\n\n"
        "**Features used:**\n"
        "- Estimated diameter (min/max)\n"
        "- Relative velocity\n"
        "- Miss distance\n"
        "- Absolute magnitude"
    )

    # Show saved metrics if available
    import json
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.subheader("📊 Model Performance")
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        st.metric("ROC-AUC",  f"{metrics['roc_auc']:.4f}")

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("🔢 Enter NEO Features")

col1, col2 = st.columns(2)

with col1:
    est_diameter_min = st.number_input(
        "Estimated Diameter Min (km)",
        min_value=0.0, max_value=100.0, value=0.3, step=0.01, format="%.4f"
    )
    relative_velocity = st.number_input(
        "Relative Velocity (km/h)",
        min_value=0.0, max_value=300000.0, value=50000.0, step=100.0
    )
    absolute_magnitude = st.number_input(
        "Absolute Magnitude (H)",
        min_value=0.0, max_value=40.0, value=20.0, step=0.1
    )

with col2:
    est_diameter_max = st.number_input(
        "Estimated Diameter Max (km)",
        min_value=0.0, max_value=100.0, value=0.7, step=0.01, format="%.4f"
    )
    miss_distance = st.number_input(
        "Miss Distance (km)",
        min_value=0.0, max_value=100_000_000.0, value=30_000_000.0, step=1000.0
    )

st.divider()

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("🚀 Predict Hazard Status", use_container_width=True):
    try:
        from predict import predict_single

        with st.spinner("Analysing object ..."):
            result = predict_single(
                est_diameter_min=est_diameter_min,
                est_diameter_max=est_diameter_max,
                relative_velocity=relative_velocity,
                miss_distance=miss_distance,
                absolute_magnitude=absolute_magnitude,
            )

        st.subheader("🎯 Prediction Result")

        if result["prediction"] == 1:
            st.error(f"### {result['label']}")
        else:
            st.success(f"### {result['label']}")

        st.metric(
            label="Probability of being Hazardous",
            value=f"{result['probability']*100:.2f}%"
        )

        # Probability bar
        st.progress(result["probability"])

    except FileNotFoundError as e:
        st.warning(
            f"⚠️ Model not found. Please run `python src/train.py` first.\n\n`{e}`"
        )
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Sprint 4 – Deployment & MLOps | NEO Hazard Prediction Project")
