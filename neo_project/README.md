# ☄️ NEO Hazard Prediction – Sprint 4: Deployment & MLOps

## Problem Statement
Predict whether a Near-Earth Object (NEO) is potentially hazardous to Earth,
using physical and orbital characteristics from the NASA NEO dataset.

## Approach
- **Algorithm:** Random Forest Classifier
- **Preprocessing:** StandardScaler, median imputation
- **Tracking:** MLflow for experiment logging
- **Frontend:** Streamlit interactive web app
- **Logging:** Python logging module (predictions.log)

## Project Structure
```
project/
├── data/
│   └── neo_cleaned.csv
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```

### 3. Launch the Streamlit app
```bash
streamlit run app/app.py
```

### 4. (Optional) View MLflow dashboard
```bash
mlflow ui
# Open http://localhost:5000
```

## Results
After training, accuracy and ROC-AUC metrics are saved in `models/metrics.json`
and also displayed in the app sidebar.

## MLOps Practices
| Practice | Tool |
|---|---|
| Version Control | GitHub |
| Experiment Tracking | MLflow |
| Logging & Monitoring | Python `logging` → predictions.log |
| Model Serialisation | joblib |
