# Customer Churn Prediction — End-to-End ML System

A production-style machine learning system that predicts customer churn using the Telco Customer Churn dataset. Built to demonstrate core ML fundamentals, MLOps practices, and deployment readiness — not just model accuracy.

## What This Project Demonstrates

- **ML Pipeline**: Data validation → feature engineering → model training → threshold tuning → evaluation
- **MLOps**: Experiment tracking and model registry with MLflow
- **Serving**: Real-time prediction API with FastAPI
- **Monitoring**: Prometheus metrics + Grafana dashboard for production observability
- **Deployment**: Fully containerized with Docker Compose (API + MLflow + Prometheus + Grafana)

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│  Preprocessing│────▶│   Training   │
│  (CSV)       │     │  Pipeline    │     │  + MLflow    │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                     │
                    Serialized pipeline    Best model + threshold
                            │                     │
                            ▼                     ▼
                     ┌────────────────────────────────┐
                     │       FastAPI Service           │
                     │  /predict  /health  /metrics    │
                     └──────────────┬─────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                    ┌───────────┐      ┌───────────┐
                    │Prometheus │─────▶│  Grafana   │
                    └───────────┘      └───────────┘
```

## Key Technical Decisions

**Shared preprocessing pipeline**: The same scikit-learn `Pipeline` object is used in training and inference. It's serialized with `joblib` and loaded by the API at startup. This eliminates train/serve skew — the most common source of silent production bugs in ML systems.

**Threshold tuning over default 0.5**: The optimal classification threshold is found by maximizing F1 on the precision-recall curve, not by using the default 0.5. For churn prediction, this matters because the business cost of missing a churning customer (false negative) differs from the cost of a false alarm (false positive).

**PR-AUC as primary metric**: With ~26% churn rate, ROC-AUC can be misleadingly high. PR-AUC gives a more honest picture of model performance on the minority class.

**Class-weight balancing**: Used `class_weight='balanced'` instead of SMOTE to handle class imbalance. Simpler, no risk of data leakage, and equivalent performance on this dataset.

**Rule-based risk factors**: The API returns human-readable risk factors (e.g., "Month-to-month contract", "Short tenure") alongside predictions. In production, this would use SHAP values — for a portfolio project, rule-based is more interpretable and demonstrates the same thinking.

## Results

| Model | PR-AUC | ROC-AUC | F1 | Threshold |
|-------|--------|---------|-----|-----------|
| Logistic Regression | 0.637 | 0.843 | 0.631 | 0.569 |
| **Random Forest** | **0.654** | **0.844** | **0.640** | **0.445** |
| Gradient Boosting | 0.634 | 0.830 | 0.628 | 0.512 |


Random Forest was selected as the best model based on PR-AUC. The tuned threshold of 0.445 (vs default 0.5) improved recall from ~0.48 to ~0.55 while maintaining precision above 0.60.

## Project Structure

```
churn-prediction/
├── src/
│   ├── data/           # Load, validate, split
│   ├── features/       # Preprocessing pipeline
│   ├── models/         # Train, evaluate, threshold tuning
│   ├── api/            # FastAPI app, schemas, Prometheus metrics
│   └── utils/          # Config loader, logger
├── scripts/
│   ├── run_eda.py      # Exploratory data analysis
│   ├── train.py        # Full training pipeline
│   └── load_test.py    # Generate API traffic for monitoring
├── configs/
│   └── model_config.yaml
├── grafana/            # Dashboard + provisioning
├── prometheus.yml
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Quick Start

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# EDA
python -m scripts.run_eda

# Train (start MLflow server first)
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db
python -m scripts.train

# Serve
uvicorn src.api.app:app --reload
```

### Full Stack (Docker Compose)

```bash
docker-compose up --build
```

This starts:
- **Churn API**: http://localhost:8000/docs
- **MLflow**: http://localhost:5001
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

**Response:**
```json
{
  "will_churn": true,
  "churn_probability": 0.7234,
  "threshold": 0.4453,
  "risk_level": "critical",
  "top_risk_factors": [
    "Month-to-month contract (no lock-in)",
    "Short tenure (1 months)",
    "Electronic check payment (highest churn method)"
  ]
}
```

## Monitoring Dashboard

The Grafana dashboard auto-provisions on startup and tracks:
- Total predictions served and churn prediction rate
- P50/P95/P99 prediction latency
- Predictions over time (churn vs retained)
- Churn probability distribution
- HTTP request rate by status code
- API health status

## What I'd Add in Production

- **SHAP explanations** for per-prediction feature importance
- **Data drift detection** using Evidently or custom statistical tests
- **A/B testing** framework for model comparison in production
- **CI/CD pipeline** with automated retraining on new data
- **Feature store** to decouple feature engineering from training
- **Load balancing** with multiple API replicas behind nginx

## Tech Stack

Python, scikit-learn, FastAPI, MLflow, Prometheus, Grafana, Docker, Docker Compose