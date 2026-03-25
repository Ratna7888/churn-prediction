"""FastAPI prediction service for customer churn with Prometheus metrics."""

import time
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import CustomerRequest, PredictionResponse, HealthResponse
from src.api.metrics import (
    PREDICTIONS_TOTAL,
    PREDICTION_PROBABILITY,
    PREDICTION_LATENCY,
    MODEL_INFO,
)
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

# Global state
model = None
pipeline = None
threshold = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and pipeline on startup, expose model info metric."""
    global model, pipeline, threshold

    config = load_config()
    model_path = Path(config["api"]["model_path"])
    pipeline_path = Path(config["api"]["pipeline_path"])

    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Run training first.")
    if not pipeline_path.exists():
        raise RuntimeError(f"Pipeline not found at {pipeline_path}. Run training first.")

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    threshold_path = model_path.parent / "threshold.txt"
    threshold = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5

    # Expose model metadata as a Prometheus gauge
    model_name = type(model).__name__
    MODEL_INFO.labels(model_name=model_name, threshold=str(round(threshold, 4))).set(1)

    logger.info(f"Loaded model: {model_name} from {model_path}")
    logger.info(f"Using threshold: {threshold}")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Churn Prediction API",
    description="Real-time customer churn prediction with monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus HTTP metrics (request count, latency histogram, in-progress gauge)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _classify_risk(probability: float) -> str:
    if probability < 0.3:
        return "low"
    elif probability < 0.5:
        return "medium"
    elif probability < 0.7:
        return "high"
    return "critical"


def _get_risk_factors(row: dict) -> list[str]:
    """
    Simple rule-based risk factor extraction.

    In production you'd use SHAP or feature importances.
    For a portfolio project, rule-based is fine and more interpretable.
    """
    factors = []
    if row.get("Contract") == "Month-to-month":
        factors.append("Month-to-month contract (no lock-in)")
    if row.get("tenure", 99) < 12:
        factors.append(f"Short tenure ({row.get('tenure')} months)")
    if row.get("InternetService") == "Fiber optic":
        factors.append("Fiber optic (higher churn segment)")
    if row.get("MonthlyCharges", 0) > 70:
        factors.append(f"High monthly charges (${row.get('MonthlyCharges'):.2f})")
    if row.get("TechSupport") == "No" and row.get("InternetService") != "No":
        factors.append("No tech support")
    if row.get("PaymentMethod") == "Electronic check":
        factors.append("Electronic check payment (highest churn method)")
    return factors[:3]  # Top 3


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerRequest):
    """
    Predict churn probability for a single customer.

    Accepts raw features and applies the same preprocessing
    pipeline used during training.
    """
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.perf_counter()

        input_dict = customer.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Preprocess (same pipeline as training)
        X = pipeline.transform(input_df)

        # Predict
        proba = model.predict_proba(X)[0, 1]
        will_churn = bool(proba >= threshold)

        # Record metrics
        elapsed = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(elapsed)
        PREDICTION_PROBABILITY.observe(proba)
        PREDICTIONS_TOTAL.labels(result="churn" if will_churn else "retained").inc()

        return PredictionResponse(
            will_churn=will_churn,
            churn_probability=round(float(proba), 6),
            threshold=threshold,
            risk_level=_classify_risk(proba),
            top_risk_factors=_get_risk_factors(input_dict),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=422, detail=str(e))