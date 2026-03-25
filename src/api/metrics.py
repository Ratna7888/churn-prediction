"""
Prometheus metrics for the churn prediction API.

Exposes:
- Default HTTP metrics (request count, latency, in-progress) via instrumentator
- Custom business metrics (predictions served, churn rate, probability distribution)

These feed into the Grafana dashboard for real-time monitoring.
"""

from prometheus_client import Counter, Histogram, Gauge


# Business metrics — these tell the story beyond just "is the API up"
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions served",
    ["result"],  # labels: churn / retained
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of churn probabilities returned by the model",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time to process a single prediction (preprocessing + inference)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

MODEL_INFO = Gauge(
    "model_info",
    "Currently loaded model metadata",
    ["model_name", "threshold"],
)