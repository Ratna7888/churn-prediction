"""Model training with MLflow logging and model registry."""

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    ),
}


def get_model(name: str):
    """Retrieve a model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]()


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights for models without class_weight param."""
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    scoring: str = "average_precision",
) -> tuple:
    """
    Train a model with cross-validation and MLflow logging.

    Logs hyperparameters, CV scores, and the fitted model artifact.
    Returns (fitted_model, cv_scores).
    """
    model = get_model(model_name)

    # Log hyperparameters
    params = model.get_params()
    for key, val in params.items():
        if val is not None and not callable(val):
            mlflow.log_param(key, val)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )

    logger.info(
        f"{model_name} | CV {scoring}: "
        f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
    )

    # Fit on full training set
    if model_name == "gradient_boosting":
        weights = compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    # Log to MLflow
    mlflow.log_param("model_name", model_name)
    mlflow.log_metric(f"cv_{scoring}_mean", cv_scores.mean())
    mlflow.log_metric(f"cv_{scoring}_std", cv_scores.std())
    mlflow.sklearn.log_model(model, artifact_path="model")

    return model, cv_scores