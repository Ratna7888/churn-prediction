"""
Sklearn preprocessing pipeline for Telco Churn dataset.

Key design decision: everything is inside a serializable Pipeline/ColumnTransformer
so training and inference share identical preprocessing. No train/serve skew.

Feature engineering rationale:
- tenure_group: buckets raw tenure into business-meaningful segments
- avg_monthly_charge: TotalCharges / tenure approximates spending consistency
- These are created BEFORE the ColumnTransformer so they flow through naturally
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features before column transformation.

    - tenure_group: categorical bucket (0-12, 13-24, 25-48, 49-60, 61+)
    - avg_monthly_charge: spending rate (TotalCharges / max(tenure, 1))
    """
    df = df.copy()

    # Drop customerID — it's a unique identifier, not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Tenure buckets — maps to typical contract renewal cycles
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ["0-12m", "13-24m", "25-48m", "49-60m", "61+m"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)

    # Average monthly spending rate
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    return df


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build a full preprocessing pipeline.

    Steps:
    1. Engineer new features (tenure_group, avg_monthly_charge)
    2. Scale numeric columns (tenure, MonthlyCharges, TotalCharges, avg_monthly_charge)
    3. One-hot encode categorical columns

    OneHotEncoder uses handle_unknown='ignore' so unseen categories
    at inference time produce all-zeros instead of crashing.
    """
    # After engineering, we have extra columns
    all_numeric = numeric_features + ["avg_monthly_charge"]
    all_categorical = categorical_features + ["tenure_group"]

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                all_categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("feature_engineering", FunctionTransformer(_engineer_features, validate=False)),
        ("column_transform", column_transformer),
    ])

    logger.info(
        f"Built pipeline — {len(all_numeric)} numeric, "
        f"{len(all_categorical)} categorical features"
    )
    return pipeline


def prepare_features(
    df: pd.DataFrame,
    pipeline: Pipeline,
    target: str = "Churn",
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Apply pipeline to a dataframe. Returns (X, y).

    Args:
        df: Raw dataframe including target column.
        pipeline: The preprocessing pipeline.
        target: Name of target column.
        fit: If True, fit_transform. If False, transform only.
    """
    y = df[target].values if target in df.columns else None
    features_df = df.drop(columns=[target], errors="ignore")

    if fit:
        X = pipeline.fit_transform(features_df)
        logger.info(f"Fit and transformed — shape: {X.shape}")
    else:
        X = pipeline.transform(features_df)
        logger.info(f"Transformed — shape: {X.shape}")

    return X, y