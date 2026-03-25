"""Load and validate the Telco Customer Churn dataset."""

from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET = "Churn"


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load CSV, clean known data issues, and validate.

    Known issues in the Telco dataset:
    - TotalCharges has whitespace strings for new customers (tenure=0)
    - Churn is 'Yes'/'No' strings, needs binary encoding
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)

    # TotalCharges has empty strings — coerce to numeric, fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    null_count = df["TotalCharges"].isnull().sum()
    if null_count > 0:
        logger.info(f"Filling {null_count} null TotalCharges with 0.0 (new customers)")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target: Yes=1, No=0
    assert TARGET in df.columns, f"Target column '{TARGET}' not found"
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    assert df[TARGET].isin([0, 1]).all(), "Target mapping failed"

    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    logger.info(f"Churn rate: {df[TARGET].mean():.3f} ({df[TARGET].sum()} churned)")
    return df