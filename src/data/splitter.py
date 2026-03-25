"""Train/test split with stratification."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split.

    Churn is ~26.5% — less extreme than fraud but still imbalanced
    enough that stratification matters for stable evaluation.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )

    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    logger.info(
        f"Train churn rate: {train_df[target].mean():.4f}, "
        f"Test churn rate: {test_df[target].mean():.4f}"
    )

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(save_path / "train.csv", index=False)
        test_df.to_csv(save_path / "test.csv", index=False)
        logger.info(f"Saved splits to {save_path}")

    return train_df, test_df