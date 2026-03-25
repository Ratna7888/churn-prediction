"""EDA script for Telco Churn dataset. Saves plots to outputs/figures/."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.loader import load_raw_data
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("eda")


def main():
    config = load_config()
    figures_dir = Path(config["evaluation"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(config["data"]["raw_path"])

    # 1. Churn distribution
    logger.info("--- Churn Distribution ---")
    counts = df["Churn"].value_counts()
    churn_pct = counts[1] / len(df) * 100
    logger.info(f"Retained: {counts[0]:,} | Churned: {counts[1]:,} ({churn_pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Retained (0)", "Churned (1)"], [counts[0], counts[1]], color=["#5DCAA5", "#E24B4A"])
    ax.set_ylabel("Count")
    ax.set_title("Churn Distribution")
    for i, v in enumerate([counts[0], counts[1]]):
        ax.text(i, v + 30, f"{v:,}", ha="center", fontsize=10)
    fig.savefig(figures_dir / "churn_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Churn by contract type — this is the #1 feature
    logger.info("--- Churn Rate by Contract ---")
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    contract_churn.plot(kind="bar", ax=ax, color=["#E24B4A", "#EF9F27", "#5DCAA5"])
    ax.set_ylabel("Churn Rate")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(contract_churn):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=10)
    fig.savefig(figures_dir / "churn_by_contract.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Tenure distribution by churn status
    logger.info("--- Tenure Distribution ---")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        [df[df["Churn"] == 0]["tenure"], df[df["Churn"] == 1]["tenure"]],
        bins=36, label=["Retained", "Churned"], stacked=False, alpha=0.7,
        color=["#5DCAA5", "#E24B4A"],
    )
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.set_title("Tenure Distribution by Churn Status")
    ax.legend()
    fig.savefig(figures_dir / "tenure_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Monthly charges by churn
    logger.info("--- Monthly Charges ---")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column="MonthlyCharges", by="Churn", ax=ax)
    ax.set_title("Monthly Charges by Churn Status")
    ax.set_xlabel("Churn (0=Retained, 1=Churned)")
    plt.suptitle("")  # remove auto-title
    fig.savefig(figures_dir / "monthly_charges_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Churn rate by internet service + tech support
    logger.info("--- Churn by Internet Service ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, col in enumerate(["InternetService", "TechSupport"]):
        rates = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=axes[idx], color="#7F77DD")
        axes[idx].set_title(f"Churn Rate by {col}")
        axes[idx].set_ylabel("Churn Rate")
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "churn_by_internet_techsupport.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. Correlation heatmap for numeric features
    logger.info("--- Numeric Correlations ---")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[numeric_cols].corr(), annot=True, cmap="RdYlBu_r",
        center=0, fmt=".2f", ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    fig.savefig(figures_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 7. Payment method churn rates
    logger.info("--- Churn by Payment Method ---")
    fig, ax = plt.subplots(figsize=(10, 5))
    pay_churn = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False)
    pay_churn.plot(kind="barh", ax=ax, color="#D85A30")
    ax.set_xlabel("Churn Rate")
    ax.set_title("Churn Rate by Payment Method")
    fig.savefig(figures_dir / "churn_by_payment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"All EDA figures saved to {figures_dir}")


if __name__ == "__main__":
    main()