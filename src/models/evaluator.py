"""Model evaluation, threshold tuning, and reporting."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Find the classification threshold that maximizes the chosen metric.

    For churn, the default 0.5 is often reasonable since the class
    imbalance is moderate (~26%), but threshold tuning still matters
    for optimizing the precision-recall tradeoff.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    if metric == "f1":
        f1_scores = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0,
        )
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        logger.info(
            f"Best threshold: {best_threshold:.4f} | "
            f"F1: {f1_scores[best_idx]:.4f} | "
            f"Precision: {precisions[best_idx]:.4f} | "
            f"Recall: {recalls[best_idx]:.4f}"
        )
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return best_threshold


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    figures_dir: str = "outputs/figures",
    reports_dir: str = "outputs/reports",
) -> dict:
    """
    Full evaluation: metrics, confusion matrix, PR curve, reports.
    Logs everything to MLflow and saves artifacts to disk.
    """
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    y_pred = (y_proba >= threshold).astype(int)

    # Core metrics
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "threshold": threshold,
        "precision_churn": report["1"]["precision"],
        "recall_churn": report["1"]["recall"],
        "true_positives": int(cm[1, 1]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_negatives": int(cm[0, 0]),
    }

    # Log to MLflow
    for key, val in metrics.items():
        mlflow.log_metric(key, val)
    mlflow.log_param("threshold", threshold)

    # Save classification report
    report_text = classification_report(y_true, y_pred, target_names=["Retained", "Churned"])
    report_path = Path(reports_dir) / f"{model_name}_report.txt"
    report_path.write_text(report_text)
    mlflow.log_artifact(str(report_path))

    logger.info(f"{model_name} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")
    logger.info(f"\n{report_text}")

    # Plots
    _plot_precision_recall_curve(y_true, y_proba, model_name, threshold, figures_dir)
    _plot_confusion_matrix(cm, model_name, figures_dir)

    return metrics


def _plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    threshold: float,
    figures_dir: str,
):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, linewidth=2, label=f"AP = {ap:.4f}")

    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(
        recalls[idx], precisions[idx],
        color="red", s=100, zorder=5,
        label=f"Threshold = {threshold:.3f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = Path(figures_dir) / f"{model_name}_pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path))


def _plot_confusion_matrix(cm: np.ndarray, model_name: str, figures_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    labels = ["Retained", "Churned"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        ylabel="Actual", xlabel="Predicted",
        title=f"Confusion Matrix — {model_name}",
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )

    path = Path(figures_dir) / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path))