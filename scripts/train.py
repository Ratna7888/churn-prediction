"""
Training pipeline — loads data, preprocesses, trains models, evaluates,
registers best model in MLflow Model Registry.
"""

from pathlib import Path
import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.data.loader import load_raw_data
from src.data.splitter import split_data
from src.features.preprocessing import build_pipeline, prepare_features
from src.models.trainer import train_model
from src.models.evaluator import find_best_threshold, evaluate_model
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train")


def main():
    config = load_config()

    # Setup MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load and split
    df = load_raw_data(config["data"]["raw_path"])
    train_df, test_df = split_data(
        df,
        target=config["features"]["target"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        save_dir=config["data"]["processed_dir"],
    )

    # Build and fit preprocessing pipeline
    pipeline = build_pipeline(
        numeric_features=config["features"]["numeric_features"],
        categorical_features=config["features"]["categorical_features"],
    )
    X_train, y_train = prepare_features(train_df, pipeline, target=config["features"]["target"], fit=True)
    X_test, y_test = prepare_features(test_df, pipeline, target=config["features"]["target"], fit=False)

    # Save fitted pipeline
    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / "preprocessing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Saved preprocessing pipeline to {pipeline_path}")

    # Train and evaluate each model
    best_model = None
    best_model_name = None
    best_pr_auc = -1.0
    best_threshold = 0.5
    best_run_id = None

    for model_name in config["training"]["models"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*50}")

        with mlflow.start_run(run_name=model_name) as run:
            model, cv_scores = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                cv_folds=config["training"]["cv_folds"],
                scoring=config["training"]["scoring"],
            )

            y_proba = model.predict_proba(X_test)[:, 1]

            threshold = find_best_threshold(
                y_test, y_proba,
                metric=config["evaluation"]["threshold_metric"],
            )

            metrics = evaluate_model(
                model_name=model_name,
                y_true=y_test,
                y_proba=y_proba,
                threshold=threshold,
                figures_dir=config["evaluation"]["figures_dir"],
                reports_dir=config["evaluation"]["reports_dir"],
            )

            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_model = model
                best_model_name = model_name
                best_threshold = threshold
                best_run_id = run.info.run_id

    # Save best model locally
    logger.info(f"\n{'='*50}")
    logger.info(f"Best model: {best_model_name} (PR-AUC: {best_pr_auc:.4f})")
    logger.info(f"{'='*50}")

    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    threshold_path = models_dir / "threshold.txt"
    threshold_path.write_text(str(best_threshold))

    logger.info(f"Saved best model to {best_model_path}")
    logger.info(f"Saved threshold ({best_threshold:.4f}) to {threshold_path}")

    # Register best model in MLflow Model Registry
    registered_name = config["mlflow"]["registered_model_name"]
    model_uri = f"runs:/{best_run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, registered_name)
        logger.info(
            f"Registered model '{registered_name}' version {mv.version} "
            f"from run {best_run_id}"
        )
    except Exception as e:
        logger.warning(f"Model registry failed (non-critical): {e}")
        logger.warning("MLflow Model Registry requires a database-backed store. Skipping.")


if __name__ == "__main__":
    main()