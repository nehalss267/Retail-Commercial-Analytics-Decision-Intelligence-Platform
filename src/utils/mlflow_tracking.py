"""MLflow experiment tracking helper."""
import mlflow
import json
from pathlib import Path

from src.config import settings


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("retailai")


def log_churn_experiment():
    """Log churn model results to MLflow."""
    setup_mlflow()

    report_path = settings.PROCESSED_DATA_DIR / "churn_report.json"
    if not report_path.exists():
        print("No churn report found")
        return

    report = json.loads(report_path.read_text())

    for model_name, metrics in report.get("results", {}).items():
        with mlflow.start_run(run_name=f"churn_{model_name.lower().replace(' ', '_')}"):
            mlflow.log_params({
                "model_type": model_name,
                "task": "churn_prediction",
            })
            mlflow.log_metrics(metrics)
            mlflow.log_dict(report.get("feature_importance", {}), "feature_importance.json")


def log_forecast_experiment():
    """Log forecasting results to MLflow."""
    setup_mlflow()

    report_path = settings.PROCESSED_DATA_DIR / "forecast_report.json"
    if not report_path.exists():
        print("No forecast report found")
        return

    report = json.loads(report_path.read_text())

    with mlflow.start_run(run_name="revenue_forecast_xgboost"):
        mlflow.log_params({
            "model_type": "XGBoost",
            "task": "revenue_forecasting",
            "horizon_days": report.get("horizon_days"),
        })
        xgb = report.get("xgboost", {})
        mlflow.log_metrics({
            "mae": xgb.get("mae"),
            "rmse": xgb.get("rmse"),
        })


def log_segmentation_experiment():
    """Log segmentation results to MLflow."""
    setup_mlflow()

    report_path = settings.PROCESSED_DATA_DIR / "segmentation_report.json"
    if not report_path.exists():
        return

    report = json.loads(report_path.read_text())
    clustering = report.get("clustering", {})

    with mlflow.start_run(run_name="rfm_clustering"):
        mlflow.log_params({
            "algorithm": "KMeans",
            "task": "customer_segmentation",
            "best_k": clustering.get("best_k"),
        })
        mlflow.log_metrics({
            "silhouette_score": clustering.get("silhouette_scores", {}).get(str(clustering.get("best_k")), 0),
        })


if __name__ == "__main__":
    print("Logging experiments to MLflow...")
    log_churn_experiment()
    log_forecast_experiment()
    log_segmentation_experiment()
    print("Done!")
