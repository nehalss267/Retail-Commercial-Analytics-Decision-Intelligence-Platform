"""Churn Prediction Tool — Predict and explain customer churn."""
from langchain_core.tools import tool
import json

from src.config import settings


@tool
def predict_churn_risk(customer_id: int) -> dict:
    """Predict churn probability and risk level for a specific customer."""
    from src.models.churn.predict import predict_single
    return predict_single(customer_id)


@tool
def get_churn_model_summary() -> dict:
    """Get churn model performance summary and top feature importance."""
    path = settings.PROCESSED_DATA_DIR / "churn_report.json"
    if path.exists():
        report = json.loads(path.read_text())
        return {
            "best_model": report.get("best_model"),
            "results": report.get("results", {}),
            "top_features": dict(list(report.get("feature_importance", {}).items())[:5]),
        }
    return {"error": "Churn report not available"}


@tool
def get_churn_explanations() -> dict:
    """Get SHAP-based feature importance explanations for churn model."""
    path = settings.PROCESSED_DATA_DIR / "churn_explainability.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"error": "Churn explanations not available. Run explain.py first."}
