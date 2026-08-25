"""API Routes — Prediction endpoints (churn, CLV)."""
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

from src.api.dependencies import load_report
from src.config import settings

router = APIRouter(prefix="/predict", tags=["predictions"])


class ChurnRequest(BaseModel):
    customer_id: int


class CLVRequest(BaseModel):
    customer_id: int


@router.post("/churn")
def predict_churn(req: ChurnRequest):
    """Predict churn probability for a customer."""
    churn_report = load_report("churn_report")
    rfm_path = settings.FEATURES_DIR / "rfm_features.parquet"
    if not rfm_path.exists():
        return {"error": "RFM features not available"}

    rfm = pd.read_parquet(rfm_path)
    row = rfm[rfm["CustomerID"] == req.customer_id]
    if row.empty:
        return {"error": f"Customer {req.customer_id} not found"}

    # Use days_since_last as proxy for churn risk
    days = row["recency"].values[0]
    churn_prob = min(1.0, days / 180)  # Simple heuristic

    risk_level = "HIGH" if churn_prob > 0.6 else "MEDIUM" if churn_prob > 0.3 else "LOW"

    return {
        "customer_id": req.customer_id,
        "churn_probability": round(float(churn_prob), 4),
        "risk_level": risk_level,
        "model_version": churn_report.get("best_model", "xgboost"),
        "days_since_last_purchase": int(days),
    }


@router.post("/clv")
def predict_clv(req: CLVRequest):
    """Estimate CLV for a customer."""
    clv_path = settings.FEATURES_DIR / "clv_features.parquet"
    if not clv_path.exists():
        return {"error": "CLV features not available"}

    clv = pd.read_parquet(clv_path)
    row = clv[clv["CustomerID"] == req.customer_id]
    if row.empty:
        return {"error": f"Customer {req.customer_id} not found"}

    return {
        "customer_id": req.customer_id,
        "predicted_clv": round(float(row["predicted_clv"].values[0]), 2),
        "clv_segment": str(row["clv_segment"].values[0]),
    }
