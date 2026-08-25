"""API Routes — Customer analytics endpoints."""
from fastapi import APIRouter
import pandas as pd

from src.api.dependencies import load_report
from src.config import settings

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("/segments")
def get_segments():
    seg = load_report("segmentation_report")
    return seg.get("segment_summary", [])


@router.get("/{customer_id}")
def get_customer(customer_id: int):
    rfm_path = settings.FEATURES_DIR / "rfm_features.parquet"
    if not rfm_path.exists():
        return {"error": "RFM data not available"}
    rfm = pd.read_parquet(rfm_path)
    row = rfm[rfm["CustomerID"] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found"}
    return row.iloc[0].to_dict()
