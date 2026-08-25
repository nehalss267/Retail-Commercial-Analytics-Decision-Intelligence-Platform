"""API Routes — Forecasting endpoints."""
from fastapi import APIRouter
import pandas as pd

from src.api.dependencies import load_report
from src.config import settings

router = APIRouter(prefix="/forecast", tags=["forecasting"])


@router.get("/summary")
def forecast_summary():
    return load_report("forecast_report")


@router.get("/next-30-days")
def forecast_30days():
    path = settings.PROCESSED_DATA_DIR / "forecast_30d.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df.to_dict(orient="records")
    return []


@router.post("/revenue")
def forecast_revenue():
    """Run a fresh revenue forecast."""
    from src.models.forecasting.revenue_forecast import run_forecasting
    result = run_forecasting()
    return result
