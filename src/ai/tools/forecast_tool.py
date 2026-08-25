"""Forecast Tool — Revenue forecasting and model comparison."""
from langchain_core.tools import tool
import json

from src.config import settings


@tool
def get_forecast_summary() -> dict:
    """Get the 30-day revenue forecast summary with model performance."""
    path = settings.PROCESSED_DATA_DIR / "forecast_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"error": "Forecast report not available"}


@tool
def get_forecast_values() -> list[dict]:
    """Get the actual 30-day forecast values for all models."""
    import pandas as pd
    path = settings.PROCESSED_DATA_DIR / "forecast_30d.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df.to_dict(orient="records")
    return []


@tool
def compare_forecast_models() -> dict:
    """Compare forecasting model performance (MAE, RMSE, MAPE)."""
    path = settings.PROCESSED_DATA_DIR / "forecast_report.json"
    if path.exists():
        report = json.loads(path.read_text())
        return {
            "xgboost": report.get("xgboost", {}),
            "naive": report.get("naive", {}),
            "moving_average": report.get("moving_average", {}),
            "forecast_total": report.get("forecast_summary", {}).get("xgboost_total_forecast"),
        }
    return {"error": "Forecast report not available"}
