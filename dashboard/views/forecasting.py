"""Forecasting page."""
import streamlit as st
import pandas as pd
import json
from pathlib import Path

from dashboard.components.cards import kpi_row
from dashboard.components.charts import forecast_line

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def load_forecast():
    path = DATA_DIR / "forecast_30d.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


@st.cache_data
def load_report():
    path = DATA_DIR / "forecast_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def render():
    st.title("Revenue Forecasting")

    forecast = load_forecast()
    report = load_report()

    if forecast is not None:
        st.subheader("30-Day Revenue Forecast")
        st.plotly_chart(
            forecast_line(forecast, ["xgboost", "naive", "moving_avg_7d"]),
            use_container_width=True,
        )

        if report:
            st.subheader("Model Performance")
            xgb = report.get("xgboost", {})
            kpi_row([
                {"label": "MAE", "value": f"£{xgb.get('mae', 0):,.0f}"},
                {"label": "RMSE", "value": f"£{xgb.get('rmse', 0):,.0f}"},
                {"label": "MAPE", "value": f"{xgb.get('mape', 0):.1f}%"},
            ])

            summary = report.get("forecast_summary", {})
            st.subheader("Forecast Summary")
            kpi_row([
                {"label": "30-Day Total", "value": f"£{summary.get('xgboost_total_forecast', 0):,.0f}"},
                {"label": "Avg Daily", "value": f"£{summary.get('xgboost_avg_forecast', 0):,.0f}"},
            ])
    else:
        st.warning("Forecast data not available. Run the forecasting pipeline first.")
