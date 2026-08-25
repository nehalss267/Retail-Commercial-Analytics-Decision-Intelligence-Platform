"""Executive Overview page."""
import streamlit as st
import pandas as pd
from pathlib import Path

from dashboard.components.cards import kpi_row, section_header
from dashboard.components.charts import revenue_trend_bar, country_bar

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def load():
    cleaned = pd.read_parquet(DATA_DIR / "cleaned_retail.parquet")
    eda = pd.read_json(DATA_DIR / "eda_report.json")
    return cleaned, eda


def render():
    st.title("Executive Overview")

    cleaned, eda = load()
    valid = cleaned[(cleaned["Revenue"] > 0) & (cleaned["HasCustomerID"])]

    # KPI cards
    total_rev = valid["Revenue"].sum()
    total_orders = valid["InvoiceNo"].nunique()
    total_customers = valid["CustomerID"].nunique()
    aov = valid.groupby("InvoiceNo")["Revenue"].sum().mean()

    kpi_row([
        {"label": "Total Revenue", "value": f"£{total_rev:,.0f}"},
        {"label": "Total Orders", "value": f"{total_orders:,}"},
        {"label": "Total Customers", "value": f"{total_customers:,}"},
        {"label": "Avg Order Value", "value": f"£{aov:,.0f}"},
    ])

    # Monthly revenue trend
    section_header("Monthly Revenue Trend")
    valid = valid.copy()
    valid["YearMonth"] = valid["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = valid.groupby("YearMonth").agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
    ).reset_index()
    st.plotly_chart(revenue_trend_bar(monthly), use_container_width=True)

    # Country breakdown
    section_header("Revenue by Country")
    country_rev = valid.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    st.plotly_chart(country_bar(country_rev), use_container_width=True)
