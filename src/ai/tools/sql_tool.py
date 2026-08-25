"""SQL Analytics Tool — Query retail data."""
from langchain_core.tools import tool
import json

from src.config import settings


def _load_report(name: str) -> dict:
    path = settings.PROCESSED_DATA_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@tool
def query_revenue_metrics() -> dict:
    """Get total revenue, orders, customers, and AOV for the business."""
    eda = _load_report("eda_report")
    summary = eda.get("revenue_trends", {}).get("summary", {})
    return {
        "total_revenue": summary.get("total_revenue"),
        "total_orders": summary.get("total_orders"),
        "total_customers": summary.get("total_customers"),
        "best_month": summary.get("best_month"),
    }


@tool
def query_customer_segments() -> dict:
    """Get customer segment distribution and statistics."""
    seg = _load_report("segmentation_report")
    return {
        "segment_counts": seg.get("segment_counts", {}),
        "segment_summary": seg.get("segment_summary", []),
    }


@tool
def query_top_products(n: int = 10) -> list[dict]:
    """Get top N products by revenue."""
    eda = _load_report("eda_report")
    products = eda.get("products", {}).get("top_10_by_revenue", [])
    return products[:n]


@tool
def query_country_revenue() -> dict:
    """Get revenue breakdown by country."""
    eda = _load_report("eda_report")
    countries = eda.get("countries", {})
    return {
        "uk_dominance_pct": countries.get("uk_dominance_pct"),
        "top_countries": countries.get("top_10", []),
    }


@tool
def query_customer_detail(customer_id: int) -> dict:
    """Get RFM details for a specific customer."""
    import pandas as pd
    rfm_path = settings.FEATURES_DIR / "rfm_features.parquet"
    if not rfm_path.exists():
        return {"error": "RFM data not available"}
    rfm = pd.read_parquet(rfm_path)
    row = rfm[rfm["CustomerID"] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found"}
    return row.iloc[0].to_dict()


@tool
def query_daily_metrics() -> dict:
    """Get daily revenue, orders, and customer metrics."""
    import pandas as pd
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]
    daily = valid.groupby(valid["InvoiceDate"].dt.date).agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
    ).agg({
        "revenue": ["mean", "sum", "min", "max"],
        "orders": ["mean", "sum"],
        "customers": ["mean", "sum"],
    }).round(2)
    return {"daily_summary": str(daily)}
