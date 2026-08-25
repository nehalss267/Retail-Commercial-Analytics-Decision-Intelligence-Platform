"""Customer Feature Engineering — Standalone customer-level features."""
import pandas as pd
import numpy as np


def build_customer_features(df: pd.DataFrame,
                            reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build comprehensive customer-level features."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])].copy()

    if reference_date is None:
        reference_date = valid["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Base aggregation
    customers = valid.groupby("CustomerID").agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        total_orders=("InvoiceNo", "nunique"),
        total_revenue=("Revenue", "sum"),
        total_items=("Quantity", "sum"),
        avg_unit_price=("UnitPrice", "mean"),
        n_countries=("Country", "nunique"),
        n_products=("StockCode", "nunique"),
    ).reset_index()

    # Temporal features
    customers["days_since_first"] = (reference_date - customers["first_purchase"]).dt.days
    customers["days_since_last"] = (reference_date - customers["last_purchase"]).dt.days
    customers["tenure_days"] = (customers["last_purchase"] - customers["first_purchase"]).dt.days
    customers["purchase_frequency"] = customers["total_orders"] / (customers["tenure_days"] + 1)

    # Value features
    customers["avg_order_value"] = customers["total_revenue"] / customers["total_orders"]
    customers["revenue_per_day"] = customers["total_revenue"] / (customers["tenure_days"] + 1)
    customers["items_per_order"] = customers["total_items"] / customers["total_orders"]
    customers["avg_item_price"] = customers["total_revenue"] / (customers["total_items"] + 1)

    # Product diversity
    customers["product_diversity"] = customers["n_products"] / customers["total_orders"]

    # Log transforms
    customers["revenue_log"] = np.log1p(customers["total_revenue"])
    customers["orders_log"] = np.log1p(customers["total_orders"])

    return customers
