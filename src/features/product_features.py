"""Product Feature Engineering — Standalone product-level features."""
import pandas as pd
import numpy as np


def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build comprehensive product-level features."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])].copy()

    products = valid.groupby(["StockCode", "Description"]).agg(
        orders=("InvoiceNo", "nunique"),
        units_sold=("Quantity", "sum"),
        revenue=("Revenue", "sum"),
        avg_price=("UnitPrice", "mean"),
        min_price=("UnitPrice", "min"),
        max_price=("UnitPrice", "max"),
        unique_customers=("CustomerID", "nunique"),
        first_sale=("InvoiceDate", "min"),
        last_sale=("InvoiceDate", "max"),
    ).reset_index()

    # Derived features
    products["price_range"] = products["max_price"] - products["min_price"]
    products["revenue_per_order"] = products["revenue"] / products["orders"]
    products["units_per_order"] = products["units_sold"] / products["orders"]
    products["customers_per_order"] = products["unique_customers"] / products["orders"]
    products["product_lifespan_days"] = (products["last_sale"] - products["first_sale"]).dt.days
    products["revenue_per_day"] = products["revenue"] / (products["product_lifespan_days"] + 1)

    # Log transforms
    products["revenue_log"] = np.log1p(products["revenue"])
    products["units_log"] = np.log1p(products["units_sold"])

    return products
