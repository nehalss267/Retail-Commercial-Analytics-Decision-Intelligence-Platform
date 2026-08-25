"""Popularity-Based Recommendation — Baseline model."""
import pandas as pd


def popularity_baseline(df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    """Most popular products by revenue."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]
    products = valid.groupby(["StockCode", "Description"]).agg(
        revenue=("Revenue", "sum"),
        quantity=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
    ).reset_index().sort_values("revenue", ascending=False)

    return products.head(top_n)[
        ["StockCode", "Description", "revenue", "quantity", "orders", "customers"]
    ].to_dict(orient="records")
