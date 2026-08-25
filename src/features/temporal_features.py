"""Temporal Feature Engineering — Time-based features."""
import pandas as pd
import numpy as np


def build_temporal_features(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Build time-series features at the specified frequency."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])].copy()

    # Aggregate to frequency
    temporal = valid.groupby(valid["InvoiceDate"].dt.to_period(freq)).agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
        units=("Quantity", "sum"),
    ).reset_index()
    temporal["InvoiceDate"] = temporal["InvoiceDate"].dt.to_timestamp()

    # Calendar features
    temporal["day_of_week"] = temporal["InvoiceDate"].dt.dayofweek
    temporal["month"] = temporal["InvoiceDate"].dt.month
    temporal["quarter"] = temporal["InvoiceDate"].dt.quarter
    temporal["year"] = temporal["InvoiceDate"].dt.year
    temporal["is_weekend"] = temporal["day_of_week"].isin([5, 6]).astype(int)

    # Lag features
    for lag in [1, 3, 7, 14, 28]:
        temporal[f"revenue_lag_{lag}"] = temporal["revenue"].shift(lag)
        temporal[f"orders_lag_{lag}"] = temporal["orders"].shift(lag)

    # Rolling features
    for window in [7, 14, 28]:
        temporal[f"revenue_roll_mean_{window}"] = temporal["revenue"].rolling(window).mean()
        temporal[f"revenue_roll_std_{window}"] = temporal["revenue"].rolling(window).std()
        temporal[f"orders_roll_mean_{window}"] = temporal["orders"].rolling(window).mean()

    # Growth
    temporal["revenue_mom"] = temporal["revenue"].pct_change()
    temporal["revenue_wow"] = temporal["revenue"].pct_change(7)

    # AOV
    temporal["aov"] = temporal["revenue"] / (temporal["orders"] + 1)

    return temporal
