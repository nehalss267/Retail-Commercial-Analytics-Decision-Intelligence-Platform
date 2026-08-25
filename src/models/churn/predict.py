"""Churn Prediction — Standalone prediction interface."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import settings


def load_churn_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned data and RFM features."""
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    rfm = pd.read_parquet(settings.FEATURES_DIR / "rfm_features.parquet")
    return df, rfm


def prepare_features(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """Prepare customer-level features for churn prediction."""
    reference_date = df["InvoiceDate"].max()
    cutoff = reference_date - pd.Timedelta(days=90)

    customers = df[df["HasCustomerID"]].groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        total_orders=("InvoiceNo", "nunique"),
        total_revenue=("Revenue", "sum"),
        total_items=("Quantity", "sum"),
        avg_unit_price=("UnitPrice", "mean"),
        n_countries=("Country", "nunique"),
        first_purchase=("InvoiceDate", "min"),
    ).reset_index()

    customers["days_since_last"] = (reference_date - customers["last_purchase"]).dt.days
    customers["tenure_days"] = (customers["last_purchase"] - customers["first_purchase"]).dt.days
    customers["is_churned"] = (customers["last_purchase"] <= cutoff).astype(int)

    rfm_cols = ["CustomerID", "recency", "frequency", "monetary", "avg_order_value", "purchase_interval"]
    customers = customers.merge(rfm[rfm_cols], on="CustomerID", how="left")
    return customers


def get_feature_columns() -> list[str]:
    """Return the feature columns used for churn prediction."""
    return [
        "total_orders", "total_revenue", "total_items", "avg_unit_price",
        "n_countries", "tenure_days", "days_since_last",
        "recency", "frequency", "monetary", "avg_order_value", "purchase_interval",
    ]


def predict_single(customer_id: int, model=None) -> dict:
    """Predict churn for a single customer."""
    from src.models.churn.train import train_churn_models

    df, rfm = load_churn_data()
    data = prepare_features(df, rfm)

    row = data[data["CustomerID"] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found"}

    if model is None:
        results = train_churn_models(data)
        # Use XGBoost by default
        from xgboost import XGBClassifier
        feature_cols = get_feature_columns()
        X = data[feature_cols].fillna(0)
        y = data["is_churned"]
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              random_state=42, eval_metric="logloss")
        model.fit(X, y)

    feature_cols = get_feature_columns()
    X_pred = row[feature_cols].fillna(0)
    prob = model.predict_proba(X_pred)[:, 1][0]

    return {
        "customer_id": customer_id,
        "churn_probability": round(float(prob), 4),
        "risk_level": "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.3 else "LOW",
        "days_since_last_purchase": int(row["days_since_last"].values[0]),
    }
