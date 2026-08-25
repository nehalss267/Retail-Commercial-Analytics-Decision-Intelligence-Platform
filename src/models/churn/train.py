"""Churn Prediction — Predict which customers will become inactive."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import json

from src.config import settings


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned data and RFM features."""
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    rfm = pd.read_parquet(settings.FEATURES_DIR / "rfm_features.parquet")
    return df, rfm


def define_churn(df: pd.DataFrame, rfm: pd.DataFrame, inactive_days: int = 90) -> pd.DataFrame:
    """Define churn label based on inactivity.

    Churned = last purchase was more than inactive_days ago
    (relative to the latest date in the dataset).
    """
    reference_date = df["InvoiceDate"].max()
    cutoff = reference_date - pd.Timedelta(days=inactive_days)

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

    # Merge RFM
    rfm_cols = ["CustomerID", "recency", "frequency", "monetary", "avg_order_value", "purchase_interval"]
    customers = customers.merge(rfm[rfm_cols], left_on="CustomerID", right_on="CustomerID", how="left")

    return customers


def train_churn_models(data: pd.DataFrame) -> dict:
    """Train and evaluate churn models."""
    feature_cols = [
        "total_orders", "total_revenue", "total_items", "avg_unit_price",
        "n_countries", "tenure_days", "days_since_last",
        "recency", "frequency", "monetary", "avg_order_value", "purchase_interval",
    ]

    X = data[feature_cols].fillna(0)
    y = data["is_churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss"),
    }

    results = {}
    for name, model in models.items():
        X_tr = X_train_scaled if name == "Logistic Regression" else X_train
        X_te = X_test_scaled if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        roc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(ap), 4),
            "precision": round(float(report["1"]["precision"]), 4),
            "recall": round(float(report["1"]["recall"]), 4),
            "f1": round(float(report["1"]["f1-score"]), 4),
            "accuracy": round(float(report["accuracy"]), 4),
        }

    # Best model feature importance (XGBoost)
    xgb_model = models["XGBoost"]
    importances = dict(zip(feature_cols, [round(float(x), 4) for x in xgb_model.feature_importances_]))

    return {"results": results, "feature_importance": importances, "best_model": "XGBoost"}


def run_churn() -> dict:
    """Run churn prediction pipeline."""
    print("Loading data...")
    df, rfm = load_data()

    print("Defining churn label...")
    data = define_churn(df, rfm)
    churn_rate = data["is_churned"].mean()
    print(f"  Churn rate: {churn_rate:.2%} ({data['is_churned'].sum()} / {len(data)})")

    print("Training models...")
    results = train_churn_models(data)

    print("\n=== CHURN MODEL RESULTS ===")
    for name, metrics in results["results"].items():
        print(f"\n{name}:")
        print(f"  ROC-AUC: {metrics['roc_auc']}")
        print(f"  PR-AUC: {metrics['pr_auc']}")
        print(f"  F1: {metrics['f1']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall: {metrics['recall']}")

    print(f"\nBest model: {results['best_model']}")
    print("\nFeature Importance (XGBoost):")
    for feat, imp in sorted(results["feature_importance"].items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp}")

    with open(settings.PROCESSED_DATA_DIR / "churn_report.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_churn()
