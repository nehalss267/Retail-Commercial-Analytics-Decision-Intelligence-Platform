"""Customer Lifetime Value estimation."""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import json

from src.config import settings


def load_features() -> pd.DataFrame:
    """Load RFM features."""
    return pd.read_parquet(settings.FEATURES_DIR / "rfm_features.parquet")


def build_clv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build CLV model features."""
    features = df[[
        "CustomerID", "recency", "frequency", "monetary",
        "avg_order_value", "purchase_interval",
    ]].copy()

    # Additional derived features
    features["monetary_log"] = np.log1p(features["monetary"])
    features["frequency_log"] = np.log1p(features["frequency"])
    features["reciprocal_recency"] = 1 / (features["recency"] + 1)

    return features


def estimate_clv(features: pd.DataFrame) -> pd.DataFrame:
    """Estimate CLV using GBM."""
    feature_cols = [
        "recency", "frequency", "monetary", "avg_order_value",
        "purchase_interval", "monetary_log", "frequency_log", "reciprocal_recency",
    ]

    X = features[feature_cols].fillna(0)
    y = features["monetary"]

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X, y)

    features = features.copy()
    features["predicted_clv"] = model.predict(X)

    # CV score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    # CLV segments
    features["clv_segment"] = pd.qcut(
        features["predicted_clv"], q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    return features, {
        "model_r2": round(float(model.score(X, y)), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "feature_importance": dict(zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_])),
    }


def run_clv() -> dict:
    """Run CLV pipeline."""
    print("Loading features...")
    rfm = load_features()

    print("Building CLV features...")
    features = build_clv_features(rfm)

    print("Estimating CLV...")
    clv, model_info = estimate_clv(features)

    # Save
    clv.to_parquet(settings.FEATURES_DIR / "clv_features.parquet", index=False)

    summary = clv.groupby("clv_segment").agg(
        n_customers=("CustomerID", "count"),
        avg_predicted_clv=("predicted_clv", "mean"),
        total_predicted_clv=("predicted_clv", "sum"),
    ).round(2)

    print("\n=== CLV SEGMENTS ===")
    print(summary)
    print(f"\nModel R²: {model_info['model_r2']}")
    print(f"CV R²: {model_info['cv_r2_mean']} ± {model_info['cv_r2_std']}")

    report = {"model_info": model_info, "segment_summary": summary.reset_index().to_dict(orient="records")}
    with open(settings.PROCESSED_DATA_DIR / "clv_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


if __name__ == "__main__":
    run_clv()
