"""Feature Pipeline — Orchestrated feature engineering."""
import json

from src.config import settings
from src.features.customer_features import build_customer_features
from src.features.product_features import build_product_features
from src.features.temporal_features import build_temporal_features
from src.features.rfm_features import compute_rfm, rfm_scores, rfm_segment


def run_feature_pipeline() -> dict:
    """Run complete feature engineering pipeline."""
    import pandas as pd

    print("Loading cleaned data...")
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")

    # Customer features
    print("Building customer features...")
    customer_features = build_customer_features(df)
    customer_features.to_parquet(settings.FEATURES_DIR / "customer_features.parquet", index=False)
    print(f"  Saved: {len(customer_features)} customers")

    # Product features
    print("Building product features...")
    product_features = build_product_features(df)
    product_features.to_parquet(settings.FEATURES_DIR / "product_features.parquet", index=False)
    print(f"  Saved: {len(product_features)} products")

    # Temporal features
    print("Building temporal features...")
    temporal_features = build_temporal_features(df)
    temporal_features.to_parquet(settings.FEATURES_DIR / "temporal_features.parquet", index=False)
    print(f"  Saved: {len(temporal_features)} time periods")

    # RFM features
    print("Building RFM features...")
    rfm = compute_rfm(df)
    rfm = rfm_scores(rfm)
    rfm = rfm_segment(rfm)
    rfm.to_parquet(settings.FEATURES_DIR / "rfm_features.parquet", index=False)
    print(f"  Saved: {len(rfm)} customers")

    result = {
        "customer_features": len(customer_features),
        "product_features": len(product_features),
        "temporal_features": len(temporal_features),
        "rfm_features": len(rfm),
    }

    with open(settings.PROCESSED_DATA_DIR / "feature_pipeline_report.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== FEATURE PIPELINE COMPLETE ===")
    print(f"  Customer features: {result['customer_features']}")
    print(f"  Product features: {result['product_features']}")
    print(f"  Temporal features: {result['temporal_features']}")
    print(f"  RFM features: {result['rfm_features']}")

    return result


if __name__ == "__main__":
    run_feature_pipeline()
