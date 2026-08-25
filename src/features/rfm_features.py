"""RFM Segmentation and Customer Clustering."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import json

from src.config import settings


def load_cleaned() -> pd.DataFrame:
    """Load cleaned data."""
    return pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")


def compute_rfm(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute RFM features per customer."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])].copy()

    if reference_date is None:
        reference_date = valid["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = valid.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
    ).reset_index()

    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    rfm["purchase_interval"] = valid.groupby("CustomerID")["InvoiceDate"].apply(
        lambda x: x.sort_values().diff().dt.days.mean() if len(x) > 1 else 0
    ).values

    return rfm


def rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign RFM quintile scores."""
    rfm = rfm.copy()
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_total"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    return rfm


def rfm_segment(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign business segment labels."""
    rfm = rfm.copy()
    rfm["segment"] = "Regular"

    rfm.loc[(rfm["r_score"] >= 4) & (rfm["f_score"] >= 4) & (rfm["m_score"] >= 4), "segment"] = "Champions"
    rfm.loc[(rfm["r_score"] >= 3) & (rfm["f_score"] >= 3) & (rfm["segment"] == "Regular"), "segment"] = "Loyal"
    rfm.loc[(rfm["r_score"] >= 4) & (rfm["f_score"] <= 2), "segment"] = "New Customers"
    rfm.loc[(rfm["r_score"] <= 2) & (rfm["f_score"] >= 3), "segment"] = "At Risk"
    rfm.loc[(rfm["r_score"] <= 2) & (rfm["f_score"] <= 2), "segment"] = "Lost"
    rfm.loc[(rfm["r_score"] == 5) & (rfm["f_score"] <= 2), "segment"] = "Promising"
    rfm.loc[(rfm["r_score"] <= 2) & (rfm["f_score"] >= 4), "segment"] = "Can't Lose"

    return rfm


def cluster_customers(rfm: pd.DataFrame, n_clusters: int = 5) -> dict:
    """K-Means clustering on RFM features."""
    features = ["recency", "frequency", "monetary"]
    X = rfm[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k using silhouette
    results = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results[k] = round(float(sil), 4)

    best_k = max(results, key=results.get)

    # Final clustering
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = km_final.fit_predict(X_scaled)

    # Cluster profiles
    profiles = rfm.groupby("cluster")[features].agg(["mean", "median", "count"]).round(2)
    profile_list = []
    for cluster in sorted(rfm["cluster"].unique()):
        cluster_data = rfm[rfm["cluster"] == cluster]
        profile_list.append({
            "cluster": int(cluster),
            "n_customers": len(cluster_data),
            "pct_of_total": round(len(cluster_data) / len(rfm) * 100, 2),
            "avg_recency": round(float(cluster_data["recency"].mean()), 1),
            "avg_frequency": round(float(cluster_data["frequency"].mean()), 1),
            "avg_monetary": round(float(cluster_data["monetary"].mean()), 2),
            "avg_order_value": round(float(cluster_data["avg_order_value"].mean()), 2),
        })

    return {
        "best_k": best_k,
        "silhouette_scores": results,
        "cluster_profiles": profile_list,
    }


def run_segmentation() -> dict:
    """Run complete RFM segmentation pipeline."""
    print("Loading data...")
    df = load_cleaned()

    print("Computing RFM...")
    rfm = compute_rfm(df)

    print("Scoring RFM...")
    rfm = rfm_scores(rfm)

    print("Assigning segments...")
    rfm = rfm_segment(rfm)

    print("Clustering...")
    cluster_results = cluster_customers(rfm)

    # Save RFM data
    rfm.to_parquet(settings.FEATURES_DIR / "rfm_features.parquet", index=False)
    print(f"Saved RFM features: {settings.FEATURES_DIR / 'rfm_features.parquet'}")

    # Segment summary
    segment_summary = rfm.groupby("segment").agg(
        n_customers=("CustomerID", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).round(2)

    print("\n=== SEGMENT SUMMARY ===")
    print(segment_summary)

    print(f"\n=== CLUSTERING ===")
    print(f"Best K: {cluster_results['best_k']}")
    for p in cluster_results["cluster_profiles"]:
        print(f"  Cluster {p['cluster']}: {p['n_customers']} customers, "
              f"avg monetary={p['avg_monetary']:,.0f}, freq={p['avg_frequency']:.1f}")

    report = {
        "segment_counts": rfm["segment"].value_counts().to_dict(),
        "segment_summary": segment_summary.reset_index().to_dict(orient="records"),
        "clustering": cluster_results,
    }

    out_path = settings.PROCESSED_DATA_DIR / "segmentation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {out_path}")

    return report


if __name__ == "__main__":
    run_segmentation()
