"""Segmentation Evaluation — Cluster quality metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def evaluate_clustering(rfm: pd.DataFrame, cluster_col: str = "cluster") -> dict:
    """Comprehensive clustering evaluation."""
    features = ["recency", "frequency", "monetary"]
    X = rfm[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = rfm[cluster_col].values

    metrics = {
        "n_clusters": len(np.unique(labels)),
        "n_samples": len(rfm),
        "silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_scaled, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_scaled, labels)), 4),
    }

    # Per-cluster stats
    cluster_stats = rfm.groupby(cluster_col)[features].agg(["mean", "std", "count"]).round(2)
    metrics["cluster_stats"] = cluster_stats.to_dict()

    return metrics
