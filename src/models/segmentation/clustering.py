"""Customer Segmentation — K-Means clustering on RFM features."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import settings


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 8)) -> tuple[int, dict]:
    """Find optimal K using silhouette score."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        scores[k] = round(float(sil), 4)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def cluster_customers(rfm: pd.DataFrame, n_clusters: int | None = None) -> dict:
    """K-Means clustering on RFM features."""
    features = ["recency", "frequency", "monetary"]
    X = rfm[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_clusters is None:
        n_clusters, silhouette_scores = find_optimal_k(X_scaled)
    else:
        silhouette_scores = {}

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = km.fit_predict(X_scaled)

    profiles = []
    for cluster in sorted(rfm["cluster"].unique()):
        data = rfm[rfm["cluster"] == cluster]
        profiles.append({
            "cluster": int(cluster),
            "n_customers": len(data),
            "pct_of_total": round(len(data) / len(rfm) * 100, 2),
            "avg_recency": round(float(data["recency"].mean()), 1),
            "avg_frequency": round(float(data["frequency"].mean()), 1),
            "avg_monetary": round(float(data["monetary"].mean()), 2),
        })

    return {
        "best_k": n_clusters,
        "silhouette_scores": silhouette_scores,
        "cluster_profiles": profiles,
    }
