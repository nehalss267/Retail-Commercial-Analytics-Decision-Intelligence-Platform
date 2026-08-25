"""Recommendation Evaluation — Ranking and quality metrics."""
import numpy as np
import pandas as pd


def precision_at_k(recommended: list[str], purchased: set[str], k: int = 10) -> float:
    """Precision@K: fraction of recommended items that are relevant."""
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return len(set(rec_k) & purchased) / len(rec_k)


def recall_at_k(recommended: list[str], purchased: set[str], k: int = 10) -> float:
    """Recall@K: fraction of relevant items that are recommended."""
    rec_k = recommended[:k]
    if not purchased:
        return 0.0
    return len(set(rec_k) & purchased) / len(purchased)


def ndcg_at_k(recommended: list[str], purchased: set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    rec_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in purchased)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(purchased), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mean_average_precision(predictions: list[list[str]], purchases: list[set[str]], k: int = 10) -> float:
    """Mean Average Precision across users."""
    aps = []
    for recs, purchased in zip(predictions, purchases):
        rec_k = recs[:k]
        hits = 0
        sum_precision = 0.0
        for i, item in enumerate(rec_k):
            if item in purchased:
                hits += 1
                sum_precision += hits / (i + 1)
        ap = sum_precision / min(len(purchased), k) if purchased else 0.0
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def catalog_coverage(all_recommendations: list[list[str]], total_items: int) -> float:
    """Fraction of catalog items recommended at least once."""
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    return len(unique_items) / total_items if total_items > 0 else 0.0
