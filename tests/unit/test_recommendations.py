"""Tests for recommendation system modules."""
import pytest
import pandas as pd
from pathlib import Path

from src.config import settings


@pytest.fixture
def cleaned_data():
    path = settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pytest.skip("Cleaned data not available")


class TestPopularity:
    def test_popularity_returns_results(self, cleaned_data):
        from src.models.recommendation.popularity import popularity_baseline
        results = popularity_baseline(cleaned_data, top_n=10)
        assert len(results) == 10
        assert "StockCode" in results[0]
        assert "revenue" in results[0]

    def test_popularity_sorted_by_revenue(self, cleaned_data):
        from src.models.recommendation.popularity import popularity_baseline
        results = popularity_baseline(cleaned_data, top_n=20)
        revenues = [r["revenue"] for r in results]
        assert revenues == sorted(revenues, reverse=True)


class TestContentBased:
    def test_content_based_returns_results(self, cleaned_data):
        from src.models.recommendation.popularity import popularity_baseline
        from src.models.recommendation.content_based import content_based
        pop = popularity_baseline(cleaned_data, top_n=1)
        top_product = pop[0]["StockCode"]
        results = content_based(cleaned_data, top_product, top_n=5)
        assert len(results) > 0
        assert "similarity_score" in results[0]


class TestEvaluation:
    def test_precision_at_k(self):
        from src.models.recommendation.evaluation import precision_at_k
        recommended = ["a", "b", "c", "d", "e"]
        purchased = {"a", "c", "e"}
        assert precision_at_k(recommended, purchased, k=5) == 3 / 5

    def test_recall_at_k(self):
        from src.models.recommendation.evaluation import recall_at_k
        recommended = ["a", "b", "c"]
        purchased = {"a", "b", "c", "d"}
        assert recall_at_k(recommended, purchased, k=3) == 3 / 4

    def test_ndcg_perfect(self):
        from src.models.recommendation.evaluation import ndcg_at_k
        recommended = ["a", "b", "c"]
        purchased = {"a", "b", "c"}
        assert ndcg_at_k(recommended, purchased, k=3) == 1.0
