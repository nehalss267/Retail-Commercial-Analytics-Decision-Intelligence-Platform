"""Tests for churn prediction."""
import json
import pytest
from pathlib import Path

from src.config import settings


@pytest.fixture
def churn_report():
    path = settings.PROCESSED_DATA_DIR / "churn_report.json"
    if path.exists():
        return json.loads(path.read_text())
    pytest.skip("Churn report not available")


class TestChurn:
    def test_has_results(self, churn_report):
        assert "results" in churn_report

    def test_has_multiple_models(self, churn_report):
        assert len(churn_report["results"]) >= 2

    def test_all_models_have_metrics(self, churn_report):
        for name, metrics in churn_report["results"].items():
            assert "roc_auc" in metrics
            assert "f1" in metrics
            assert 0 <= metrics["roc_auc"] <= 1

    def test_has_best_model(self, churn_report):
        assert "best_model" in churn_report

    def test_has_feature_importance(self, churn_report):
        assert "feature_importance" in churn_report
        assert len(churn_report["feature_importance"]) > 0
