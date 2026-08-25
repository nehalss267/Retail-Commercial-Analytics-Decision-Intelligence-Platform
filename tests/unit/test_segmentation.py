"""Tests for RFM segmentation."""
import pandas as pd
import pytest
from pathlib import Path

from src.config import settings


@pytest.fixture
def rfm_data():
    path = settings.FEATURES_DIR / "rfm_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pytest.skip("RFM features not available")


class TestRFM:
    def test_has_rfm_columns(self, rfm_data):
        for col in ["recency", "frequency", "monetary"]:
            assert col in rfm_data.columns

    def test_recency_positive(self, rfm_data):
        assert (rfm_data["recency"] >= 0).all()

    def test_frequency_positive(self, rfm_data):
        assert (rfm_data["frequency"] > 0).all()

    def test_monetary_positive(self, rfm_data):
        assert (rfm_data["monetary"] > 0).all()

    def test_has_segments(self, rfm_data):
        assert "segment" in rfm_data.columns
        assert rfm_data["segment"].nunique() >= 5

    def test_has_rfm_scores(self, rfm_data):
        for col in ["r_score", "f_score", "m_score"]:
            assert col in rfm_data.columns
            assert rfm_data[col].min() >= 1
            assert rfm_data[col].max() <= 5
