"""Tests for feature engineering modules."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import settings


@pytest.fixture
def cleaned_data():
    path = settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pytest.skip("Cleaned data not available")


@pytest.fixture
def rfm_features():
    path = settings.FEATURES_DIR / "rfm_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pytest.skip("RFM features not available")


class TestCustomerFeatures:
    def test_builds_correctly(self, cleaned_data):
        from src.features.customer_features import build_customer_features
        features = build_customer_features(cleaned_data)
        assert len(features) > 0
        assert "CustomerID" in features.columns

    def test_has_required_columns(self, cleaned_data):
        from src.features.customer_features import build_customer_features
        features = build_customer_features(cleaned_data)
        required = ["total_orders", "total_revenue", "avg_order_value", "days_since_last"]
        for col in required:
            assert col in features.columns

    def test_no_null_customer_ids(self, cleaned_data):
        from src.features.customer_features import build_customer_features
        features = build_customer_features(cleaned_data)
        assert features["CustomerID"].isna().sum() == 0


class TestProductFeatures:
    def test_builds_correctly(self, cleaned_data):
        from src.features.product_features import build_product_features
        features = build_product_features(cleaned_data)
        assert len(features) > 0
        assert "StockCode" in features.columns

    def test_has_required_columns(self, cleaned_data):
        from src.features.product_features import build_product_features
        features = build_product_features(cleaned_data)
        required = ["orders", "units_sold", "revenue", "avg_price"]
        for col in required:
            assert col in features.columns


class TestTemporalFeatures:
    def test_builds_correctly(self, cleaned_data):
        from src.features.temporal_features import build_temporal_features
        features = build_temporal_features(cleaned_data)
        assert len(features) > 0

    def test_has_lag_features(self, cleaned_data):
        from src.features.temporal_features import build_temporal_features
        features = build_temporal_features(cleaned_data)
        lag_cols = [c for c in features.columns if "lag" in c]
        assert len(lag_cols) > 0

    def test_has_rolling_features(self, cleaned_data):
        from src.features.temporal_features import build_temporal_features
        features = build_temporal_features(cleaned_data)
        roll_cols = [c for c in features.columns if "roll" in c]
        assert len(roll_cols) > 0
