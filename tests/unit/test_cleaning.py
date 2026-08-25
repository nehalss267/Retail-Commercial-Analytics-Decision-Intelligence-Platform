"""Tests for data preprocessing and cleaning."""
import pandas as pd
import pytest
from pathlib import Path

from src.config import settings


@pytest.fixture
def raw_data():
    """Load raw data for testing."""
    path = settings.RAW_DATA_DIR / "Online Retail.xlsx"
    return pd.read_excel(path, engine="openpyxl")


@pytest.fixture
def cleaned_data():
    """Load cleaned data for testing."""
    path = settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pytest.skip("Cleaned data not available")


class TestRawData:
    def test_loads_correctly(self, raw_data):
        assert len(raw_data) > 0
        assert raw_data.shape[1] == 8

    def test_has_expected_columns(self, raw_data):
        expected = {"InvoiceNo", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
        assert expected == set(raw_data.columns)

    def test_quantity_is_numeric(self, raw_data):
        assert pd.api.types.is_numeric_dtype(raw_data["Quantity"])

    def test_unit_price_is_numeric(self, raw_data):
        assert pd.api.types.is_numeric_dtype(raw_data["UnitPrice"])


class TestCleanedData:
    def test_no_exact_duplicates(self, cleaned_data):
        assert cleaned_data.duplicated().sum() == 0

    def test_has_revenue_column(self, cleaned_data):
        assert "Revenue" in cleaned_data.columns

    def test_has_flags(self, cleaned_data):
        for col in ["IsCancellation", "IsReturn", "HasCustomerID"]:
            assert col in cleaned_data.columns

    def test_invoice_no_is_string(self, cleaned_data):
        assert cleaned_data["InvoiceNo"].dtype == object

    def test_no_null_invoice_dates(self, cleaned_data):
        assert cleaned_data["InvoiceDate"].isna().sum() == 0
