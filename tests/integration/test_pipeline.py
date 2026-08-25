"""Integration tests for end-to-end pipeline."""
import json
import pytest
from pathlib import Path
import pandas as pd

from src.config import settings


class TestDataPipeline:
    """Test that data flows correctly through the pipeline."""

    def test_raw_data_exists(self):
        path = settings.RAW_DATA_DIR / "Online Retail.xlsx"
        assert path.exists(), f"Raw data not found at {path}"

    def test_cleaned_data_exists(self):
        path = settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet"
        assert path.exists(), f"Cleaned data not found at {path}"

    def test_cleaned_data_quality(self):
        path = settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet"
        df = pd.read_parquet(path)
        assert len(df) > 0, "Cleaned data is empty"
        assert "Revenue" in df.columns, "Revenue column missing"
        assert "HasCustomerID" in df.columns, "HasCustomerID column missing"
        assert df["Revenue"].sum() > 0, "Total revenue is zero"

    def test_features_exist(self):
        feature_files = ["rfm_features.parquet", "clv_features.parquet"]
        for fname in feature_files:
            path = settings.FEATURES_DIR / fname
            assert path.exists(), f"Feature file missing: {fname}"
            df = pd.read_parquet(path)
            assert len(df) > 0, f"Feature file is empty: {fname}"


class TestModelPipeline:
    """Test that model outputs are consistent."""

    def test_churn_report_structure(self):
        path = settings.PROCESSED_DATA_DIR / "churn_report.json"
        if not path.exists():
            pytest.skip("Churn report not generated")
        report = json.loads(path.read_text())
        assert "best_model" in report
        assert "results" in report
        assert report["best_model"] in report["results"]

    def test_forecast_output(self):
        path = settings.PROCESSED_DATA_DIR / "forecast_30d.csv"
        if not path.exists():
            pytest.skip("Forecast not generated")
        df = pd.read_csv(path, parse_dates=["date"])
        assert len(df) == 30, f"Expected 30 forecast days, got {len(df)}"
        assert "xgboost" in df.columns
        assert (df["xgboost"] >= 0).all(), "Negative forecast values"

    def test_recommendations_report(self):
        path = settings.PROCESSED_DATA_DIR / "recommendations_report.json"
        if not path.exists():
            pytest.skip("Recommendations not generated")
        report = json.loads(path.read_text())
        assert "popularity_top_20" in report
        assert "content_based_for_product" in report

    def test_optimization_report(self):
        path = settings.PROCESSED_DATA_DIR / "optimization_report.json"
        if not path.exists():
            pytest.skip("Optimization not generated")
        report = json.loads(path.read_text())
        assert "budget" in report
        assert report["budget"] == 100

    def test_segmentation_report(self):
        path = settings.PROCESSED_DATA_DIR / "segmentation_report.json"
        if not path.exists():
            pytest.skip("Segmentation not generated")
        report = json.loads(path.read_text())
        assert "clustering" in report
        assert "best_k" in report["clustering"]


class TestAgentPipeline:
    """Test the GenAI agent end-to-end."""

    def test_agent_importable(self):
        from src.ai.agent import process_query, get_available_tools
        assert callable(process_query)
        assert callable(get_available_tools)

    def test_agent_has_tools(self):
        from src.ai.agent import get_available_tools
        tools = get_available_tools()
        assert len(tools) > 0, "No tools registered"
        for tool in tools:
            assert "name" in tool
            assert "description" in tool

    def test_agent_revenue_query(self):
        from src.ai.agent import process_query
        result = process_query("What is the total revenue?")
        assert "answer" in result
        assert "route" in result
        assert result["route"] == "sql"
        assert len(result["answer"]) > 0

    def test_agent_churn_query(self):
        from src.ai.agent import process_query
        result = process_query("What is the churn rate?")
        assert "answer" in result
        assert "route" in result
        assert result["route"] == "churn"

    def test_agent_forecast_query(self):
        from src.ai.agent import process_query
        result = process_query("What does the forecast look like?")
        assert "answer" in result
        assert "route" in result
        assert result["route"] == "forecast"

    def test_agent_recommendation_query(self):
        from src.ai.agent import process_query
        result = process_query("What products should I recommend?")
        assert "answer" in result
        assert "route" in result
        assert result["route"] == "recommendation"

    def test_agent_optimization_query(self):
        from src.ai.agent import process_query
        result = process_query("Which customers should we target?")
        assert "answer" in result
        assert "route" in result
        assert result["route"] == "optimization"


class TestRAGPipeline:
    """Test the RAG knowledge base."""

    def test_knowledge_base_exists(self):
        kb_path = settings.BASE_DIR / "data" / "processed" / "chroma_db"
        if not kb_path.exists():
            pytest.skip("Knowledge base not initialized")

    def test_rag_importable(self):
        from src.ai.rag.retrieval import retrieve_context
        assert callable(retrieve_context)

    def test_rag_ingestion_importable(self):
        from src.ai.rag.ingestion import initialize_knowledge_base
        assert callable(initialize_knowledge_base)
