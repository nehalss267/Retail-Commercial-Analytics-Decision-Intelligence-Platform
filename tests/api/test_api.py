"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestExecutive:
    def test_summary(self):
        response = client.get("/executive/summary")
        assert response.status_code == 200


class TestSegments:
    def test_segments(self):
        response = client.get("/segments")
        assert response.status_code == 200


class TestChurn:
    def test_churn_summary(self):
        response = client.get("/churn/summary")
        assert response.status_code == 200


class TestForecast:
    def test_forecast_summary(self):
        response = client.get("/forecast/summary")
        assert response.status_code == 200


class TestStatistics:
    def test_statistics(self):
        response = client.get("/statistics")
        assert response.status_code == 200


class TestRecommendations:
    def test_popular(self):
        response = client.get("/recommendations/popular")
        assert response.status_code == 200


class TestAgent:
    def test_query(self):
        response = client.post("/agent/query", json={"question": "What is the total revenue?"})
        assert response.status_code == 200
        assert "answer" in response.json()
