"""Tests for forecasting modules."""
import pytest
import numpy as np
import pandas as pd
from src.models.forecasting.baselines import naive_forecast, moving_average_forecast
from src.models.forecasting.evaluation import evaluate_forecast, compare_models


class TestBaselines:
    def test_naive_forecast(self):
        series = pd.Series([1, 2, 3, 4, 5])
        result = naive_forecast(series, horizon=3)
        assert len(result) == 3
        assert all(v == 5.0 for v in result)

    def test_moving_average_forecast(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = moving_average_forecast(series, horizon=3, window=7)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


class TestEvaluation:
    def test_evaluate_forecast(self):
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 190, 310, 380, 520])
        result = evaluate_forecast(actual, predicted)
        assert "mae" in result
        assert "rmse" in result
        assert "mape" in result
        assert result["mae"] > 0
        assert result["n_points"] == 5

    def test_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        result = evaluate_forecast(actual, predicted)
        assert result["mae"] == 0
        assert result["rmse"] == 0

    def test_compare_models(self):
        actual = np.array([100, 200, 300])
        predictions = {
            "model_a": np.array([110, 190, 310]),
            "model_b": np.array([150, 150, 150]),
        }
        result = compare_models(actual, predictions)
        assert len(result) == 2
        assert "model_a" in result.index
        assert "model_b" in result.index
