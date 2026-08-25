"""Tests for statistical analysis modules."""
import pytest
import numpy as np
from src.statistics.confidence_intervals import mean_ci, proportion_ci, difference_ci
from src.statistics.power_analysis import power_two_sample, required_sample_size


class TestConfidenceIntervals:
    def test_mean_ci(self):
        data = np.random.normal(50, 10, 100)
        result = mean_ci(data)
        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["mean"] < result["ci_upper"]

    def test_proportion_ci(self):
        result = proportion_ci(30, 100)
        assert 0 <= result["proportion"] <= 1
        assert result["ci_lower"] < result["proportion"] < result["ci_upper"]

    def test_difference_ci(self):
        result = difference_ci(10, 8, 4, 3, 50, 50)
        assert "difference" in result
        assert result["difference"] == 2.0


class TestPowerAnalysis:
    def test_power_increases_with_effect(self):
        p1 = power_two_sample(0, 0.5, 1, 1, 50, 50)
        p2 = power_two_sample(0, 1.0, 1, 1, 50, 50)
        assert p2["power"] > p1["power"]

    def test_required_sample_size(self):
        result = required_sample_size(0.5, power=0.8)
        assert result["n_per_group"] > 0

    def test_power_increases_with_n(self):
        p1 = power_two_sample(0, 0.5, 1, 1, 20, 20)
        p2 = power_two_sample(0, 0.5, 1, 1, 100, 100)
        assert p2["power"] > p1["power"]
