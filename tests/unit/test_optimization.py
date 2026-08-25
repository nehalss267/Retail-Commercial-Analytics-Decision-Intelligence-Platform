"""Tests for optimization modules."""
import pytest
import pandas as pd
import numpy as np
from src.optimization.constraints import (
    budget_constraint, segment_constraints, discount_constraint
)


@pytest.fixture
def sample_targets():
    return pd.DataFrame({
        "CustomerID": range(1, 51),
        "segment": np.random.choice(["Champions", "Loyal", "At Risk", "Lost"], 50),
        "predicted_clv": np.random.uniform(100, 5000, 50),
        "expected_incremental_value": np.random.uniform(10, 500, 50),
        "discount_rate": np.random.uniform(0.05, 0.2, 50),
    })


class TestConstraints:
    def test_budget_constraint(self, sample_targets):
        result = budget_constraint(sample_targets, budget=10)
        assert len(result) == 10
        # Should be the top 10 by incremental value
        assert result["expected_incremental_value"].is_monotonic_decreasing

    def test_segment_constraints_exclude(self, sample_targets):
        result = segment_constraints(sample_targets, exclude_segments=["Champions"])
        assert "Champions" not in result["segment"].values

    def test_segment_constraints_include(self, sample_targets):
        result = segment_constraints(sample_targets, include_segments=["Loyal", "At Risk"])
        assert set(result["segment"].unique()).issubset({"Loyal", "At Risk"})

    def test_discount_constraint(self, sample_targets):
        result = discount_constraint(sample_targets, max_discount_pct=10.0)
        assert result["discount_applied"].max() <= 0.10
