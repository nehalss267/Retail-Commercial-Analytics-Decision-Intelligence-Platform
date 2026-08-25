"""Optimization Constraints — Constraint definitions."""
import numpy as np
import pandas as pd


def budget_constraint(targets: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Select top-N targets within budget."""
    return targets.nlargest(budget, "expected_incremental_value")


def segment_constraints(targets: pd.DataFrame, exclude_segments: list[str] | None = None,
                        include_segments: list[str] | None = None) -> pd.DataFrame:
    """Filter targets by segment."""
    if exclude_segments:
        targets = targets[~targets["segment"].isin(exclude_segments)]
    if include_segments:
        targets = targets[targets["segment"].isin(include_segments)]
    return targets


def discount_constraint(targets: pd.DataFrame, max_discount_pct: float) -> pd.DataFrame:
    """Cap discount at maximum percentage."""
    targets = targets.copy()
    targets["discount_applied"] = np.minimum(targets.get("discount_rate", 0.1), max_discount_pct / 100)
    return targets


def minimum_value_constraint(targets: pd.DataFrame, min_clv: float = 0) -> pd.DataFrame:
    """Filter to customers above minimum CLV threshold."""
    if "predicted_clv" in targets.columns:
        return targets[targets["predicted_clv"] >= min_clv]
    return targets
