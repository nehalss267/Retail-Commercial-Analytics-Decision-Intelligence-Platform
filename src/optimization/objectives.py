"""Optimization Objectives — Objective function definitions."""
import numpy as np
import pandas as pd


def maximize_incremental_revenue(targets: pd.DataFrame) -> np.ndarray:
    """Objective: maximize expected incremental revenue."""
    return targets["expected_incremental_value"].values


def maximize_roi(targets: pd.DataFrame) -> np.ndarray:
    """Objective: maximize expected ROI."""
    cost = targets["predicted_clv"] * 0.15
    return np.where(cost > 0, targets["expected_incremental_value"] / cost, 0)


def maximize_coverage(targets: pd.DataFrame) -> np.ndarray:
    """Objective: maximize customer segment coverage."""
    return np.ones(len(targets))
