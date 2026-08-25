"""Matching Methods — Propensity score and covariate matching."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


def propensity_score_model(data: pd.DataFrame, feature_cols: list[str],
                           treatment_col: str = "treatment") -> pd.DataFrame:
    """Estimate propensity scores."""
    X = data[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = data[treatment_col].astype(int)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    result = data.copy()
    result["propensity_score"] = model.predict_proba(X_scaled)[:, 1]
    return result


def nearest_neighbor_matching(data: pd.DataFrame, caliper: float = 0.1,
                              treatment_col: str = "treatment") -> dict:
    """Nearest-neighbor propensity score matching with caliper."""
    treated = data[data[treatment_col] == 1].copy()
    control = data[data[treatment_col] == 0].copy()

    treated_scores = treated["propensity_score"].values.reshape(-1, 1)
    control_scores = control["propensity_score"].values.reshape(-1, 1)

    distances = cdist(treated_scores, control_scores, metric="euclidean")
    matched_indices = distances.argmin(axis=1)
    matched_distances = distances.min(axis=1)

    valid_match = matched_distances <= caliper

    return {
        "matched_treated": treated.iloc[valid_match],
        "matched_control": control.iloc[matched_indices[valid_match]],
        "n_treated": int(valid_match.sum()),
        "n_control": int(valid_match.sum()),
        "n_dropped": int((~valid_match).sum()),
        "caliper": caliper,
    }
