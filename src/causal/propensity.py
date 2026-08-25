"""Causal Inference — Propensity Score Matching and Treatment Effects."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import json

from src.config import settings


def load_experiment_data() -> pd.DataFrame:
    """Load experiment results (synthetic scenario)."""
    return pd.read_parquet(settings.PROCESSED_DATA_DIR / "experiment_results.parquet")


def propensity_score_model(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate propensity scores for treatment assignment."""
    feature_cols = ["orders", "avg_revenue"]
    data = data.copy()
    data["days_since_last"] = (data["last_purchase"].max() - data["last_purchase"]).dt.days
    feature_cols.append("days_since_last")
    X = data[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = data["treatment"].astype(int)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    data = data.copy()
    data["propensity_score"] = model.predict_proba(X_scaled)[:, 1]

    return data, {
        "features": feature_cols,
        "coefficients": dict(zip(feature_cols, [round(float(x), 4) for x in model.coef_[0]])),
        "intercept": round(float(model.intercept_[0]), 4),
    }


def propensity_matching(data: pd.DataFrame, caliper: float = 0.1) -> dict:
    """Nearest-neighbor propensity score matching."""
    treated = data[data["treatment"] == True].copy()
    control = data[data["treatment"] == False].copy()

    # Match each treated to nearest control by propensity score
    treated_scores = treated["propensity_score"].values.reshape(-1, 1)
    control_scores = control["propensity_score"].values.reshape(-1, 1)

    distances = cdist(treated_scores, control_scores, metric="euclidean")
    matched_indices = distances.argmin(axis=1)
    matched_distances = distances.min(axis=1)

    # Apply caliper
    valid_match = matched_distances <= caliper

    matched_treated = treated.iloc[valid_match].copy()
    matched_control = control.iloc[matched_indices[valid_match]].copy()

    return {
        "n_treated": len(matched_treated),
        "n_control": len(matched_control),
        "n_dropped": int((~valid_match).sum()),
        "caliper": caliper,
        "matched_treated": matched_treated,
        "matched_control": matched_control,
    }


def estimate_treatment_effect(matched: dict) -> dict:
    """Estimate Average Treatment Effect on the Treated (ATT)."""
    treated_rev = matched["matched_treated"]["simulated_revenue"]
    control_rev = matched["matched_control"]["simulated_revenue"]

    att = treated_rev.mean() - control_rev.mean()
    se = np.sqrt(treated_rev.var() / len(treated_rev) + control_rev.var() / len(control_rev))
    ci_95 = (att - 1.96 * se, att + 1.96 * se)

    from scipy import stats
    t_stat, p_value = stats.ttest_ind(treated_rev, control_rev, equal_var=False)

    return {
        "method": "Propensity Score Matching",
        "data_source": "synthetic_scenario",
        "att": round(float(att), 2),
        "att_pct": round(float(att / control_rev.mean() * 100), 2),
        "ci_95_lower": round(float(ci_95[0]), 2),
        "ci_95_upper": round(float(ci_95[1]), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(f"{p_value:.4e}"),
        "significant": bool(p_value < 0.05),
        "n_treated": matched["n_treated"],
        "n_control": matched["n_control"],
        "n_dropped_by_caliper": matched["n_dropped"],
    }


def run_causal() -> dict:
    """Run causal inference pipeline."""
    print("Loading experiment data...")
    data = load_experiment_data()

    print("Fitting propensity score model...")
    data, prop_model = propensity_score_model(data)

    print("Propensity score matching...")
    matched = propensity_matching(data)

    print("Estimating treatment effect...")
    ate = estimate_treatment_effect(matched)

    results = {
        "propensity_model": prop_model,
        "matching": {
            "n_treated": matched["n_treated"],
            "n_control": matched["n_control"],
            "n_dropped": matched["n_dropped"],
        },
        "treatment_effect": ate,
    }

    with open(settings.PROCESSED_DATA_DIR / "causal_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== CAUSAL INFERENCE RESULTS ===")
    print(f"Method: {ate['method']}")
    print(f"ATT: £{ate['att']} ({ate['att_pct']}% uplift)")
    print(f"95% CI: [{ate['ci_95_lower']}, {ate['ci_95_upper']}]")
    print(f"p-value: {ate['p_value']}")
    print(f"Significant: {ate['significant']}")

    return results


if __name__ == "__main__":
    run_causal()
