"""Difference-in-Differences — Causal inference for panel data."""
import pandas as pd
import numpy as np
from scipy import stats


def difference_in_differences(data: pd.DataFrame,
                              outcome_col: str,
                              treatment_col: str = "treatment",
                              period_col: str = "period",
                              time_col: str = "time") -> dict:
    """Two-way Difference-in-Differences estimation.

    Requires panel data with pre/post periods and control/treatment groups.
    """
    # Pre/Post periods
    pre = data[data[period_col] == "pre"]
    post = data[data[period_col] == "post"]

    # Group means
    treated_pre = pre[pre[treatment_col] == 1][outcome_col]
    treated_post = post[post[treatment_col] == 1][outcome_col]
    control_pre = pre[pre[treatment_col] == 0][outcome_col]
    control_post = post[post[treatment_col] == 0][outcome_col]

    # DiD estimate
    did = (treated_post.mean() - treated_pre.mean()) - (control_post.mean() - control_pre.mean())

    # Standard error (robust)
    se = np.sqrt(
        treated_post.var() / len(treated_post) +
        treated_pre.var() / len(treated_pre) +
        control_post.var() / len(control_post) +
        control_pre.var() / len(control_pre)
    )

    t_stat = did / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(data) - 4))
    ci = (did - 1.96 * se, did + 1.96 * se)

    return {
        "method": "Difference-in-Differences",
        "data_source": "synthetic_scenario",
        "did_estimate": round(float(did), 4),
        "std_error": round(float(se), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(f"{p_value:.4e}"),
        "significant": bool(p_value < 0.05),
        "ci_95_lower": round(float(ci[0]), 4),
        "ci_95_upper": round(float(ci[1]), 4),
        "treated_pre_mean": round(float(treated_pre.mean()), 4),
        "treated_post_mean": round(float(treated_post.mean()), 4),
        "control_pre_mean": round(float(control_pre.mean()), 4),
        "control_post_mean": round(float(control_post.mean()), 4),
        "n_treated": int(len(treated_pre)),
        "n_control": int(len(control_pre)),
    }


def simulate_did_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate panel data for DiD analysis from transaction data."""
    rng = np.random.default_rng(seed)

    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    customers = valid.groupby("CustomerID").agg(
        orders=("InvoiceNo", "nunique"),
        avg_revenue=("Revenue", "mean"),
    ).reset_index()

    # Select eligible customers (3+ orders)
    eligible = customers[customers["orders"] >= 3].copy()

    # Random treatment assignment
    eligible["treatment"] = (rng.random(len(eligible)) < 0.5).astype(int)

    # Create pre/post periods
    rows = []
    for _, row in eligible.iterrows():
        base = row["avg_revenue"]
        # Pre period
        rows.append({
            "CustomerID": row["CustomerID"],
            "treatment": row["treatment"],
            "period": "pre",
            "revenue": base + rng.normal(0, base * 0.1),
        })
        # Post period (treatment gets uplift)
        uplift = base * 0.12 * row["treatment"] if row["treatment"] else 0
        rows.append({
            "CustomerID": row["CustomerID"],
            "treatment": row["treatment"],
            "period": "post",
            "revenue": base + uplift + rng.normal(0, base * 0.1),
        })

    return pd.DataFrame(rows)
