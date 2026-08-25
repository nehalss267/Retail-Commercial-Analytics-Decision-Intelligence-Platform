"""A/B Testing and Experimentation Module.

Since UCI is observational, we simulate a promotion experiment
using customer/product history as baseline.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

from src.config import settings


def load_cleaned() -> pd.DataFrame:
    return pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")


def simulate_experiment(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate a promotion experiment.

    - Eligible customers: those with 2+ orders
    - Treatment: simulated 15% discount exposure
    - Effect: ~8% revenue uplift for treated (with noise)
    """
    rng = np.random.default_rng(seed)

    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])].copy()

    # Get customer metrics
    cust = valid.groupby("CustomerID").agg(
        orders=("InvoiceNo", "nunique"),
        avg_revenue=("Revenue", "mean"),
        last_purchase=("InvoiceDate", "max"),
    ).reset_index()

    # Eligibility: 2+ orders
    eligible = cust[cust["orders"] >= 2].copy()
    print(f"  Eligible customers: {len(eligible)}")

    # Random assignment 50/50
    eligible["treatment"] = rng.random(len(eligible)) < 0.5
    assignment = eligible[["CustomerID", "treatment"]].copy()

    # Simulate outcome: treatment gets ~8% uplift with noise
    eligible["base_revenue"] = eligible["avg_revenue"]
    eligible["treatment_effect"] = np.where(
        eligible["treatment"],
        eligible["base_revenue"] * rng.normal(0.08, 0.03, len(eligible)),
        0.0,
    )
    eligible["simulated_revenue"] = eligible["base_revenue"] + eligible["treatment_effect"]
    eligible["data_source"] = "synthetic_scenario"

    return eligible


def ab_test_analysis(experiment: pd.DataFrame) -> dict:
    """Analyze A/B test results."""
    control = experiment[~experiment["treatment"]]["simulated_revenue"]
    treatment = experiment[experiment["treatment"]]["simulated_revenue"]

    # Metrics
    control_mean = control.mean()
    treatment_mean = treatment.mean()
    absolute_uplift = treatment_mean - control_mean
    relative_uplift = (absolute_uplift / control_mean) * 100

    # t-test
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    # Confidence interval for difference
    se = np.sqrt(control.var() / len(control) + treatment.var() / len(treatment))
    ci_95 = (absolute_uplift - 1.96 * se, absolute_uplift + 1.96 * se)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((control.std()**2 + treatment.std()**2) / 2)
    cohens_d = absolute_uplift / pooled_std

    # Power (approximate using normal approximation)
    from scipy.stats import norm
    alpha = 0.05
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = (abs(absolute_uplift) / se) - z_alpha
    power = norm.cdf(z_beta)

    # Required sample size per group
    effect_size_d = cohens_d
    n_required = int(((z_alpha + norm.ppf(0.8)) / effect_size_d) ** 2) if effect_size_d > 0 else 0

    return {
        "experiment": "Simulated Promotion Experiment",
        "data_source": "synthetic_scenario",
        "control_n": len(control),
        "treatment_n": len(treatment),
        "control_mean_revenue": round(float(control_mean), 2),
        "treatment_mean_revenue": round(float(treatment_mean), 2),
        "absolute_uplift": round(float(absolute_uplift), 2),
        "relative_uplift_pct": round(float(relative_uplift), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(f"{p_value:.4e}"),
        "significant": bool(p_value < 0.05),
        "ci_95_lower": round(float(ci_95[0]), 2),
        "ci_95_upper": round(float(ci_95[1]), 2),
        "effect_size_cohens_d": round(float(cohens_d), 4),
        "statistical_power": round(float(power), 4),
        "min_sample_size_per_group": n_required,
    }


def run_experimentation() -> dict:
    """Run full experimentation pipeline."""
    df = load_cleaned()

    print("Simulating promotion experiment...")
    experiment = simulate_experiment(df)

    print("Running A/B test analysis...")
    results = ab_test_analysis(experiment)

    # Save
    experiment.to_parquet(settings.PROCESSED_DATA_DIR / "experiment_results.parquet", index=False)

    with open(settings.PROCESSED_DATA_DIR / "experiment_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== EXPERIMENT RESULTS ===")
    print(f"Control: {results['control_n']} customers, mean £{results['control_mean_revenue']}")
    print(f"Treatment: {results['treatment_n']} customers, mean £{results['treatment_mean_revenue']}")
    print(f"Uplift: {results['relative_uplift_pct']}% (p={results['p_value']})")
    print(f"Power: {results['statistical_power']}")

    return results


if __name__ == "__main__":
    run_experimentation()
