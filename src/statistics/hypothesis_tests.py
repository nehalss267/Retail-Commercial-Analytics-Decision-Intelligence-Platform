"""Statistical Analysis — hypothesis tests, CI, ANOVA."""
import pandas as pd
import numpy as np
from scipy import stats
import json

from src.config import settings


def load_cleaned() -> pd.DataFrame:
    return pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")


def confidence_interval(data: pd.Series, confidence: float = 0.95) -> dict:
    """Calculate confidence interval for mean."""
    n = len(data)
    mean = data.mean()
    se = stats.sem(data)
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=se)
    return {
        "mean": round(float(mean), 4),
        "ci_lower": round(float(ci[0]), 4),
        "ci_upper": round(float(ci[1]), 4),
        "std": round(float(data.std()), 4),
        "n": n,
    }


def test_high_vs_low_value_customers(df: pd.DataFrame) -> dict:
    """Do high-value customers have different order values?"""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    # Split by median revenue per customer
    customer_rev = valid.groupby("CustomerID")["Revenue"].median()
    median_rev = customer_rev.median()

    high_custs = customer_rev[customer_rev >= median_rev].index
    low_custs = customer_rev[customer_rev < median_rev].index

    high_orders = valid[valid["CustomerID"].isin(high_custs)]["Revenue"]
    low_orders = valid[valid["CustomerID"].isin(low_custs)]["Revenue"]

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(high_orders, low_orders, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((high_orders.std()**2 + low_orders.std()**2) / 2)
    cohens_d = (high_orders.mean() - low_orders.mean()) / pooled_std

    return {
        "test": "Welch's t-test: High vs Low Value Customer Order Value",
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(f"{p_value:.2e}"),
        "significant": p_value < 0.05,
        "effect_size_cohens_d": round(float(cohens_d), 4),
        "high_value_mean": round(float(high_orders.mean()), 2),
        "low_value_mean": round(float(low_orders.mean()), 2),
        "high_value_ci": confidence_interval(high_orders),
        "low_value_ci": confidence_interval(low_orders),
    }


def test_country_aov(df: pd.DataFrame) -> dict:
    """Do countries differ significantly in AOV? (ANOVA)"""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"]) & (df["Country"] == "United Kingdom")]
    other = df[(df["Revenue"] > 0) & (df["HasCustomerID"]) & (df["Country"] != "United Kingdom")]

    uk_rev = valid["Revenue"]
    other_rev = other["Revenue"]

    # Top 5 countries
    top_countries = df[df["HasCustomerID"] & (df["Revenue"] > 0)]["Country"].value_counts().head(5).index.tolist()
    groups = [df[(df["Country"] == c) & (df["Revenue"] > 0)]["Revenue"] for c in top_countries]

    f_stat, p_value = stats.f_oneway(*groups)

    # Kruskal-Wallis (non-parametric)
    h_stat, kw_p = stats.kruskal(*groups)

    return {
        "test": "One-way ANOVA: AOV across top 5 countries",
        "countries": top_countries,
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(f"{p_value:.2e}"),
        "kruskal_h": round(float(h_stat), 4),
        "kruskal_p": float(f"{kw_p:.2e}"),
        "significant": p_value < 0.05,
        "uk_mean_revenue": round(float(uk_rev.mean()), 2),
        "non_uk_mean_revenue": round(float(other_rev.mean()), 2),
    }


def test_segment_differences(df: pd.DataFrame) -> dict:
    """Do customer segments have different monetary values? (ANOVA on RFM segments)"""
    rfm = pd.read_parquet(settings.FEATURES_DIR / "rfm_features.parquet")
    rfm = rfm[rfm["monetary"] > 0]

    segments = rfm["segment"].unique()
    groups = [rfm[rfm["segment"] == s]["monetary"] for s in segments]

    f_stat, p_value = stats.f_oneway(*groups)
    h_stat, kw_p = stats.kruskal(*groups)

    segment_stats = rfm.groupby("segment").agg(
        n=("CustomerID", "count"),
        mean_monetary=("monetary", "mean"),
        median_monetary=("monetary", "median"),
        std_monetary=("monetary", "std"),
    ).round(2)

    return {
        "test": "One-way ANOVA: Monetary value across RFM segments",
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(f"{p_value:.2e}"),
        "kruskal_h": round(float(h_stat), 4),
        "kruskal_p": float(f"{kw_p:.2e}"),
        "significant": p_value < 0.05,
        "segment_stats": segment_stats.reset_index().to_dict(orient="records"),
    }


def run_statistics() -> dict:
    """Run all statistical tests."""
    df = load_cleaned()

    print("1. High vs Low value customer test...")
    hv_lv = test_high_vs_low_value_customers(df)

    print("2. Country AOV test (ANOVA)...")
    country = test_country_aov(df)

    print("3. Segment differences test...")
    segment = test_segment_differences(df)

    results = {
        "high_vs_low_value": hv_lv,
        "country_aov": country,
        "segment_differences": segment,
    }

    with open(settings.PROCESSED_DATA_DIR / "statistics_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== STATISTICAL RESULTS ===")
    for name, r in results.items():
        print(f"\n{r['test']}:")
        print(f"  Significant: {r['significant']}")
        if "p_value" in r:
            print(f"  p-value: {r['p_value']}")

    return results


if __name__ == "__main__":
    run_statistics()
