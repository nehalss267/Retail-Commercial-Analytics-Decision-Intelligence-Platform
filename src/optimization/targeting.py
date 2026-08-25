"""Prescriptive Analytics — Budget-constrained customer targeting optimization."""
import pandas as pd
import numpy as np
import json

from src.config import settings


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load churn predictions, CLV, and experiment data."""
    churn_path = settings.PROCESSED_DATA_DIR / "churn_report.json"
    clv_path = settings.FEATURES_DIR / "clv_features.parquet"
    rfm_path = settings.FEATURES_DIR / "rfm_features.parquet"

    rfm = pd.read_parquet(rfm_path)
    clv = pd.read_parquet(clv_path) if clv_path.exists() else rfm.copy()

    return rfm, clv


def optimize_targeting(rfm: pd.DataFrame, clv: pd.DataFrame, budget: int = 100,
                       max_discount_pct: float = 15.0) -> dict:
    """Select customers to target for maximum expected incremental revenue.

    Objective: maximize expected incremental revenue
    Subject to: budget (number of customers), eligibility, discount cap
    """
    # Merge CLV into RFM
    if "predicted_clv" in clv.columns:
        merged = rfm.merge(
            clv[["CustomerID", "predicted_clv", "clv_segment"]],
            on="CustomerID", how="left"
        )
    else:
        merged = rfm.copy()
        merged["predicted_clv"] = merged["monetary"]
        merged["clv_segment"] = "Unknown"

    # Scoring: high CLV + at risk = high priority
    merged["churn_score"] = np.where(
        merged["segment"].isin(["At Risk", "Can't Lose", "Lost"]), 1.0,
        np.where(merged["segment"] == "Regular", 0.5, 0.2)
    )

    merged["expected_incremental_value"] = (
        merged["predicted_clv"] * merged["churn_score"] * (max_discount_pct / 100)
    )

    # Budget constraint: select top N by expected incremental value
    eligible = merged[merged["segment"] != "Champions"].copy()  # Don't target champions
    targeted = eligible.nlargest(budget, "expected_incremental_value")

    total_incremental = targeted["expected_incremental_value"].sum()
    total_cost = targeted["predicted_clv"].sum() * (max_discount_pct / 100)

    return {
        "budget": budget,
        "max_discount_pct": max_discount_pct,
        "n_targeted": len(targeted),
        "total_expected_incremental_revenue": round(float(total_incremental), 2),
        "total_estimated_cost": round(float(total_cost), 2),
        "expected_roi": round(float(total_incremental / total_cost * 100), 2) if total_cost > 0 else 0,
        "target_segments": targeted["segment"].value_counts().to_dict(),
        "targets": targeted[["CustomerID", "segment", "monetary", "predicted_clv",
                              "expected_incremental_value"]].to_dict(orient="records"),
    }


def run_optimization() -> dict:
    """Run prescriptive analytics pipeline."""
    rfm, clv = load_data()

    print("Running optimization...")
    result = optimize_targeting(rfm, clv, budget=100, max_discount_pct=15.0)

    with open(settings.PROCESSED_DATA_DIR / "optimization_report.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Budget: {result['budget']} customers")
    print(f"Targeted: {result['n_targeted']}")
    print(f"Expected incremental revenue: £{result['total_expected_incremental_revenue']:,.2f}")
    print(f"Expected ROI: {result['expected_roi']}%")
    print(f"Target segments: {result['target_segments']}")

    return result


if __name__ == "__main__":
    run_optimization()
