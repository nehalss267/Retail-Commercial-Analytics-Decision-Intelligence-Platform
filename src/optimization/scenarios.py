"""Scenario Analysis — Compare different optimization strategies."""
import pandas as pd
import numpy as np


def compare_scenarios(rfm: pd.DataFrame, clv: pd.DataFrame,
                      scenarios: list[dict]) -> list[dict]:
    """Run and compare multiple targeting scenarios.

    Each scenario dict: {name, budget, max_discount_pct, exclude_segments}
    """
    from src.optimization.targeting import optimize_targeting

    results = []
    for scenario in scenarios:
        result = optimize_targeting(
            rfm, clv,
            budget=scenario.get("budget", 100),
            max_discount_pct=scenario.get("max_discount_pct", 15.0),
        )
        result["scenario_name"] = scenario.get("name", "Unnamed")
        results.append(result)
    return results


def sensitivity_analysis(rfm: pd.DataFrame, clv: pd.DataFrame,
                         budget_range: list[int] | None = None,
                         discount_range: list[float] | None = None) -> list[dict]:
    """Sensitivity analysis across budget and discount levels."""
    from src.optimization.targeting import optimize_targeting

    if budget_range is None:
        budget_range = [25, 50, 100, 200, 500]
    if discount_range is None:
        discount_range = [5.0, 10.0, 15.0, 20.0, 25.0]

    results = []
    for budget in budget_range:
        for discount in discount_range:
            r = optimize_targeting(rfm, clv, budget=budget, max_discount_pct=discount)
            results.append({
                "budget": budget,
                "discount_pct": discount,
                "incremental_revenue": r["total_expected_incremental_revenue"],
                "roi": r["expected_roi"],
                "n_targeted": r["n_targeted"],
            })
    return results
