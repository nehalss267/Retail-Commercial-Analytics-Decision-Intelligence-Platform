"""Optimization Tool — Prescriptive analytics and customer targeting."""
from langchain_core.tools import tool
import json

from src.config import settings


@tool
def get_optimization_results() -> dict:
    """Get optimization results: targeted customers, expected ROI, incremental revenue."""
    path = settings.PROCESSED_DATA_DIR / "optimization_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"error": "Optimization report not available"}


@tool
def explain_optimization_methodology() -> str:
    """Explain the optimization methodology."""
    return (
        "Prescriptive analytics combines predictions with business constraints to recommend actions. "
        "Objective: maximize expected incremental revenue. "
        "Constraints: campaign budget (number of customers), customer eligibility, "
        "maximum discount rate (15%). "
        "Scoring: CLV × churn risk × discount rate. "
        "Champions are excluded from targeting (they don't need retention incentives). "
        "The optimizer selects the top-N customers by expected incremental value."
    )


@tool
def get_sensitivity_analysis() -> dict:
    """Get sensitivity analysis across different budget and discount levels."""
    path = settings.PROCESSED_DATA_DIR / "sensitivity_analysis.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"error": "Sensitivity analysis not available. Run scenarios.py first."}
