"""Causal Tool — Causal inference and treatment effect estimation."""
from langchain_core.tools import tool
import json

from src.config import settings


@tool
def get_causal_results() -> dict:
    """Get causal inference results from propensity score matching."""
    path = settings.PROCESSED_DATA_DIR / "causal_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"error": "Causal report not available"}


@tool
def explain_causal_methodology() -> str:
    """Explain the causal inference methodology used."""
    return (
        "Causal inference uses propensity score matching to estimate "
        "Average Treatment Effect on the Treated (ATT). "
        "Steps: 1) Confounder identification, 2) Propensity score model (Logistic Regression), "
        "3) Nearest-neighbor matching with caliper, 4) ATT estimation with 95% CI. "
        "All treatment data is synthetic_scenario, clearly labeled to prevent "
        "confusion with historical observations."
    )
