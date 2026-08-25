"""Experiment Tool — A/B testing and experiment analysis."""
from langchain_core.tools import tool
import json

from src.config import settings


@tool
def get_experiment_results() -> dict:
    """Get A/B test experiment results including uplift, p-value, and confidence interval."""
    path = settings.PROCESSED_DATA_DIR / "experiment_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"error": "Experiment report not available"}


@tool
def explain_experiment_methodology() -> str:
    """Explain the A/B testing methodology used."""
    return (
        "The experiment simulates a promotion campaign on eligible customers (2+ orders). "
        "Customers are randomly assigned 50/50 to control (no intervention) or treatment (15% discount). "
        "The simulated effect is ~8% revenue uplift with noise. "
        "Analysis uses Welch's t-test, 95% confidence intervals, Cohen's d effect size, "
        "and statistical power calculation. "
        "All experiment data is synthetic_scenario, not historical observations."
    )
