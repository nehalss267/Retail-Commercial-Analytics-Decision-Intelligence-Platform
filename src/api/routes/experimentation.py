"""API Routes — Experimentation endpoints."""
from fastapi import APIRouter

from src.api.dependencies import load_report

router = APIRouter(prefix="/experiment", tags=["experimentation"])


@router.get("/results")
def experiment_results():
    return load_report("experiment_report")


@router.post("/analyze")
def analyze_experiment():
    """Run a fresh experiment analysis."""
    from src.experimentation.ab_test import run_experimentation
    return run_experimentation()
