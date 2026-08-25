"""API Routes — Optimization endpoints."""
from fastapi import APIRouter

from src.api.dependencies import load_report

router = APIRouter(prefix="/optimization", tags=["optimization"])


@router.get("/results")
def optimization_results():
    return load_report("optimization_report")


@router.post("/run")
def run_optimization():
    """Run optimization with custom parameters."""
    from src.optimization.targeting import run_optimization
    return run_optimization()
