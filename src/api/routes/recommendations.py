"""API Routes — Recommendation endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel

from src.api.dependencies import load_report

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendRequest(BaseModel):
    customer_id: int
    top_n: int = 10


@router.get("/popular")
def popular_products():
    recs = load_report("recommendations_report")
    return recs.get("popularity_top_20", [])[:10]


@router.get("/{customer_id}")
def recommend_for_customer(customer_id: int):
    recs = load_report("recommendations_report")
    collab = recs.get("collaborative_for_customer", {})
    return collab


@router.post("")
def recommend(req: RecommendRequest):
    """Get product recommendations for a customer."""
    from src.models.recommendation.recommend import collaborative_filtering, load_cleaned
    df = load_cleaned()
    recs = collaborative_filtering(df, req.customer_id, top_n=req.top_n)
    return {
        "customer_id": req.customer_id,
        "recommendations": recs,
    }
