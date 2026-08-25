"""Recommendation Tool — Product recommendation engine."""
from langchain_core.tools import tool
import json

from src.config import settings


@tool
def get_popular_products(n: int = 10) -> list[dict]:
    """Get the most popular products by revenue."""
    path = settings.PROCESSED_DATA_DIR / "recommendations_report.json"
    if path.exists():
        recs = json.loads(path.read_text())
        return recs.get("popularity_top_20", [])[:n]
    return []


@tool
def get_product_recommendations(product_code: str, n: int = 5) -> list[dict]:
    """Get products similar to a given product (content-based)."""
    from src.models.recommendation.content_based import content_based
    import pandas as pd
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    return content_based(df, product_code, top_n=n)


@tool
def get_customer_recommendations(customer_id: int, n: int = 5) -> list[dict]:
    """Get product recommendations for a specific customer (collaborative filtering)."""
    from src.models.recommendation.collaborative import collaborative_filtering
    import pandas as pd
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    return collaborative_filtering(df, customer_id, top_n=n)
