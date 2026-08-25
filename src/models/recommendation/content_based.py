"""Content-Based Recommendation — Product similarity."""
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def content_based(df: pd.DataFrame, stock_code: str, top_n: int = 10) -> list[dict]:
    """Recommend products similar to a given product based on purchase patterns."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    pivot = valid.pivot_table(
        index="StockCode", columns="CustomerID",
        values="Revenue", aggfunc="sum", fill_value=0
    )

    if stock_code not in pivot.index:
        return []

    sim_matrix = cosine_similarity(pivot.values)
    sim_df = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)

    similarities = sim_df[stock_code].drop(stock_code).sort_values(ascending=False).head(top_n)

    results = []
    for code, score in similarities.items():
        desc = valid[valid["StockCode"] == code]["Description"].iloc[0] if len(valid[valid["StockCode"] == code]) > 0 else ""
        results.append({
            "stock_code": code,
            "description": str(desc)[:80],
            "similarity_score": round(float(score), 4),
        })
    return results
