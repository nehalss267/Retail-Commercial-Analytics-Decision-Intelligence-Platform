"""Collaborative Filtering — User-based recommendation."""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def collaborative_filtering(df: pd.DataFrame, customer_id: int, top_n: int = 10) -> list[dict]:
    """User-based collaborative filtering."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    pivot = valid.pivot_table(
        index="CustomerID", columns="StockCode",
        values="Revenue", aggfunc="sum", fill_value=0
    )

    if customer_id not in pivot.index:
        return []

    sparse = csr_matrix(pivot.values)
    sim = cosine_similarity(sparse)
    sim_df = pd.DataFrame(sim, index=pivot.index, columns=pivot.index)

    user_sims = sim_df[customer_id].drop(customer_id).sort_values(ascending=False).head(20)

    user_products = set(pivot.loc[customer_id][pivot.loc[customer_id] > 0].index)
    all_products = set(pivot.columns)

    scores = {}
    for product in all_products - user_products:
        sim_scores = user_sims.values
        product_scores = pivot.loc[user_sims.index, product].values
        score = np.dot(sim_scores, product_scores)
        scores[product] = score

    top_products = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

    results = []
    for code, score in top_products:
        desc = valid[valid["StockCode"] == code]["Description"].iloc[0] if len(valid[valid["StockCode"] == code]) > 0 else ""
        results.append({
            "stock_code": code,
            "description": str(desc)[:80],
            "score": round(float(score), 2),
        })
    return results
