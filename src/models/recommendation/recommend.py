"""Recommendation System — Popularity, Content-Based, Collaborative Filtering."""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import json

from src.config import settings


def load_cleaned() -> pd.DataFrame:
    return pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")


def popularity_baseline(df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    """Most popular products by revenue."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]
    products = valid.groupby(["StockCode", "Description"]).agg(
        revenue=("Revenue", "sum"),
        quantity=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
    ).reset_index().sort_values("revenue", ascending=False)

    return products.head(top_n)[
        ["StockCode", "Description", "revenue", "quantity", "orders", "customers"]
    ].to_dict(orient="records")


def content_based(df: pd.DataFrame, stock_code: str, top_n: int = 10) -> list[dict]:
    """Recommend products similar to a given product based on purchase patterns."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    # Product-customer matrix
    pivot = valid.pivot_table(
        index="StockCode", columns="CustomerID",
        values="Revenue", aggfunc="sum", fill_value=0
    )

    if stock_code not in pivot.index:
        return []

    # Cosine similarity
    sim_matrix = cosine_similarity(pivot.values)
    sim_df = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)

    # Get top similar products (excluding self)
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


def collaborative_filtering(df: pd.DataFrame, customer_id: int, top_n: int = 10) -> list[dict]:
    """User-based collaborative filtering."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    # User-item matrix
    pivot = valid.pivot_table(
        index="CustomerID", columns="StockCode",
        values="Revenue", aggfunc="sum", fill_value=0
    )

    if customer_id not in pivot.index:
        return []

    # Sparse matrix for similarity
    sparse = csr_matrix(pivot.values)
    sim = cosine_similarity(sparse)
    sim_df = pd.DataFrame(sim, index=pivot.index, columns=pivot.index)

    # Find similar users
    user_sims = sim_df[customer_id].drop(customer_id).sort_values(ascending=False).head(20)

    # Products the target customer hasn't bought
    user_products = set(pivot.loc[customer_id][pivot.loc[customer_id] > 0].index)
    all_products = set(pivot.columns)

    # Score unseen products by similar users' purchases
    scores = {}
    for product in all_products - user_products:
        sim_scores = user_sims.values
        product_scores = pivot.loc[user_sims.index, product].values
        score = np.dot(sim_scores, product_scores)
        scores[product] = score

    # Top N
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


def run_recommendations() -> dict:
    """Run full recommendation pipeline."""
    df = load_cleaned()

    print("1. Popularity baseline...")
    pop = popularity_baseline(df)

    print("2. Content-based (for top product)...")
    top_product = pop[0]["StockCode"]
    content = content_based(df, top_product)

    print("3. Collaborative filtering (for random customer)...")
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]
    sample_customer = int(valid["CustomerID"].value_counts().head(100).sample(1).index[0])
    collab = collaborative_filtering(df, sample_customer)

    results = {
        "popularity_top_20": pop,
        "content_based_for_product": {"product": top_product, "recommendations": content},
        "collaborative_for_customer": {"customer_id": sample_customer, "recommendations": collab},
    }

    with open(settings.PROCESSED_DATA_DIR / "recommendations_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== POPULARITY TOP 5 ===")
    for p in pop[:5]:
        print(f"  {p['StockCode']}: {p['Description'][:50]} — £{p['revenue']:,.0f}")

    print(f"\n=== CONTENT-BASED for {top_product} ===")
    for r in content[:5]:
        print(f"  {r['stock_code']}: {r['description'][:50]} (sim={r['similarity_score']})")

    print(f"\n=== COLLABORATIVE for customer {sample_customer} ===")
    for r in collab[:5]:
        print(f"  {r['stock_code']}: {r['description'][:50]} (score={r['score']})")

    return results


if __name__ == "__main__":
    run_recommendations()
