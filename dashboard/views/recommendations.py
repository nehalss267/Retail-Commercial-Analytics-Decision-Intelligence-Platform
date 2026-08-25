"""Recommendations page."""
import streamlit as st
import json
from pathlib import Path

from dashboard.components.tables import display_dataframe
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def load_recs():
    path = DATA_DIR / "recommendations_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def render():
    st.title("Recommendation System")

    recs = load_recs()
    if recs is None:
        st.warning("Recommendations report not available.")
        return

    # Popular products
    st.subheader("Popular Products")
    pop = pd.DataFrame(recs.get("popularity_top_20", []))
    if not pop.empty:
        display_dataframe(pop[["StockCode", "Description", "revenue", "quantity", "orders"]])

    # Content-based
    content_data = recs.get("content_based_for_product", {})
    if content_data:
        st.subheader(f"Content-Based Recommendations for {content_data.get('product', 'N/A')}")
        content = pd.DataFrame(content_data.get("recommendations", []))
        if not content.empty:
            display_dataframe(content)

    # Collaborative
    collab_data = recs.get("collaborative_for_customer", {})
    if collab_data:
        st.subheader(f"Collaborative Filtering for Customer {collab_data.get('customer_id', 'N/A')}")
        collab = pd.DataFrame(collab_data.get("recommendations", []))
        if not collab.empty:
            display_dataframe(collab)
