"""Customer Intelligence page."""
import streamlit as st
import pandas as pd
from pathlib import Path

from dashboard.components.charts import segment_pie, rfm_scatter
from dashboard.components.tables import display_metric_table

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@st.cache_data
def load_rfm():
    path = DATA_DIR / "features" / "rfm_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def render():
    st.title("Customer Intelligence")

    rfm = load_rfm()
    if rfm is None:
        st.warning("RFM features not available. Run the segmentation pipeline first.")
        return

    # Segment distribution
    section_header = "Customer Segments"
    st.subheader(section_header)
    seg_counts = rfm["segment"].value_counts()
    st.plotly_chart(segment_pie(seg_counts.values, seg_counts.index, "RFM Segments"),
                    use_container_width=True)

    # Segment details
    st.subheader("Segment Profiles")
    seg_stats = rfm.groupby("segment").agg(
        n=("CustomerID", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).round(2)
    display_metric_table(seg_stats)

    # RFM scatter
    st.subheader("RFM Scatter (Recency vs Monetary)")
    st.plotly_chart(rfm_scatter(rfm), use_container_width=True)
