"""Optimization page."""
import streamlit as st
import json
from pathlib import Path

from dashboard.components.cards import kpi_row
from dashboard.components.tables import display_metric_table
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def load_opt():
    path = DATA_DIR / "optimization_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def render():
    st.title("Prescriptive Analytics & Optimization")

    opt = load_opt()
    if opt is None:
        st.warning("Optimization report not available.")
        return

    kpi_row([
        {"label": "Budget", "value": f"{opt.get('budget', 0)} customers"},
        {"label": "Targeted", "value": f"{opt.get('n_targeted', 0)}"},
        {"label": "Expected Incremental Revenue", "value": f"£{opt.get('total_expected_incremental_revenue', 0):,.2f}"},
        {"label": "Expected ROI", "value": f"{opt.get('expected_roi', 0)}%"},
    ])

    st.subheader("Target Segments")
    segments = opt.get("target_segments", {})
    if segments:
        seg_df = pd.DataFrame(list(segments.items()), columns=["Segment", "Count"])
        st.bar_chart(seg_df.set_index("Segment"))

    st.subheader("Targeted Customers")
    targets = pd.DataFrame(opt.get("targets", []))
    if not targets.empty:
        display_metric_table(targets)
