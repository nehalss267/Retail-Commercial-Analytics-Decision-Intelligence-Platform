"""Experimentation page."""
import streamlit as st
import json
from pathlib import Path

from dashboard.components.cards import kpi_row
from dashboard.components.tables import display_metric_table
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def load_report():
    path = DATA_DIR / "experiment_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def render():
    st.title("Experimentation & A/B Testing")

    report = load_report()
    if report is None:
        st.warning("Experiment report not available.")
        return

    st.info(f"**{report.get('experiment', 'N/A')}** — Data source: {report.get('data_source', 'N/A')}")

    kpi_row([
        {"label": "Control N", "value": f"{report.get('control_n', 0):,}"},
        {"label": "Treatment N", "value": f"{report.get('treatment_n', 0):,}"},
        {"label": "Relative Uplift", "value": f"{report.get('relative_uplift_pct', 0):.2f}%"},
        {"label": "p-value", "value": report.get("p_value", "N/A")},
    ])

    kpi_row([
        {"label": "Absolute Uplift", "value": f"£{report.get('absolute_uplift', 0):.2f}"},
        {"label": "Effect Size (Cohen's d)", "value": f"{report.get('effect_size_cohens_d', 0):.4f}"},
        {"label": "Statistical Power", "value": f"{report.get('statistical_power', 0):.4f}"},
        {"label": "Significant", "value": "Yes" if report.get("significant") else "No"},
    ])

    st.subheader("Confidence Interval")
    ci_lower = report.get("ci_95_lower", 0)
    ci_upper = report.get("ci_95_upper", 0)
    st.write(f"95% CI for uplift: [£{ci_lower:.2f}, £{ci_upper:.2f}]")

    st.subheader("Full Results")
    display_metric_table(pd.DataFrame([report]))
