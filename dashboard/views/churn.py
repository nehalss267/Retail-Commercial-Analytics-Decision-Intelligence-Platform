"""Churn & CLV page."""
import streamlit as st
import json
import pandas as pd
from pathlib import Path

from dashboard.components.cards import kpi_row
from dashboard.components.tables import display_metric_table

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@st.cache_data
def load_churn():
    path = DATA_DIR / "processed" / "churn_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_clv():
    path = DATA_DIR / "processed" / "clv_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def render():
    st.title("Churn & Customer Lifetime Value")

    # Churn section
    st.header("Churn Prediction")
    churn = load_churn()
    if churn:
        best = churn.get("best_model", "N/A")
        results = churn.get("results", {})
        if best in results:
            m = results[best]
            kpi_row([
                {"label": "Best Model", "value": best},
                {"label": "ROC-AUC", "value": f"{m.get('roc_auc', 0):.4f}"},
                {"label": "F1 Score", "value": f"{m.get('f1', 0):.4f}"},
                {"label": "PR-AUC", "value": f"{m.get('pr_auc', 0):.4f}"},
            ])

        st.subheader("Model Comparison")
        comparison = pd.DataFrame(results).T
        display_metric_table(comparison)

        st.subheader("Feature Importance (XGBoost)")
        fi = churn.get("feature_importance", {})
        fi_df = pd.DataFrame(
            list(fi.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)
        st.bar_chart(fi_df.set_index("Feature"))
    else:
        st.warning("Churn report not available.")

    # CLV section
    st.header("Customer Lifetime Value")
    clv = load_clv()
    if clv:
        model_info = clv.get("model_info", {})
        kpi_row([
            {"label": "Model R²", "value": f"{model_info.get('model_r2', 0):.4f}"},
            {"label": "CV R² Mean", "value": f"{model_info.get('cv_r2_mean', 0):.4f}"},
        ])

        st.subheader("CLV Segments")
        seg_summary = pd.DataFrame(clv.get("segment_summary", []))
        if not seg_summary.empty:
            display_metric_table(seg_summary)
    else:
        st.warning("CLV report not available.")
