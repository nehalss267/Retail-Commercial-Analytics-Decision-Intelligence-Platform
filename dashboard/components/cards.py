"""Reusable KPI card components for the dashboard."""
import streamlit as st


def kpi_card(label: str, value: str, delta: str | None = None, delta_color: str = "normal"):
    """Render a single KPI metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def kpi_row(metrics: list[dict], cols: int = 4):
    """Render a row of KPI cards.

    Each metric dict should have: label, value, and optionally delta.
    """
    columns = st.columns(cols)
    for i, m in enumerate(metrics):
        with columns[i % cols]:
            kpi_card(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
            )


def info_card(title: str, content: str, icon: str = ""):
    """Render an info card using st.info."""
    label = f"{icon} {title}" if icon else title
    st.info(f"**{label}**\n\n{content}")


def section_header(title: str, description: str = ""):
    """Render a section header with optional description."""
    st.subheader(title)
    if description:
        st.caption(description)
