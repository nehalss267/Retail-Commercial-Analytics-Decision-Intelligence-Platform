"""Reusable table components for the dashboard."""
import streamlit as st
import pandas as pd


def display_dataframe(df: pd.DataFrame, title: str = "", use_container_width: bool = True):
    """Display a dataframe with optional title."""
    if title:
        st.subheader(title)
    st.dataframe(df, use_container_width=use_container_width)


def display_metric_table(df: pd.DataFrame, title: str = ""):
    """Display a formatted metric table."""
    if title:
        st.subheader(title)
    st.dataframe(
        df.style.format(precision=2, na_rep="—"),
        use_container_width=True,
    )


def display_json_table(data: dict | list, title: str = ""):
    """Display JSON data as a table."""
    if title:
        st.subheader(title)
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
