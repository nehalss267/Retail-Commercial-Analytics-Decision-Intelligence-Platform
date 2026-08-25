"""RetailAI Dashboard — Main entry point with multi-page navigation."""
import streamlit as st

st.set_page_config(
    page_title="RetailAI",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("RetailAI")
st.sidebar.caption("Decision Intelligence Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Customer Intelligence",
        "Product Analytics",
        "Churn & CLV",
        "Forecasting",
        "Experimentation",
        "Recommendations",
        "Optimization",
        "AI Copilot",
    ],
)

# Route to page
from dashboard.views import (
    executive, customers, products, churn,
    forecasting, experimentation, recommendations, optimization, copilot,
)

pages = {
    "Executive Overview": executive,
    "Customer Intelligence": customers,
    "Product Analytics": products,
    "Churn & CLV": churn,
    "Forecasting": forecasting,
    "Experimentation": experimentation,
    "Recommendations": recommendations,
    "Optimization": optimization,
    "AI Copilot": copilot,
}

pages[page].render()
