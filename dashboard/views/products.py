"""Product Analytics page."""
import streamlit as st
import pandas as pd
from pathlib import Path

from dashboard.components.charts import bar_chart, histogram
from dashboard.components.tables import display_dataframe

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def load():
    return pd.read_parquet(DATA_DIR / "cleaned_retail.parquet")


def render():
    st.title("Product Analytics")

    cleaned = load()
    valid = cleaned[(cleaned["Revenue"] > 0) & (cleaned["HasCustomerID"])]

    # Top products
    st.subheader("Top Products by Revenue")
    products = valid.groupby(["StockCode", "Description"]).agg(
        revenue=("Revenue", "sum"),
        quantity=("Quantity", "sum"),
    ).reset_index().sort_values("revenue", ascending=False).head(20)

    fig = bar_chart(
        products["Description"], products["revenue"],
        title="Top 20 Products", x_label="Product", y_label="Revenue (£)",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Revenue distribution
    st.subheader("Order Value Distribution")
    order_totals = valid.groupby("InvoiceNo")["Revenue"].sum()
    st.plotly_chart(
        histogram(order_totals, nbins=50, title="Order Value Distribution",
                  labels={"value": "Order Value (£)"}),
        use_container_width=True,
    )
