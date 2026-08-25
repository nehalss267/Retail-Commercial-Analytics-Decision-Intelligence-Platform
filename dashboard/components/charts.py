"""Reusable chart components for the dashboard."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def revenue_trend_bar(monthly: pd.DataFrame, x: str = "YearMonth", y: str = "revenue"):
    """Bar chart of monthly revenue."""
    fig = px.bar(
        monthly, x=x, y=y,
        title="Revenue by Month",
        labels={y: "Revenue (£)", x: "Month"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def country_bar(country_rev: pd.Series, top_n: int = 10):
    """Bar chart of revenue by country."""
    data = country_rev.head(top_n)
    fig = px.bar(
        x=data.index, y=data.values,
        labels={"x": "Country", "y": "Revenue (£)"},
        title=f"Top {top_n} Countries by Revenue",
        color_discrete_sequence=["#EF553B"],
    )
    return fig


def segment_pie(values: pd.Series, names: pd.Series, title: str = "Distribution"):
    """Pie chart for segment distribution."""
    fig = px.pie(values=values, names=names, title=title)
    return fig


def rfm_scatter(rfm: pd.DataFrame, x: str = "recency", y: str = "monetary",
                color: str = "segment", size: str = "frequency"):
    """Scatter plot for RFM analysis."""
    fig = px.scatter(
        rfm, x=x, y=y, color=color, size=size,
        hover_data=["CustomerID"],
        title="Customer Segments (RFM)",
    )
    return fig


def forecast_line(forecast: pd.DataFrame, models: list[str], title: str = "30-Day Forecast"):
    """Line chart comparing forecast models."""
    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    for i, model in enumerate(models):
        if model in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast["date"], y=forecast[model],
                name=model.replace("_", " ").title(),
                mode="lines",
                line=dict(color=colors[i % len(colors)]),
            ))
    fig.update_layout(title=title, yaxis_title="Revenue (£)")
    return fig


def histogram(values: pd.Series, nbins: int = 50, title: str = "Distribution",
              labels: dict | None = None):
    """Histogram chart."""
    fig = px.histogram(values, nbins=nbins, title=title, labels=labels or {})
    return fig


def bar_chart(x, y, title: str = "", x_label: str = "", y_label: str = "",
              color: str | None = None):
    """Generic bar chart."""
    df = pd.DataFrame({x.name if hasattr(x, 'name') else "x": x,
                       y.name if hasattr(y, 'name') else "y": y})
    fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title,
                 labels={df.columns[0]: x_label, df.columns[1]: y_label},
                 color=color)
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, title: str = "",
               labels: dict | None = None):
    """Generic line chart."""
    fig = px.line(df, x=x, y=y, title=title, labels=labels or {})
    return fig
