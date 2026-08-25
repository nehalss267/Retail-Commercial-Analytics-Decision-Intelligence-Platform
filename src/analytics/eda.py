"""Exploratory Data Analysis for UCI Online Retail dataset.

Works directly from cleaned parquet (no database required).
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.config import settings


def load_cleaned() -> pd.DataFrame:
    """Load cleaned data from parquet."""
    path = settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet"
    return pd.read_parquet(path)


def revenue_trends(df: pd.DataFrame) -> dict:
    """Monthly revenue trends."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]
    valid = valid.copy()
    valid["YearMonth"] = valid["InvoiceDate"].dt.to_period("M")

    monthly = valid.groupby("YearMonth").agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
        avg_order_value=("Revenue", "mean"),
    ).reset_index()

    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    # Month-over-month growth
    monthly["revenue_growth_pct"] = monthly["revenue"].pct_change() * 100

    return {
        "monthly": monthly.to_dict(orient="records"),
        "summary": {
            "total_revenue": round(float(valid["Revenue"].sum()), 2),
            "total_orders": int(valid["InvoiceNo"].nunique()),
            "total_customers": int(valid["CustomerID"].nunique()),
            "avg_monthly_revenue": round(float(monthly["revenue"].mean()), 2),
            "best_month": str(monthly.loc[monthly["revenue"].idxmax(), "YearMonth"]),
            "worst_month": str(monthly.loc[monthly["revenue"].idxmin(), "YearMonth"]),
        }
    }


def customer_distribution(df: pd.DataFrame) -> dict:
    """Customer spending distribution."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]
    customer_spend = valid.groupby("CustomerID")["Revenue"].sum()

    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    spend_percentiles = {f"p{int(p*100)}": round(float(customer_spend.quantile(p)), 2) for p in percentiles}

    return {
        "n_customers": len(customer_spend),
        "total_revenue": round(float(customer_spend.sum()), 2),
        "mean_spend": round(float(customer_spend.mean()), 2),
        "median_spend": round(float(customer_spend.median()), 2),
        "std_spend": round(float(customer_spend.std()), 2),
        "spend_percentiles": spend_percentiles,
        "top_10_customers_revenue_pct": round(float(customer_spend.nlargest(10).sum() / customer_spend.sum() * 100), 2),
    }


def product_analysis(df: pd.DataFrame) -> dict:
    """Product popularity and revenue analysis."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    product_revenue = valid.groupby(["StockCode", "Description"]).agg(
        revenue=("Revenue", "sum"),
        quantity=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"),
    ).reset_index().sort_values("revenue", ascending=False)

    total_revenue = product_revenue["revenue"].sum()

    return {
        "n_products": len(product_revenue),
        "top_10_by_revenue": product_revenue.head(10)[
            ["StockCode", "Description", "revenue", "quantity", "orders"]
        ].to_dict(orient="records"),
        "top_10_by_quantity": product_revenue.nlargest(10, "quantity")[
            ["StockCode", "Description", "revenue", "quantity"]
        ].to_dict(orient="records"),
        "pct_revenue_top_10_products": round(
            float(product_revenue.head(10)["revenue"].sum() / total_revenue * 100), 2
        ),
    }


def country_analysis(df: pd.DataFrame) -> dict:
    """Country-level analysis."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    country_stats = valid.groupby("Country").agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
        avg_order_value=("Revenue", "mean"),
    ).reset_index().sort_values("revenue", ascending=False)

    total_revenue = country_stats["revenue"].sum()
    country_stats["revenue_pct"] = round(country_stats["revenue"] / total_revenue * 100, 2)

    return {
        "n_countries": len(country_stats),
        "top_15": country_stats.head(15).to_dict(orient="records"),
        "uk_dominance_pct": round(float(
            country_stats[country_stats["Country"] == "United Kingdom"]["revenue"].sum() / total_revenue * 100
        ), 2),
    }


def pareto_analysis(df: pd.DataFrame) -> dict:
    """Pareto analysis — what % of customers generate what % of revenue."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]
    customer_spend = valid.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)
    total = customer_spend.sum()

    cumulative = customer_spend.cumsum() / total * 100
    n_customers = len(customer_spend)

    # Find % of customers that generate 80% of revenue
    pct_80 = float(cumulative[cumulative <= 80].count() / n_customers * 100)

    # Percentile buckets
    buckets = []
    for pct in [10, 20, 30, 40, 50]:
        n = int(n_customers * pct / 100)
        rev = customer_spend.head(n).sum()
        buckets.append({
            "pct_customers": pct,
            "revenue_pct": round(float(rev / total * 100), 2),
        })

    return {
        "pct_customers_generating_80pct_revenue": round(pct_80, 2),
        "revenue_concentration": buckets,
    }


def repeat_purchase_analysis(df: pd.DataFrame) -> dict:
    """Repeat purchase behavior."""
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]
    customer_orders = valid.groupby("CustomerID")["InvoiceNo"].nunique()

    one_time = (customer_orders == 1).sum()
    repeat = (customer_orders > 1).sum()

    # Repeat frequency distribution
    freq_dist = customer_orders.value_counts().sort_index().head(20)

    return {
        "total_customers": len(customer_orders),
        "one_time_buyers": int(one_time),
        "repeat_buyers": int(repeat),
        "one_time_pct": round(one_time / len(customer_orders) * 100, 2),
        "repeat_pct": round(repeat / len(customer_orders) * 100, 2),
        "avg_orders_per_customer": round(float(customer_orders.mean()), 2),
        "max_orders": int(customer_orders.max()),
        "frequency_distribution": {str(k): int(v) for k, v in freq_dist.items()},
    }


def run_eda() -> dict:
    """Run complete EDA."""
    print("Loading cleaned data...")
    df = load_cleaned()
    print(f"Loaded {len(df):,} rows\n")

    print("1. Revenue trends...")
    rev = revenue_trends(df)

    print("2. Customer distribution...")
    cust = customer_distribution(df)

    print("3. Product analysis...")
    prod = product_analysis(df)

    print("4. Country analysis...")
    country = country_analysis(df)

    print("5. Pareto analysis...")
    pareto = pareto_analysis(df)

    print("6. Repeat purchase analysis...")
    repeat = repeat_purchase_analysis(df)

    report = {
        "revenue_trends": rev,
        "customer_distribution": cust,
        "products": prod,
        "countries": country,
        "pareto": pareto,
        "repeat_purchase": repeat,
    }

    # Save report
    out_path = settings.PROCESSED_DATA_DIR / "eda_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"Total Revenue: £{rev['summary']['total_revenue']:,.2f}")
    print(f"Total Orders: {rev['summary']['total_orders']:,}")
    print(f"Total Customers: {rev['summary']['total_customers']:,}")
    print(f"Best Month: {rev['summary']['best_month']}")
    print(f"UK Revenue Dominance: {country['uk_dominance_pct']}%")
    print(f"Pareto (80% revenue from {pareto['pct_customers_generating_80pct_revenue']}% customers)")
    print(f"Repeat Buyers: {repeat['repeat_pct']}%")
    print(f"One-time Buyers: {repeat['one_time_pct']}%")

    return report


if __name__ == "__main__":
    run_eda()
