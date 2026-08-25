"""Data quality analysis for UCI Online Retail dataset."""
from pathlib import Path

import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.config import settings


def load_raw() -> pd.DataFrame:
    """Load raw data from Excel."""
    file_path = settings.RAW_DATA_DIR / "Online Retail.xlsx"
    return pd.read_excel(file_path, engine="openpyxl")


def schema_analysis(df: pd.DataFrame) -> dict:
    """Analyze schema and data types."""
    info = {
        "shape": list(df.shape),
        "columns": {},
    }
    for col in df.columns:
        info["columns"][col] = {
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "n_unique": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
        }
    return info


def missing_values_analysis(df: pd.DataFrame) -> dict:
    """Detailed missing values analysis."""
    missing = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            missing[col] = {
                "count": null_count,
                "pct": round(null_count / len(df) * 100, 2),
            }
    return missing


def duplicate_analysis(df: pd.DataFrame) -> dict:
    """Detect duplicate records."""
    exact_dupes = df.duplicated().sum()
    # Business key duplicates (excluding row identity)
    business_key_cols = ["InvoiceNo", "StockCode", "Description", "Quantity",
                         "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    business_dupes = df.duplicated(subset=business_key_cols).sum()

    return {
        "exact_duplicates": int(exact_dupes),
        "exact_duplicate_pct": round(exact_dupes / len(df) * 100, 2),
        "business_key_duplicates": int(business_dupes),
        "business_key_duplicate_pct": round(business_dupes / len(df) * 100, 2),
    }


def cancellation_analysis(df: pd.DataFrame) -> dict:
    """Detect cancellations and returns."""
    invoice_str = df["InvoiceNo"].astype(str)
    cancellations = df[invoice_str.str.startswith("C")]
    negative_qty = df[df["Quantity"] < 0]
    negative_price = df[df["UnitPrice"] < 0]

    return {
        "cancellations_count": len(cancellations),
        "cancellations_pct": round(len(cancellations) / len(df) * 100, 2),
        "negative_quantity_count": len(negative_qty),
        "negative_price_count": len(negative_price),
        "total_returns_approx": len(cancellations),
    }


def invalid_data_analysis(df: pd.DataFrame) -> dict:
    """Detect invalid quantities and prices."""
    zero_qty = (df["Quantity"] == 0).sum()
    zero_price = (df["UnitPrice"] == 0).sum()
    neg_qty = (df["Quantity"] < 0).sum()
    neg_price = (df["UnitPrice"] < 0).sum()

    # Description is placeholder in UCI dataset - many are "Manual" or dotcom
    description_stats = {
        "null_descriptions": int(df["Description"].isna().sum()),
        "top_descriptions": df["Description"].value_counts().head(10).to_dict(),
    }

    return {
        "zero_quantity": int(zero_qty),
        "zero_price": int(zero_price),
        "negative_quantity": int(neg_qty),
        "negative_price": int(neg_price),
        "description_stats": description_stats,
    }


def temporal_analysis(df: pd.DataFrame) -> dict:
    """Analyze date ranges and temporal patterns."""
    dates = pd.to_datetime(df["InvoiceDate"])

    return {
        "min_date": str(dates.min()),
        "max_date": str(dates.max()),
        "date_range_days": (dates.max() - dates.min()).days,
        "months_span": int((dates.max().to_period("M") - dates.min().to_period("M")).n) + 1,
    }


def customer_analysis(df: pd.DataFrame) -> dict:
    """Analyze customer coverage."""
    total_rows = len(df)
    has_customer = df["CustomerID"].notna().sum()
    no_customer = df["CustomerID"].isna().sum()

    customers_with_id = df[df["CustomerID"].notna()]
    n_customers = customers_with_id["CustomerID"].nunique()

    return {
        "total_rows": total_rows,
        "rows_with_customer_id": int(has_customer),
        "rows_without_customer_id": int(no_customer),
        "pct_without_customer_id": round(no_customer / total_rows * 100, 2),
        "unique_customers": int(n_customers),
    }


def country_analysis(df: pd.DataFrame) -> dict:
    """Analyze country distribution."""
    country_counts = df["Country"].value_counts()
    country_rev = df.groupby("Country").apply(
        lambda x: (x["Quantity"] * x["UnitPrice"]).sum(), include_groups=False
    ).sort_values(ascending=False)

    return {
        "n_countries": int(country_counts.shape[0]),
        "top_10_by_transactions": country_counts.head(10).to_dict(),
        "top_10_by_revenue": country_rev.head(10).to_dict(),
    }


def revenue_analysis(df: pd.DataFrame) -> dict:
    """Basic revenue statistics."""
    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    positive_rev = df[df["Revenue"] > 0]

    return {
        "total_revenue_all": round(float(df["Revenue"].sum()), 2),
        "total_revenue_positive": round(float(positive_rev["Revenue"].sum()), 2),
        "mean_revenue_per_line": round(float(df["Revenue"].mean()), 2),
        "median_revenue_per_line": round(float(df["Revenue"].median()), 2),
    }


def stockcode_analysis(df: pd.DataFrame) -> dict:
    """Analyze product codes."""
    codes = df["StockCode"].astype(str)
    # Non-standard codes (e.g. POST, DOT, M, etc.)
    non_numeric = codes[~codes.str.match(r"^\d+$")]

    return {
        "unique_stock_codes": int(df["StockCode"].nunique()),
        "non_standard_codes": non_numeric.value_counts().head(15).to_dict(),
    }


def run_full_analysis(output_dir: Path | None = None) -> dict:
    """Run complete data quality analysis."""
    print("Loading data...")
    df = load_raw()
    print(f"Loaded {len(df):,} rows\n")

    print("1. Schema analysis...")
    schema = schema_analysis(df)

    print("2. Missing values...")
    missing = missing_values_analysis(df)

    print("3. Duplicates...")
    dupes = duplicate_analysis(df)

    print("4. Cancellations/returns...")
    cancellations = cancellation_analysis(df)

    print("5. Invalid data...")
    invalid = invalid_data_analysis(df)

    print("6. Temporal analysis...")
    temporal = temporal_analysis(df)

    print("7. Customer analysis...")
    customers = customer_analysis(df)

    print("8. Country analysis...")
    countries = country_analysis(df)

    print("9. Revenue analysis...")
    revenue = revenue_analysis(df)

    print("10. Stock code analysis...")
    stockcodes = stockcode_analysis(df)

    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset": "UCI Online Retail",
        "schema": schema,
        "missing_values": missing,
        "duplicates": dupes,
        "cancellations": cancellations,
        "invalid_data": invalid,
        "temporal": temporal,
        "customers": customers,
        "countries": countries,
        "revenue": revenue,
        "stock_codes": stockcodes,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "data_quality_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved: {report_path}")

    return report


if __name__ == "__main__":
    report = run_full_analysis(settings.PROCESSED_DATA_DIR)

    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)
    print(f"Rows: {report['schema']['shape'][0]:,}")
    print(f"Columns: {report['schema']['shape'][1]}")
    print(f"Date range: {report['temporal']['min_date']} to {report['temporal']['max_date']}")
    print(f"Unique customers: {report['customers']['unique_customers']:,}")
    print(f"Unique products: {report['stock_codes']['unique_stock_codes']:,}")
    print(f"Countries: {report['countries']['n_countries']}")
    print(f"Cancellations: {report['cancellations']['cancellations_count']:,} ({report['cancellations']['cancellations_pct']}%)")
    print(f"Rows without CustomerID: {report['customers']['rows_without_customer_id']:,} ({report['customers']['pct_without_customer_id']}%)")
