"""Cleaning pipeline for UCI Online Retail dataset.

Steps:
1. Load raw data
2. Convert types (InvoiceNo → string, CustomerID → nullable Int64)
3. Derive Revenue = Quantity × UnitPrice
4. Flag cancellations (InvoiceNo starts with 'C')
5. Flag returns (Quantity < 0)
6. Separate clean transactions from cancellations/returns
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import settings


def load_raw() -> pd.DataFrame:
    """Load raw Excel data."""
    path = settings.RAW_DATA_DIR / "Online Retail.xlsx"
    return pd.read_excel(path, engine="openpyxl")


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column types."""
    df = df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["Country"] = df["Country"].astype(str).str.strip()
    return df


def derive_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add business flags."""
    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].str.startswith("C", na=False)
    df["IsReturn"] = df["Quantity"] < 0
    df["HasCustomerID"] = df["CustomerID"].notna()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"  Removed {removed:,} exact duplicates")
    return df


def clean_pipeline() -> pd.DataFrame:
    """Run full cleaning pipeline."""
    print("Step 1: Loading raw data...")
    df = load_raw()
    print(f"  Loaded {len(df):,} rows")

    print("Step 2: Converting types...")
    df = convert_types(df)

    print("Step 3: Deriving flags...")
    df = derive_flags(df)

    print("Step 4: Removing duplicates...")
    df = remove_duplicates(df)

    print("Step 5: Saving cleaned data...")
    out_path = settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Summary
    cancellations = df["IsCancellation"].sum()
    returns = df["IsReturn"].sum()
    no_customer = df["HasCustomerID"].sum() == False

    print(f"\n=== CLEANED DATA SUMMARY ===")
    print(f"Rows: {len(df):,}")
    print(f"Cancellations: {cancellations:,}")
    print(f"Returns: {returns:,}")
    print(f"Without CustomerID: {no_customer:,}")
    print(f"Revenue (total): £{df['Revenue'].sum():,.2f}")

    return df


if __name__ == "__main__":
    clean_pipeline()
