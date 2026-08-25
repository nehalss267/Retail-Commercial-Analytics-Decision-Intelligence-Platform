from pathlib import Path

import pandas as pd

from src.config.settings import RAW_DATA_DIR


def load_raw_data() -> pd.DataFrame:
    """Load the UCI Online Retail dataset from Excel."""
    file_path = RAW_DATA_DIR / "Online Retail.xlsx"
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data not found: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def save_raw_parquet(df: pd.DataFrame) -> None:
    """Save raw data as parquet for faster future loads."""
    out_path = RAW_DATA_DIR / "online_retail.parquet"
    df.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"Saved parquet: {out_path}")


if __name__ == "__main__":
    df = load_raw_data()
    save_raw_parquet(df)
    print("\nSchema:")
    print(df.dtypes)
    print(f"\nFirst 5 rows:")
    print(df.head())
