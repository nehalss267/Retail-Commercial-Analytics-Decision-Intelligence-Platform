"""I/O Utilities — Consistent data loading and saving."""
import pandas as pd
import json
from pathlib import Path

from src.config import settings


def load_parquet(filename: str, subdir: str = "processed") -> pd.DataFrame:
    """Load a parquet file from the data directory."""
    path = settings.BASE_DIR / "data" / subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, filename: str, subdir: str = "processed"):
    """Save a dataframe to parquet."""
    path = settings.BASE_DIR / "data" / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_json(filename: str, subdir: str = "processed") -> dict:
    """Load a JSON file."""
    path = settings.BASE_DIR / "data" / subdir / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json(data: dict, filename: str, subdir: str = "processed"):
    """Save data to JSON."""
    path = settings.BASE_DIR / "data" / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
