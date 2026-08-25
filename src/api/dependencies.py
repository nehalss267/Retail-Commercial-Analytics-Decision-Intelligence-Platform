"""FastAPI dependencies — shared utilities for route modules."""
import json
from pathlib import Path

from src.config import settings


DATA_DIR = settings.PROCESSED_DATA_DIR


def load_report(name: str) -> dict:
    """Load a JSON report from the processed data directory."""
    path = DATA_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}
