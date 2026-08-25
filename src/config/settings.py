from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://retailai:retailai_secret@localhost:5432/retailai")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "data" / "features"
SAMPLES_DIR = BASE_DIR / "data" / "samples"
