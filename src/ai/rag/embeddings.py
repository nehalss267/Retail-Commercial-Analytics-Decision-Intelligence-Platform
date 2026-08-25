"""RAG Embeddings — Embedding model configuration with fallback."""
from pathlib import Path

from src.config import settings


def get_embeddings():
    """Get the embedding model for RAG. Falls back to None if unavailable."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder=str(settings.BASE_DIR / "data" / "embeddings_cache"),
        )
    except Exception:
        return None
