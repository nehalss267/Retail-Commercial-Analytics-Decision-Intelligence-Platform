"""RAG Ingestion — Load business knowledge into ChromaDB."""
import chromadb
from pathlib import Path

from src.config import settings
from src.ai.rag.embeddings import get_embeddings


CHROMA_DIR = settings.BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "retailai_knowledge"


def get_client() -> chromadb.PersistentClient:
    """Get ChromaDB persistent client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def ingest_documents(documents: list[dict], collection_name: str = COLLECTION_NAME):
    """Ingest documents into ChromaDB.

    Each document dict should have: id, text, metadata
    """
    client = get_client()
    embeddings = get_embeddings()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]

    # Embed texts
    embedded = embeddings.embed_documents(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embedded,
        metadatas=metadatas,
    )

    return collection.count()


def ingest_from_files(file_paths: list[Path], collection_name: str = COLLECTION_NAME):
    """Ingest markdown/text files into the knowledge base."""
    documents = []
    for path in file_paths:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        # Split into chunks
        chunks = _split_text(content, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{path.stem}_{i}",
                "text": chunk,
                "metadata": {"source": str(path), "chunk": i},
            })

    return ingest_documents(documents, collection_name)


def _split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def initialize_knowledge_base():
    """Initialize the knowledge base from project documentation."""
    doc_dir = settings.BASE_DIR / "docs"
    files = [
        doc_dir / "architecture.md",
        doc_dir / "roadmap.md",
    ]

    # Also add data dictionary if available
    dd_path = settings.PROCESSED_DATA_DIR / "data_dictionary.md"
    if dd_path.exists():
        files.append(dd_path)

    count = ingest_from_files([f for f in files if f.exists()])
    print(f"Ingested {count} document chunks into knowledge base")
    return count


if __name__ == "__main__":
    initialize_knowledge_base()
