"""RAG Retrieval — Query the knowledge base with graceful fallback."""
import chromadb

from src.config import settings
from src.ai.rag.ingestion import get_client, COLLECTION_NAME


def retrieve_context(query: str, n_results: int = 3,
                     collection_name: str = COLLECTION_NAME) -> list[dict]:
    """Retrieve relevant context from the knowledge base.

    Returns empty list if embeddings are not available or collection is empty.
    """
    try:
        from src.ai.rag.embeddings import get_embeddings
        embeddings = get_embeddings()
        if embeddings is None:
            return []
    except Exception:
        return []

    client = get_client()

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return []

    try:
        query_embedding = embeddings.embed_query(query)
    except Exception:
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    contexts = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            contexts.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "relevance": round(1 - dist, 4),
            })

    return contexts


def format_context_for_prompt(contexts: list[dict]) -> str:
    """Format retrieved context for inclusion in a prompt."""
    if not contexts:
        return "No relevant knowledge base documents found."

    parts = []
    for i, ctx in enumerate(contexts, 1):
        source = ctx["source"].split("/")[-1]
        parts.append(f"[Source: {source}] {ctx['text']}")

    return "\n\n---\n\n".join(parts)
