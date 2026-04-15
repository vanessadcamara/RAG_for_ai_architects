
from typing import List, Dict, Any
 
import chromadb
 
from config import TOP_K, SIMILARITY_THRESHOLD
from src.embeddings import generate_embedding, verify_ollama

def retrieve_chunks(
    user_query: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Execute semantic search in ChromaDB for a user question.
 
    Args:
        user_query:    User's question text
        collection:    Opened ChromaDB collection
        top_k:         Maximum number of chunks to retrieve
        threshold:     Maximum distance to consider relevant
 
    Returns:
        List of dicts with {text, source, chunk_index, distance, relevancia}
        Ordered from most to least relevant.
    """
    if collection.count() == 0:
        raise ValueError("The collection is empty. Run the indexing pipeline first.")
 
    print(f"Searching...")
 
    # Convert the user's question into the same vector space as the chunks
    query_vector = generate_embedding(user_query)

    # Query ChromaDB
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),  # don't request more than exists
        include=["documents", "metadatas", "distances"]
    )
 
    # Process the results
    relevant_chunks = []
    documents  = result["documents"][0]
    metadatas   = result["metadatas"][0]
    distances  = result["distances"][0]
 
    for text, meta, dist in zip(documents, metadatas, distances):
        if dist <= threshold:  # filter out less relevant chunks
            relevant_chunks.append({
                "text":        text,
                "source":      meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance":    round(dist, 4),
                "relevancia":  round(1 - dist, 4)  # inverted for human readability
            })
 
    print(f"   → {len(relevant_chunks)} chunk(s) relevante(s) encontrado(s) "
          f"(threshold={threshold})\n")
 
    return relevant_chunks
 
 
def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats the retrieved chunks into a context block ready
    to be inserted into the LLM prompt.
 
    Includes the source of each chunk for traceability.
    """
    if not chunks:
        return "No relevant context found."
 
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['source']} | relevance: {chunk['relevancia']:.2f}]\n"
            f"{chunk['text']}"
        )
 
    return "\n\n---\n\n".join(parts)