import time
from turtle import delay
from typing import List, Dict, Any
 
import requests
 
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL
 
def verify_ollama() -> bool:
    try: 
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        available = any(EMBEDDING_MODEL in m for m in models)
 
        if available:
            print(f"Ollama OK model '{EMBEDDING_MODEL}' available.")
        else:
            print(f"  Ollama running, but '{EMBEDDING_MODEL}' not found.")
            print(f"   Run: ollama pull {EMBEDDING_MODEL}")
 
        return available
 
    except requests.exceptions.ConnectionError:
        print(" Ollama is not running!")
        print("   Start with: ollama serve")
        return False
    
def generate_embedding(text: str) -> List[float]:
    """
    Sends a text to Ollama and receives back the embedding vector.
 
    Args:
        text: Text of the chunk to be vectorized
 
    Returns:
        List of floats representing the embedding
    """
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }
 
    ans = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json=payload,
        timeout=60
    )
    ans.raise_for_status()
    return ans.json()["embedding"]

def generate_embeddings_batch(chunks: List[Dict[str, Any]], logger) -> List[Dict[str, Any]]:
    total = len(chunks)
    print(f"Generating embeddings for {total} chunk(s) with '{EMBEDDING_MODEL}'...")
    print("   (this may take a few minutes on the first run)\n")
 
    chunks_with_embedding = []
    errors = 0
 
    for i, chunk in enumerate(chunks):
        try:
            embedding = generate_embedding(chunk["text"])
            chunks_with_embedding.append({**chunk, "embedding": embedding})
 
            # Progress every 10 chunks
            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                print(f"  [{i+1:>4}/{total}] {pct:.1f}% completed", end="\r")
 
            if delay > 0:
                time.sleep(delay)
 
        except Exception as e:
            print(f"\n  Error in chunk {i} ({chunk.get('source', '?')}): {e}")
            errors += 1
 
    print(f"\n\n Embeddings generated: {len(chunks_with_embedding)} | Errors: {errors}\n")
    return chunks_with_embedding