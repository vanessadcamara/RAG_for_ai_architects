import hashlib
from typing import List, Dict, Any
 
import chromadb
from chromadb.config import Settings
 
from config import CHROMA_DIR, COLLECTION_NAME
 
def get_collection(
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME
) -> chromadb.Collection:
    """
    Creates or opens a persistent collection in ChromaDB.
    The `persist_directory` parameter ensures that vectors are saved to disk,
    so you don't have to re-index every time you run the pipeline.

    Args:
        chroma_dir: Directory where ChromaDB will store data
        collection_name: Name of the collection to create/open
    Returns:
        A ready-to-use ChromaDB Collection object
    """
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False) 
    )
 
    # get_or_create
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  
    )
 
    count = collection.count()
    print(f"Collection '{collection_name}' opened | {count} document(s) existingly indexed.\n")
    return collection
 
def generate_id(font: str, chunk_index: int) -> str:
    """
    Generates a unique and deterministic ID for each chunk.
    Using a hash prevents duplicate IDs even with long file names.
    """
    content = f"{font}::{chunk_index}"
    return hashlib.md5(content.encode()).hexdigest()
 
def index_chunks(
    chunks_with_embedding: List[Dict[str, Any]],
    collection: chromadb.Collection,
    batch_size: int = 100
) -> int:
    """
    Inserts chunks into ChromaDB in batches for efficiency.
    Deduplication strategy:
      The ID is generated from (source, chunk_index). If a chunk with
      the same ID already exists in Chroma, the `upsert` simply updates,
      without creating a duplicate.
      
    Args:
        chunks_with_embedding: List of chunks with 'embedding' key (step 2)
        collection:            Opened ChromaDB collection
        batch_size:            How many chunks to insert at a time
 
    Returns:
        Total number of chunks inserted/updated
    """
    total = len(chunks_with_embedding)
    inserted = 0
 
    print(f"Indexing {total} chunk(s) into ChromaDB (batch_size={batch_size})...")
 
    for i in range(0, total, batch_size):
        batch = chunks_with_embedding[i : i + batch_size]
 
        ids         = [generate_id(c["source"], c["chunk_index"]) for c in batch]
        embeddings  = [c["embedding"] for c in batch]
        documents   = [c["text"] for c in batch]
        metadatas   = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch
        ]
 
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
 
        inserted += len(batch)
        pct = inserted / total * 100
        print(f"  [{inserted:>4}/{total}] {pct:.1f}%", end="\r")
 
    print(f"Indexing completed! Total in collection: {collection.count()}\n")
    return inserted
 
def list_fonts(collection: chromadb.Collection) -> List[str]:
    results = collection.get(include=["metadatas"])
    
    metadatas = results["metadatas"]
    
    fonts = []
    
    for metadata in metadatas:
        font = metadata.get("source")
        if font:
            fonts.append(font)
    
    unique_fonts = list(set(fonts))
    sorted_fonts = sorted(unique_fonts)
    
    return sorted_fonts