import os
import re
from pathlib import Path
from typing import List, Dict, Any
 
import pdfplumber  
 
from config import DADOS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def read_txt(path_file: Path) -> str:
    with open(path_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
 
def read_pdf(path_file: Path) -> str:
    ''' Reads a PDF file and extracts text from each page, 
    returning a single string with page breaks. '''

    text_pages = []
    with pdfplumber.open(path_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:  
                text_pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(text_pages)
 
def load_documents(folder: str = DADOS_DIR) -> List[Dict[str, Any]]:
    """
    Returns :
        List of dicts with:
          - 'text': raw content of the file
          - 'font': file name (used as metadata in the vector)
          - 'type': 'txt' or 'pdf'
    """
    documents = []
    folder_path = Path(folder)
 
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder}' not found.")
 
    files = list(folder_path.glob("**/*.txt")) + list(folder_path.glob("**/*.pdf"))
 
    if not files:
        raise ValueError(f"No .txt or .pdf files found in '{folder}'.")
 
    print(f"{len(files)} file(s) found in '{folder}'")
 
    for file in sorted(files):
        ext = file.suffix.lower()
        print(f"  Reading: {file.name}", end=" ")
 
        try:
            if ext == ".txt":
                text = read_txt(file)
            elif ext == ".pdf":
                text = read_pdf(file)
            else:
                print("(ignored)")
                continue
 
            if not text.strip():
                print("(empty skipped)")
                continue
 
            documents.append({
                "text": text,
                "font": file.name,
                "type": ext.lstrip(".")
            })
            print(f"({len(text):,} chars)")
 
        except Exception as e:
            print(f"Error: {e}")
 
    print(f"\n✅ {len(documents)} document(s) successfully loaded.\n")
    return documents

# Chunking step

def clean_text(text: str) -> str:
    ''' Cleans text by removing excessive whitespace and newlines. '''
    text = re.sub(r'\r\n', '\n', text)           
    text = re.sub(r'\n{3,}', '\n\n', text)       
    text = re.sub(r'[ \t]{2,}', ' ', text)       
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', '', text)  # ctrl chars
    return text.strip()

def create_chunks(
    text: str,
    font: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Divides text into overlapping chunks for embedding. (slidding window)
 
    Strategy:
      Advances `chunk_size - chunk_overlap` characters at each iteration.
      The overlap preserves context at the boundary between chunks.
 
    Args:
        text:         Cleaned text of the document
        font:         File name (for metadata)
        chunk_size:    Maximum size of each chunk (in chars)
        chunk_overlap: Number of chars from the previous chunk repeated in the next
 
    Returns:
        List of dicts with {text, source, chunk_index}
    """
    text = clean_text(text)
    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    idx = 0
 
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
 
        # Try to break at the last space/newline to avoid cutting words
        if end < len(text):
            last_space = max(chunk.rfind('\n'), chunk.rfind(' '))
            if last_space > chunk_size // 2:  # only adjust if something reasonable is found
                chunk = chunk[:last_space]
 
        if chunk.strip():  # ignore empty chunks
            chunks.append({
                "text": chunk.strip(),
                "source": font,
                "chunk_index": idx
            })
            idx += 1
 
        start += step
 
    return chunks
 

def process_documents(folder: str = DADOS_DIR) -> List[Dict[str, Any]]:
    """
    Complete ingestion pipeline:
      1. Loads documents from the folder
      2. Generates chunks for each document
      3. Returns a unified list of chunks with metadata
 
    Usage:
        from src.ingestion import processar_documentos
        chunks = process_documents()
    """
    documents = load_documents(folder)
    all_chunks = []
 
    for doc in documents:
        chunks = create_chunks(
            text=doc["text"],
            font=doc["font"]
        )
        all_chunks.extend(chunks)
        print(f"  {doc['font']} → {len(chunks)} chunk(s)")
 
    print(f"\n Total de chunks gerados: {len(all_chunks)}\n")
    return all_chunks