import os
import json
from typing import List, Dict

import numpy as np
import faiss
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    DATA_DIR,
    STORAGE_DIR,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
)
from loaders import load_documents

client = OpenAI(api_key=OPENAI_API_KEY)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Simple character-based chunking (good starter approach).
    """
    text = text.replace("\r", "\n")
    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    vectors = []
    batch_size = 96

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])

    arr = np.array(vectors, dtype="float32")
    return arr

def main():
    os.makedirs(STORAGE_DIR, exist_ok=True)

    docs = load_documents(DATA_DIR)
    all_chunks: List[Dict] = []

    for d in docs:
        doc_chunks = chunk_text(d["text"])
        for idx, ch in enumerate(doc_chunks):
            all_chunks.append({
                "id": f'{d["source"]}::chunk_{idx}',
                "source": d["source"],
                "text": ch
            })

    if not all_chunks:
        raise RuntimeError("No documents found. Add PDF/TXT files to data/documents/ first.")

    texts = [c["text"] for c in all_chunks]
    vectors = embed_texts(texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Ingested {len(docs)} docs")
    print(f"✅ Created {len(all_chunks)} chunks")
    print(f"✅ Saved index -> {FAISS_INDEX_PATH}")
    print(f"✅ Saved chunks -> {CHUNKS_PATH}")

if __name__ == "__main__":
    main()
