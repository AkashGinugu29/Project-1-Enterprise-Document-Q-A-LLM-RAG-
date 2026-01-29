import json
from typing import List, Dict, Tuple

import numpy as np
import faiss
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_EMBED_MODEL,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
)

client = OpenAI(api_key=OPENAI_API_KEY)

def load_store() -> Tuple[faiss.Index, List[Dict]]:
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def embed_query(question: str) -> np.ndarray:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[question])
    v = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(v)
    return v

def retrieve(question: str, k: int = 5) -> List[Dict]:
    index, chunks = load_store()
    qv = embed_query(question)
    scores, ids = index.search(qv, k)

    results = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx == -1:
            continue
        c = chunks[int(idx)]
        results.append({
            "rank": rank,
            "score": float(score),
            "source": c["source"],
            "chunk_id": c["id"],
            "text": c["text"],
        })
    return results

def answer(question: str, contexts: List[Dict]) -> Dict:
    """
    Answer using ONLY retrieved context. Include citations like [1], [2].
    """
    context_block = "\n\n".join(
        [f"[{i+1}] ({c['source']} | {c['chunk_id']})\n{c['text']}"
         for i, c in enumerate(contexts)]
    )

    system_msg = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Always include citations like [1], [2] that refer to the numbered context chunks."
    )

    user_msg = f"Question:\n{question}\n\nContext:\n{context_block}"

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    return {
        "answer": resp.choices[0].message.content,
        "contexts": contexts,
    }
