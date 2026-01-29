from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import retrieve, answer

app = FastAPI(
    title="Enterprise Document Q&A (RAG)",
    description="Retrieval-Augmented Generation API for document-based Q&A",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        contexts = retrieve(request.question, k=request.top_k)

        if not contexts:
            return {
                "answer": "I don't know. No relevant context found.",
                "contexts": [],
            }

        return answer(request.question, contexts)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
