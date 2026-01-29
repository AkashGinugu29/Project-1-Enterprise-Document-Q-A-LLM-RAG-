# Enterprise Document Q&A (LLM + RAG)

A retrieval-augmented generation (RAG) system that enables natural-language Q&A over enterprise documents (PDF/TXT).
Built using FastAPI, FAISS vector search, and OpenAI embeddings and LLMs. Answers are grounded in retrieved source content.

## Features
- PDF and TXT document ingestion
- Chunking and embedding generation
- FAISS-based vector search for retrieval
- FastAPI inference endpoint (`/query`)
- Citation-backed answers to reduce hallucinations
- Docker-ready for consistent deployment

## Architecture Overview
Document Ingestion → Chunking → Embeddings → FAISS Index  
User Query → Query Embedding → Top-K Retrieval → LLM Answer with Citations

## Getting Started

### 1. Add documents
Place PDF or TXT files in:
data/documents/

### 2. Configure environment
Create a `.env` file from `.env.example` and add your OpenAI API key.

### 3. Build the vector index
python src/ingest.py

### 4. Run the API
uvicorn src.app:app --reload

### 5. Query the system
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","top_k":5}'

## Notes
- Embeddings are normalized for cosine similarity.
- The system only answers using retrieved context and returns citations.
- This project demonstrates end-to-end LLM system design, deployment, and retrieval grounding.
```

