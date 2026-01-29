import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "documents")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.json") 
