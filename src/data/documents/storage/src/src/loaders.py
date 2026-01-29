import os
from typing import List, Dict
from pypdf import PdfReader

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)

def load_documents(folder: str) -> List[Dict]:
    documents = []

    if not os.path.exists(folder):
        return documents

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if os.path.isdir(filepath):
            continue

        if filename.lower().endswith(".pdf"):
            text = load_pdf(filepath)
        elif filename.lower().endswith(".txt"):
            text = load_txt(filepath)
        else:
            continue

        documents.append({
            "source": filename,
            "text": text
        })

    return documents
