"""
Build a ChromaDB vector store from Deloitte industry outlook PDFs.

Usage:
    python src/build_vectordb.py

Requires OPENAI_API_KEY in .env for embedding generation.
"""

import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# --- Config ---
ROOT_DIR = Path(__file__).parent.parent
REPORTS_DIR = ROOT_DIR / "data" / "reports"
CHROMA_DIR = ROOT_DIR / "data" / "vectordb"
COLLECTION_NAME = "industry_reports"

# Map PDF filenames to industry tags
INDUSTRY_MAP = {
    "2025 Manufacturing": "Manufacturing",
    "2025 Renewable Energy": "Renewable Energy",
    "2025 technology": "Technology",
    "2026 Chemical": "Chemical",
    "2026 Engineering and Construction": "Engineering & Construction",
    "2026 Oil and Gas": "Oil & Gas",
    "2026 Power and Utilities": "Power & Utilities",
    "2026 Renewable Energy": "Renewable Energy",
}


def detect_industry(filename: str) -> str:
    """Match filename to an industry tag."""
    for key, industry in INDUSTRY_MAP.items():
        if key.lower() in filename.lower():
            return industry
    return "General"


def extract_chunks(pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Extract text from PDF and split into overlapping chunks."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Clean up text
    full_text = re.sub(r'\n+', '\n', full_text)
    full_text = re.sub(r' +', ' ', full_text)

    # Split into chunks
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk_text = full_text[start:end].strip()
        if len(chunk_text) > 50:  # Skip very short chunks
            chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def get_embeddings(client: OpenAI, texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """Generate embeddings using OpenAI API in batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def main():
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        return

    openai_client = OpenAI(api_key=api_key)

    # Initialize ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Process each PDF
    pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")

    total_chunks = 0
    for pdf_path in pdf_files:
        industry = detect_industry(pdf_path.name)
        print(f"Processing: {pdf_path.name}")
        print(f"  Industry: {industry}")

        chunks = extract_chunks(str(pdf_path))
        print(f"  Chunks: {len(chunks)}")

        if not chunks:
            continue

        # Generate embeddings
        embeddings = get_embeddings(openai_client, chunks)

        # Add to ChromaDB
        ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "industry": industry,
                "source": pdf_path.name,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"  Added to vector store\n")

    print(f"Done! Total chunks: {total_chunks}")
    print(f"Vector store saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
