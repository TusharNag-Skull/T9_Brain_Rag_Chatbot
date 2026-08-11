"""Simple embedding service for Think9 Brain.

Converts text chunks into numerical vectors using a
local HuggingFace sentence-transformers model.
"""

import sys
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

# Allow importing from src/ when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.document_loader import load_documents
from preprocessing.chunker import chunk_documents


def create_embedding_model():
    """Create and return the HuggingFace embedding model."""
    # all-MiniLM-L6-v2 is small, fast, and good enough for a prototype
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


def embed_documents(chunks, embedding_model):
    """Generate an embedding vector for each chunk's text.

    Args:
        chunks: list of dicts from chunk_documents()
        embedding_model: model returned by create_embedding_model()

    Returns:
        A list of embedding vectors (one list of floats per chunk).
    """
    # Pull only the text field from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # LangChain turns each text into a numeric vector
    embeddings = embedding_model.embed_documents(texts)
    return embeddings


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    embedding_model = create_embedding_model()
    embeddings = embed_documents(chunks, embedding_model)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding vector length: {len(embeddings[0])}")
    print(f"First 5 values of the first embedding: {embeddings[0][:5]}")