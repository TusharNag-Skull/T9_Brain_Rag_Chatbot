"""Simple Chroma vector store for Think9 Brain.

Takes text chunks + an embedding model and stores them
locally so we can run similarity search later.
"""

import sys
from pathlib import Path

import chromadb
from langchain_chroma import Chroma

# Allow importing from src/ when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.document_loader import load_documents
from preprocessing.chunker import chunk_documents
from embeddings.embedding_service import create_embedding_model


def create_vector_store(chunks, embedding_model):
    """Create or load a local Chroma vector store from text chunks.

    Args:
        chunks: list of dicts from chunk_documents()
        embedding_model: model from create_embedding_model()

    Returns:
        A Chroma vector store saved under data/vectorstore/
    """
    # Save the database inside the project
    project_root = Path(__file__).resolve().parents[2]
    persist_directory = str(project_root / "data" / "vectorstore")

    # Keep the persistent collection so existing vector store objects remain valid.
    client = chromadb.PersistentClient(path=persist_directory)
    existing_names = [collection.name for collection in client.list_collections()]

    # Check whether the collection already has data
    collection_is_ready = False
    if "think9_brain" in existing_names:
        existing_collection = client.get_collection("think9_brain")
        if existing_collection.count() > 0:
            collection_is_ready = True

    # Connect to the persistent collection (creates it if missing)
    vector_store = Chroma(
        collection_name="think9_brain",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    # Reuse existing data instead of deleting and rebuilding
    if collection_is_ready:
        return vector_store

    # Populate only when the collection is empty or new
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]

    # Stable chunk_id values prevent duplicate inserts
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return vector_store


def search_vector_store(vector_store, query, k=3):
    """Find the top k chunks most similar to the query.

    Args:
        vector_store: Chroma store from create_vector_store()
        query: natural-language question or search text
        k: how many results to return

    Returns:
        A list of LangChain Document objects
    """
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(chunks, embedding_model)

    query = "Why was the packaging vendor selected?"
    results = search_vector_store(vector_store, query, k=3)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print("=" * 40)

    for i, doc in enumerate(results, start=1):
        meta = doc.metadata
        # Strip BOM so Windows consoles can print the text safely
        text = doc.page_content.lstrip("\ufeff")
        print(f"Result {i}")
        print(f"text: {text}")
        print(f"source: {meta.get('source')}")
        print(f"brand: {meta.get('brand')}")
        print(f"document_type: {meta.get('document_type')}")
        print(f"chunk_index: {meta.get('chunk_index')}")
        print(f"chunk_id: {meta.get('chunk_id')}")
        print("-" * 40)